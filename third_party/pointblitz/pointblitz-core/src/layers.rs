//! SkyLens additions: everything a map-style scene needs next to the points, drawn into one
//! caller-owned pass so points, meshes and lines share a depth buffer.
//!
//! Frame recipe (all layers use the same camera-relative convention as the points: positions are
//! absolute ENU metres, the GPU sees `position - camera.eye`):
//!
//! ```ignore
//! let style = FrameStyle { clear: [0.039, 0.067, 0.157, 1.0], fog: Some(Fog { .. }) };
//! points.prepare(device, queue, size, &camera, &[&scene], style.fog);
//! meshes.prepare(queue, size, &camera, style.fog);
//! lines.prepare(queue, size, &camera, style.fog);
//! {
//!     let mut pass = begin_pass(encoder, color, depth, &style);
//!     meshes.draw_opaque(&mut pass);
//!     points.draw(&mut pass, &[&scene], None);
//!     meshes.draw_transparent(&mut pass);
//!     lines.draw(&mut pass);
//! }
//! ```
//!
//! Depth is reversed-Z (`Depth32Float`, cleared to 0, compare `Greater`), exactly like the point
//! pipeline. The signatures in this file are the contract the SkyLens screens code against.
//!
//! Colours: every colour given to this module (tints, light colours, emissive, vertex colours,
//! texture texels) is display-encoded sRGB, i.e. CSS hex / 255, like the point colours. Meshes and
//! lines decode to linear, light and blend like three.js (`MeshLambertMaterial`,
//! `MeshBasicMaterial`, `LineBasicMaterial`), and encode again when the colour target is not an
//! `*Srgb` format. `FrameStyle::clear` and `Fog::color` are values for the colour target as is
//! (what a clear colour means for that format), so fog fades exactly into the background. Points
//! keep the original behaviour: their bytes are written unchanged.
//!
//! WebGL2: uniforms are read only in vertex stages, no storage buffers, every uniform binding is a
//! single small slot addressed with dynamic offsets aligned to
//! `min_uniform_buffer_offset_alignment`, textures are shrunk to `max_texture_dimension_2d`.

pub(crate) mod math;

use crate::camera::Camera;
use crate::renderer::DEPTH_FORMAT;
use bytemuck::Zeroable;
use glam::DVec3;
use math::{
    Candidate, FrameUniform, LineObject, MeshObject, Planned, frame_uniform, line_pipeline,
    mesh_object, mesh_pipeline, pack_lines, pack_mesh, pack_slots, plan, slot_stride,
};
use std::collections::BTreeMap;
use wgpu::util::DeviceExt;

/// Linear distance fog, applied by every layer (like three.js `Fog`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fog {
    pub color: [f32; 3],
    /// Metres from the eye where fog starts and where it is total.
    pub near: f32,
    pub far: f32,
}

/// Per-frame look shared by all layers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameStyle {
    /// Background colour, the value written to the colour target as is (see the module docs).
    pub clear: [f64; 4],
    pub fog: Option<Fog>,
}

/// Opens the one render pass of a frame: clears colour to `style.clear`, depth to 0, and KEEPS
/// depth stored so later passes (e.g. overlays) may test against it.
pub fn begin_pass<'e>(
    encoder: &'e mut wgpu::CommandEncoder,
    color: &wgpu::TextureView,
    depth: &wgpu::TextureView,
    style: &FrameStyle,
) -> wgpu::RenderPass<'e> {
    let [r, g, b, a] = style.clear;
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("pointblitz frame"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: color,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color { r, g, b, a }),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth,
            depth_ops: Some(wgpu::Operations {
                // Reversed-Z: 0 is infinitely far.
                load: wgpu::LoadOp::Clear(0.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        ..Default::default()
    })
}

// ---------------------------------------------------------------------------------------------
// Shared GPU plumbing
// ---------------------------------------------------------------------------------------------

fn uniform_layout(device: &wgpu::Device, label: &str, dynamic: bool) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            // Vertex only: the fragment stages get what they need through varyings.
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: dynamic,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

/// The per-frame uniform (camera, fog, output encoding) of one layer renderer.
struct FrameSlot {
    layout: wgpu::BindGroupLayout,
    buf: wgpu::Buffer,
    bg: wgpu::BindGroup,
    encode_srgb: bool,
}

impl FrameSlot {
    fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let layout = uniform_layout(device, "layer frame", false);
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("layer frame"),
            size: size_of::<FrameUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("layer frame"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            }],
        });
        Self {
            layout,
            buf,
            bg,
            // An sRGB target encodes in hardware; anything else gets display values from the shader.
            encode_srgb: !color_format.is_srgb(),
        }
    }

    fn write(&self, queue: &wgpu::Queue, size: [u32; 2], camera: &Camera, fog: Option<Fog>) {
        let aspect = f64::from(size[0].max(1)) / f64::from(size[1].max(1));
        let u = frame_uniform(
            camera.view_proj(aspect).to_cols_array_2d(),
            fog,
            self.encode_srgb,
        );
        queue.write_buffer(&self.buf, 0, bytemuck::bytes_of(&u));
    }
}

/// Per-object uniform slots in one buffer, bound with a dynamic offset. Each binding is one slot,
/// far below WebGL2's 16 KiB `max_uniform_buffer_binding_size`.
struct Slots {
    layout: wgpu::BindGroupLayout,
    buf: wgpu::Buffer,
    bg: wgpu::BindGroup,
    capacity: usize,
    stride: u64,
    size: u64,
}

impl Slots {
    fn new(device: &wgpu::Device, label: &str, size: u64) -> Self {
        let layout = uniform_layout(device, label, true);
        let align = u64::from(device.limits().min_uniform_buffer_offset_alignment);
        let stride = slot_stride(size, align);
        let capacity = 16;
        let (buf, bg) = Self::alloc(device, &layout, stride, size, capacity);
        Self {
            layout,
            buf,
            bg,
            capacity,
            stride,
            size,
        }
    }

    fn alloc(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        stride: u64,
        size: u64,
        capacity: usize,
    ) -> (wgpu::Buffer, wgpu::BindGroup) {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("layer objects"),
            size: stride * capacity as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("layer objects"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buf,
                    offset: 0,
                    size: wgpu::BufferSize::new(size),
                }),
            }],
        });
        (buf, bg)
    }

    /// Makes room for `n` slots. Returns true when the buffer was replaced (its contents are gone,
    /// so whatever `prepare` planned must not be drawn).
    fn reserve(&mut self, device: &wgpu::Device, n: usize) -> bool {
        if n <= self.capacity {
            return false;
        }
        self.capacity = n.next_power_of_two();
        (self.buf, self.bg) =
            Self::alloc(device, &self.layout, self.stride, self.size, self.capacity);
        true
    }

    fn write<T: bytemuck::Pod>(&self, queue: &wgpu::Queue, items: &[T]) {
        if !items.is_empty() {
            queue.write_buffer(&self.buf, 0, &pack_slots(items, self.stride as usize));
        }
    }

    fn offset(&self, slot: u32) -> u32 {
        (u64::from(slot) * self.stride) as u32
    }
}

fn blend_state(blend: Blend) -> Option<wgpu::BlendState> {
    use wgpu::{BlendComponent, BlendFactor as F, BlendOperation::Add};
    match blend {
        Blend::Opaque => None,
        // three.js NormalBlending (non-premultiplied).
        Blend::Alpha => Some(wgpu::BlendState {
            color: BlendComponent {
                src_factor: F::SrcAlpha,
                dst_factor: F::OneMinusSrcAlpha,
                operation: Add,
            },
            alpha: BlendComponent {
                src_factor: F::One,
                dst_factor: F::OneMinusSrcAlpha,
                operation: Add,
            },
        }),
        // three.js AdditiveBlending (non-premultiplied): colour * alpha added; target alpha kept.
        Blend::Additive => Some(wgpu::BlendState {
            color: BlendComponent {
                src_factor: F::SrcAlpha,
                dst_factor: F::One,
                operation: Add,
            },
            alpha: BlendComponent {
                src_factor: F::Zero,
                dst_factor: F::One,
                operation: Add,
            },
        }),
    }
}

/// Reversed-Z depth state; blended geometry tests but does not write.
fn depth_state(test: bool, write: bool) -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: Some(write),
        depth_compare: Some(if test {
            wgpu::CompareFunction::Greater
        } else {
            wgpu::CompareFunction::Always
        }),
        stencil: Default::default(),
        bias: Default::default(),
    }
}

fn shader(device: &wgpu::Device, label: &str, body: &str) -> wgpu::ShaderModule {
    let source = format!("{}\n{}", include_str!("color.wgsl"), body);
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}

/// Non-empty contents for a buffer (an empty slice cannot be bound).
fn non_empty<'a>(bytes: &'a [u8], fallback: &'a [u8]) -> &'a [u8] {
    if bytes.is_empty() { fallback } else { bytes }
}

// ---------------------------------------------------------------------------------------------
// Meshes
// ---------------------------------------------------------------------------------------------

/// How a mesh is lit. Mirrors the three.js materials the SkyLens tower used.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Shading {
    /// Colour (vertex colour x texture x tint) as is.
    Unlit,
    /// `MeshLambertMaterial` under a hemisphere light plus one directional sun, with emissive.
    ///
    /// Same numbers as three.js (r155+ physical units, so `diffuse / pi` per light): `sky` and
    /// `ground` are the `HemisphereLight` colours with +Z (ENU up) as its up axis, `sun_dir` points
    /// from the scene towards the sun (three.js `light.position - target.position`).
    Lambert {
        sky: [f32; 3],
        ground: [f32; 3],
        hemi_intensity: f32,
        sun_dir: [f32; 3],
        sun_color: [f32; 3],
        sun_intensity: f32,
        emissive: [f32; 3],
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Blend {
    Opaque,
    /// Alpha blended, depth tested, no depth write.
    Alpha,
    /// Additive (`AdditiveBlending`), depth tested, no depth write.
    Additive,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Material {
    pub shading: Shading,
    /// Multiplied with vertex colours and the texture. Alpha is the opacity.
    pub tint: [f32; 4],
    pub blend: Blend,
    /// Draw back faces too.
    pub double_sided: bool,
}

/// RGBA8 sRGB image, row-major, top row first.
#[derive(Debug, Clone)]
pub struct TextureImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// One mesh in absolute ENU metres. Missing normals are computed (smooth when `flat == false`,
/// per-face when `flat == true`).
#[derive(Debug, Clone, Default)]
pub struct MeshData {
    pub positions: Vec<[f64; 3]>,
    pub normals: Option<Vec<[f32; 3]>>,
    /// RGBA8 per vertex; white when absent.
    pub colors: Option<Vec<[u8; 4]>>,
    /// Texture coordinates; required when a texture is given.
    pub uvs: Option<Vec<[f32; 2]>>,
    pub indices: Vec<u32>,
    pub flat: bool,
}

struct GpuMesh {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    /// Absolute ENU origin the vertices are relative to (bounding-box centre).
    origin: DVec3,
    /// The texture is kept with its bind group so both live exactly as long as the mesh.
    texture: Option<(wgpu::Texture, wgpu::BindGroup)>,
    material: Material,
    visible: bool,
    /// Bumped by every `upsert`, so a draw list made before a replacement skips the new geometry
    /// until the next `prepare` has written its uniform.
    version: u64,
}

/// Owns GPU meshes by caller-chosen id.
pub struct MeshRenderer {
    /// Indexed by `math::mesh_pipeline(blend, double_sided)`.
    pipelines: Vec<wgpu::RenderPipeline>,
    frame: FrameSlot,
    slots: Slots,
    texture_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// 1x1 white texture for untextured meshes (one shader for both). Made by the first `upsert`,
    /// because `new` has no queue to upload it with.
    white: Option<(wgpu::Texture, wgpu::BindGroup)>,
    meshes: BTreeMap<u64, GpuMesh>,
    next_version: u64,
    opaque: Vec<Planned>,
    transparent: Vec<Planned>,
}

impl MeshRenderer {
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let module = shader(device, "pointblitz meshes", include_str!("mesh.wgsl"));
        let frame = FrameSlot::new(device, color_format);
        let slots = Slots::new(device, "mesh objects", size_of::<MeshObject>() as u64);
        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mesh texture"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("meshes"),
            bind_group_layouts: &[
                Some(&frame.layout),
                Some(&slots.layout),
                Some(&texture_layout),
            ],
            immediate_size: 0,
        });
        let attributes = wgpu::vertex_attr_array![
            0 => Float32x3,
            1 => Float32x3,
            2 => Float32x2,
            3 => Unorm8x4,
        ];
        let mut pipelines = Vec::with_capacity(6);
        for blend in [Blend::Opaque, Blend::Alpha, Blend::Additive] {
            for double_sided in [false, true] {
                debug_assert_eq!(pipelines.len(), mesh_pipeline(blend, double_sided));
                pipelines.push(
                    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: Some("meshes"),
                        layout: Some(&layout),
                        vertex: wgpu::VertexState {
                            module: &module,
                            entry_point: Some("vs"),
                            compilation_options: Default::default(),
                            buffers: &[Some(wgpu::VertexBufferLayout {
                                array_stride: size_of::<math::MeshVertex>() as u64,
                                step_mode: wgpu::VertexStepMode::Vertex,
                                attributes: &attributes,
                            })],
                        },
                        primitive: wgpu::PrimitiveState {
                            topology: wgpu::PrimitiveTopology::TriangleList,
                            front_face: wgpu::FrontFace::Ccw,
                            cull_mode: if double_sided {
                                None
                            } else {
                                Some(wgpu::Face::Back)
                            },
                            ..Default::default()
                        },
                        depth_stencil: Some(depth_state(true, blend == Blend::Opaque)),
                        multisample: Default::default(),
                        fragment: Some(wgpu::FragmentState {
                            module: &module,
                            entry_point: Some("fs"),
                            compilation_options: Default::default(),
                            targets: &[Some(wgpu::ColorTargetState {
                                format: color_format,
                                blend: blend_state(blend),
                                write_mask: wgpu::ColorWrites::ALL,
                            })],
                        }),
                        multiview_mask: None,
                        cache: None,
                    }),
                );
            }
        }
        // three.js textures: linear filtering with mipmaps; clamp so drapes do not bleed at edges.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mesh texture"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        Self {
            pipelines,
            frame,
            slots,
            texture_layout,
            sampler,
            white: None,
            meshes: BTreeMap::new(),
            next_version: 0,
            opaque: Vec::new(),
            transparent: Vec::new(),
        }
    }

    /// Inserts or replaces mesh `id`. The texture, if any, is uploaded once and owned by the mesh.
    pub fn upsert(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: u64,
        mesh: &MeshData,
        texture: Option<&TextureImage>,
        material: Material,
    ) {
        assert!(
            texture.is_none() || mesh.uvs.is_some(),
            "a textured mesh needs uvs"
        );
        let packed = pack_mesh(mesh);
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh vertices"),
            contents: non_empty(bytemuck::cast_slice(&packed.vertices), &[0; 36]),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh indices"),
            contents: non_empty(bytemuck::cast_slice(&packed.indices), &[0; 4]),
            usage: wgpu::BufferUsages::INDEX,
        });
        let texture = texture.map(|img| self.upload_texture(device, queue, img));
        if self.white.is_none() {
            let white = math::MipChain {
                width: 1,
                height: 1,
                levels: 1,
                data: vec![255; 4],
            };
            self.white = Some(self.texture_from_chain(device, queue, &white));
        }
        self.next_version += 1;
        // A replaced mesh keeps its visibility (e.g. a hidden layer being refreshed stays hidden).
        let visible = self.meshes.get(&id).is_none_or(|m| m.visible);
        self.meshes.insert(
            id,
            GpuMesh {
                vertices,
                indices,
                index_count: packed.indices.len() as u32,
                origin: packed.origin,
                texture,
                material,
                visible,
                version: self.next_version,
            },
        );
        if self.slots.reserve(device, self.meshes.len()) {
            self.opaque.clear();
            self.transparent.clear();
        }
    }

    fn upload_texture(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        img: &TextureImage,
    ) -> (wgpu::Texture, wgpu::BindGroup) {
        let chain = math::mip_chain(img, device.limits().max_texture_dimension_2d);
        self.texture_from_chain(device, queue, &chain)
    }

    fn texture_from_chain(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        chain: &math::MipChain,
    ) -> (wgpu::Texture, wgpu::BindGroup) {
        let texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("mesh texture"),
                size: wgpu::Extent3d {
                    width: chain.width,
                    height: chain.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: chain.levels,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                // Sampling decodes sRGB, so the shader filters and lights in linear.
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &chain.data,
        );
        let view = texture.create_view(&Default::default());
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh texture"),
            layout: &self.texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        (texture, bg)
    }

    /// Changes only the material (e.g. a display-mode switch) without re-uploading geometry.
    pub fn set_material(&mut self, id: u64, material: Material) {
        if let Some(m) = self.meshes.get_mut(&id) {
            m.material = material;
        }
    }

    pub fn set_visible(&mut self, id: u64, visible: bool) {
        if let Some(m) = self.meshes.get_mut(&id) {
            m.visible = visible;
        }
    }

    pub fn remove(&mut self, id: u64) {
        self.meshes.remove(&id);
    }

    pub fn clear(&mut self) {
        self.meshes.clear();
        self.opaque.clear();
        self.transparent.clear();
    }

    pub fn len(&self) -> usize {
        self.meshes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Uploads camera, light and fog uniforms for this frame.
    pub fn prepare(
        &mut self,
        queue: &wgpu::Queue,
        size: [u32; 2],
        camera: &Camera,
        fog: Option<Fog>,
    ) {
        self.frame.write(queue, size, camera, fog);
        let candidates: Vec<Candidate> = self
            .meshes
            .iter()
            .map(|(&id, m)| Candidate {
                id,
                version: m.version,
                visible: m.visible && m.index_count > 0,
                blend: m.material.blend,
                pipeline: mesh_pipeline(m.material.blend, m.material.double_sided),
                distance: (m.origin - camera.eye).length(),
            })
            .collect();
        let (opaque, transparent) = plan(&candidates);
        let mut objects = vec![MeshObject::zeroed(); opaque.len() + transparent.len()];
        for p in opaque.iter().chain(&transparent) {
            let m = &self.meshes[&p.id];
            objects[p.slot as usize] = mesh_object(&m.material, m.origin - camera.eye);
        }
        self.slots.write(queue, &objects);
        self.opaque = opaque;
        self.transparent = transparent;
    }

    /// Draws `Blend::Opaque` meshes.
    pub fn draw_opaque(&self, pass: &mut wgpu::RenderPass<'_>) {
        self.draw_list(pass, &self.opaque);
    }

    /// Draws `Blend::Alpha` and `Blend::Additive` meshes (after opaque geometry and points).
    pub fn draw_transparent(&self, pass: &mut wgpu::RenderPass<'_>) {
        self.draw_list(pass, &self.transparent);
    }

    fn draw_list(&self, pass: &mut wgpu::RenderPass<'_>, list: &[Planned]) {
        if list.is_empty() {
            return;
        }
        pass.set_bind_group(0, &self.frame.bg, &[]);
        let mut bound = None;
        for p in list {
            let Some(m) = self.meshes.get(&p.id) else {
                continue;
            };
            if m.version != p.version || !m.visible || m.index_count == 0 {
                continue;
            }
            if bound != Some(p.pipeline) {
                pass.set_pipeline(&self.pipelines[p.pipeline]);
                bound = Some(p.pipeline);
            }
            pass.set_bind_group(1, &self.slots.bg, &[self.slots.offset(p.slot)]);
            let Some((_, texture)) = m.texture.as_ref().or(self.white.as_ref()) else {
                continue;
            };
            pass.set_bind_group(2, texture, &[]);
            pass.set_vertex_buffer(0, m.vertices.slice(..));
            pass.set_index_buffer(m.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..m.index_count, 0, 0..1);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Lines
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Topology {
    /// Pairs of points.
    List,
    /// One connected polyline.
    Strip,
}

/// A set of 1-pixel lines (like three.js `LineBasicMaterial` / `LineSegments`).
#[derive(Debug, Clone)]
pub struct LineSet {
    pub positions: Vec<[f64; 3]>,
    pub topology: Topology,
    /// Per-vertex RGBA8; overrides `color` when present (used for fading trails).
    pub colors: Option<Vec<[u8; 4]>>,
    pub color: [f32; 4],
    pub blend: Blend,
    /// `false` draws on top of everything (no depth test).
    pub depth_test: bool,
}

struct GpuLines {
    buffer: wgpu::Buffer,
    draw_count: u32,
    origin: DVec3,
    tint: [f32; 4],
    blend: Blend,
    depth_test: bool,
    topology: Topology,
    visible: bool,
    version: u64,
}

/// Owns GPU line sets by caller-chosen id.
pub struct LineRenderer {
    /// Indexed by `math::line_pipeline(blend, depth_test, topology)`.
    pipelines: Vec<wgpu::RenderPipeline>,
    frame: FrameSlot,
    slots: Slots,
    sets: BTreeMap<u64, GpuLines>,
    next_version: u64,
    /// Depth-tested sets (opaque, then blended far to near), then sets drawn on top.
    planned: Vec<Planned>,
}

impl LineRenderer {
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let module = shader(device, "pointblitz lines", include_str!("lines.wgsl"));
        let frame = FrameSlot::new(device, color_format);
        let slots = Slots::new(device, "line objects", size_of::<LineObject>() as u64);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lines"),
            bind_group_layouts: &[Some(&frame.layout), Some(&slots.layout)],
            immediate_size: 0,
        });
        let attributes = wgpu::vertex_attr_array![0 => Float32x3, 1 => Unorm8x4];
        let mut pipelines = Vec::with_capacity(12);
        for blend in [Blend::Opaque, Blend::Alpha, Blend::Additive] {
            for depth_test in [false, true] {
                for topology in [Topology::List, Topology::Strip] {
                    debug_assert_eq!(pipelines.len(), line_pipeline(blend, depth_test, topology));
                    pipelines.push(device.create_render_pipeline(
                        &wgpu::RenderPipelineDescriptor {
                            label: Some("lines"),
                            layout: Some(&layout),
                            vertex: wgpu::VertexState {
                                module: &module,
                                entry_point: Some("vs"),
                                compilation_options: Default::default(),
                                buffers: &[Some(wgpu::VertexBufferLayout {
                                    array_stride: size_of::<math::LineVertex>() as u64,
                                    step_mode: wgpu::VertexStepMode::Vertex,
                                    attributes: &attributes,
                                })],
                            },
                            primitive: wgpu::PrimitiveState {
                                topology: match topology {
                                    Topology::List => wgpu::PrimitiveTopology::LineList,
                                    Topology::Strip => wgpu::PrimitiveTopology::LineStrip,
                                },
                                ..Default::default()
                            },
                            depth_stencil: Some(depth_state(
                                depth_test,
                                depth_test && blend == Blend::Opaque,
                            )),
                            multisample: Default::default(),
                            fragment: Some(wgpu::FragmentState {
                                module: &module,
                                entry_point: Some("fs"),
                                compilation_options: Default::default(),
                                targets: &[Some(wgpu::ColorTargetState {
                                    format: color_format,
                                    blend: blend_state(blend),
                                    write_mask: wgpu::ColorWrites::ALL,
                                })],
                            }),
                            multiview_mask: None,
                            cache: None,
                        },
                    ));
                }
            }
        }
        Self {
            pipelines,
            frame,
            slots,
            sets: BTreeMap::new(),
            next_version: 0,
            planned: Vec::new(),
        }
    }

    pub fn upsert(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, id: u64, lines: &LineSet) {
        let _ = queue;
        let packed = pack_lines(lines);
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("line vertices"),
            contents: non_empty(bytemuck::cast_slice(&packed.vertices), &[0; 16]),
            usage: wgpu::BufferUsages::VERTEX,
        });
        self.next_version += 1;
        let visible = self.sets.get(&id).is_none_or(|s| s.visible);
        self.sets.insert(
            id,
            GpuLines {
                buffer,
                draw_count: packed.draw_count,
                origin: packed.origin,
                tint: packed.tint,
                blend: lines.blend,
                depth_test: lines.depth_test,
                topology: lines.topology,
                visible,
                version: self.next_version,
            },
        );
        if self.slots.reserve(device, self.sets.len()) {
            self.planned.clear();
        }
    }

    /// Hides or shows line set `id` without re-uploading it (SkyLens helper).
    pub fn set_visible(&mut self, id: u64, visible: bool) {
        if let Some(s) = self.sets.get_mut(&id) {
            s.visible = visible;
        }
    }

    pub fn remove(&mut self, id: u64) {
        self.sets.remove(&id);
    }

    pub fn clear(&mut self) {
        self.sets.clear();
        self.planned.clear();
    }

    /// Number of line sets held (SkyLens helper).
    pub fn len(&self) -> usize {
        self.sets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sets.is_empty()
    }

    pub fn prepare(
        &mut self,
        queue: &wgpu::Queue,
        size: [u32; 2],
        camera: &Camera,
        fog: Option<Fog>,
    ) {
        self.frame.write(queue, size, camera, fog);
        let candidates: Vec<Candidate> = self
            .sets
            .iter()
            .map(|(&id, s)| Candidate {
                id,
                version: s.version,
                visible: s.visible && s.draw_count > 0,
                blend: s.blend,
                pipeline: line_pipeline(s.blend, s.depth_test, s.topology),
                distance: (s.origin - camera.eye).length(),
            })
            .collect();
        let (opaque, blended) = plan(&candidates);
        let mut objects = vec![LineObject::zeroed(); opaque.len() + blended.len()];
        let mut planned: Vec<Planned> = opaque.into_iter().chain(blended).collect();
        for p in &planned {
            let s = &self.sets[&p.id];
            let d = (s.origin - camera.eye).as_vec3();
            objects[p.slot as usize] = LineObject {
                offset: [d.x, d.y, d.z, 0.0],
                tint: s.tint,
            };
        }
        // Sets without depth test go last so they really are on top (stable: order kept).
        planned.sort_by_key(|p| !self.sets[&p.id].depth_test);
        self.slots.write(queue, &objects);
        self.planned = planned;
    }

    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        if self.planned.is_empty() {
            return;
        }
        pass.set_bind_group(0, &self.frame.bg, &[]);
        let mut bound = None;
        for p in &self.planned {
            let Some(s) = self.sets.get(&p.id) else {
                continue;
            };
            if s.version != p.version || !s.visible || s.draw_count == 0 {
                continue;
            }
            if bound != Some(p.pipeline) {
                pass.set_pipeline(&self.pipelines[p.pipeline]);
                bound = Some(p.pipeline);
            }
            pass.set_bind_group(1, &self.slots.bg, &[self.slots.offset(p.slot)]);
            pass.set_vertex_buffer(0, s.buffer.slice(..));
            pass.draw(0..s.draw_count, 0..1);
        }
    }
}
