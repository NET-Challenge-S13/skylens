//! The point pipeline: one draw per chunk, instanced quads, reversed-Z depth, no sorting
//! (decisions 0006, 0007, 0013).

use crate::camera::Camera;
use crate::scene::Scene;
use bytemuck::{Pod, Zeroable};

/// Depth format used by every PointBlitz target.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    viewport_px: [f32; 2],
    point_px: f32,
    _pad: f32,
}

/// Per-chunk uniform slot; dynamic offsets must be multiples of the device alignment.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ChunkUniform {
    offset: [f32; 4],
    scale: [f32; 4],
}

pub struct Renderer {
    /// Point pipelines for strides 12 and 8 (decision 0051).
    pipelines: [wgpu::RenderPipeline; 2],
    camera_buf: wgpu::Buffer,
    camera_bg: wgpu::BindGroup,
    chunk_layout: wgpu::BindGroupLayout,
    chunk_buf: wgpu::Buffer,
    chunk_bg: wgpu::BindGroup,
    chunk_capacity: usize,
    chunk_stride: u64,
    /// Point diameter in pixels (decision 0007: 2 px).
    pub point_px: f32,
    /// GPU timestamps for the next `render` calls' pass (decision 0049): begin at `index`, end at
    /// `index + 1`. Needs `Features::TIMESTAMP_QUERY`.
    timestamps: Option<(wgpu::QuerySet, u32)>,
}

impl Renderer {
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pointblitz points"),
            source: wgpu::ShaderSource::Wgsl(include_str!("points.wgsl").into()),
        });
        let uniform_entry = |dynamic: bool| wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: dynamic,
                min_binding_size: None,
            },
            count: None,
        };
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera"),
            entries: &[uniform_entry(false)],
        });
        let chunk_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("chunk"),
            entries: &[uniform_entry(true)],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("points"),
            bind_group_layouts: &[Some(&camera_layout), Some(&chunk_layout)],
            immediate_size: 0,
        });
        // One pipeline per point stride of chunk format v2 (decision 0051).
        let attrs12 = wgpu::vertex_attr_array![0 => Uint16x4, 1 => Unorm8x4];
        let attrs8 = wgpu::vertex_attr_array![0 => Uint16x4];
        let pipeline = |stride: u64, entry: &str, attributes: &[wgpu::VertexAttribute]| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("points"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: stride,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes,
                    })],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    // Reversed-Z: nearer is larger.
                    depth_compare: Some(wgpu::CompareFunction::Greater),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let pipelines = [pipeline(12, "vs12", &attrs12), pipeline(8, "vs8", &attrs8)];

        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera"),
            size: size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera"),
            layout: &camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buf.as_entire_binding(),
            }],
        });
        let align = u64::from(device.limits().min_uniform_buffer_offset_alignment);
        let chunk_stride = (size_of::<ChunkUniform>() as u64).div_ceil(align) * align;
        let (chunk_buf, chunk_bg) = Self::chunk_slots(device, &chunk_layout, chunk_stride, 64);

        Self {
            pipelines,
            camera_buf,
            camera_bg,
            chunk_layout,
            chunk_buf,
            chunk_bg,
            chunk_capacity: 64,
            chunk_stride,
            point_px: 2.0,
            timestamps: None,
        }
    }

    fn chunk_slots(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        stride: u64,
        capacity: usize,
    ) -> (wgpu::Buffer, wgpu::BindGroup) {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk offsets"),
            size: stride * capacity as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("chunk offsets"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buf,
                    offset: 0,
                    size: wgpu::BufferSize::new(size_of::<ChunkUniform>() as u64),
                }),
            }],
        });
        (buf, bg)
    }

    /// Makes the next `render` calls write the pass's GPU start and end time into `set` at `index` and
    /// `index + 1` (decision 0049); `None` stops it. The caller resolves the set.
    pub fn set_timestamp_writes(&mut self, writes: Option<(wgpu::QuerySet, u32)>) {
        self.timestamps = writes;
    }

    /// Records one frame: clears colour to black and depth to 0, then draws every visible chunk.
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        color: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        size: [u32; 2],
        camera: &Camera,
        scene: &Scene,
    ) {
        let aspect = f64::from(size[0]) / f64::from(size[1]);
        let cam = CameraUniform {
            view_proj: camera.view_proj(aspect).to_cols_array_2d(),
            viewport_px: [size[0] as f32, size[1] as f32],
            point_px: self.point_px,
            _pad: 0.0,
        };
        queue.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&cam));

        let chunks = scene.visible_chunks();
        if chunks.len() > self.chunk_capacity {
            self.chunk_capacity = chunks.len().next_power_of_two();
            (self.chunk_buf, self.chunk_bg) = Self::chunk_slots(
                device,
                &self.chunk_layout,
                self.chunk_stride,
                self.chunk_capacity,
            );
        }
        let mut slots = vec![0u8; self.chunk_stride as usize * chunks.len()];
        for (i, c) in chunks.iter().enumerate() {
            let h = &c.header;
            let min = glam::DVec3::from(h.bbox_min.map(f64::from));
            let size = glam::DVec3::from(h.bbox_max.map(f64::from)) - min;
            let d = c.origin + min - camera.eye;
            let s = size / 65535.0;
            let u = ChunkUniform {
                offset: [d.x as f32, d.y as f32, d.z as f32, 0.0],
                scale: [s.x as f32, s.y as f32, s.z as f32, 0.0],
            };
            let at = i * self.chunk_stride as usize;
            slots[at..at + size_of::<ChunkUniform>()].copy_from_slice(bytemuck::bytes_of(&u));
        }
        if !slots.is_empty() {
            queue.write_buffer(&self.chunk_buf, 0, &slots);
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("points"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.0),
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: self.timestamps.as_ref().map(|(set, index)| {
                wgpu::RenderPassTimestampWrites {
                    query_set: set,
                    beginning_of_pass_write_index: Some(*index),
                    end_of_pass_write_index: Some(*index + 1),
                }
            }),
            ..Default::default()
        });
        pass.set_bind_group(0, &self.camera_bg, &[]);
        let mut bound = None;
        for (i, c) in chunks.iter().enumerate() {
            let p = usize::from(c.header.stride != 12);
            if bound != Some(p) {
                pass.set_pipeline(&self.pipelines[p]);
                bound = Some(p);
            }
            pass.set_bind_group(1, &self.chunk_bg, &[(i as u64 * self.chunk_stride) as u32]);
            pass.set_vertex_buffer(0, c.buffer.slice(..));
            pass.draw(0..4, 0..c.points);
        }
    }

    // ---- SkyLens additions (see layers.rs) --------------------------------------------------

    /// Point size: fixed pixels (the original behaviour) or world metres with perspective
    /// attenuation (three.js `PointsMaterial { sizeAttenuation: true }`).
    pub fn set_point_size(&mut self, size: PointSize) {
        let _ = size;
        todo!("pointblitz layers")
    }

    /// Uploads camera, fog and per-chunk uniforms for every scene drawn this frame. Several
    /// independent scenes (e.g. one per reconstruction segment) share one frame.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        size: [u32; 2],
        camera: &Camera,
        scenes: &[&Scene],
        fog: Option<crate::layers::Fog>,
    ) {
        let _ = (device, queue, size, camera, scenes, fog);
        todo!("pointblitz layers")
    }

    /// Draws the scenes given to the last `prepare`, in the same order, into a caller-owned pass
    /// (see `layers::begin_pass`). `alpha` scales every point's opacity per scene (reveal fades);
    /// `None` means fully opaque.
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>, scenes: &[&Scene], alpha: Option<&[f32]>) {
        let _ = (pass, scenes, alpha);
        todo!("pointblitz layers")
    }
}

/// See `Renderer::set_point_size`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PointSize {
    Pixels(f32),
    Metres(f32),
}
