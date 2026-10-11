//! The point pipeline: one draw per chunk, instanced quads, reversed-Z depth, no sorting
//! (decisions 0006, 0007, 0013).

use crate::camera::Camera;
use crate::layers::math::fog_params;
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
    /// World diameter in metres; 0 keeps the fixed pixel size (SkyLens).
    point_m: f32,
    /// SkyLens fog: rgb in the target's space, a = 1 when on.
    fog_color: [f32; 4],
    /// x, y: fog scale and bias; z: focal length in pixels; w: unused.
    params: [f32; 4],
}

/// Per-chunk uniform slot; dynamic offsets must be multiples of the device alignment.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ChunkUniform {
    offset: [f32; 4],
    scale: [f32; 4],
}

pub struct Renderer {
    /// Point pipelines for strides 12 and 8 (decision 0051): opaque, then faded (SkyLens: per-scene
    /// alpha through the blend constant).
    pipelines: [wgpu::RenderPipeline; 4],
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
    /// World point diameter in metres (`PointSize::Metres`); 0 means `point_px` is used.
    point_m: f32,
    /// Per scene given to the last `prepare`/`render`: first chunk slot and chunks written.
    prepared: Vec<(usize, usize)>,
}

impl Renderer {
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let source = format!("{}\n{}", include_str!("color.wgsl"), include_str!("points.wgsl"));
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pointblitz points"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
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
        // Faded: out = src * a + dst * (1 - a) with `a` the pass's blend constant, so one pipeline
        // serves every per-scene alpha without a buffer write between draws.
        let faded = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::Constant,
            dst_factor: wgpu::BlendFactor::OneMinusConstant,
            operation: wgpu::BlendOperation::Add,
        };
        let faded = Some(wgpu::BlendState {
            color: faded,
            alpha: faded,
        });
        // One pipeline per point stride of chunk format v2 (decision 0051).
        let attrs12 = wgpu::vertex_attr_array![0 => Uint16x4, 1 => Unorm8x4];
        let attrs8 = wgpu::vertex_attr_array![0 => Uint16x4];
        let pipeline = |stride: u64,
                        entry: &str,
                        attributes: &[wgpu::VertexAttribute],
                        blend: Option<wgpu::BlendState>| {
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
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let pipelines = [
            pipeline(12, "vs12", &attrs12, None),
            pipeline(8, "vs8", &attrs8, None),
            pipeline(12, "vs12", &attrs12, faded),
            pipeline(8, "vs8", &attrs8, faded),
        ];

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
            point_m: 0.0,
            prepared: Vec::new(),
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
        self.write_frame(device, queue, size, camera, &[scene], None);

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
        self.draw(&mut pass, &[scene], None);
    }

    /// Writes the camera uniform and one chunk slot per visible chunk of every scene, scenes in
    /// order, and remembers each scene's slot range for `draw`.
    fn write_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        size: [u32; 2],
        camera: &Camera,
        scenes: &[&Scene],
        fog: Option<crate::layers::Fog>,
    ) {
        let (w, h) = (size[0].max(1), size[1].max(1));
        let aspect = f64::from(w) / f64::from(h);
        let f = fog_params(fog);
        let cam = CameraUniform {
            view_proj: camera.view_proj(aspect).to_cols_array_2d(),
            viewport_px: [w as f32, h as f32],
            point_px: self.point_px,
            point_m: self.point_m,
            fog_color: f.color,
            params: [f.scale, f.bias, focal_px(camera.fov_y_deg, h) as f32, 0.0],
        };
        queue.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&cam));

        let counts: Vec<usize> = scenes.iter().map(|s| s.visible_chunks().len()).collect();
        self.prepared = slot_ranges(&counts);
        let total: usize = counts.iter().sum();
        if total > self.chunk_capacity {
            self.chunk_capacity = total.next_power_of_two();
            (self.chunk_buf, self.chunk_bg) = Self::chunk_slots(
                device,
                &self.chunk_layout,
                self.chunk_stride,
                self.chunk_capacity,
            );
        }
        let mut slots = vec![0u8; self.chunk_stride as usize * total];
        let chunks = scenes.iter().flat_map(|s| s.visible_chunks());
        for (i, c) in chunks.enumerate() {
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
    }

    // ---- SkyLens additions (see layers.rs) --------------------------------------------------

    /// Point size: fixed pixels (the original behaviour) or world metres with perspective
    /// attenuation (three.js `PointsMaterial { sizeAttenuation: true }`).
    pub fn set_point_size(&mut self, size: PointSize) {
        match size {
            PointSize::Pixels(px) => {
                self.point_px = px;
                self.point_m = 0.0;
            }
            PointSize::Metres(m) => self.point_m = m.max(0.0),
        }
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
        self.write_frame(device, queue, size, camera, scenes, fog);
    }

    /// Draws the scenes given to the last `prepare`, in the same order, into a caller-owned pass
    /// (see `layers::begin_pass`). `alpha` scales every point's opacity per scene (reveal fades);
    /// `None` means fully opaque.
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>, scenes: &[&Scene], alpha: Option<&[f32]>) {
        pass.set_bind_group(0, &self.camera_bg, &[]);
        let mut bound = None;
        for (s, scene) in scenes.iter().enumerate() {
            // A scene that was not prepared has no camera-relative slots: skip it.
            let Some(&(first, count)) = self.prepared.get(s) else {
                break;
            };
            let faded = match fade(alpha, s) {
                Fade::Skip => continue,
                Fade::Opaque => false,
                Fade::Blend(a) => {
                    let a = f64::from(a);
                    pass.set_blend_constant(wgpu::Color {
                        r: a,
                        g: a,
                        b: a,
                        a,
                    });
                    true
                }
            };
            // Chunks inserted after `prepare` wait for the next frame.
            for (i, c) in scene.visible_chunks().iter().take(count).enumerate() {
                let p = usize::from(c.header.stride != 12) + 2 * usize::from(faded);
                if bound != Some(p) {
                    pass.set_pipeline(&self.pipelines[p]);
                    bound = Some(p);
                }
                let slot = (first + i) as u64 * self.chunk_stride;
                pass.set_bind_group(1, &self.chunk_bg, &[slot as u32]);
                pass.set_vertex_buffer(0, c.buffer.slice(..));
                pass.draw(0..4, 0..c.points);
            }
        }
    }
}

/// See `Renderer::set_point_size`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PointSize {
    Pixels(f32),
    Metres(f32),
}

/// Focal length in pixels for a vertical field of view over `height` pixels.
fn focal_px(fov_y_deg: f64, height: u32) -> f64 {
    f64::from(height) / 2.0 / (fov_y_deg.to_radians() / 2.0).tan()
}

/// CPU mirror of the point shader's diameter for a point at view depth `depth` metres.
#[cfg_attr(not(test), allow(dead_code))]
fn diameter_px(size: PointSize, focal_px: f32, depth: f32) -> f32 {
    match size {
        PointSize::Pixels(px) => px,
        PointSize::Metres(m) if m > 0.0 => (m * focal_px / depth.max(1e-6)).clamp(1.0, 512.0),
        PointSize::Metres(_) => 0.0,
    }
}

/// First slot and count per scene when scenes are laid out back to back.
fn slot_ranges(counts: &[usize]) -> Vec<(usize, usize)> {
    let mut first = 0;
    counts
        .iter()
        .map(|&n| {
            let r = (first, n);
            first += n;
            r
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Fade {
    Skip,
    /// Opaque fast path: no blending.
    Opaque,
    Blend(f32),
}

/// Per-scene alpha: missing or >= 1 is opaque, <= 0 (or NaN) is not drawn.
fn fade(alpha: Option<&[f32]>, scene: usize) -> Fade {
    match alpha.and_then(|a| a.get(scene)).copied() {
        None => Fade::Opaque,
        Some(a) if a >= 1.0 => Fade::Opaque,
        Some(a) if a > 0.0 => Fade::Blend(a),
        Some(_) => Fade::Skip,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_uniform_layout_matches_the_wgsl_struct() {
        // mat4 64 + vec2 8 + f32 4 + f32 4 + vec4 16 + vec4 16.
        assert_eq!(size_of::<CameraUniform>(), 112);
        assert_eq!(size_of::<ChunkUniform>(), 32);
    }

    #[test]
    fn focal_length_matches_the_projection() {
        // 90 degrees over 1000 px: tan(45) = 1, focal = 500 px.
        assert!((focal_px(90.0, 1000) - 500.0).abs() < 1e-9);
        // Agrees with Camera::project_px: a point 1 m right at 10 m depth is focal / 10 px off.
        let cam = Camera::new([0.0, -10.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 50.0);
        let a = cam.project_px([0.0, 0.0, 0.0], 800, 600).unwrap();
        let b = cam.project_px([1.0, 0.0, 0.0], 800, 600).unwrap();
        assert!((b[0] - a[0] - focal_px(50.0, 600) / 10.0).abs() < 1e-6);
    }

    #[test]
    fn point_sizes_in_pixels_and_metres() {
        assert_eq!(diameter_px(PointSize::Pixels(2.0), 500.0, 1000.0), 2.0);
        // 0.5 m at 10 m with a 500 px focal length: 25 px; twice as far, half as big.
        assert_eq!(diameter_px(PointSize::Metres(0.5), 500.0, 10.0), 25.0);
        assert_eq!(diameter_px(PointSize::Metres(0.5), 500.0, 20.0), 12.5);
        // Clamped: never below a pixel, never screen-filling.
        assert_eq!(diameter_px(PointSize::Metres(0.5), 500.0, 1e6), 1.0);
        assert_eq!(diameter_px(PointSize::Metres(0.5), 500.0, 1e-3), 512.0);
    }

    #[test]
    fn scenes_get_back_to_back_slot_ranges() {
        assert_eq!(slot_ranges(&[3, 0, 2]), vec![(0, 3), (3, 0), (3, 2)]);
        assert!(slot_ranges(&[]).is_empty());
    }

    #[test]
    fn per_scene_alpha_picks_opaque_blend_or_skip() {
        assert_eq!(fade(None, 0), Fade::Opaque);
        let a = [1.0, 0.25, 0.0, f32::NAN, 2.0];
        assert_eq!(fade(Some(&a), 0), Fade::Opaque);
        assert_eq!(fade(Some(&a), 1), Fade::Blend(0.25));
        assert_eq!(fade(Some(&a), 2), Fade::Skip);
        assert_eq!(fade(Some(&a), 3), Fade::Skip);
        assert_eq!(fade(Some(&a), 4), Fade::Opaque);
        // Shorter alpha slice: the rest is opaque.
        assert_eq!(fade(Some(&a), 9), Fade::Opaque);
    }
}
