//! Animated point layer: per-point fade-in, fade-out and glide, interpolated on the GPU.
//!
//! Points are addressed by a caller-chosen `u64` key. `apply(adds, moves, removes, now)` changes
//! slots and uploads only the touched ranges; the vertex shader reads `now` from a uniform, so an
//! animation costs nothing per frame. See `book` for the CPU bookkeeping.
//!
//! Frame: `prepare(..)` (uniforms + recycle), then `draw(&mut pass)` inside the pass from
//! [`crate::begin_pass`], next to the other layers. Colours are drawn as given (like [`crate::Renderer`]).

pub mod book;

use crate::camera::Camera;
use crate::layers::Fog;
use crate::layers::math::fog_params;
use crate::renderer::{DEPTH_FORMAT, PointSize, focal_px};
use book::{Add, ApplyStats, Book, Move, REC_SIZE, FADE_IN_S, FADE_OUT_S, GLIDE_S};
use bytemuck::{Pod, Zeroable};

pub use book::{FRAME_BUDGET, PointRec};

/// How points are coloured.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColorMode {
    /// The colour given at add/move time.
    Rgb,
    /// The legacy blue-cyan-green-yellow-red ramp over world height `low..high` (metres, ENU z).
    Height { low: f32, high: f32 },
    /// One colour for every point (display values 0..1).
    Flat([f32; 3]),
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AnimUniform {
    view_proj: [[f32; 4]; 4],
    viewport_px: [f32; 2],
    point_px: f32,
    point_m: f32,
    fog_color: [f32; 4],
    params: [f32; 4],
    origin: [f32; 4],
    color_mode: [f32; 4],
    flat_color: [f32; 4],
    durations: [f32; 4],
}

pub struct AnimatedPoints {
    pipeline: wgpu::RenderPipeline,
    buf: wgpu::Buffer,
    bg: wgpu::BindGroup,
    verts: wgpu::Buffer,
    /// Slots the vertex buffer holds.
    gpu_capacity: usize,
    book: Book,
    point_px: f32,
    point_m: f32,
    color_mode: ColorMode,
}

fn vertex_buffer(device: &wgpu::Device, slots: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("animated points"),
        size: (slots * REC_SIZE) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

impl AnimatedPoints {
    /// `origin`: absolute ENU position that slot coordinates are stored relative to (pick the scene
    /// centre; f32 precision is per layer-origin distance). `capacity`: initial slot count; it grows
    /// by itself when full.
    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        origin: [f64; 3],
        capacity: usize,
    ) -> Self {
        let source = format!("{}\n{}", include_str!("color.wgsl"), include_str!("animated.wgsl"));
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pointblitz animated points"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("animated points"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("animated points"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let attrs = wgpu::vertex_attr_array![
            0 => Float32x3, 1 => Float32x3, 2 => Uint8x4, 3 => Float32
        ];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("animated points"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: REC_SIZE as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &attrs,
                })],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
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
        });
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("animated points uniform"),
            size: size_of::<AnimUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("animated points"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }],
        });
        let max = (device.limits().max_buffer_size / REC_SIZE as u64).min(1 << 27) as usize;
        let book = Book::new(origin, capacity, max);
        let gpu_capacity = book.capacity();
        Self {
            pipeline,
            buf,
            bg,
            verts: vertex_buffer(device, gpu_capacity),
            gpu_capacity,
            book,
            point_px: 2.0,
            point_m: 0.0,
            color_mode: ColorMode::Rgb,
        }
    }

    /// Live points (fading-out ones not counted).
    pub fn len(&self) -> usize {
        self.book.len()
    }
    pub fn is_empty(&self) -> bool {
        self.book.is_empty()
    }
    /// Slot capacity of the GPU buffer.
    pub fn capacity(&self) -> usize {
        self.gpu_capacity
    }
    /// The CPU bookkeeping (read only), e.g. for `slot_of`.
    pub fn book(&self) -> &Book {
        &self.book
    }

    /// Fixed pixels or world metres, like [`crate::Renderer::set_point_size`].
    pub fn set_point_size(&mut self, size: PointSize) {
        match size {
            PointSize::Pixels(px) => {
                self.point_px = px;
                self.point_m = 0.0;
            }
            PointSize::Metres(m) => self.point_m = m.max(0.0),
        }
    }

    /// Applies at the next `prepare`/`draw`; no buffer is rewritten (it is a uniform).
    pub fn set_color_mode(&mut self, mode: ColorMode) {
        self.color_mode = mode;
    }

    /// Reallocates the vertex buffer to `capacity` slots (never below the slots in use) and
    /// re-uploads the shadow copy. `apply` grows by itself, so this is for reserving ahead.
    pub fn set_capacity(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, capacity: usize) {
        self.book.set_capacity(capacity);
        self.sync(device, queue);
    }

    /// Moves the layer origin (all slots are re-sent). Cheap enough for a rare recentre.
    pub fn set_origin(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, origin: [f64; 3]) {
        self.book.set_origin(origin);
        self.sync(device, queue);
    }

    /// Adds fade in (150 ms), moves glide (700 ms, ease-out) from where the point is drawn, removes
    /// fade out (150 ms) and free their slot afterwards. `now` is seconds on any monotonic clock
    /// and must be the same clock as `prepare`. Only touched slot ranges are uploaded; keep one
    /// call within [`FRAME_BUDGET`] points.
    pub fn apply(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        adds: &[Add],
        moves: &[Move],
        removes: &[u64],
        now: f64,
    ) -> ApplyStats {
        let mut st = self.book.apply(adds, moves, removes, now);
        let (uploaded, ranges, realloc) = self.sync(device, queue);
        st.uploaded = uploaded;
        st.ranges = ranges;
        st.reallocated = realloc;
        st
    }

    /// Uploads dirty ranges (all of them after a capacity change); returns slots, calls, realloc.
    fn sync(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> (usize, usize, bool) {
        let mut realloc = false;
        if self.book.capacity() != self.gpu_capacity {
            self.gpu_capacity = self.book.capacity();
            self.verts = vertex_buffer(device, self.gpu_capacity);
            realloc = true;
        }
        let ranges = self.book.take_dirty();
        let recs = self.book.recs();
        let mut slots = 0;
        for r in &ranges {
            let bytes: &[u8] = bytemuck::cast_slice(&recs[r.clone()]);
            queue.write_buffer(&self.verts, (r.start * REC_SIZE) as u64, bytes);
            slots += r.len();
        }
        (slots, ranges.len(), realloc)
    }

    /// Per-frame uniforms: camera, fog, `now`. Also frees finished fade-outs.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        size: [u32; 2],
        camera: &Camera,
        now: f64,
        fog: Option<Fog>,
    ) {
        self.book.recycle(now);
        self.sync(device, queue);
        let t = self.book.time(now);
        let (w, h) = (size[0].max(1), size[1].max(1));
        let f = fog_params(fog);
        let origin = self.book.origin();
        let off = origin - camera.eye;
        let (mode, lo, hi, flat) = match self.color_mode {
            ColorMode::Rgb => (0.0, 0.0, 1.0, [0.0; 3]),
            ColorMode::Height { low, high } => (1.0, low, high, [0.0; 3]),
            ColorMode::Flat(c) => (2.0, 0.0, 1.0, c),
        };
        let u = AnimUniform {
            view_proj: camera.view_proj(f64::from(w) / f64::from(h)).to_cols_array_2d(),
            viewport_px: [w as f32, h as f32],
            point_px: self.point_px,
            point_m: self.point_m,
            fog_color: f.color,
            params: [f.scale, f.bias, focal_px(camera.fov_y_deg, h) as f32, t],
            origin: [off.x as f32, off.y as f32, off.z as f32, origin.z as f32],
            color_mode: [mode, lo, hi, 0.0],
            flat_color: [flat[0], flat[1], flat[2], 1.0],
            durations: [FADE_IN_S, FADE_OUT_S, GLIDE_S, 0.0],
        };
        queue.write_buffer(&self.buf, 0, bytemuck::bytes_of(&u));
    }

    /// Draws into the frame's pass (after `prepare`).
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        let n = self.book.draw_count();
        if n == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bg, &[]);
        pass.set_vertex_buffer(0, self.verts.slice(..(u64::from(n) * REC_SIZE as u64)));
        pass.draw(0..4, 0..n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anim_uniform_layout_matches_the_wgsl_struct() {
        // mat4 64 + vec2 8 + 2 f32 8 + 6 vec4 96.
        assert_eq!(size_of::<AnimUniform>(), 176);
    }
}
