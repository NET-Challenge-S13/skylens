//! Slot-indexed point cloud (SkyLens): the GPU side of the live status board. One instance per
//! server slot id, glide and fade evaluated per point on the GPU from the timestamps in
//! [`SlotInstance`], so a frame costs one uniform write and one draw call.
//!
//! Frame recipe (the caller owns the pass, see `layers::begin_pass`):
//!
//! ```ignore
//! if cloud.ensure_capacity(&device, slots.len()) { /* buffer was reallocated: */ cloud.upload(&queue, &slots, &[0..slots.len()]); }
//! cloud.upload(&queue, &slots, &changed_ranges);
//! cloud.set_used(slots.len());
//! cloud.prepare(&queue, size, &camera, style.fog, &params);
//! cloud.draw(&mut pass);
//! ```
//!
//! Positions are absolute ENU metres; the shader gets `position - camera.eye` as one f32 offset
//! rounded from f64. Depth is reversed-Z like every other layer. WebGL2-safe: no storage buffers,
//! uniform read in the vertex stage only, 4 varyings.

use bytemuck::{Pod, Zeroable};
use pointblitz::core::renderer::DEPTH_FORMAT;
use pointblitz::core::{Camera, Fog, PointSize};
use std::ops::Range;

/// Timestamp meaning "long ago" (a finished animation) in [`SlotInstance::tm`].
pub const T_LONG: f32 = -1.0e3;
/// Timestamp meaning "never" (add not started / not dying) in [`SlotInstance::tm`].
pub const T_NEVER: f32 = 1.0e9;

/// One slot of the cloud, as uploaded (40 bytes). Positions are ENU metres about the anchor.
///
/// - `pos`: target position; `from`: position the glide starts from.
/// - `tm`: `[t_move, t_add, t_die]` in seconds on the board clock. The shown position is
///   `mix(from, pos, ease((now - t_move) / glide))`; alpha is
///   `clamp((now - t_add) / fade) * clamp(1 - (now - t_die) / fade)`. A slot never added has
///   `t_add = T_NEVER` (invisible).
/// - `rgbz`: colour and owning zone.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SlotInstance {
    pub pos: [f32; 3],
    pub from: [f32; 3],
    pub tm: [f32; 3],
    pub rgbz: [u8; 4],
}

impl SlotInstance {
    /// An empty (never added) slot.
    pub const EMPTY: SlotInstance =
        SlotInstance { pos: [0.0; 3], from: [0.0; 3], tm: [T_LONG, T_NEVER, T_NEVER], rgbz: [0; 4] };
}

pub const SLOT_STRIDE: usize = std::mem::size_of::<SlotInstance>();

/// Colour modes of [`SlotParams::mode`].
pub const MODE_RGB: u32 = 0;
pub const MODE_ZONE: u32 = 1;
pub const MODE_HEIGHT: u32 = 2;

/// Per-frame inputs of the slot cloud.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlotParams {
    /// Board clock in seconds, the same clock as `SlotInstance::tm`.
    pub now: f32,
    /// Seconds a move takes (the reference uses 0.7).
    pub glide_s: f32,
    /// Seconds an add fades in and a remove fades out (the reference uses 0.15).
    pub fade_s: f32,
    /// [`MODE_RGB`], [`MODE_ZONE`] or [`MODE_HEIGHT`].
    pub mode: u32,
    /// Height ramp range in ENU z metres (mode 2).
    pub height_low: f32,
    pub height_high: f32,
    pub point_size: PointSize,
}

impl Default for SlotParams {
    fn default() -> Self {
        SlotParams {
            now: 0.0,
            glide_s: 0.7,
            fade_s: 0.15,
            mode: MODE_RGB,
            height_low: 0.0,
            height_high: 30.0,
            point_size: PointSize::Pixels(2.0),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SlotUniform {
    view_proj: [[f32; 4]; 4],
    viewport_px: [f32; 2],
    point_px: f32,
    point_m: f32,
    fog_color: [f32; 4],
    params: [f32; 4],
    offset: [f32; 4],
    time: [f32; 4],
    height: [f32; 4],
}

/// Fog uniform terms (colour with alpha 0 when off, scale and bias of the linear ramp).
fn fog_params(fog: Option<Fog>) -> ([f32; 4], f32, f32) {
    match fog {
        None => ([0.0; 4], 0.0, 0.0),
        Some(f) => {
            let span = (f.far - f.near).max(1e-6);
            ([f.color[0], f.color[1], f.color[2], 1.0], 1.0 / span, -f.near / span)
        }
    }
}

/// Smallest buffer, in slots.
const MIN_CAPACITY: usize = 1024;

/// New capacity for `needed` slots from `current`: unchanged when it fits, else grown by 1.3x
/// (at least `needed`, at least [`MIN_CAPACITY`]).
fn grown_capacity(current: usize, needed: usize) -> usize {
    if needed <= current {
        return current;
    }
    let grown = (current as f64 * 1.3).ceil() as usize;
    grown.max(needed).max(MIN_CAPACITY)
}

/// CPU mirror of the shader's ease-in-out quad.
pub fn ease(u: f32) -> f32 {
    let u = u.clamp(0.0, 1.0);
    if u < 0.5 { 2.0 * u * u } else { 1.0 - (-2.0 * u + 2.0).powi(2) / 2.0 }
}

/// CPU mirror of the shader's position and alpha for a slot at time `now`.
pub fn slot_state(s: &SlotInstance, now: f32, glide: f32, fade: f32) -> ([f32; 3], f32) {
    let e = ease((now - s.tm[0]) / glide.max(1e-4));
    let p = std::array::from_fn(|i| s.from[i] + (s.pos[i] - s.from[i]) * e);
    let fade = fade.max(1e-4);
    let a = ((now - s.tm[1]) / fade).clamp(0.0, 1.0) * (1.0 - (now - s.tm[2]) / fade).clamp(0.0, 1.0);
    (p, a)
}

/// Whether the shader draws a slot at all (alpha above 0.01).
pub fn slot_drawn(s: &SlotInstance, now: f32, fade: f32) -> bool {
    slot_state(s, now, 1.0, fade).1 > 0.01
}

pub struct SlotCloud {
    pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    buffer: wgpu::Buffer,
    capacity: usize,
    used: usize,
}

impl SlotCloud {
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let source = include_str!("slot.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("skylens slot cloud"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let layout_bg = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("slot cloud"),
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
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("slot cloud"),
            bind_group_layouts: &[Some(&layout_bg)],
            immediate_size: 0,
        });
        let attrs = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x3, 3 => Uint32];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("slot cloud"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: SLOT_STRIDE as u64,
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
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("slot cloud uniform"),
            size: size_of::<SlotUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("slot cloud"),
            layout: &layout_bg,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() }],
        });
        SlotCloud {
            pipeline,
            uniform,
            bind_group,
            buffer: Self::alloc(device, 0),
            capacity: 0,
            used: 0,
        }
    }

    fn alloc(device: &wgpu::Device, slots: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("slot cloud instances"),
            size: (slots.max(1) * SLOT_STRIDE) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Slots the GPU buffer holds.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Slots drawn (instance count).
    pub fn used(&self) -> usize {
        self.used
    }

    /// Makes room for `n` slots. Returns true when the GPU buffer was reallocated (grown 1.3x):
    /// its contents are gone, so the caller must upload every slot again.
    pub fn ensure_capacity(&mut self, device: &wgpu::Device, n: usize) -> bool {
        let cap = grown_capacity(self.capacity, n);
        if cap == self.capacity {
            return false;
        }
        self.buffer = Self::alloc(device, cap);
        self.capacity = cap;
        true
    }

    /// Writes the given slot ranges (indices into `slots`, which is the whole CPU array).
    /// Ranges are clamped to `slots` and to the capacity.
    pub fn upload(&self, queue: &wgpu::Queue, slots: &[SlotInstance], ranges: &[Range<usize>]) {
        let end = slots.len().min(self.capacity);
        for r in ranges {
            let (a, b) = (r.start.min(end), r.end.min(end));
            if a < b {
                queue.write_buffer(&self.buffer, (a * SLOT_STRIDE) as u64, bytemuck::cast_slice(&slots[a..b]));
            }
        }
    }

    /// Number of slots to draw (slot ids 0..n).
    pub fn set_used(&mut self, n: usize) {
        self.used = n;
    }

    /// Writes the frame uniform: camera, fog, point size, clock and colour mode.
    pub fn prepare(&mut self, queue: &wgpu::Queue, size: [u32; 2], camera: &Camera, fog: Option<Fog>, params: &SlotParams) {
        let (w, h) = (size[0].max(1), size[1].max(1));
        let (fog_color, fog_scale, fog_bias) = fog_params(fog);
        let (point_px, point_m) = match params.point_size {
            PointSize::Pixels(px) => (px, 0.0),
            PointSize::Metres(m) => (2.0, m.max(0.0)),
        };
        let focal = f64::from(h) / 2.0 / (camera.fov_y_deg.to_radians() / 2.0).tan();
        let off = -camera.eye;
        let u = SlotUniform {
            view_proj: camera.view_proj(f64::from(w) / f64::from(h)).to_cols_array_2d(),
            viewport_px: [w as f32, h as f32],
            point_px,
            point_m,
            fog_color,
            params: [fog_scale, fog_bias, focal as f32, 0.0],
            offset: [off.x as f32, off.y as f32, off.z as f32, 0.0],
            time: [params.now, params.glide_s, params.fade_s, params.mode as f32],
            height: [params.height_low, params.height_high, 0.0, 0.0],
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));
    }

    /// Draws into a caller-owned pass (see `layers::begin_pass`). Nothing when `used` is 0.
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        let n = self.used.min(self.capacity);
        if n == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.buffer.slice(..(n * SLOT_STRIDE) as u64));
        pass.draw(0..4, 0..n as u32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot(tm: [f32; 3]) -> SlotInstance {
        SlotInstance { pos: [4.0, 0.0, 0.0], from: [0.0; 3], tm, rgbz: [255; 4] }
    }

    #[test]
    fn capacity_grows_by_1_3() {
        assert_eq!(grown_capacity(0, 0), 0);
        assert_eq!(grown_capacity(0, 10), MIN_CAPACITY);
        assert_eq!(grown_capacity(2000, 2000), 2000);
        assert_eq!(grown_capacity(2000, 2001), 2600);
        assert_eq!(grown_capacity(2000, 9000), 9000);
    }

    #[test]
    fn ease_is_in_out_quad() {
        assert_eq!(ease(0.0), 0.0);
        assert_eq!(ease(1.0), 1.0);
        assert!((ease(0.5) - 0.5).abs() < 1e-6);
        assert!((ease(0.25) - 0.125).abs() < 1e-6);
        assert!((ease(0.75) - 0.875).abs() < 1e-6);
        assert_eq!(ease(2.0), 1.0);
    }

    #[test]
    fn glide_interpolates_with_the_ease() {
        let s = slot([10.0, T_LONG, T_NEVER]);
        assert_eq!(slot_state(&s, 9.0, 0.7, 0.15).0[0], 0.0);
        assert!((slot_state(&s, 10.175, 0.7, 0.15).0[0] - 0.5).abs() < 1e-5);
        assert_eq!(slot_state(&s, 11.0, 0.7, 0.15).0[0], 4.0);
    }

    #[test]
    fn alpha_fades_in_and_out() {
        let s = slot([T_LONG, 10.0, T_NEVER]);
        assert_eq!(slot_state(&s, 10.0, 0.7, 0.15).1, 0.0);
        assert!((slot_state(&s, 10.075, 0.7, 0.15).1 - 0.5).abs() < 1e-5);
        assert_eq!(slot_state(&s, 11.0, 0.7, 0.15).1, 1.0);
        let d = slot([T_LONG, T_LONG, 10.0]);
        assert_eq!(slot_state(&d, 9.0, 0.7, 0.15).1, 1.0);
        assert!((slot_state(&d, 10.075, 0.7, 0.15).1 - 0.5).abs() < 1e-5);
        assert!(!slot_drawn(&d, 11.0, 0.15));
        assert!(!slot_drawn(&SlotInstance::EMPTY, 11.0, 0.15));
        assert!(slot_drawn(&slot([T_LONG, T_LONG, T_NEVER]), 11.0, 0.15));
    }

    #[test]
    fn uniform_is_160_bytes() {
        assert_eq!(size_of::<SlotUniform>(), 160);
    }
}
