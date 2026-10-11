//! Offscreen rendering and readback: fixed-viewpoint captures, tests, server-side frames.
//!
//! Colours: PLY colours are sRGB bytes. The capture target is `Rgba8Unorm` (no sRGB conversion),
//! so the bytes written are the bytes read — the same pixels three.js produces with its default
//! colour management (sRGB in, sRGB out).

use crate::camera::Camera;
use crate::renderer::{DEPTH_FORMAT, Renderer};
use crate::scene::Scene;
use std::time::{Duration, Instant};

pub const CAPTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Creates a device on the default (high-performance) adapter without a window.
/// Returns None when no adapter is available (e.g. CI without a GPU).
pub fn device() -> Option<(wgpu::Device, wgpu::Queue, wgpu::AdapterInfo)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster_block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .ok()?;
    let info = adapter.get_info();
    let (device, queue) =
        pollster_block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;
    Some((device, queue, info))
}

/// Minimal executor so the core does not depend on an async runtime.
fn pollster_block_on<F: std::future::Future>(f: F) -> F::Output {
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake, Waker};
    struct ThreadWaker(std::thread::Thread);
    impl Wake for ThreadWaker {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = Waker::from(Arc::new(ThreadWaker(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    let mut f = std::pin::pin!(f);
    loop {
        match f.as_mut().poll(&mut cx) {
            Poll::Ready(v) => return v,
            Poll::Pending => std::thread::park(),
        }
    }
}

pub struct Headless {
    pub size: [u32; 2],
    pub renderer: Renderer,
    color: wgpu::Texture,
    depth: wgpu::Texture,
    readback: wgpu::Buffer,
    padded_row: u32,
}

/// Timing of one synchronised frame (decision 0020): submit = CPU encode + submit,
/// total = until the GPU finished.
#[derive(Debug, Clone, Copy)]
pub struct FrameTiming {
    pub cpu: Duration,
    pub total: Duration,
}

impl Headless {
    pub fn new(device: &wgpu::Device, size: [u32; 2]) -> Self {
        let tex = |format, usage, label| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size[0],
                    height: size[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let color = tex(
            CAPTURE_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            "capture colour",
        );
        let depth = tex(
            DEPTH_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
            "capture depth",
        );
        let padded_row = (size[0] * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("capture readback"),
            size: u64::from(padded_row) * u64::from(size[1]),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            size,
            renderer: Renderer::new(device, CAPTURE_FORMAT),
            color,
            depth,
            readback,
            padded_row,
        }
    }

    /// Renders one frame and waits for the GPU (a synchronised frame, decision 0020).
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera: &Camera,
        scene: &Scene,
    ) -> FrameTiming {
        let t0 = Instant::now();
        let mut encoder = device.create_command_encoder(&Default::default());
        let cv = self.color.create_view(&Default::default());
        let dv = self.depth.create_view(&Default::default());
        self.renderer.render(
            device,
            queue,
            &mut encoder,
            &cv,
            &dv,
            self.size,
            camera,
            scene,
        );
        let index = queue.submit([encoder.finish()]);
        let cpu = t0.elapsed();
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(index),
                timeout: None,
            })
            .expect("GPU wait");
        FrameTiming {
            cpu,
            total: t0.elapsed(),
        }
    }

    /// Renders and returns tightly packed RGBA8 pixels (row-major, top row first).
    pub fn capture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera: &Camera,
        scene: &Scene,
    ) -> Vec<u8> {
        self.render(device, queue, camera, scene);
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            self.color.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_row),
                    rows_per_image: Some(self.size[1]),
                },
            },
            wgpu::Extent3d {
                width: self.size[0],
                height: self.size[1],
                depth_or_array_layers: 1,
            },
        );
        let index = queue.submit([encoder.finish()]);
        self.readback
            .map_async(wgpu::MapMode::Read, .., |r| r.expect("map readback"));
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(index),
                timeout: None,
            })
            .expect("GPU wait");
        let mut out = Vec::with_capacity((self.size[0] * self.size[1] * 4) as usize);
        {
            let view = self.readback.get_mapped_range(..).expect("mapped");
            for row in view.chunks_exact(self.padded_row as usize) {
                out.extend_from_slice(&row[..self.size[0] as usize * 4]);
            }
        }
        self.readback.unmap();
        out
    }
}
