//! GPU time of a run of frames from pass timestamps (decision 0049).
//!
//! [`GpuTimer::next`] points the renderer's next pass at a fresh begin/end pair, [`GpuTimer::resolve`]
//! copies the timestamps to a mappable buffer at the end of the run, and after the caller has mapped
//! [`GpuTimer::readback`], [`GpuTimer::take_ms`] sums the passes. Mapping is left to the caller so a
//! native loop can block and a browser can await.

use crate::Renderer;

pub struct GpuTimer {
    set: wgpu::QuerySet,
    resolved: wgpu::Buffer,
    readback: wgpu::Buffer,
    /// Frames the set holds (two timestamps each).
    capacity: u32,
    /// Frames written since the last `take_ms`.
    used: u32,
    /// Nanoseconds per timestamp tick.
    period_ns: f32,
}

impl GpuTimer {
    /// `None` when the device was created without `Features::TIMESTAMP_QUERY`.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, frames: u32) -> Option<Self> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let capacity = frames.max(1);
        let bytes = u64::from(capacity) * 2 * 8;
        Some(Self {
            set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("frame timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: capacity * 2,
            }),
            resolved: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("frame timestamps resolved"),
                size: bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("frame timestamps readback"),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            capacity,
            used: 0,
            period_ns: queue.get_timestamp_period(),
        })
    }

    /// Times the renderer's next pass. Past the capacity the pass is not timed.
    pub fn next(&mut self, renderer: &mut Renderer) {
        if self.used < self.capacity {
            renderer.set_timestamp_writes(Some((self.set.clone(), self.used * 2)));
            self.used += 1;
        } else {
            renderer.set_timestamp_writes(None);
        }
    }

    /// Stops timing and copies this run's timestamps to [`Self::readback`].
    pub fn resolve(&mut self, renderer: &mut Renderer, encoder: &mut wgpu::CommandEncoder) {
        renderer.set_timestamp_writes(None);
        if self.used == 0 {
            return;
        }
        encoder.resolve_query_set(&self.set, 0..self.used * 2, &self.resolved, 0);
        encoder.copy_buffer_to_buffer(
            &self.resolved,
            0,
            &self.readback,
            0,
            u64::from(self.used) * 16,
        );
    }

    /// The buffer to map (read) after the resolve was submitted.
    pub fn readback(&self) -> &wgpu::Buffer {
        &self.readback
    }

    /// Sum of the timed passes in ms, read from the mapped [`Self::readback`], which it unmaps.
    /// Returns the number of passes too.
    pub fn take_ms(&mut self) -> (f64, u32) {
        let n = self.used;
        self.used = 0;
        if n == 0 {
            return (0.0, 0);
        }
        let ticks: u64 = {
            let view = self
                .readback
                .get_mapped_range(..u64::from(n) * 16)
                .expect("mapped");
            view.as_chunks::<16>()
                .0
                .iter()
                .map(|pair| {
                    let (begin, end) = pair.split_at(8);
                    let tick = |b: &[u8]| u64::from_le_bytes(b.try_into().expect("8 bytes"));
                    tick(end).saturating_sub(tick(begin))
                })
                .sum()
        };
        self.readback.unmap();
        (ticks as f64 * f64::from(self.period_ns) / 1e6, n)
    }
}
