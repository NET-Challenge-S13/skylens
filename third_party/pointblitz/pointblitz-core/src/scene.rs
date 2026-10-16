//! Chunks on the GPU, grouped in generations (decision 0009).
//!
//! - A chunk of the generation on screen is appended (preview snapshots add points).
//! - A chunk of a newer generation goes to a pending set; the screen keeps showing the current
//!   generation until the pending one's last chunk arrives, then the two swap in one step and the
//!   old buffers are released (refined snapshots replace everything, without a blank frame).
//! - Before anything is on screen, the first generation is shown as it arrives.

use glam::DVec3;
use pointblitz_io::chunk::{
    self, ChunkError, ChunkHeader, FLAG_FIRST_PASS_COMPLETE, FLAG_LAST_IN_GENERATION,
};
use wgpu::util::DeviceExt;

pub struct GpuChunk {
    pub buffer: wgpu::Buffer,
    pub points: u32,
    pub origin: DVec3,
    pub header: ChunkHeader,
}

/// What an inserted chunk did to the scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inserted {
    /// Added to the generation on screen.
    Appended,
    /// Stored for a newer generation that is not complete yet.
    Pending,
    /// Completed a newer generation, which is now on screen.
    Swapped { from: Option<u32>, to: u32 },
    /// Older than what is on screen; dropped.
    Stale,
}

#[derive(Default)]
pub struct Scene {
    current: Option<u32>,
    shown: Vec<GpuChunk>,
    pending: Option<(u32, Vec<GpuChunk>)>,
}

impl Scene {
    pub fn new() -> Self {
        Self::default()
    }

    /// Generation on screen.
    pub fn generation(&self) -> Option<u32> {
        self.current
    }

    pub fn visible_chunks(&self) -> &[GpuChunk] {
        &self.shown
    }

    /// Points on screen.
    pub fn points(&self) -> u64 {
        self.shown.iter().map(|c| u64::from(c.points)).sum()
    }

    /// GPU bytes held by vertex buffers (on screen + pending).
    pub fn gpu_bytes(&self) -> u64 {
        let pending = self.pending.iter().flat_map(|(_, v)| v);
        self.shown
            .iter()
            .chain(pending)
            .map(|c| c.buffer.size())
            .sum()
    }

    /// Validates a chunk, uploads its body as a vertex buffer and applies the generation rules.
    pub fn insert(&mut self, device: &wgpu::Device, bytes: &[u8]) -> Result<Inserted, ChunkError> {
        let header = chunk::decode_header(bytes)?;
        let generation = header.generation;
        // A new generation is shown once its coarse first pass is in (decision 0051) or, for a
        // delivery without one, at its last chunk; the rest is filled in as it arrives.
        let last = header.flags & (FLAG_LAST_IN_GENERATION | FLAG_FIRST_PASS_COMPLETE) != 0;

        if self.current.is_some_and(|cur| generation < cur) {
            return Ok(Inserted::Stale);
        }
        let gpu = upload(device, bytes, header);

        match self.current {
            None => {
                self.current = Some(generation);
                self.shown.push(gpu);
                Ok(Inserted::Appended)
            }
            Some(cur) if cur == generation => {
                self.shown.push(gpu);
                Ok(Inserted::Appended)
            }
            Some(cur) => {
                match &mut self.pending {
                    Some((g, v)) if *g == generation => v.push(gpu),
                    Some((g, _)) if *g > generation => return Ok(Inserted::Stale),
                    _ => self.pending = Some((generation, vec![gpu])),
                }
                if last {
                    let (g, v) = self.pending.take().expect("pending set above");
                    self.shown = v; // old buffers drop here
                    self.current = Some(g);
                    Ok(Inserted::Swapped {
                        from: Some(cur),
                        to: g,
                    })
                } else {
                    Ok(Inserted::Pending)
                }
            }
        }
    }
}

fn upload(device: &wgpu::Device, bytes: &[u8], header: ChunkHeader) -> GpuChunk {
    let body = chunk::vertex_bytes(bytes);
    // Zero-point chunks still need a non-empty buffer to be valid.
    let contents: &[u8] = if body.is_empty() { &[0u8; 12] } else { body };
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chunk"),
        contents,
        usage: wgpu::BufferUsages::VERTEX,
    });
    GpuChunk {
        buffer,
        points: header.point_count,
        origin: DVec3::from(header.origin),
        header,
    }
}
