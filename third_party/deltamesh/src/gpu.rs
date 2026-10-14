//! GPU device selection and the accumulation-only GPU path ("splat").
//!
//! Requires the `gpu` feature. wgpu runs on Metal on macOS and on Vulkan or DX12 elsewhere.
//!
//! [`GpuCtx`] owns the selected device and queue and is shared by the accumulation, extraction and
//! binning modules. [`GpuSplat`] accumulates on the GPU and merges on the CPU:
//!
//! - Each brick of 8^3 voxels is handled by 8 workgroups of 64 threads, one thread per voxel. A thread
//!   walks the list of bins that touch its brick in a fixed order and produces this input's
//!   contribution (sum of w*d, sum of w, sum of w*rgb). No two threads add into the same voxel, so the
//!   shader needs no atomics and results are deterministic on a given device.
//! - The CPU adds the returned contributions into the target layer. Preview layers are still dropped
//!   as a whole on replacement, so correctness matches the CPU path.
//! - Results are not bitwise identical to the CPU path: the summation order and the GPU `exp`
//!   implementation differ.
//!
//! Device selection skips software rasterizers (`DeviceType::Cpu`, e.g. llvmpipe) and prefers NVIDIA
//! (vendor 0x10de), then discrete, integrated and virtual GPUs. Setting `DELTAMESH_GPU_ADAPTER` to part
//! of an adapter name restricts selection to matching adapters.
//!
//! Open one device per process and share it through [`GpuCtx::shared`]. Some drivers deadlock when
//! several devices are created and destroyed concurrently in one process; device creation and adapter
//! enumeration are serialized by a process-wide lock, but destruction cannot be, so long-running
//! programs should not open and drop devices repeatedly.

use std::sync::Mutex;

/// Number of bricks sent to the GPU per dispatch.
///
/// The result buffer holds bricks x 512 voxels x 32 bytes, i.e. 32 MiB for 2048 bricks.
pub const CHUNK_BRICKS: usize = 2048;

/// Splat shader: one workgroup of 64 threads covers one eighth of a brick.
///
/// Bindings: `bins` holds three `vec4` per bin (position, normal, color); `idx` is the concatenated
/// per-brick list of bin indices addressed by `Tile::start`/`Tile::count`; `outv` holds two `vec4` per
/// voxel: (sum w*d, sum w, sum w*r, sum w*g) and (sum w*b, 0, 0, 0).
const SHADER: &str = r#"
struct Params { voxel: f32, r2: f32, inv2s2: f32, ntiles: u32 };
struct Tile { ox: i32, oy: i32, oz: i32, start: u32, count: u32, a: u32, b: u32, c: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> bins: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> idx: array<u32>;
@group(0) @binding(3) var<storage, read> tiles: array<Tile>;
@group(0) @binding(4) var<storage, read_write> outv: array<vec4<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = wg.x / 8u;
  if (t >= P.ntiles) { return; }
  let vi = (wg.x % 8u) * 64u + li;
  let tile = tiles[t];
  let x = vec3<f32>(f32(tile.ox + i32(vi & 7u)) * P.voxel,
                    f32(tile.oy + i32((vi >> 3u) & 7u)) * P.voxel,
                    f32(tile.oz + i32(vi >> 6u)) * P.voxel);
  var wd = 0.0; var w = 0.0; var c = vec3<f32>(0.0, 0.0, 0.0);
  let end = tile.start + tile.count;
  for (var k = tile.start; k < end; k = k + 1u) {
    let b = idx[k] * 3u;
    let d = x - bins[b].xyz;
    let q = (d.x * d.x + d.y * d.y) + d.z * d.z;
    if (q <= P.r2) {
      let n = bins[b + 1u].xyz;
      let ww = exp(-q * P.inv2s2);
      wd = wd + ww * ((n.x * d.x + n.y * d.y) + n.z * d.z);
      w = w + ww;
      c = c + ww * bins[b + 2u].xyz;
    }
  }
  let o = (t * 512u + vi) * 2u;
  outv[o] = vec4<f32>(wd, w, c.x, c.y);
  outv[o + 1u] = vec4<f32>(c.z, 0.0, 0.0, 0.0);
}
"#;

/// Uniform parameters of the splat shader; layout matches `Params` in [`SHADER`].
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    voxel: f32,
    r2: f32,
    inv2s2: f32,
    ntiles: u32,
}

/// One brick to accumulate on the GPU.
///
/// `start` and `count` select the brick's range in the bin index list passed to
/// [`GpuSplat::splat`]. The layout matches `Tile` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Tile {
    /// Global voxel x coordinate of the brick origin.
    pub ox: i32,
    /// Global voxel y coordinate of the brick origin.
    pub oy: i32,
    /// Global voxel z coordinate of the brick origin.
    pub oz: i32,
    /// First entry of this brick in the bin index list.
    pub start: u32,
    /// Number of entries of this brick in the bin index list.
    pub count: u32,
    /// Padding to 32 bytes; ignored by the shader.
    pub pad: [u32; 3],
}

/// Description of the selected GPU adapter, for logging.
#[derive(Clone, Debug)]
pub struct GpuInfo {
    /// Adapter name reported by the driver.
    pub name: String,
    /// Backend in use (`Metal`, `Vulkan`, ...).
    pub backend: String,
    /// Device type (`DiscreteGpu`, `IntegratedGpu`, ...).
    pub device_type: String,
    /// PCI vendor ID.
    pub vendor: u32,
}

impl std::fmt::Display for GpuInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} ({}, {}, vendor 0x{:04x})",
            self.name, self.backend, self.device_type, self.vendor
        )
    }
}

/// Persistent buffers of [`GpuSplat`], sized for [`CHUNK_BRICKS`] bricks.
struct Bufs {
    tiles: wgpu::Buffer,
    out: wgpu::Buffer,
    staging: wgpu::Buffer,
    params: wgpu::Buffer,
}

/// Time breakdown of one [`GpuSplat::splat`] call, in milliseconds.
///
/// `upload` covers buffer creation and writes on the CPU side, `wait` covers submission through
/// mapping the results (compute and copy), and `sink` covers the result callback.
#[derive(Clone, Copy, Debug, Default)]
pub struct SplatTimes {
    /// Buffer creation and writes on the CPU side.
    pub upload: f64,
    /// Submission, compute, copy and mapping of results.
    pub wait: f64,
    /// Time spent in the result callback.
    pub sink: f64,
}

/// Selected GPU device and queue, shared by the accumulation, extraction and binning modules.
pub struct GpuCtx {
    /// Opened device.
    pub device: wgpu::Device,
    /// Queue of [`GpuCtx::device`].
    pub queue: wgpu::Queue,
    /// Description of the selected adapter.
    pub info: GpuInfo,
    /// Largest size in bytes that a single storage buffer binding may have.
    pub max_binding: u64,
    /// Serializes submission and waiting so several users in one process can share the device.
    ///
    /// Hold it only around GPU submission and waiting, never while calling into rayon: a rayon worker
    /// that holds the lock can steal a task that wants the same lock and deadlock.
    pub lock: Mutex<()>,
    /// Whether subgroup operations are enabled. Used by the binning sort; the shader checks the
    /// actual subgroup size at run time.
    pub subgroups: bool,
    /// Whether mappable storage buffers (`MAPPABLE_PRIMARY_BUFFERS`) are enabled on an integrated GPU
    /// with unified memory.
    ///
    /// When set, binning skips the upload and readback copies and extraction reads results without a
    /// copy. Disabled by `DELTAMESH_NO_UMA` or `DELTAMESH_GPU_MAPPED=0`.
    pub uma: bool,
}

/// Accumulation-only GPU path: computes per-voxel contributions on the GPU for the CPU to merge.
pub struct GpuSplat {
    /// Device the pipeline was built on.
    pub ctx: std::sync::Arc<GpuCtx>,
    pipeline: wgpu::ComputePipeline,
    bufs: Bufs,
    /// Copy of [`GpuCtx::info`].
    pub info: GpuInfo,
}

/// Ranks an adapter for selection; higher is better.
///
/// Returns `None` for software adapters, which are never selected.
fn score(info: &wgpu::AdapterInfo) -> Option<i32> {
    use wgpu::DeviceType::*;
    let base = match info.device_type {
        Cpu => return None,
        DiscreteGpu => 30,
        IntegratedGpu => 20,
        VirtualGpu => 10,
        Other => 0,
    };
    Some(base + if info.vendor == 0x10de { 100 } else { 0 })
}

/// Process-wide lock around adapter enumeration and device creation.
///
/// Creating and destroying several devices concurrently in one process can deadlock inside the
/// driver or loader, so device opening and enumeration happen one at a time. Destruction cannot be
/// serialized this way; open one device per process and share it via [`GpuCtx::shared`].
static OPEN: Mutex<()> = Mutex::new(());

/// Lists all adapters with their selection score, best first. Intended for diagnostics.
///
/// Software adapters are included with a score of `None`.
pub fn list_adapters() -> Vec<(GpuInfo, Option<i32>)> {
    let _o = OPEN.lock().unwrap_or_else(|e| e.into_inner());
    let inst = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let mut v: Vec<_> = pollster::block_on(inst.enumerate_adapters(wgpu::Backends::PRIMARY))
        .into_iter()
        .map(|a| {
            let i = a.get_info();
            (to_info(&i), score(&i))
        })
        .collect();
    v.sort_by_key(|(_, s)| std::cmp::Reverse(s.unwrap_or(-1)));
    v
}

/// Converts wgpu adapter info into a [`GpuInfo`].
fn to_info(i: &wgpu::AdapterInfo) -> GpuInfo {
    GpuInfo {
        name: i.name.clone(),
        backend: format!("{:?}", i.backend),
        device_type: format!("{:?}", i.device_type),
        vendor: i.vendor,
    }
}

impl GpuCtx {
    /// Selects the best available GPU and opens a device on it.
    ///
    /// Requests all adapter limits, plus `SUBGROUP` when supported (unless `DELTAMESH_NO_SUBGROUPS` is
    /// set) and `MAPPABLE_PRIMARY_BUFFERS` on integrated GPUs (see [`GpuCtx::uma`]).
    ///
    /// Prefer [`GpuCtx::shared`]; see the module documentation for why a process should open only
    /// one device.
    ///
    /// # Errors
    ///
    /// Returns an error if no hardware adapter is available or the device cannot be opened. Callers
    /// fall back to the CPU path.
    pub fn new() -> Result<std::sync::Arc<Self>, String> {
        let _o = OPEN.lock().unwrap_or_else(|e| e.into_inner());
        let inst =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let want = std::env::var("DELTAMESH_GPU_ADAPTER")
            .ok()
            .map(|s| s.to_lowercase());
        let mut cands: Vec<(i32, wgpu::Adapter)> =
            pollster::block_on(inst.enumerate_adapters(wgpu::Backends::PRIMARY))
                .into_iter()
                .filter_map(|a| {
                    let i = a.get_info();
                    let s = score(&i)?;
                    if let Some(w) = &want {
                        if !i.name.to_lowercase().contains(w) {
                            return None;
                        }
                    }
                    Some((s, a))
                })
                .collect();
        cands.sort_by_key(|(s, _)| std::cmp::Reverse(*s));
        let (_, adapter) = cands
            .into_iter()
            .next()
            .ok_or_else(|| "no usable GPU adapter (software adapters are excluded)".to_string())?;
        let ainfo = adapter.get_info();
        let limits = adapter.limits();
        let subgroups = adapter.features().contains(wgpu::Features::SUBGROUP)
            && std::env::var_os("DELTAMESH_NO_SUBGROUPS").is_none();
        let uma = ainfo.device_type == wgpu::DeviceType::IntegratedGpu
            && adapter
                .features()
                .contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS)
            && std::env::var_os("DELTAMESH_NO_UMA").is_none()
            && std::env::var("DELTAMESH_GPU_MAPPED").map_or(true, |v| v != "0");
        let mut features = wgpu::Features::empty();
        if subgroups {
            features |= wgpu::Features::SUBGROUP;
        }
        if uma {
            features |= wgpu::Features::MAPPABLE_PRIMARY_BUFFERS;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("deltamesh"),
            required_features: features,
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .map_err(|e| format!("failed to open GPU device: {e}"))?;
        let max_binding = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size);
        Ok(std::sync::Arc::new(Self {
            device,
            queue,
            info: to_info(&ainfo),
            max_binding,
            lock: Mutex::new(()),
            subgroups,
            uma,
        }))
    }

    /// Returns the process-wide shared device, opening it on first use.
    ///
    /// The device is never released, so repeated calls do not create and destroy devices. All tests
    /// use this. Because the device lives until process exit, some drivers may crash during their own
    /// teardown at exit; programs that control their shutdown can open a device with [`GpuCtx::new`]
    /// and drop it before exiting.
    ///
    /// # Errors
    ///
    /// Returns the error from the first [`GpuCtx::new`] call; the result is cached.
    pub fn shared() -> Result<std::sync::Arc<Self>, String> {
        static S: std::sync::OnceLock<Result<std::sync::Arc<GpuCtx>, String>> =
            std::sync::OnceLock::new();
        S.get_or_init(Self::new).clone()
    }

    /// Reads the first `bytes` bytes of `buf` back to the CPU.
    ///
    /// Copies into a temporary staging buffer, maps it and blocks until done. Intended for tests and
    /// small readbacks. `buf` must have `COPY_SRC` usage.
    ///
    /// # Errors
    ///
    /// Returns an error if waiting on the device or mapping the staging buffer fails.
    pub fn read_buffer(&self, buf: &wgpu::Buffer, bytes: u64) -> Result<Vec<u8>, String> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read"),
            size: bytes.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("read"),
            });
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
        self.queue.submit(Some(enc.finish()));
        let slice = staging.slice(0..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("GPU wait failed: {e}"))?;
        rx.recv()
            .map_err(|e| format!("{e}"))?
            .map_err(|e| format!("{e}"))?;
        let v = slice
            .get_mapped_range()
            .map_err(|e| format!("{e:?}"))?
            .to_vec();
        staging.unmap();
        Ok(v)
    }
}

impl GpuSplat {
    /// Opens a new device with [`GpuCtx::new`] and builds the splat pipeline on it.
    ///
    /// # Errors
    ///
    /// Returns an error if no GPU is available; callers fall back to the CPU path.
    pub fn new() -> Result<Self, String> {
        Self::with_ctx(GpuCtx::new()?)
    }

    /// Builds the splat pipeline and its persistent buffers on an already opened device.
    ///
    /// # Errors
    ///
    /// Currently always succeeds; the `Result` keeps the signature uniform with [`GpuSplat::new`].
    pub fn with_ctx(ctx: std::sync::Arc<GpuCtx>) -> Result<Self, String> {
        let device = &ctx.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("splat"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("splat"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let out_size = (CHUNK_BRICKS * 512 * 32) as u64;
        let mk = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let bufs = Bufs {
            tiles: mk(
                "tiles",
                (CHUNK_BRICKS * std::mem::size_of::<Tile>()) as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
            out: mk(
                "out",
                out_size,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            ),
            staging: mk(
                "staging",
                out_size,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
            params: mk(
                "params",
                16,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
        };
        let info = ctx.info.clone();
        Ok(Self {
            ctx,
            pipeline,
            bufs,
            info,
        })
    }

    /// Computes per-voxel contributions for every brick in `tiles`.
    ///
    /// Bricks are processed in chunks of [`CHUNK_BRICKS`]. The device lock is held only while a chunk
    /// is submitted, waited on and read back; `sink` runs outside the lock because it may use rayon
    /// (see [`GpuCtx::lock`]). The shared buffers (params, tiles, out, staging) are protected by the
    /// same lock.
    ///
    /// # Arguments
    ///
    /// * `bins` - three `vec4` per bin: position, normal, color.
    /// * `idx` - concatenated per-brick bin index lists, addressed by [`Tile::start`] and [`Tile::count`].
    /// * `tiles` - bricks to accumulate.
    /// * `voxel` - voxel size.
    /// * `r2` - squared support radius; bins farther than this from a voxel are skipped.
    /// * `inv2s2` - `1 / (2 sigma^2)` of the Gaussian weight.
    /// * `sink` - called once per chunk as `sink(first_tile, &out)`, where `out` holds, for each brick
    ///   of the chunk, 512 voxels x 8 `f32` laid out as in the shader's `outv`.
    ///
    /// # Returns
    ///
    /// The time spent in each phase.
    ///
    /// # Errors
    ///
    /// Returns an error if `bins` or `idx` exceed the device's storage binding limit, or if waiting on
    /// or mapping the results fails. Callers fall back to the CPU path.
    #[allow(clippy::too_many_arguments)]
    pub fn splat(
        &self,
        bins: &[[f32; 4]],
        idx: &[u32],
        tiles: &[Tile],
        voxel: f32,
        r2: f32,
        inv2s2: f32,
        mut sink: impl FnMut(usize, &[f32]),
    ) -> Result<SplatTimes, String> {
        let mut tm = SplatTimes::default();
        let ms = |t: std::time::Instant| t.elapsed().as_secs_f64() * 1e3;
        if tiles.is_empty() {
            return Ok(tm);
        }
        let bins_bytes = (bins.len().max(1) * 16) as u64;
        let idx_bytes = (idx.len().max(1) * 4) as u64;
        if bins_bytes > self.ctx.max_binding || idx_bytes > self.ctx.max_binding {
            return Err(format!(
                "GPU buffer limit exceeded (bins {bins_bytes} B, index list {idx_bytes} B, limit {} B)",
                self.ctx.max_binding
            ));
        }
        let t_up = std::time::Instant::now();
        let mk = |label, data: &[u8]| {
            let b = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (data.len().max(16) as u64 + 3) & !3,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            if !data.is_empty() {
                self.ctx.queue.write_buffer(&b, 0, data);
            }
            b
        };
        let bins_buf = mk("bins", bytemuck::cast_slice(bins));
        let idx_buf = mk("idx", bytemuck::cast_slice(idx));
        tm.upload += ms(t_up);
        let layout = self.pipeline.get_bind_group_layout(0);
        let bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("splat"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.bufs.params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: bins_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: idx_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.bufs.tiles.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.bufs.out.as_entire_binding(),
                    },
                ],
            });
        for (ci, chunk) in tiles.chunks(CHUNK_BRICKS).enumerate() {
            let n = chunk.len();
            let g = self.ctx.lock.lock().unwrap();
            let t_up = std::time::Instant::now();
            self.ctx
                .queue
                .write_buffer(&self.bufs.tiles, 0, bytemuck::cast_slice(chunk));
            let p = Params {
                voxel,
                r2,
                inv2s2,
                ntiles: n as u32,
            };
            self.ctx
                .queue
                .write_buffer(&self.bufs.params, 0, bytemuck::bytes_of(&p));
            let bytes = (n * 512 * 32) as u64;
            tm.upload += ms(t_up);
            let t_wait = std::time::Instant::now();
            let mut enc = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("splat"),
                });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("splat"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups((n * 8) as u32, 1, 1);
            }
            enc.copy_buffer_to_buffer(&self.bufs.out, 0, &self.bufs.staging, 0, bytes);
            self.ctx.queue.submit(Some(enc.finish()));
            let slice = self.bufs.staging.slice(0..bytes);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            self.ctx
                .device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| format!("GPU wait failed: {e}"))?;
            rx.recv()
                .map_err(|e| format!("failed to receive GPU map result: {e}"))?
                .map_err(|e| format!("failed to map GPU results: {e}"))?;
            tm.wait += ms(t_wait);
            let t_sink = std::time::Instant::now();
            let data: Vec<f32> = {
                let view = slice
                    .get_mapped_range()
                    .map_err(|e| format!("failed to read GPU results: {e:?}"))?;
                bytemuck::cast_slice(&view).to_vec()
            };
            self.bufs.staging.unmap();
            drop(g);
            sink(ci * CHUNK_BRICKS, &data);
            tm.sink += ms(t_sink);
        }
        Ok(tm)
    }
}
