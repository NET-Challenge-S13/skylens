//! GPU point binning and normal estimation.
//!
//! [`GpuBinner::bin_points`] has the same contract as [`crate::bins::bin_points`]: same bin order,
//! [`Bin`] fields and [`BinStats`]. The shaders live in `gpu_bins.wgsl`, with the subgroup scatter in
//! `gpu_bins_sg.wgsl` and the orientation neighbor lists in `gpu_bins_nb.wgsl`.
//!
//! Steps:
//! 1. The CPU writes points into a mapped upload buffer in parallel and, using the CPU binning rule,
//!    computes the per-axis cell range and the number of valid points at the same time.
//! 2. The GPU packs each cell key into 64 bits as per-axis offsets (z in the low bits, then y, then x;
//!    the top bit marks invalid points). Only the bits in use are sorted, by an LSD radix sort with
//!    8-bit digits whose values are point indices; the sort is stable, so points within a cell keep
//!    their input order. Each pass builds per-tile digit histograms, scans them, and scatters each
//!    tile after partitioning it stably in shared memory by two 4-bit rounds.
//! 3. Cell-start flags are scanned to produce cell starts and unique keys.
//! 4. One thread per cell sums its points in input order (f32 positions relative to the cell origin,
//!    integer color sums).
//! 5. For cells that need an estimated normal, the covariance of the neighboring cells is gathered by
//!    binary searching the z range of every (dx, dy) row in the sorted keys; the smallest eigenvector,
//!    flipped towards +z, becomes the normal. Cells with too few neighbors are dropped.
//! 6. The host reads back the cell count (first sync) and then all results at once (second sync).
//!    With unified memory both happen in a single sync without copies.
//!
//! Atomics are used only for integer counts and error flags. All floating-point sums run in a fixed
//! order, so results are bitwise identical across runs on the same device.

use crate::Point;
use crate::bins::{Bin, BinStats, NormalPolicy};
use crate::gpu::GpuCtx;
use crate::orient::Lists;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

/// Main binning shader.
const SHADER: &str = include_str!("gpu_bins.wgsl");
/// Subgroup variant of the radix scatter; compiled appended to [`SHADER`].
const SHADER_SG: &str = include_str!("gpu_bins_sg.wgsl");
/// Orientation neighbor list kernels; compiled appended to [`SHADER`].
const SHADER_NB: &str = include_str!("gpu_bins_nb.wgsl");
/// Tile size of the radix sort and the scan; must match the shader.
const TILE: usize = 1024;
/// Bits per radix sort pass.
const RADIX_BITS: u32 = 8;
/// u32 words per cell result: position (3), normal (3), color (3), count | state << 30.
const BIN_WORDS: usize = 10;
/// Number of uniform buffer slots, one per dispatch configuration.
const SLOTS: usize = 32;
/// Cell state: normal taken from the input.
const ST_FIXED: u32 = 0;
/// Cell state: normal estimated from neighbors.
const ST_EST: u32 = 2;
/// Cell state: dropped (estimation failed).
const ST_DROP: u32 = 3;

/// Byte distance between uniform slots: `size` rounded up to a multiple of the device's
/// `min_uniform_buffer_offset_alignment`, so every dynamic offset is aligned.
fn uniform_stride(min_align: u64, size: u64) -> u64 {
    let a = min_align.max(1);
    size.div_ceil(a) * a
}

/// Uniform parameters of one dispatch; layout and field meanings match `U` in `gpu_bins.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct U {
    n: u32,
    nb: u32,
    shift: u32,
    flag: u32,
    stride: u32,
    policy: u32,
    radius: i32,
    min_nb: u32,
    size: f32,
    inv: f32,
    bx: u32,
    by: u32,
    bz: u32,
    vbit: u32,
    mnx: i32,
    mny: i32,
    mnz: i32,
    rx: i32,
    ry: i32,
    rz: i32,
    orad: i32,
    snz: f32,
    stab: u32,
    pad: u32,
}

/// Maximum number of entries of the slab table.
///
/// If the x range is wider, the table is not built and neighbor searches use plain binary search.
const SLAB_MAX: usize = 1 << 20;

/// Compute pipelines of [`GpuBinner`].
struct Pipes {
    pack: wgpu::ComputePipeline,
    count: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
    scan_tile: wgpu::ComputePipeline,
    scan_top: wgpu::ComputePipeline,
    scan_add: wgpu::ComputePipeline,
    flags: wgpu::ComputePipeline,
    segs: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
    estimate: wgpu::ComputePipeline,
    slab: wgpu::ComputePipeline,
    /// Subgroup scatter; present only when the device supports subgroups.
    scatter_sg: Option<wgpu::ComputePipeline>,
    /// Neighbor list kernels, present only with unified memory: estimate plus mark and count, fill,
    /// and the layout of bind group 1.
    nb: Option<(
        wgpu::ComputePipeline,
        wgpu::ComputePipeline,
        wgpu::BindGroupLayout,
    )>,
}

/// Buffers reused across calls for up to `cap` points.
struct Bufs {
    cap: usize,
    pts: wgpu::Buffer,
    ka: wgpu::Buffer,
    va: wgpu::Buffer,
    kb: wgpu::Buffer,
    vb: wgpu::Buffer,
    hist: wgpu::Buffer,
    sums: wgpu::Buffer,
    info: wgpu::Buffer,
    bins: wgpu::Buffer,
    segs: wgpu::Buffer,
    /// Placeholder buffers for unused bindings. There are two so that writable bindings never
    /// alias.
    dummy: wgpu::Buffer,
    dummy2: wgpu::Buffer,
    small: wgpu::Buffer,
    /// Upload buffer the CPU maps and writes points into directly.
    upload: wgpu::Buffer,
    /// Readback buffer and its size in bytes; grown on demand.
    stage: Option<(u64, wgpu::Buffer)>,
    /// Slab table: capacity in u32 entries and buffer.
    slab: (usize, wgpu::Buffer),
    /// Neighbor list buffers: final index, list start, list start per final index. Sized for `cap`
    /// only when the neighbor list kernels exist.
    fin: wgpu::Buffer,
    noff: wgpu::Buffer,
    jo: wgpu::Buffer,
    /// Neighbor list: capacity in u32 entries, GPU-only buffer and mappable copy. Grown on the next
    /// call when too small.
    ///
    /// Scattered GPU writes to mappable shared memory are slow, so the kernel writes to a GPU-only
    /// buffer that is then copied to the mappable one.
    nlist: Option<(usize, wgpu::Buffer, wgpu::Buffer)>,
}

/// GPU implementation of [`crate::bins::bin_points`].
///
/// Buffers are cached between calls and protected by an internal mutex; calls on the same binner
/// are serialized.
pub struct GpuBinner {
    ctx: Arc<GpuCtx>,
    layout: wgpu::BindGroupLayout,
    pipes: Pipes,
    uni: wgpu::Buffer,
    align: u64,
    bufs: Mutex<Option<Bufs>>,
    /// Whether to use the subgroup scatter. Cleared when the shader reports an unsupported subgroup
    /// layout.
    sg: std::sync::atomic::AtomicBool,
    /// Whether to map buffers directly on unified memory (see [`GpuCtx::uma`]).
    uma: bool,
    /// Requested neighbor list capacity in u32 entries; raised when the previous call overflowed.
    nl_want: std::sync::atomic::AtomicUsize,
}

/// Number of tiles needed for `n` elements.
fn tiles(n: usize) -> usize {
    n.div_ceil(TILE)
}

/// Splits a workgroup count into a 2D grid with at most 65535 groups per axis.
///
/// Tests use a smaller limit so the 2D path is exercised too.
fn groups(n: usize) -> (u32, u32) {
    const MAX_X: usize = if cfg!(test) { 1000 } else { 65535 };
    let n = n.max(1);
    if n <= MAX_X {
        (n as u32, 1)
    } else {
        (MAX_X as u32, n.div_ceil(MAX_X) as u32)
    }
}

/// Number of bits needed to store values in `0..=range`.
fn bits_for(range: u32) -> u32 {
    32 - range.leading_zeros()
}

/// Maps `slice` with `mode` and blocks until the mapping completes.
fn wait_map(ctx: &GpuCtx, slice: wgpu::BufferSlice<'_>, mode: wgpu::MapMode) -> Result<(), String> {
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(mode, move |r| {
        let _ = tx.send(r);
    });
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| format!("GPU wait failed: {e}"))?;
    rx.recv()
        .map_err(|e| format!("failed to receive GPU map result: {e}"))?
        .map_err(|e| format!("failed to map GPU results: {e}"))
}

/// Maps several buffer slices for reading and waits once for all of them.
fn wait_map2(ctx: &GpuCtx, slices: &[wgpu::BufferSlice<'_>]) -> Result<(), String> {
    let (tx, rx) = std::sync::mpsc::channel();
    for sl in slices {
        let tx = tx.clone();
        sl.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
    }
    drop(tx);
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| format!("GPU wait failed: {e}"))?;
    for _ in slices {
        rx.recv()
            .map_err(|e| format!("failed to receive GPU map result: {e}"))?
            .map_err(|e| format!("failed to map GPU results: {e}"))?;
    }
    Ok(())
}

/// Failure reported by [`check`].
enum Check {
    /// The subgroup scatter does not fit this device; disable it and run again.
    Retry,
    /// Unrecoverable error; returned to the caller.
    Fail(String),
}
use Check::{Fail, Retry};

/// Validates the counters read back from the GPU `info` buffer.
///
/// `m` is the number of cells, `nv` the number of valid points, `bad` the key range flag and `sg_bad`
/// the subgroup layout flag.
fn check(m: usize, nv: usize, bad: i32, sg_bad: i32) -> Result<(), Check> {
    if sg_bad != 0 {
        return Err(Retry);
    }
    if bad != 0 {
        return Err(Fail(
            "GPU cell key is outside the range computed on the CPU".into(),
        ));
    }
    if m == 0 || m > nv {
        return Err(Fail(format!(
            "invalid GPU cell count ({m}, valid points {nv})"
        )));
    }
    Ok(())
}

impl GpuBinner {
    /// Creates a binner on `ctx` with all optional modes enabled where the device supports them.
    ///
    /// # Errors
    ///
    /// See [`GpuBinner::with_modes`].
    pub fn new(ctx: Arc<GpuCtx>) -> Result<Self, String> {
        Self::with_modes(ctx, true, true)
    }

    /// Creates a binner with the subgroup scatter and unified-memory mapping individually disabled.
    ///
    /// Both modes produce bitwise identical results; this exists for testing and benchmarking. The
    /// neighbor list kernels need unified memory and at least 16 storage buffers per stage.
    ///
    /// # Errors
    ///
    /// Returns an error if the device has fewer than 12 storage buffers per shader stage or less than
    /// 23 KiB of workgroup storage (`k_scatter` needs about 22.5 KiB).
    pub fn with_modes(ctx: Arc<GpuCtx>, subgroups: bool, uma: bool) -> Result<Self, String> {
        let device = &ctx.device;
        let lim = device.limits();
        if lim.max_storage_buffers_per_shader_stage < 12 {
            return Err(format!(
                "not enough storage buffer bindings ({} < 12)",
                lim.max_storage_buffers_per_shader_stage
            ));
        }
        if lim.max_compute_workgroup_storage_size < 23 * 1024 {
            return Err("not enough workgroup storage".into());
        }
        let want_nb = uma && ctx.uma && lim.max_storage_buffers_per_shader_stage >= 16;
        let src = if want_nb {
            format!("{SHADER}\n{SHADER_NB}")
        } else {
            SHADER.to_string()
        };
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_bins"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let st = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<U>() as u64),
            },
            count: None,
        }];
        entries.push(st(1, true));
        for b in 2..=12 {
            entries.push(st(b, false));
        }
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_bins"),
            entries: &entries,
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu_bins"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let mk_in = |module: &wgpu::ShaderModule, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let mk = |entry: &str| mk_in(&module, entry);
        let nb = want_nb.then(|| {
            let l1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu_bins nb"),
                entries: &(0..4).map(|b| st(b, false)).collect::<Vec<_>>(),
            });
            let pl2 = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gpu_bins nb"),
                bind_group_layouts: &[Some(&layout), Some(&l1)],
                immediate_size: 0,
            });
            let mk2 = |entry: &str| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pl2),
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
            };
            (mk2("k_estimate_nb"), mk2("k_nfill"), l1)
        });
        let scatter_sg = (subgroups && ctx.subgroups).then(|| {
            let src = format!("{SHADER}\n{SHADER_SG}");
            let m = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gpu_bins_sg"),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            mk_in(&m, "k_scatter_sg")
        });
        let pipes = Pipes {
            pack: mk("k_pack"),
            count: mk("k_count"),
            scatter: mk("k_scatter"),
            scan_tile: mk("k_scan_tile"),
            scan_top: mk("k_scan_top"),
            scan_add: mk("k_scan_add"),
            flags: mk("k_flags"),
            segs: mk("k_segs"),
            reduce: mk("k_reduce"),
            estimate: mk("k_estimate"),
            slab: mk("k_slab"),
            scatter_sg,
            nb,
        };
        let align = uniform_stride(
            lim.min_uniform_buffer_offset_alignment as u64,
            std::mem::size_of::<U>() as u64,
        );
        let uni = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_bins uni"),
            size: align * SLOTS as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sg = std::sync::atomic::AtomicBool::new(pipes.scatter_sg.is_some());
        let uma = uma && ctx.uma;
        Ok(Self {
            ctx,
            layout,
            pipes,
            uni,
            align,
            bufs: Mutex::new(None),
            sg,
            uma,
            nl_want: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    /// Allocates the buffer set for up to `cap` points.
    ///
    /// With unified memory the point, result and info buffers are mapped by the CPU directly, so no
    /// upload or readback copies are needed. `sums` holds enough tiles for either the point scan or
    /// the histogram scan, plus the total.
    fn make_bufs(&self, cap: usize) -> Bufs {
        let d = &self.ctx.device;
        use wgpu::BufferUsages as B;
        let mk = |label, size: usize, usage| {
            d.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (size.max(16) as u64 + 15) & !15,
                usage,
                mapped_at_creation: false,
            })
        };
        let s = B::STORAGE;
        let nb = tiles(cap);
        let uma = self.uma;
        let (pts_u, bins_u, info_u) = if uma {
            (
                s | B::MAP_WRITE,
                s | B::MAP_READ | B::COPY_SRC,
                s | B::COPY_DST | B::COPY_SRC | B::MAP_READ,
            )
        } else {
            (
                s | B::COPY_DST,
                s | B::COPY_SRC,
                s | B::COPY_DST | B::COPY_SRC,
            )
        };
        Bufs {
            cap,
            pts: mk("pts", cap * 7 * 4, pts_u),
            ka: mk("ka", cap * 8, s),
            va: mk("va", cap * 4, s),
            kb: mk("kb", cap * 8, s),
            vb: mk("vb", cap * 4, s),
            hist: mk("hist", 256 * nb * 4, s),
            sums: mk("sums", (tiles(cap).max(tiles(256 * nb)) + 2) * 4, s),
            info: mk("info", 64, info_u),
            bins: mk("bins", cap * BIN_WORDS * 4, bins_u),
            segs: mk("segs", (cap + 1) * 4, s),
            dummy: mk("dummy", 16, s),
            dummy2: mk("dummy2", 16, s),
            small: mk("small", 64, B::MAP_READ | B::COPY_DST),
            upload: mk(
                "upload",
                if uma { 16 } else { cap * 7 * 4 },
                B::MAP_WRITE | B::COPY_SRC,
            ),
            stage: None,
            slab: (4, mk("slab", 16, s)),
            fin: mk("fin", if self.pipes.nb.is_some() { cap * 4 } else { 16 }, s),
            noff: mk(
                "noff",
                if self.pipes.nb.is_some() { cap * 4 } else { 16 },
                s,
            ),
            jo: if self.pipes.nb.is_some() {
                mk("jo", (cap + 1) * 4, s | B::MAP_READ)
            } else {
                mk("jo", 16, s)
            },
            nlist: None,
        }
    }

    /// Creates bind group 0 with the given sort, scan and unique-key buffers.
    ///
    /// The sort buffers swap roles between passes, and after sorting the free pair is reused for
    /// cell-start flags and unique keys.
    #[allow(clippy::too_many_arguments)]
    fn bind<'a>(
        &'a self,
        b: &'a Bufs,
        kin: &'a wgpu::Buffer,
        vin: &'a wgpu::Buffer,
        kout: &'a wgpu::Buffer,
        vout: &'a wgpu::Buffer,
        sdata: &'a wgpu::Buffer,
        ukeys: &'a wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let e = |binding: u32, buf: &'a wgpu::Buffer| wgpu::BindGroupEntry {
            binding,
            resource: buf.as_entire_binding(),
        };
        self.ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gpu_bins"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.uni,
                            offset: 0,
                            size: wgpu::BufferSize::new(std::mem::size_of::<U>() as u64),
                        }),
                    },
                    e(1, &b.pts),
                    e(2, kin),
                    e(3, vin),
                    e(4, kout),
                    e(5, vout),
                    e(6, sdata),
                    e(7, &b.sums),
                    e(8, &b.info),
                    e(9, &b.bins),
                    e(10, &b.segs),
                    e(11, ukeys),
                    e(12, &b.slab.1),
                ],
            })
    }

    /// Bins `pts` into cells of edge `size`, with the same contract as [`crate::bins::bin_points`].
    ///
    /// # Errors
    ///
    /// Returns an error if the input exceeds the device buffer limits or the key range, or if a GPU
    /// operation fails. Callers fall back to the CPU path.
    pub fn bin_points(
        &self,
        pts: &[Point],
        size: f32,
        policy: NormalPolicy,
        normal_radius: i32,
        normal_min_neighbors: usize,
    ) -> Result<(Vec<Bin>, BinStats), String> {
        self.bin_points_nb(
            pts,
            size,
            policy,
            normal_radius,
            normal_min_neighbors,
            None,
            |_, _, _| (),
        )
        .map(|r| (r.0, r.1))
    }

    /// [`GpuBinner::bin_points`] that also builds neighbor lists for normal orientation.
    ///
    /// The binned result is passed to `f(bins, stats, lists)` and `f`'s return value is returned
    /// alongside it. The lists borrow the mapped GPU result directly, without a copy; the device lock
    /// is released while `f` runs. Lists are built only with [`NormalPolicy::Estimate`], on the
    /// unified-memory path, and when the orientation radius equals `normal_radius`. Otherwise, or if
    /// the list buffer overflowed or a bin position does not map back to its cell, `f` receives
    /// `None` and should search neighbors on the CPU.
    ///
    /// The device lock (`GpuCtx::lock`) is held only around GPU submission and waiting, never while
    /// rayon runs; the cached buffers are protected by the binner's own mutex for the whole call.
    ///
    /// # Arguments
    ///
    /// * `orient` - `(radius, seed_nz)` of the orientation step, or `None` to skip the lists.
    /// * `f` - consumer of the result; not called when an error is returned.
    ///
    /// # Errors
    ///
    /// Same as [`GpuBinner::bin_points`].
    #[allow(clippy::too_many_arguments)]
    pub fn bin_points_nb<R, F>(
        &self,
        pts: &[Point],
        size: f32,
        policy: NormalPolicy,
        normal_radius: i32,
        normal_min_neighbors: usize,
        orient: Option<(i32, f32)>,
        f: F,
    ) -> Result<(Vec<Bin>, BinStats, R), String>
    where
        F: FnOnce(&mut [Bin], &BinStats, Option<Lists<'_>>) -> R,
    {
        let n = pts.len();
        if n == 0 {
            let (mut b, st) = (Vec::new(), BinStats::default());
            let r = f(&mut b, &st, None);
            return Ok((b, st, r));
        }
        if n >= 1 << 30 {
            return Err(format!("too many points ({n})"));
        }
        let stride = if policy == NormalPolicy::Trust { 7 } else { 4 };
        let maxb = self.ctx.max_binding as usize;
        if n * 7 * 4 > maxb || n * BIN_WORDS * 4 > maxb {
            return Err(format!(
                "GPU buffer limit exceeded ({n} points, limit {maxb} B)"
            ));
        }
        let ctx = &*self.ctx;
        let mut lk;
        let mut guard = self.bufs.lock().unwrap();
        if guard.as_ref().is_none_or(|b| b.cap < n) {
            let cap = (n + n / 4).min(maxb / (BIN_WORDS * 4)).max(n);
            *guard = Some(self.make_bufs(cap));
        }
        let b = guard.as_mut().unwrap();

        let prof = std::env::var_os("DELTAMESH_BINS_PROF").is_some();
        let t0 = std::time::Instant::now();
        let mut marks: Vec<(&str, f64)> = Vec::new();
        let mut mark = |name: &'static str| {
            if prof {
                marks.push((name, t0.elapsed().as_secs_f64() * 1e3));
            }
        };
        let inv = 1.0 / size;
        let up_bytes = (n * stride * 4) as u64;
        let uma = self.uma;
        let up_buf = if uma { &b.pts } else { &b.upload };
        let up_slice = up_buf.slice(0..up_bytes);
        wait_map(ctx, up_slice, wgpu::MapMode::Write)?;
        let (mn, mx, nv) = {
            let mut view = up_slice
                .get_mapped_range_mut()
                .map_err(|e| format!("{e:?}"))?;
            const CH: usize = 4096;
            let mut parts = Vec::with_capacity(n.div_ceil(CH));
            let mut rest = view.slice(..);
            for ps in pts.chunks(CH) {
                let (a, r) = rest.split_at(ps.len() * stride * 4);
                parts.push((SendW(a), ps));
                rest = r;
            }
            parts
                .into_par_iter()
                .map(|(w, ps)| {
                    let mut w = w.0;
                    let mut tmp = vec![0u32; ps.len() * stride];
                    let (mut mn, mut mx, mut c) = ([i32::MAX; 3], [i32::MIN; 3], 0usize);
                    for (o, p) in tmp.chunks_exact_mut(stride).zip(ps.iter()) {
                        o[0] = p.pos[0].to_bits();
                        o[1] = p.pos[1].to_bits();
                        o[2] = p.pos[2].to_bits();
                        o[3] = p.rgb[0] as u32 | (p.rgb[1] as u32) << 8 | (p.rgb[2] as u32) << 16;
                        if stride == 7 {
                            o[4] = p.normal[0].to_bits();
                            o[5] = p.normal[1].to_bits();
                            o[6] = p.normal[2].to_bits();
                        }
                        if p.pos[0].is_finite() && p.pos[1].is_finite() && p.pos[2].is_finite() {
                            for a in 0..3 {
                                let k = (p.pos[a] * inv).floor() as i32;
                                mn[a] = mn[a].min(k);
                                mx[a] = mx[a].max(k);
                            }
                            c += 1;
                        }
                    }
                    w.copy_from_slice(bytemuck::cast_slice(&tmp));
                    (mn, mx, c)
                })
                .reduce(
                    || ([i32::MAX; 3], [i32::MIN; 3], 0),
                    |a, b| {
                        (
                            [a.0[0].min(b.0[0]), a.0[1].min(b.0[1]), a.0[2].min(b.0[2])],
                            [a.1[0].max(b.1[0]), a.1[1].max(b.1[1]), a.1[2].max(b.1[2])],
                            a.2 + b.2,
                        )
                    },
                )
        };
        up_buf.unmap();
        if nv == 0 {
            drop(guard);
            let (mut b, st) = (Vec::new(), BinStats::default());
            let r = f(&mut b, &st, None);
            return Ok((b, st, r));
        }
        let rng = |a: usize| (mx[a] as i64 - mn[a] as i64) as u32;
        let (bx, by, bz) = (bits_for(rng(0)), bits_for(rng(1)), bits_for(rng(2)));
        if bx.max(by).max(bz) > 30 || bx + by + bz > 63 {
            return Err(format!("cell key range too wide ({bx}+{by}+{bz} bits)"));
        }
        let kbits = bx + by + bz + 1;
        let passes = kbits.div_ceil(RADIX_BITS) as usize;
        let base = U {
            n: n as u32,
            stride: stride as u32,
            policy: match policy {
                NormalPolicy::Ignore => 0,
                NormalPolicy::Trust => 1,
                NormalPolicy::Estimate => 2,
            },
            radius: normal_radius,
            min_nb: normal_min_neighbors.min(u32::MAX as usize) as u32,
            size,
            inv,
            bx,
            by,
            bz,
            vbit: kbits - 1,
            mnx: mn[0],
            mny: mn[1],
            mnz: mn[2],
            rx: rng(0) as i32,
            ry: rng(1) as i32,
            rz: rng(2) as i32,
            ..Default::default()
        };
        let slab_n = rng(0) as usize + 2;
        let stab = policy != NormalPolicy::Ignore && slab_n <= SLAB_MAX;
        if stab && b.slab.0 < slab_n {
            let c = (slab_n + slab_n / 4).min(SLAB_MAX);
            b.slab = (
                c,
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("slab"),
                    size: (c * 4) as u64,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                }),
            );
        }
        let base = U {
            stab: stab as u32,
            ..base
        };
        lk = Some(ctx.lock.lock().unwrap());
        ctx.queue
            .write_buffer(&b.info, 0, bytemuck::cast_slice(&[0i32; 16]));
        let bg_even = self.bind(b, &b.ka, &b.va, &b.kb, &b.vb, &b.hist, &b.dummy);
        let bg_odd = self.bind(b, &b.kb, &b.vb, &b.ka, &b.va, &b.hist, &b.dummy);

        mark("upload");
        let nbs = tiles(n);
        let hist_len = 256 * nbs;
        let mut slots: Vec<U> = Vec::new();
        slots.push(base);
        slots.push(U {
            n: hist_len as u32,
            nb: tiles(hist_len) as u32,
            ..base
        });
        for p in 0..passes {
            slots.push(U {
                nb: nbs as u32,
                shift: p as u32 * RADIX_BITS,
                ..base
            });
        }
        let s_flag = slots.len();
        slots.push(U {
            n: nv as u32,
            nb: tiles(nv) as u32,
            flag: 1,
            ..base
        });
        let s_seg = slots.len();
        slots.push(U {
            n: nv as u32,
            ..base
        });
        let nbp = match (orient, &self.pipes.nb) {
            (Some(o), Some(p))
                if policy == NormalPolicy::Estimate && self.uma && o.0 == normal_radius =>
            {
                Some((o, p))
            }
            _ => None,
        };
        let s_nb = slots.len();
        if let Some(((orad, snz), _)) = nbp {
            slots.push(U {
                n: nv as u32,
                orad,
                snz,
                ..base
            });
            slots.push(U {
                n: nv as u32,
                nb: tiles(nv) as u32,
                ..base
            });
            slots.push(U {
                n: nv as u32,
                nb: tiles(nv) as u32,
                flag: 2,
                ..base
            });
        }
        assert!(slots.len() <= SLOTS);
        let mut ub = vec![0u8; slots.len() * self.align as usize];
        for (i, s) in slots.iter().enumerate() {
            let o = i * self.align as usize;
            ub[o..o + std::mem::size_of::<U>()].copy_from_slice(bytemuck::bytes_of(s));
        }
        ctx.queue.write_buffer(&self.uni, 0, &ub);
        let off = |s: usize| (s as u64 * self.align) as u32;

        let (kfin, vfin, kother, vother) = if passes.is_multiple_of(2) {
            (&b.ka, &b.va, &b.kb, &b.vb)
        } else {
            (&b.kb, &b.vb, &b.ka, &b.va)
        };
        let bg_seg = self.bind(b, kfin, vfin, &b.dummy, &b.dummy2, vother, kother);

        let nl_cap = if nbp.is_some() {
            let want = self
                .nl_want
                .load(std::sync::atomic::Ordering::Relaxed)
                .max(nv * 6)
                .max(1 << 20)
                .min(maxb / 4);
            if b.nlist.as_ref().is_none_or(|(c, _, _)| *c < want) {
                let c = want + want / 4;
                let c = c.min(maxb / 4);
                let mk = |label, usage| {
                    ctx.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(label),
                        size: (c * 4) as u64,
                        usage,
                        mapped_at_creation: false,
                    })
                };
                use wgpu::BufferUsages as B;
                b.nlist = Some((
                    c,
                    mk("gpu_bins nlist", B::STORAGE | B::COPY_SRC),
                    mk("gpu_bins nlist read", B::MAP_READ | B::COPY_DST),
                ));
            }
            b.nlist.as_ref().unwrap().0
        } else {
            0
        };
        let nb_groups = nbp.map(|(_, p)| {
            let nl = &b.nlist.as_ref().unwrap().1;
            let g1 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gpu_bins nb"),
                layout: &p.2,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: b.fin.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b.noff.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: nl.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: b.jo.as_entire_binding(),
                    },
                ],
            });
            (
                g1,
                self.bind(b, kfin, vfin, &b.dummy, &b.dummy2, &b.fin, kother),
                self.bind(b, kfin, vfin, &b.dummy, &b.dummy2, &b.noff, kother),
            )
        });
        let use_sg = self.sg.load(std::sync::atomic::Ordering::Relaxed);
        let scatter = if use_sg {
            self.pipes.scatter_sg.as_ref().unwrap()
        } else {
            &self.pipes.scatter
        };
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu_bins 2"),
            });
        if !uma {
            enc.copy_buffer_to_buffer(&b.upload, 0, &b.pts, 0, up_bytes);
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            let d1 = |pass: &mut wgpu::ComputePass<'_>,
                      pipe: &wgpu::ComputePipeline,
                      bg: &wgpu::BindGroup,
                      slot: usize,
                      wgs: usize| {
                pass.set_pipeline(pipe);
                pass.set_bind_group(0, bg, &[off(slot)]);
                let (x, y) = groups(wgs);
                pass.dispatch_workgroups(x, y, 1);
            };
            let scan = |pass: &mut wgpu::ComputePass<'_>,
                        bg: &wgpu::BindGroup,
                        slot: usize,
                        len: usize| {
                d1(pass, &self.pipes.scan_tile, bg, slot, tiles(len));
                d1(pass, &self.pipes.scan_top, bg, slot, 1);
                d1(pass, &self.pipes.scan_add, bg, slot, len.div_ceil(256));
            };
            d1(&mut pass, &self.pipes.pack, &bg_even, 0, n.div_ceil(256));
            for p in 0..passes {
                let bg = if p % 2 == 0 { &bg_even } else { &bg_odd };
                d1(&mut pass, &self.pipes.count, bg, 2 + p, nbs);
                scan(&mut pass, bg, 1, hist_len);
                d1(&mut pass, scatter, bg, 2 + p, nbs);
            }
            d1(
                &mut pass,
                &self.pipes.flags,
                &bg_seg,
                s_seg,
                nv.div_ceil(256),
            );
            scan(&mut pass, &bg_seg, s_flag, nv);
            d1(
                &mut pass,
                &self.pipes.segs,
                &bg_seg,
                s_seg,
                nv.div_ceil(256),
            );
            d1(
                &mut pass,
                &self.pipes.reduce,
                &bg_seg,
                s_seg,
                nv.div_ceil(256),
            );
            if stab {
                d1(
                    &mut pass,
                    &self.pipes.slab,
                    &bg_seg,
                    s_seg,
                    slab_n.div_ceil(256),
                );
            }
            if policy != NormalPolicy::Ignore && nbp.is_none() {
                d1(
                    &mut pass,
                    &self.pipes.estimate,
                    &bg_seg,
                    s_seg,
                    nv.div_ceil(64),
                );
            }
            if let (Some((_, p)), Some((g1, bg_fin, bg_noff))) = (nbp, &nb_groups) {
                let d2 = |pass: &mut wgpu::ComputePass<'_>, pipe: &wgpu::ComputePipeline| {
                    pass.set_pipeline(pipe);
                    pass.set_bind_group(0, &bg_seg, &[off(s_nb)]);
                    pass.set_bind_group(1, g1, &[]);
                    let (x, y) = groups(nv.div_ceil(64));
                    pass.dispatch_workgroups(x, y, 1);
                };
                d2(&mut pass, &p.0);
                scan(&mut pass, bg_fin, s_nb + 1, nv);
                scan(&mut pass, bg_noff, s_nb + 2, nv);
                d2(&mut pass, &p.1);
            }
        }
        if uma {
            if nbp.is_some() {
                let nl = b.nlist.as_ref().unwrap();
                enc.copy_buffer_to_buffer(&nl.1, 0, &nl.2, 0, (nl_cap * 4) as u64);
            }
            ctx.queue.submit(Some(enc.finish()));
            let isl = b.info.slice(0..64);
            let bsl = b.bins.slice(0..(nv * BIN_WORDS * 4) as u64);
            let nsl = nbp.map(|_| {
                (
                    b.jo.slice(0..((nv + 1) * 4) as u64),
                    b.nlist.as_ref().unwrap().2.slice(0..(nl_cap * 4) as u64),
                )
            });
            match nsl {
                Some((j, l)) => wait_map2(ctx, &[isl, bsl, j, l])?,
                None => wait_map2(ctx, &[isl, bsl])?,
            }
            lk.take();
            mark("GPU");
            let v = *bytemuck::from_bytes::<[i32; 16]>(
                &isl.get_mapped_range().map_err(|e| format!("{e:?}"))?[..],
            );
            b.info.unmap();
            let (m, bad, sg_bad) = (v[7] as usize, v[8], v[9]);
            let res = check(m, nv, bad, sg_bad).map(|()| {
                mark("map");
                let view = bsl.get_mapped_range();
                view.map(|view| {
                    convert(
                        &bytemuck::cast_slice::<u8, u32>(&view[..])[..m * BIN_WORDS],
                        policy,
                    )
                })
            });
            b.bins.unmap();
            let lsl = nsl.zip(nbp);
            let unmap_lists = || {
                if lsl.is_some() {
                    b.jo.unmap();
                    b.nlist.as_ref().unwrap().2.unmap();
                }
            };
            let mut out = match res {
                Ok(Ok(out)) => out,
                Ok(Err(e)) => {
                    unmap_lists();
                    return Err(format!("{e:?}"));
                }
                Err(Retry) => {
                    unmap_lists();
                    self.sg.store(false, std::sync::atomic::Ordering::Relaxed);
                    drop(guard);
                    lk.take();
                    return self.bin_points_nb(
                        pts,
                        size,
                        policy,
                        normal_radius,
                        normal_min_neighbors,
                        orient,
                        f,
                    );
                }
                Err(Fail(e)) => {
                    unmap_lists();
                    return Err(e);
                }
            };
            mark("convert");
            if prof {
                eprintln!(
                    "gpu_bins: {n} points, {m} cells, {passes} sort passes (unified memory): {marks:?}"
                );
            }
            lk.take();
            let Some(((j, l), ((orad, snz), _))) = lsl else {
                let r = f(&mut out.0, &out.1, None);
                return Ok((out.0, out.1, r));
            };
            let total = v[11] as usize;
            if total > nl_cap {
                self.nl_want
                    .store(total, std::sync::atomic::Ordering::Relaxed);
            }
            let r = {
                let keep = out.0.len();
                let views = if v[10] == 0 && total <= nl_cap {
                    j.get_mapped_range().ok().zip(l.get_mapped_range().ok())
                } else {
                    None
                };
                let lists = views.as_ref().and_then(|(jv, lv)| {
                    let off = &bytemuck::cast_slice::<u8, u32>(&jv[..])[..keep + 1];
                    let list = &bytemuck::cast_slice::<u8, u32>(&lv[..])[..total];
                    (off[keep] as usize == total).then_some(Lists {
                        radius: orad,
                        seed_nz: snz,
                        off,
                        list,
                    })
                });
                f(&mut out.0, &out.1, lists)
            };
            unmap_lists();
            return Ok((out.0, out.1, r));
        }
        enc.copy_buffer_to_buffer(&b.info, 0, &b.small, 0, 64);
        ctx.queue.submit(Some(enc.finish()));
        let (m, bad, sg_bad) = {
            let sl = b.small.slice(0..64);
            wait_map(ctx, sl, wgpu::MapMode::Read)?;
            let v = *bytemuck::from_bytes::<[i32; 16]>(
                &sl.get_mapped_range().map_err(|e| format!("{e:?}"))?[..],
            );
            b.small.unmap();
            (v[7] as usize, v[8], v[9])
        };
        match check(m, nv, bad, sg_bad) {
            Ok(()) => {}
            Err(Retry) => {
                self.sg.store(false, std::sync::atomic::Ordering::Relaxed);
                drop(guard);
                lk.take();
                return self.bin_points_nb(
                    pts,
                    size,
                    policy,
                    normal_radius,
                    normal_min_neighbors,
                    orient,
                    f,
                );
            }
            Err(Fail(e)) => return Err(e),
        }

        mark("sort+reduce+estimate");
        let bytes = (m * BIN_WORDS * 4) as u64;
        if b.stage.as_ref().is_none_or(|(c, _)| *c < bytes) {
            let c = (bytes + bytes / 4 + 255) & !255;
            let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu_bins stage"),
                size: c,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            b.stage = Some((c, buf));
        }
        let stage = &b.stage.as_ref().unwrap().1;
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu_bins 3"),
            });
        enc.copy_buffer_to_buffer(&b.bins, 0, stage, 0, bytes);
        ctx.queue.submit(Some(enc.finish()));
        let sl = stage.slice(0..bytes);
        wait_map(ctx, sl, wgpu::MapMode::Read)?;
        lk.take();
        mark("readback");
        let out = {
            let view = sl.get_mapped_range().map_err(|e| format!("{e:?}"))?;
            let rec: &[u32] = bytemuck::cast_slice(&view[..]);
            convert(rec, policy)
        };
        stage.unmap();
        mark("convert");
        if prof {
            eprintln!("gpu_bins: {n} points, {m} cells, {passes} sort passes: {marks:?}");
        }
        lk.take();
        let mut out = out;
        let r = f(&mut out.0, &out.1, None);
        Ok((out.0, out.1, r))
    }
}

/// Wrapper that lets disjoint write-only ranges of a mapped buffer move to rayon workers.
///
/// wgpu's `WriteOnly<[T]>` does not implement `Send` for unsized `T`, but it has the same meaning as
/// `&mut [u8]`.
struct SendW<'a>(wgpu::WriteOnly<'a, [u8]>);
// SAFETY: each wrapper owns an exclusive, non-overlapping range, and `u8` is `Send`.
unsafe impl Send for SendW<'_> {}

/// State bits of a cell record.
fn st_of(r: &[u32]) -> u32 {
    r[9] >> 30
}

/// Decodes a cell record into a [`Bin`].
fn to_bin(r: &[u32]) -> Bin {
    let f = |i: usize| f32::from_bits(r[i]);
    Bin {
        pos: [f(0), f(1), f(2)],
        normal: [f(3), f(4), f(5)],
        rgb: [f(6), f(7), f(8)],
        count: r[9] & 0x3fff_ffff,
    }
}

/// Converts GPU cell records into the CPU result format.
///
/// Dropped cells are removed, and `fixed` is filled when at least one cell was estimated or dropped.
/// Kept cells are counted per chunk first, so each chunk writes its own non-overlapping range of the
/// preallocated output in parallel.
fn convert(rec: &[u32], policy: NormalPolicy) -> (Vec<Bin>, BinStats) {
    const CH: usize = 8192;
    let counts: Vec<(usize, usize)> = rec
        .par_chunks(CH * BIN_WORDS)
        .map(|c| {
            let mut e = (0usize, 0usize);
            for r in c.chunks_exact(BIN_WORDS) {
                match st_of(r) {
                    ST_EST => e.0 += 1,
                    ST_DROP => e.1 += 1,
                    _ => {}
                }
            }
            e
        })
        .collect();
    let est: usize = counts.iter().map(|c| c.0).sum();
    let drop: usize = counts.iter().map(|c| c.1).sum();
    let m = rec.len() / BIN_WORDS;
    let keep = m - drop;
    let with_fixed = policy != NormalPolicy::Ignore && est + drop > 0;
    let mut bins: Vec<Bin> = Vec::with_capacity(keep);
    let mut fixed: Vec<bool> = if with_fixed {
        Vec::with_capacity(keep)
    } else {
        Vec::new()
    };
    {
        let mut bs = &mut bins.spare_capacity_mut()[..keep];
        let mut fs = &mut fixed.spare_capacity_mut()[..if with_fixed { keep } else { 0 }];
        let mut jobs = Vec::with_capacity(counts.len());
        for (ci, c) in rec.chunks(CH * BIN_WORDS).enumerate() {
            let k = c.len() / BIN_WORDS - counts[ci].1;
            let (a, r) = std::mem::take(&mut bs).split_at_mut(k);
            bs = r;
            let f = if with_fixed {
                let (a, r) = std::mem::take(&mut fs).split_at_mut(k);
                fs = r;
                a
            } else {
                &mut []
            };
            jobs.push((c, a, f));
        }
        jobs.into_par_iter().for_each(|(c, out, fx)| {
            let mut j = 0;
            for r in c.chunks_exact(BIN_WORDS) {
                let st = st_of(r);
                if st == ST_DROP {
                    continue;
                }
                out[j].write(to_bin(r));
                if !fx.is_empty() {
                    fx[j].write(st == ST_FIXED);
                }
                j += 1;
            }
        });
    }
    // SAFETY: the parallel loop above initialized every element of 0..keep, since each chunk's range
    // length equals its number of kept cells.
    unsafe {
        bins.set_len(keep);
        if with_fixed {
            fixed.set_len(keep);
        }
    }
    let stats = if with_fixed {
        BinStats {
            estimated: est,
            dropped: drop,
            fixed,
        }
    } else {
        BinStats::default()
    };
    (bins, stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bins;

    #[test]
    fn uniform_stride_is_a_multiple_of_the_alignment() {
        // The 96-byte `U` on a device asking for 64-byte offsets needs a 128-byte stride.
        assert_eq!(uniform_stride(64, 96), 128);
        assert_eq!(uniform_stride(256, 96), 256);
        assert_eq!(uniform_stride(32, 96), 96);
        assert_eq!(uniform_stride(0, 96), 96);
        assert_eq!(uniform_stride(64, std::mem::size_of::<U>() as u64) % 64, 0);
    }

    /// Binner on the shared device, or `None` when no GPU is available.
    fn binner() -> Option<GpuBinner> {
        let ctx = GpuCtx::shared().ok()?;
        Some(GpuBinner::new(ctx).expect("GpuBinner"))
    }

    /// Deterministic xorshift generator of uniform values in `[0, 1)`.
    fn rng(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// Random points mixing a dense cluster, negative coordinates, NaN and infinite positions, and
    /// invalid normals.
    fn random_points(n: usize, seed: u64) -> Vec<Point> {
        let mut f = rng(seed);
        (0..n)
            .map(|i| {
                let mut pos = [
                    (f() * 60.0 - 30.0) as f32,
                    (f() * 40.0 - 20.0) as f32,
                    (f() * 8.0 - 2.0) as f32,
                ];
                if i % 3 == 0 {
                    pos = [
                        (f() * 0.5) as f32 + 1.0,
                        (f() * 0.5) as f32,
                        (f() * 0.3) as f32,
                    ];
                }
                if i % 997 == 0 {
                    pos[i % 3] = if i % 2 == 0 { f32::NAN } else { f32::INFINITY };
                }
                let rgb = [
                    (f() * 255.0) as u8,
                    (f() * 255.0) as u8,
                    (f() * 255.0) as u8,
                ];
                let normal = match i % 7 {
                    0 => [f32::NAN; 3],
                    1 => [1e15, 0.0, f32::INFINITY],
                    2 => [0.1, 0.1, 0.1],
                    _ => [(f() - 0.5) as f32, (f() - 0.5) as f32, f() as f32 + 0.3],
                };
                Point { pos, rgb, normal }
            })
            .collect()
    }

    /// Several tilted noisy planes, one of them vertical, with negative coordinates. Only some of the
    /// input normals are valid; others are NaN or flipped.
    fn planes(seed: u64) -> Vec<Point> {
        let mut f = rng(seed);
        let mut v = Vec::new();
        let defs: [([f32; 3], [f32; 3], [f32; 3]); 3] = [
            ([-20.0, -15.0, -3.0], [1.0, 0.0, 0.15], [0.0, 1.0, -0.1]),
            ([5.0, -5.0, 2.0], [1.0, 0.2, 0.0], [0.0, 0.0, 1.0]),
            ([-8.0, 4.0, 0.5], [0.7, 0.7, 0.0], [-0.7, 0.7, 0.05]),
        ];
        for (o, a, b) in defs {
            let nrm = {
                let c = [
                    a[1] * b[2] - a[2] * b[1],
                    a[2] * b[0] - a[0] * b[2],
                    a[0] * b[1] - a[1] * b[0],
                ];
                let l = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
                [c[0] / l, c[1] / l, c[2] / l]
            };
            for _ in 0..150_000 {
                let (s, t) = (f() as f32 * 12.0, f() as f32 * 9.0);
                let e = (f() as f32 - 0.5) * 0.004;
                let pos = [
                    o[0] + a[0] * s + b[0] * t + nrm[0] * e,
                    o[1] + a[1] * s + b[1] * t + nrm[1] * e,
                    o[2] + a[2] * s + b[2] * t + nrm[2] * e,
                ];
                let k = (f() * 10.0) as u32;
                let normal = if k < 6 {
                    nrm
                } else if k < 8 {
                    [f32::NAN, 0.0, 1.0]
                } else {
                    [-nrm[0], -nrm[1], -nrm[2]]
                };
                v.push(Point {
                    pos,
                    rgb: [(f() * 255.0) as u8, 20, 200],
                    normal,
                });
            }
        }
        v
    }

    /// Reads a binary PLY with 27-byte vertices (x y z f32, r g b u8, nx ny nz f32).
    fn read_ply(path: &std::path::Path) -> Option<Vec<Point>> {
        let data = std::fs::read(path).ok()?;
        let end = data.windows(11).position(|w| w == b"end_header\n")? + 11;
        let head = std::str::from_utf8(&data[..end]).ok()?;
        let n: usize = head
            .lines()
            .find_map(|l| l.strip_prefix("element vertex "))?
            .trim()
            .parse()
            .ok()?;
        let body = &data[end..];
        assert!(body.len() >= n * 27);
        let f = |b: &[u8]| f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        Some(
            body[..n * 27]
                .chunks_exact(27)
                .map(|r| Point {
                    pos: [f(&r[0..]), f(&r[4..]), f(&r[8..])],
                    rgb: [r[12], r[13], r[14]],
                    normal: [f(&r[15..]), f(&r[19..]), f(&r[23..])],
                })
                .collect(),
        )
    }

    /// Reads `file` from the directory in `DELTAMESH_DATA`, or returns `None` when the variable is
    /// unset or the file is missing.
    fn read_data(file: &str) -> Option<Vec<Point>> {
        let dir = std::env::var_os("DELTAMESH_DATA")?;
        read_ply(&std::path::Path::new(&dir).join(file))
    }

    /// Matches each output bin (dropped cells excluded) to its entry in the full cell list by
    /// position bits; dropped cells map to `None`.
    fn align(all: &[Bin], out: &[Bin], fixed: &[bool]) -> Vec<Option<(Bin, bool)>> {
        let mut j = 0;
        let mut v = Vec::with_capacity(all.len());
        for a in all {
            if j < out.len() && out[j].pos.map(f32::to_bits) == a.pos.map(f32::to_bits) {
                v.push(Some((out[j], fixed.get(j).copied().unwrap_or(true))));
                j += 1;
            } else {
                v.push(None);
            }
        }
        assert_eq!(
            j,
            out.len(),
            "output is not a subsequence of the full cell list"
        );
        v
    }

    /// Bit pattern of a binning result, for exact comparisons.
    fn bits(r: &(Vec<Bin>, BinStats)) -> (Vec<[u32; 10]>, usize, usize, Vec<bool>) {
        let v =
            r.0.iter()
                .map(|b| {
                    let mut o = [0u32; 10];
                    for i in 0..3 {
                        o[i] = b.pos[i].to_bits();
                        o[3 + i] = b.normal[i].to_bits();
                        o[6 + i] = b.rgb[i].to_bits();
                    }
                    o[9] = b.count;
                    o
                })
                .collect();
        (v, r.1.estimated, r.1.dropped, r.1.fixed.clone())
    }

    /// Compares GPU binning with the CPU implementation.
    ///
    /// Positions, colors and counts are compared for every cell with the Ignore policy, since they do
    /// not depend on the policy. For other policies the GPU result must also be deterministic.
    ///
    /// Returns the number of cells, the fraction of estimated normals within 1 degree of the CPU
    /// result, and the fraction of cells whose drop/fixed state matches.
    fn compare(
        g: &GpuBinner,
        pts: &[Point],
        size: f32,
        policy: NormalPolicy,
        tag: &str,
    ) -> (usize, f64, f64) {
        let (r, mn) = (2, 6);
        let inv = 1.0 / size;
        let mut keys: Vec<[i32; 3]> = pts
            .par_iter()
            .filter(|p| p.pos.iter().all(|x| x.is_finite()))
            .map(|p| {
                [
                    (p.pos[0] * inv).floor() as i32,
                    (p.pos[1] * inv).floor() as i32,
                    (p.pos[2] * inv).floor() as i32,
                ]
            })
            .collect();
        keys.par_sort_unstable();
        keys.dedup();
        let (call, _) = bins::bin_points(pts, size, NormalPolicy::Ignore, r, mn);
        let (gall, gs) = g
            .bin_points(pts, size, NormalPolicy::Ignore, r, mn)
            .unwrap();
        assert!(gs.fixed.is_empty() && gs.estimated == 0 && gs.dropped == 0);
        assert_eq!(call.len(), keys.len());
        assert_eq!(gall.len(), keys.len(), "{tag}: cell count");
        let mut maxd = 0f32;
        for i in 0..keys.len() {
            let (c, q) = (&call[i], &gall[i]);
            assert_eq!(c.count, q.count, "{tag}: count {i} {:?}", keys[i]);
            for a in 0..3 {
                let d = (c.pos[a] - q.pos[a]).abs();
                maxd = maxd.max(d);
                assert!(
                    d < 1e-4,
                    "{tag}: position {i} {:?} {:?} {:?}",
                    keys[i],
                    c.pos,
                    q.pos
                );
                assert!((c.rgb[a] - q.rgb[a]).abs() < 0.5, "{tag}: color {i}");
            }
        }
        if policy == NormalPolicy::Ignore {
            return (keys.len(), 1.0, 1.0);
        }
        let (cout, cs) = bins::bin_points(pts, size, policy, r, mn);
        let gres = g.bin_points(pts, size, policy, r, mn).unwrap();
        let gres2 = g.bin_points(pts, size, policy, r, mn).unwrap();
        assert!(bits(&gres) == bits(&gres2), "{tag}: two runs differ");
        let (gout, gs) = gres;
        assert_eq!(gs.estimated + gs.dropped > 0, !gs.fixed.is_empty());
        let ca = align(&call, &cout, &cs.fixed);
        let ga = align(&gall, &gout, &gs.fixed);
        let (mut same_mask, mut n_est, mut est_ok, mut max_trust) = (0usize, 0usize, 0usize, 0f32);
        for (c, q) in ca.iter().zip(&ga) {
            match (c, q) {
                (None, None) => same_mask += 1,
                (Some((cb, cf)), Some((qb, qf))) => {
                    if cf == qf {
                        same_mask += 1;
                    }
                    if *cf && *qf {
                        for a in 0..3 {
                            max_trust = max_trust.max((cb.normal[a] - qb.normal[a]).abs());
                        }
                    } else if !cf && !qf {
                        n_est += 1;
                        let d: f32 = (0..3).map(|a| cb.normal[a] * qb.normal[a]).sum();
                        if d > 1f32.to_radians().cos() {
                            est_ok += 1;
                        }
                    }
                }
                _ => {}
            }
        }
        let n = keys.len();
        let mask_r = same_mask as f64 / n as f64;
        let est_r = if n_est == 0 {
            1.0
        } else {
            est_ok as f64 / n_est as f64
        };
        eprintln!(
            "{tag}: cells {n}, max position diff {maxd:.2e} | CPU estimated {} dropped {} / GPU estimated {} dropped {} | state match {:.5}, estimated within 1 deg {:.5} ({n_est}), max trusted normal diff {max_trust:.2e}",
            cs.estimated, cs.dropped, gs.estimated, gs.dropped, mask_r, est_r
        );
        assert!(max_trust < 1e-5, "{tag}: trusted normal diff {max_trust}");
        (n, est_r, mask_r)
    }

    /// GPU binning matches the CPU on synthetic inputs for all policies and several cell sizes.
    ///
    /// Random clouds have many nearly isotropic cells, so their estimated normal directions are only
    /// reported, not asserted.
    #[test]
    fn matches_cpu_synthetic() {
        let Some(g) = binner() else { return };
        let rp = random_points(300_000, 0x9e37_79b9_7f4a_7c15);
        let pl = planes(42);
        for &size in &[0.1f32, 0.05, 0.025] {
            for policy in [
                NormalPolicy::Ignore,
                NormalPolicy::Trust,
                NormalPolicy::Estimate,
            ] {
                let (_, e, m) =
                    compare(&g, &pl, size, policy, &format!("planes {size} {policy:?}"));
                assert!(e >= 0.999 && m >= 0.999, "planes {size} {policy:?}");
                let (_, _, m) =
                    compare(&g, &rp, size, policy, &format!("random {size} {policy:?}"));
                assert!(m >= 0.999, "random {size} {policy:?}");
            }
        }
    }

    /// Results with the subgroup scatter and unified-memory mapping enabled are bitwise identical to
    /// results with both disabled.
    #[test]
    fn modes_match_plain() {
        let Some(ctx) = GpuCtx::shared().ok() else {
            return;
        };
        let fast = GpuBinner::new(ctx.clone()).unwrap();
        let plain = GpuBinner::with_modes(ctx.clone(), false, false).unwrap();
        eprintln!(
            "subgroups {} unified memory {}",
            fast.sg.load(std::sync::atomic::Ordering::Relaxed),
            fast.uma
        );
        let mut sets = vec![random_points(200_000, 7), planes(3)];
        if let Some(p) = read_data("r0_preview_new.ply") {
            sets.push(p);
        }
        for pts in &sets {
            for &size in &[0.1f32, 0.025] {
                for policy in [NormalPolicy::Trust, NormalPolicy::Estimate] {
                    let a = fast.bin_points(pts, size, policy, 2, 6).unwrap();
                    let b = plain.bin_points(pts, size, policy, 2, 6).unwrap();
                    assert!(bits(&a) == bits(&b), "{size} {policy:?}");
                }
            }
        }
    }

    /// Orientation using the GPU neighbor lists is bitwise identical to orientation with the CPU
    /// neighbor search, with and without oracle answers.
    #[test]
    fn nb_lists_orient_matches_cpu_neighbors() {
        use crate::orient::{Orient, orient_normals_lists, orient_normals_par};
        let oracle_ans = |n: usize| {
            (0..n)
                .map(|i| match i % 11 {
                    0 => Some(-0.8),
                    1 => Some(0.9),
                    2 => Some(0.1),
                    _ => None,
                })
                .collect::<Vec<_>>()
        };
        let Some(g) = binner() else { return };
        if g.pipes.nb.is_none() {
            eprintln!("neighbor list kernels unavailable (no unified memory), skipping");
            return;
        }
        let mode = Orient::default();
        let Orient::Propagate {
            radius, seed_nz, ..
        } = mode
        else {
            unreachable!()
        };
        let mut sets = vec![
            ("planes", planes(5)),
            ("random", random_points(150_000, 11)),
        ];
        for f in ["r0_preview_new.ply", "r3_preview_new.ply"] {
            if let Some(p) = read_data(f) {
                sets.push((f, p));
            }
        }
        let mut used = 0;
        for (tag, pts) in &sets {
            for &size in &[0.1f32, 0.05, 0.025] {
                let bits = |v: &[Bin]| {
                    v.iter()
                        .map(|q| q.normal.map(f32::to_bits))
                        .collect::<Vec<_>>()
                };
                let (b, bs, r) = g
                    .bin_points_nb(
                        pts,
                        size,
                        NormalPolicy::Estimate,
                        2,
                        6,
                        Some((radius, seed_nz)),
                        |b, bs, l| {
                            let l = l?;
                            let mut out = Vec::new();
                            for with_oracle in [false, true] {
                                let ans = with_oracle.then(|| oracle_ans(b.len()));
                                let mut y = b.to_vec();
                                orient_normals_par(
                                    &mut y,
                                    &bs.fixed,
                                    size,
                                    mode,
                                    move |_| ans,
                                    Some(l),
                                );
                                out.push(bits(&y));
                            }
                            Some(out)
                        },
                    )
                    .unwrap();
                let Some(r) = r else {
                    eprintln!("{tag} {size}: no lists");
                    continue;
                };
                used += 1;
                for (k, with_oracle) in [false, true].into_iter().enumerate() {
                    let ans = with_oracle.then(|| oracle_ans(b.len()));
                    let mut x = b.clone();
                    orient_normals_lists(&mut x, &bs.fixed, size, mode, ans, None);
                    assert!(bits(&x) == r[k], "{tag} {size} oracle {with_oracle}");
                }
            }
        }
        assert!(used >= 6, "too few runs used the lists ({used})");
    }

    /// Benchmarks the binner modes, interleaving runs in one process to reduce machine noise.
    ///
    /// Reads input from `DELTAMESH_DATA`. Run with
    /// `cargo test --release -p deltamesh --features gpu gpu_bins::tests::bench_modes -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_modes() {
        let Some(ctx) = GpuCtx::shared().ok() else {
            return;
        };
        let modes = [(true, true), (true, false), (false, false)];
        let bs: Vec<GpuBinner> = modes
            .iter()
            .map(|&(a, b)| GpuBinner::with_modes(ctx.clone(), a, b).unwrap())
            .collect();
        for (file, policy) in [
            ("r0_refined.ply", NormalPolicy::Trust),
            ("r0_preview_new.ply", NormalPolicy::Estimate),
        ] {
            let Some(pts) = read_data(file) else {
                continue;
            };
            for &size in &[0.1f32, 0.025] {
                let mut t = vec![Vec::new(); bs.len()];
                for _ in 0..11 {
                    for (k, g) in bs.iter().enumerate() {
                        let s = std::time::Instant::now();
                        drop(g.bin_points(&pts, size, policy, 2, 6).unwrap());
                        t[k].push(s.elapsed().as_secs_f64() * 1e3);
                    }
                }
                let med: Vec<String> = t
                    .iter_mut()
                    .zip(&modes)
                    .map(|(v, m)| {
                        v.sort_by(f64::total_cmp);
                        format!("sg{} uma{} {:.2}", m.0 as u8, m.1 as u8, v[v.len() / 2])
                    })
                    .collect();
                eprintln!("{file} {policy:?} {size}: {}", med.join(" | "));
            }
        }
    }

    /// Benchmarks binning plus orientation with and without GPU neighbor lists, interleaving runs.
    ///
    /// Reads input from `DELTAMESH_DATA`. Run with
    /// `cargo test --release -p deltamesh --features gpu gpu_bins::tests::bench_nb -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_nb() {
        use crate::orient::{Orient, orient_normals_par};
        let Some(g) = binner() else { return };
        for file in ["r0_preview_new.ply", "r1_preview_new.ply"] {
            let Some(pts) = read_data(file) else {
                continue;
            };
            for &size in &[0.1f32, 0.025] {
                let o = Some((2, 0.7f32));
                let mut t = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
                for _ in 0..11 {
                    for with in [false, true] {
                        let s = std::time::Instant::now();
                        let (_, _, to) = g
                            .bin_points_nb(
                                &pts,
                                size,
                                NormalPolicy::Estimate,
                                2,
                                6,
                                if with { o } else { None },
                                |b, bs, l| {
                                    let s = std::time::Instant::now();
                                    orient_normals_par(
                                        b,
                                        &bs.fixed,
                                        size,
                                        Orient::default(),
                                        |_| None,
                                        l,
                                    );
                                    s.elapsed().as_secs_f64() * 1e3
                                },
                            )
                            .unwrap();
                        let tb = s.elapsed().as_secs_f64() * 1e3 - to;
                        let k = with as usize * 2;
                        t[k].push(tb);
                        t[k + 1].push(to);
                    }
                }
                let m: Vec<f64> = t
                    .iter_mut()
                    .map(|v| {
                        v.sort_by(f64::total_cmp);
                        v[v.len() / 2]
                    })
                    .collect();
                eprintln!(
                    "{file} {size}: without lists bin {:.2} + orient {:.2} = {:.2} | with lists bin {:.2} + orient {:.2} = {:.2}",
                    m[0],
                    m[1],
                    m[0] + m[1],
                    m[2],
                    m[3],
                    m[2] + m[3]
                );
            }
        }
    }

    /// Empty input, only invalid points, a single cell, an over-wide key range, and buffer reuse after
    /// a larger input.
    #[test]
    fn edge_cases() {
        let Some(g) = binner() else { return };
        assert!(
            g.bin_points(&[], 0.1, NormalPolicy::Trust, 2, 6)
                .unwrap()
                .0
                .is_empty()
        );
        let bad = [Point {
            pos: [f32::NAN, 0.0, 0.0],
            ..Default::default()
        }];
        assert!(
            g.bin_points(&bad, 0.1, NormalPolicy::Trust, 2, 6)
                .unwrap()
                .0
                .is_empty()
        );
        let one = [Point {
            pos: [-1.05, 2.0, -3.0],
            rgb: [1, 2, 3],
            normal: [0.0, 0.0, 1.0],
        }];
        let (b, s) = g.bin_points(&one, 0.1, NormalPolicy::Trust, 2, 6).unwrap();
        assert_eq!(b.len(), 1);
        assert!(s.fixed.is_empty());
        let (b, s) = g
            .bin_points(&one, 0.1, NormalPolicy::Estimate, 2, 6)
            .unwrap();
        assert!(b.is_empty() && s.dropped == 1);
        let far = [
            Point {
                pos: [-3e9, -3e9, -3e9],
                ..Default::default()
            },
            Point {
                pos: [3e9, 3e9, 3e9],
                ..Default::default()
            },
        ];
        assert!(
            g.bin_points(&far, 0.001, NormalPolicy::Ignore, 2, 6)
                .is_err()
        );
        let rp = random_points(50_000, 3);
        compare(&g, &rp, 0.2, NormalPolicy::Trust, "small after");
    }

    /// GPU binning matches the CPU on real captures from `DELTAMESH_DATA`; skipped when unset.
    #[test]
    fn matches_cpu_real() {
        if std::env::var_os("DELTAMESH_DATA").is_none() {
            return;
        }
        let Some(g) = binner() else { return };
        for (file, policy) in [
            ("r0_refined.ply", NormalPolicy::Trust),
            ("r0_preview_new.ply", NormalPolicy::Estimate),
        ] {
            let Some(pts) = read_data(file) else {
                eprintln!("{file} not found, skipping");
                continue;
            };
            for &size in &[0.1f32, 0.05, 0.025] {
                let (_, e, m) = compare(&g, &pts, size, policy, &format!("{file} {size}"));
                assert!(
                    e >= 0.999 && m >= 0.999,
                    "{file} {size}: estimated {e} state {m}"
                );
            }
        }
    }

    /// Benchmarks GPU binning against the CPU with one thread and with all threads.
    ///
    /// Reads input from `DELTAMESH_DATA`. Run with
    /// `cargo test --release -p deltamesh --features gpu gpu_bins::tests::bench -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench() {
        let Some(g) = binner() else { return };
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let med = |f: &mut dyn FnMut() -> f64| {
            let mut v: Vec<f64> = (0..5).map(|_| f()).collect();
            v.sort_by(f64::total_cmp);
            v[2]
        };
        let t = |f: &mut dyn FnMut()| {
            let s = std::time::Instant::now();
            f();
            s.elapsed().as_secs_f64() * 1e3
        };
        eprintln!(
            "GPU: {}, CPU threads {}",
            g.ctx.info,
            rayon::current_num_threads()
        );
        for file in ["r0_refined.ply", "r0_preview_new.ply"] {
            let Some(pts) = read_data(file) else {
                continue;
            };
            for policy in [NormalPolicy::Trust, NormalPolicy::Estimate] {
                for &size in &[0.1f32, 0.05, 0.025] {
                    g.bin_points(&pts, size, policy, 2, 6).unwrap();
                    let gpu = med(&mut || {
                        t(&mut || drop(g.bin_points(&pts, size, policy, 2, 6).unwrap()))
                    });
                    let c1 =
                        med(&mut || {
                            t(&mut || {
                                one.install(|| drop(bins::bin_points(&pts, size, policy, 2, 6)))
                            })
                        });
                    let cn =
                        med(&mut || t(&mut || drop(bins::bin_points(&pts, size, policy, 2, 6))));
                    let nb = g.bin_points(&pts, size, policy, 2, 6).unwrap().0.len();
                    eprintln!(
                        "{file:20} {policy:?}\t{size}\tcells {nb:8}\tGPU {gpu:7.1} ms\tCPU1 {c1:7.1} ms\tCPU all {cn:7.1} ms"
                    );
                }
            }
        }
    }
}
