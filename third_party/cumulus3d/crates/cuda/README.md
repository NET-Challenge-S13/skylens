English | [한국어](https://github.com/AH100-1/cumulus3d/blob/main/crates/cuda/README.ko.md)

# cumulus3d-cuda

GPU backends: multi-view stereo (PatchMatch), SIFT descriptor matching, and the SIFT scale space, computed with CUDA.
CUDA C kernels are compiled at run time through cudarc (driver API + NVRTC), and the crate plugs in by implementing
backend traits defined in other crates.

API doc comments are currently in Korean; English translation is planned.

**Pipeline stage**: the compute parts of feature extraction (`cumulus3d_features::SiftEngine` → `CudaSift`), matching (`cumulus3d_matching::MatcherBackend` → `CudaMatcher`),
and densification (`cumulus3d_dense::PatchMatchBackend` → `CudaPatchMatch`). `cumulus3d stream --gpu` uses these backends.

## Main entry points

| Function/type | Role | Input → output |
|---|---|---|
| `is_available` | Check that the driver and NVRTC libraries and a device are present (false if not, no panic) | `()` → `bool` |
| `CudaPatchMatch::try_default` | PatchMatch backend on device 0 with default options | `()` → `Result<CudaPatchMatch, GpuError>` |
| `CudaPatchMatch` | `PatchMatchBackend` implementation. Pass it to `cumulus3d_dense::densify` | `&DenseScene` … → `DenseOutput` (via dense) |
| `CudaMatcher::try_default` | Descriptor matching backend on device 0 | `()` → `Result<CudaMatcher, GpuError>` |
| `CudaMatcher` | `MatcherBackend` implementation (integer dot product, row and column top-2) | two descriptor sets → `(Vec<Top2>, Vec<Top2>)` |
| `CudaSift::try_default` | SIFT engine on device 0 | `()` → `Result<CudaSift, GpuError>` |
| `CudaSift` | `SiftEngine` implementation (GPU scale space + CPU detection and description, bit-identical to `CpuSift`) | `&GrayImage`, `&SiftOptions` → `Result<SiftOutput>` |
| `CudaDevice::new` | Create a context and a dedicated stream for device `ordinal` (shared by several backends) | `usize` → `Result<Arc<CudaDevice>, GpuError>` |

## Public items

Submodules are private and every item is re-exported at the crate root (`cumulus3d_cuda::`).

| Item | Kind | Role |
|---|---|---|
| `is_available` | fn | Whether CUDA is usable |
| `GpuError` | enum | `NotAvailable`, `Driver` (cudarc driver error), `Compile` (NVRTC failure) |
| `CudaDevice` | struct | One GPU's context and dedicated stream, with the NVRTC architecture matched to its compute capability |
| `CudaDevice::new` | fn | Open a device → `Arc<CudaDevice>` |
| `CudaDevice::name` | fn | Device name |
| `CudaPatchMatch` | struct | `cumulus3d_dense::PatchMatchBackend` implementation (per-scale run, evaluation, readback, upsampling, median filter) |
| `CudaPatchMatch::new` / `try_default` | fn | Construct from a device and options / device 0 with default options |
| `CudaPatchMatch::device` / `options` | fn | Device in use / options |
| `CudaPatchMatchOptions` | struct | `hw_interp`(true), `batch_pixels`(2,200,000), `max_batch`(64), `min_blocks`(2), `fast_math`(true) |
| `CudaMatcher` | struct | `cumulus3d_matching::MatcherBackend` implementation |
| `CudaMatcher::new` / `try_default` | fn | Construct from a device / device 0 |
| `CudaMatcher::top2_gpu` | fn | GPU row and column top-2 (returns errors as-is; the trait's `top2` warns and computes on the CPU on error) |
| `CudaSift` | struct | `cumulus3d_features::SiftEngine` implementation (on device errors, warns and computes with `CpuSift`) |
| `CudaSift::new` / `try_default` | fn | Construct from a device / device 0 |

## Example

Same code as the doc-test in the crate documentation. Does nothing if CUDA is unavailable.

```rust
use cumulus3d_cuda::{is_available, CudaMatcher};
use cumulus3d_matching::MatcherBackend;

// False on machines without CUDA, so nothing happens there.
if is_available() {
    let matcher = CudaMatcher::try_default().expect("CUDA device 0");
    let (d1, d2) = (vec![1u8; 128 * 4], vec![1u8; 128 * 3]); // 4 and 3 SIFT descriptors of 128 bytes
    let (rows, cols) = matcher.top2(&d1, 4, &d2, 3);
    assert_eq!((rows.len(), cols.len()), (4, 3));
}
```

Passing the GPU backend to densification:

```rust,no_run
use cumulus3d_cuda::CudaPatchMatch;
use cumulus3d_dense::{densify, DenseScene, DensifyOptions};

fn run(scene: &DenseScene) -> Result<(), Box<dyn std::error::Error>> {
    let pm = CudaPatchMatch::try_default()?; // device 0, default options
    let out = densify(scene, &DensifyOptions::default(), &pm, None)?;
    println!("{} dense points", out.cloud.len());
    Ok(())
}
```

## Feature flags and hardware

- No feature flags.
- Running requires an NVIDIA GPU with the **CUDA 12.x driver** (`libcuda`) and NVRTC (`libnvrtc`). The library path must be
  in `LD_LIBRARY_PATH` (e.g. `/usr/local/cuda-12.4/lib64`).
- cudarc is used with `dynamic-loading`, so **the crate builds and tests on machines without CUDA**. Without a device, `is_available()` is false and
  the GPU tests (`tests/*_gpu.rs`) are skipped.
- Kernels are compiled with NVRTC once per device (about 1 s). Known compute capabilities: 7.0 – 9.0.
- Environment variables: `CUMULUS3D_PM_MIN_BLOCKS`, `CUMULUS3D_PM_FAST_MATH` (PatchMatch option defaults),
  `CUMULUS3D_CUDA_TRACE=1` (prints kernel resources and per-call timings).
- `unsafe` is used only where cudarc requires it (kernel launches, pinned host memory, texture objects, `DeviceRepr`).
- PatchMatch produces bit-identical results for the same device and input (counter-based random numbers).
