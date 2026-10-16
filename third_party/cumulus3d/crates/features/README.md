English | [한국어](https://github.com/AH100-1/cumulus3d/blob/main/crates/features/README.ko.md)

# cumulus3d-features

Image reading (decoding, EXIF, grayscale conversion, downscaling, orientation rotation), initial camera parameter selection, and deterministic SIFT feature extraction.

> API doc comments are currently in Korean; English translation is planned.

**Pipeline stage**: the first stage (feature extraction). Takes a batch of images, assigns cameras, and incrementally writes keypoints and descriptors
to `cumulus3d_core::FeatureStore`. `cumulus3d-matching` reads this store.
SIFT computation sits behind the `SiftEngine` trait, so it can be swapped for a GPU backend (`CudaSift` in `cumulus3d-cuda`).

## Main entry points

| Function/type | Role | Input → output |
|---|---|---|
| `FeatureExtractor::new()` / `with_backend(Arc<dyn SiftEngine>)` | Create an extractor (CPU default / given backend) | backend → `FeatureExtractor` |
| `FeatureExtractor::extract_inputs` | Add decoded images to the store (for per-position calls) | `&FeatureStore`, `Vec<ImageSource>`, `&ExtractionOptions` → `Result<Vec<ImageReport>>` |
| `FeatureExtractor::extract_files` | Read a list of files (EXIF + decoding, in parallel) and add them to the store | `&FeatureStore`, root folder, name list, `&ExtractionOptions` → `Result<Vec<ImageReport>>` |
| `ExtractionOptions` / `ReaderOptions` / `SiftOptions` | Extraction, reading, and SIFT options | Start from `Default` and modify fields |
| `CpuSift::new()` + `SiftEngine::extract` | Extract SIFT from one grayscale image | `&GrayImage`, `&SiftOptions` → `Result<SiftOutput>` |
| `read_gray` | Image file → 8-bit grayscale image | path → `Result<GrayImage>` |
| `ExifInfo::read` / `from_bytes` | EXIF summary (focal length, sensor, orientation, GPS) | path or bytes → `ExifInfo` |
| `init_camera` | Initialize a new camera from EXIF/defaults | model, W, H, `&ExifInfo`, parameters (optional), factor → `Result<Camera>` |
| `extract_for_camera` | Downscale → EXIF orientation rotation → SIFT → convert to camera-size coordinates | backend, `&GrayImage`, orientation, camera size, max size, `&SiftOptions` → `Result<(Vec<Keypoint>, SiftOutput)>` |
| `read_image_list` | Read an image list file | path → `Result<Vec<String>>` |

## Public items

"Root" in the tables marks items re-exported at the crate root as `cumulus3d_features::name`.

### `extract`

| Item | Kind | Role |
|---|---|---|
| `FeatureExtractor` (root) | struct | SIFT backend + store writer. Cheap to `Clone` |
| `FeatureExtractor::new` / `with_backend` / `backend` | fn | Create with the CPU backend / create with a given backend / backend in use |
| `FeatureExtractor::extract_files` | fn | Read a file list (processed in lexicographic name order), extract and write. With `Existing(N)`, fails immediately if camera N does not exist; per-image failures go into the report |
| `FeatureExtractor::extract_inputs` | fn | Extract and write a list of decoded images (ids assigned in lexicographic name order) |
| `ExtractionOptions` (root) | struct | `reader`, `sift`, `sequential_images` |
| `ReaderOptions` (root) | struct | Camera model (default `OpenCv`), grouping mode, fixed parameters, default focal factor (1.2), max image size (3200), strict size check, recording EXIF GPS position priors |
| `CameraMode` (root) | enum | Camera grouping: `PerImage`, `Single`, `PerFolder` (default), `Existing(CameraId)` |
| `ImageSource` (root) | struct | In-memory input image: `name`, `gray`, `exif` |
| `ImageReport` (root) | struct | Processing report for one image: `name`, `status` |
| `ImageStatus` (root) | enum | `Extracted{image_id, camera_id, num_features}`, `AlreadyExists{image_id}`, `Failed{error}` |
| `extract_for_camera` (root) | fn | Extract one image, then convert keypoints to camera-size coordinates |
| `read_image_list` (root) | fn | List file: trims whitespace on each line, skips empty lines |
| `image_folder` (root) | fn | Parent folder of a name (`"camF/x.jpg"` → `"camF"`) |

### `gray`

| Item | Kind | Role |
|---|---|---|
| `GrayImage` (root) | struct | Row-major 8-bit grayscale image (`width`, `height`, `data`) |
| `GrayImage::new` / `filled` / `from_f32` | fn | Create from a buffer (length checked) / solid fill / from a `[0,1]` float function |
| `GrayImage::get` / `rotate_ccw` / `resized` | fn | Pixel value / counter-clockwise 90°×k rotation / Lanczos3 resampling |
| `read_gray` (root) | fn | Decode a file → grayscale (no automatic EXIF rotation) |
| `to_gray` | fn | `image::DynamicImage` → grayscale (drops alpha, 16-bit → 8-bit) |
| `rgb_to_gray` (root) | fn | `round(0.2126 R + 0.7152 G + 0.0722 B)` |
| `limited_size` (root) | fn | Size after applying the max-size limit |
| `rotate_keypoint_ccw` (root) | fn | Rotate a keypoint counter-clockwise by 90°×k |
| `orientation_to_rotation` | fn | EXIF Orientation → number of counter-clockwise rotations |
| `orientation_gravity` | fn | EXIF Orientation → gravity direction in image coordinates |

### `exif_info`

| Item | Kind | Role |
|---|---|---|
| `ExifInfo` (root) | struct | Make, model, focal length (mm, 35mm equivalent), focal plane resolution, orientation, GPS (latitude, longitude, altitude) |
| `ExifInfo::read` / `from_bytes` | fn | Read from a file / in-memory bytes (empty values if there is no EXIF) |

### `camera_init`

| Item | Kind | Role |
|---|---|---|
| `FocalEstimate` (root) | struct | Focal length (`focal`) and whether it came from EXIF (`prior`) |
| `infer_focal` (root) | fn | Focal rule: 35mm equivalent → focal plane resolution → sensor width table → factor × max(W, H) |
| `init_camera` (root) | fn | New camera: principal point (W/2, H/2), zero distortion; given parameters are used as is |
| `lookup_sensor_width` (root) | fn | Look up sensor width (mm) by make and model in the built-in table |

### `sift`

| Item | Kind | Role |
|---|---|---|
| `SiftEngine` (root) | trait | SIFT backend: `name()`, `extract(&GrayImage, &SiftOptions)` |
| `CpuSift` (root) | struct | CPU (rayon) backend; reuses pyramid buffers across images. `new()` |
| `SiftOptions` (root) | struct | Max features (8192), first octave (−1), number of octaves (auto), number of levels (3), thresholds, number of orientations, normalization, selection mode, etc. |
| `SiftFeature` (root) | struct | One feature (x, y, scale, orientation, octave, level, response). `keypoint()` gives an affine keypoint |
| `SiftOutput` (root) | struct | `features` + `descriptors` (row i ↔ feature i). `keypoints()`, `len()`, `is_empty()` |
| `FeatureSelection` (root) | enum | Max-feature limiting: `CompatLevels` (per level, default), `TopK`, `SpatialGrid{cells}` |
| `DescriptorNormalization` (root) | enum | Descriptor normalization: `L1Root` (default), `L2` |

### `sift::pyramid` (low level)

| Item | Kind | Role |
|---|---|---|
| `BufferPool` | struct | Reusable f32 buffer pool: `take`, `give` |
| `gaussian_kernel` | fn | 1D Gaussian kernel (width [5, 33], sums to 1) |
| `gaussian_blur` | fn | Separable Gaussian blur (edge replication) |
| `upsample2` / `decimate2` | fn | 2× upsampling / 1/2 decimation keeping even rows and columns |
| `Octave` | struct | Gaussian levels of one octave. `level(l)` |
| `ScaleSpace` | struct | Scale-space constants (S, k, σ₀). `new`, `sigma`, `sigma_inc` |
| `auto_num_octaves` | fn | Automatically determine the number of octaves |
| `build_pyramid` | fn | Build the Gaussian pyramid |

### `sift::detect` (low level)

| Item | Kind | Role |
|---|---|---|
| `Candidate` | struct | Detected point (octave coordinates, σ, refined DoG response) |
| `DetectParams` | struct | Detection parameters (thresholds, refinement iterations, singular Hessian handling, σ₀, k) |
| `detect_level` | fn | DoG extremum detection, edge suppression, and subpixel refinement at one detection level |
| `dog` | fn | Compute the difference of Gaussians (DoG) |

### `sift::orient` (low level)

| Item | Kind | Role |
|---|---|---|
| `Gradient` | struct | Gradients of a Gaussian level. `lazy` (on the fly), `precomputed` (precomputed, bit-identical results), `width`, `at` |
| `OrientParams` | struct | Orientation assignment parameters |
| `quantize_angle` / `dequantize_angle` | fn | Angle ↔ 16-bit quantization |
| `orientations` | fn | Dominant orientations of a keypoint |
| `descriptor` | fn | 128-dimensional uint8 descriptor |
| `normalize_quantize` | fn | Descriptor normalization and quantization |

## Example

Run SIFT on a synthetic image and write it to the store (same code as the crate-level doc-test).

```rust
use cumulus3d_core::FeatureStore;
use cumulus3d_features::{
    CpuSift, ExifInfo, ExtractionOptions, FeatureExtractor, GrayImage, ImageSource, ImageStatus, SiftEngine,
    SiftOptions,
};

// Synthetic image: three bright blobs.
let blob = |x: usize, y: usize, cx: f32, cy: f32, s: f32| {
    let (dx, dy) = (x as f32 - cx, y as f32 - cy);
    (-(dx * dx + dy * dy) / (2.0 * s * s)).exp()
};
let img = GrayImage::from_f32(320, 240, |x, y| {
    0.2 + 0.6 * (blob(x, y, 80.0, 60.0, 6.0) + blob(x, y, 200.0, 150.0, 9.0) + blob(x, y, 260.0, 70.0, 5.0))
});

// 1) Call the SIFT backend directly: feature i ↔ descriptor row i.
let out = CpuSift::new().extract(&img, &SiftOptions::default())?;
assert!(!out.is_empty());
assert_eq!(out.features.len(), out.descriptors.len());

// 2) Write to the store (camera assignment + keypoints and descriptors).
let store = FeatureStore::new();
let src = ImageSource { name: "camF/0001.jpg".into(), gray: img, exif: ExifInfo::default() };
let reports = FeatureExtractor::new().extract_inputs(&store, vec![src], &ExtractionOptions::default())?;
assert!(matches!(reports[0].status, ImageStatus::Extracted { .. }));
assert_eq!(store.num_images(), 1);
```

To read from files, use `FeatureExtractor::extract_files(&store, root, &names, &opts)`.
Benchmark: `cargo run --release -p cumulus3d-features --example bench_sift -- <image|synthetic> 2048 1152`.

## Feature flags and hardware

- No feature flags. Pure CPU (rayon) implementation; no special hardware required.
- To run SIFT on the GPU, pass `CudaSift` from `cumulus3d-cuda` to `FeatureExtractor::with_backend` (requires a CUDA 12.x driver).

## Behavior notes

- Output is fully deterministic: ascending octave → ascending level → row-major detection order, with a second orientation immediately following its first.
- All improvement options that change results are off by default: `truncate_width_to_4 = false`, `refinement_iterations = 5`,
  `reject_singular_refinement`, `orientation_bin_interpolation`, `FeatureSelection::TopK`/`SpatialGrid`, `num_octaves = Some(n)`.
- If the name already exists with both keypoints and descriptors, it is skipped (`AlreadyExists`). If only the row exists, only the features are filled in.
- When `read_pose_priors` is on and EXIF GPS is present, a position prior (latitude, longitude, altitude) is recorded in the store.
- The store has no rig/frame rows. In the reconstruction stage, `Reconstruction::add_image_own_frame` creates a frame per image.
