//! Rust twin of `utils/tiling.py`.
//!
//! Native-scale tiling. Instead of squashing a frame to 512x512, the frame is
//! scaled so its short side is ~765 px (VisDrone's native 1360x765) and the
//! model sees overlapping 512x512 windows of it:
//!
//! - [`scale_to_short_side`] scales the frame (and boxes), choosing area
//!   interpolation when shrinking and bilinear when enlarging, as `cv2.resize`
//!   with `INTER_AREA` / `INTER_LINEAR` does.
//! - [`tile_starts`] / [`tile_grid`] lay an overlapping grid with at least
//!   `min_overlap` px overlap.
//! - [`tile_detections_to_full`] maps detections back, and
//!   [`merge_tile_detections`] removes duplicates found by neighbouring tiles.
//!
//! Boxes are xyxy pixels (dataset contract); detections are `[x, y, w, h,
//! score]` centre format ([`Detection`]).
//!
//! OpenCV is not linked. [`resize`] reimplements the two `cv2.resize` modes on
//! `f32` images: the per-axis source indices and weights follow OpenCV's
//! `resize` (`computeResizeAreaTab` for shrinking, the half-pixel bilinear
//! tables otherwise, including OpenCV's "area" coefficients when `INTER_AREA`
//! enlarges), and accumulation is in `f32` in the same horizontal-then-vertical
//! order. Results agree with OpenCV to float rounding (about 1e-6), not
//! bit-for-bit: OpenCV may vectorise the sums in a different order.
//!
//! The training-side cache builder (`build_scaled_cache`) and the random-crop
//! dataset (`ScaledTileCropDataset`) are not ported; only [`crop_tile`]
//! reproduces their zero padding for frames smaller than a tile.

use crate::utils::metrics::{Detection, box_iou};

/// Target short side of the native-scale frame.
pub const SHORT_SIDE: u32 = 765;
/// Model window size.
pub const TILE_SIZE: u32 = 512;
/// Minimum overlap between neighbouring windows.
pub const MIN_OVERLAP: u32 = 64;

/// Row-major `f32` image, interleaved channels (HWC), as numpy holds it.
#[derive(Debug, Clone, PartialEq)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub data: Vec<f32>,
}

impl Image {
    pub fn new(width: usize, height: usize, channels: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), width * height * channels, "image buffer size mismatch");
        Self { width, height, channels, data }
    }

    pub fn zeros(width: usize, height: usize, channels: usize) -> Self {
        Self { width, height, channels, data: vec![0.0; width * height * channels] }
    }

    #[inline]
    pub fn at(&self, x: usize, y: usize, c: usize) -> f32 {
        self.data[(y * self.width + x) * self.channels + c]
    }
}

/// `cv2.resize` interpolation modes that the Python pipeline uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    /// `cv2.INTER_LINEAR`.
    Linear,
    /// `cv2.INTER_AREA`.
    Area,
}

/// Per destination index: (source index, weight) pairs.
type AxisTable = Vec<Vec<(usize, f32)>>;

/// Half-pixel bilinear table (`INTER_LINEAR`), or OpenCV's "area"
/// coefficients used when `INTER_AREA` enlarges.
fn linear_table(src: usize, dst: usize, area_coeffs: bool) -> AxisTable {
    let scale = src as f64 / dst as f64;
    let inv_scale = dst as f64 / src as f64;
    (0..dst)
        .map(|dx| {
            let (mut sx, mut fx) = if area_coeffs {
                let sx = (dx as f64 * scale).floor() as i64;
                let fx = ((dx + 1) as f64 - (sx + 1) as f64 * inv_scale) as f32;
                (sx, if fx <= 0.0 { 0.0 } else { fx - fx.floor() })
            } else {
                let f = ((dx as f64 + 0.5) * scale - 0.5) as f32;
                let sx = f.floor();
                (sx as i64, f - sx)
            };
            if sx < 0 {
                sx = 0;
                fx = 0.0;
            }
            if sx + 1 >= src as i64 {
                sx = src as i64 - 1;
                fx = 0.0;
            }
            let sx = sx as usize;
            let next = (sx + 1).min(src - 1);
            vec![(sx, 1.0 - fx), (next, fx)]
        })
        .collect()
}

/// OpenCV `computeResizeAreaTab`: exact pixel-area coverage when shrinking.
fn area_table(src: usize, dst: usize) -> AxisTable {
    let scale = src as f64 / dst as f64;
    (0..dst)
        .map(|dx| {
            let mut tab = Vec::new();
            let fsx1 = dx as f64 * scale;
            let fsx2 = fsx1 + scale;
            let cell = scale.min(src as f64 - fsx1);
            let mut sx1 = fsx1.ceil() as i64;
            let mut sx2 = fsx2.floor() as i64;
            sx2 = sx2.min(src as i64 - 1);
            sx1 = sx1.min(sx2);
            if sx1 as f64 - fsx1 > 1e-3 {
                tab.push(((sx1 - 1) as usize, ((sx1 as f64 - fsx1) / cell) as f32));
            }
            for sx in sx1..sx2 {
                tab.push((sx as usize, (1.0 / cell) as f32));
            }
            if fsx2 - sx2 as f64 > 1e-3 {
                let w = (fsx2 - sx2 as f64).min(1.0).min(cell) / cell;
                tab.push((sx2 as usize, w as f32));
            }
            tab
        })
        .collect()
}

/// Separable resampling: horizontal pass on every source row, then vertical.
fn apply_tables(img: &Image, xt: &AxisTable, yt: &AxisTable) -> Image {
    let (c, nw, nh) = (img.channels, xt.len(), yt.len());
    let mut tmp = vec![0.0f32; img.height * nw * c];
    for y in 0..img.height {
        let src_row = &img.data[y * img.width * c..(y + 1) * img.width * c];
        let dst_row = &mut tmp[y * nw * c..(y + 1) * nw * c];
        for (dx, taps) in xt.iter().enumerate() {
            for ch in 0..c {
                let mut s = 0.0f32;
                for &(sx, w) in taps {
                    s += src_row[sx * c + ch] * w;
                }
                dst_row[dx * c + ch] = s;
            }
        }
    }
    let mut out = vec![0.0f32; nh * nw * c];
    for (dy, taps) in yt.iter().enumerate() {
        let dst_row = &mut out[dy * nw * c..(dy + 1) * nw * c];
        for &(sy, w) in taps {
            let src_row = &tmp[sy * nw * c..(sy + 1) * nw * c];
            for (d, s) in dst_row.iter_mut().zip(src_row) {
                *d += s * w;
            }
        }
    }
    Image::new(nw, nh, c, out)
}

/// `cv2.resize(img, (width, height), interpolation=...)` on an `f32` image.
pub fn resize(img: &Image, width: usize, height: usize, interpolation: Interpolation) -> Image {
    assert!(width > 0 && height > 0, "resize target must be non-empty");
    assert!(img.width > 0 && img.height > 0, "resize source must be non-empty");
    if (width, height) == (img.width, img.height) {
        return img.clone();
    }
    let shrinks_both = img.width >= width && img.height >= height;
    let (xt, yt) = match interpolation {
        Interpolation::Area if shrinks_both => {
            (area_table(img.width, width), area_table(img.height, height))
        }
        Interpolation::Area => {
            (linear_table(img.width, width, true), linear_table(img.height, height, true))
        }
        Interpolation::Linear => {
            (linear_table(img.width, width, false), linear_table(img.height, height, false))
        }
    };
    apply_tables(img, &xt, &yt)
}

/// Python `round()`: round half to even.
#[inline]
fn py_round(v: f64) -> i64 {
    v.round_ties_even() as i64
}

/// Scale keeping aspect ratio so `min(h, w) == short` (`scale_to_short_side`).
/// Returns `(image, boxes, scale)`. Boxes are xyxy and scaled per axis by
/// `new / old` in `f32`, as numpy does on a float32 array.
pub fn scale_to_short_side(
    image: &Image,
    boxes: Option<&[[f32; 4]]>,
    short: u32,
) -> (Image, Option<Vec<[f32; 4]>>, f64) {
    let (h, w) = (image.height, image.width);
    let s = f64::from(short) / h.min(w) as f64;
    let nw = py_round(w as f64 * s) as usize;
    let nh = py_round(h as f64 * s) as usize;
    let out = if (nw, nh) != (w, h) {
        let interp = if s < 1.0 { Interpolation::Area } else { Interpolation::Linear };
        resize(image, nw, nh, interp)
    } else {
        image.clone()
    };
    let boxes = boxes.map(|b| {
        let (fx, fy) = ((nw as f64 / w as f64) as f32, (nh as f64 / h as f64) as f32);
        b.iter().map(|r| [r[0] * fx, r[1] * fy, r[2] * fx, r[3] * fy]).collect()
    });
    (out, boxes, s)
}

/// Evenly spaced window starts covering `[0, length)` with at least
/// `min_overlap` px overlap (`tile_starts`).
pub fn tile_starts(length: u32, tile: u32, min_overlap: u32) -> Vec<u32> {
    if length <= tile {
        return vec![0];
    }
    let n = ((f64::from(length) - f64::from(min_overlap)) / (f64::from(tile) - f64::from(min_overlap)))
        .ceil() as i64;
    let n = n.max(2);
    let span = f64::from(length - tile);
    (0..n).map(|i| py_round(i as f64 * span / (n - 1) as f64) as u32).collect()
}

/// `(x0, y0)` of every tile, row-major (`tile_grid`).
pub fn tile_grid(w: u32, h: u32, tile: u32, min_overlap: u32) -> Vec<(u32, u32)> {
    let xs = tile_starts(w, tile, min_overlap);
    tile_starts(h, tile, min_overlap)
        .into_iter()
        .flat_map(|y| xs.iter().map(move |&x| (x, y)))
        .collect()
}

/// Clip xyxy boxes to a tile and express them in tile coordinates
/// (`clip_boxes_to_tile`). A box is kept when at least `min_visible` of its
/// area lies in the tile and the clipped box is at least `min_size` px wide and
/// tall. Arithmetic is `f32`, as in the numpy version.
pub fn clip_boxes_to_tile(
    boxes: &[[f32; 4]],
    x0: f32,
    y0: f32,
    tile_w: f32,
    tile_h: f32,
    min_visible: f32,
    min_size: f32,
) -> Vec<[f32; 4]> {
    boxes
        .iter()
        .filter_map(|b| {
            let area = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
            let c = [
                (b[0] - x0).clamp(0.0, tile_w),
                (b[1] - y0).clamp(0.0, tile_h),
                (b[2] - x0).clamp(0.0, tile_w),
                (b[3] - y0).clamp(0.0, tile_h),
            ];
            let (cw, ch) = (c[2] - c[0], c[3] - c[1]);
            let vis = if area > 0.0 { cw * ch / area.max(1e-9) } else { 0.0 };
            (vis >= min_visible && cw >= min_size && ch >= min_size).then_some(c)
        })
        .collect()
}

/// Shift detections from tile to full-image coordinates
/// (`tile_detections_to_full`).
pub fn tile_detections_to_full(det: &[Detection], x0: f64, y0: f64) -> Vec<Detection> {
    det.iter().map(|d| [d[0] + x0, d[1] + y0, d[2], d[3], d[4]]).collect()
}

/// Greedy de-duplication across tiles, highest score wins
/// (`merge_tile_detections`).
///
/// `dets[i]` are full-image detections of tile `i`. A detection is suppressed
/// only by a higher-scoring one from a *different* tile (the heatmap max-pool
/// already de-duplicates within a tile) when their box IoU exceeds
/// `iou_threshold` or their centres are within `center_distance` px.
pub fn merge_tile_detections(
    dets: &[Vec<Detection>],
    iou_threshold: f64,
    center_distance: f64,
) -> Vec<Detection> {
    let mut all: Vec<(Detection, usize)> = dets
        .iter()
        .enumerate()
        .flat_map(|(t, d)| d.iter().map(move |&r| (r, t)))
        .collect();
    // np.argsort(-score, kind="stable")
    all.sort_by(|a, b| b.0[4].total_cmp(&a.0[4]));

    let n = all.len();
    let mut suppressed = vec![false; n];
    let mut keep = Vec::new();
    for i in 0..n {
        if suppressed[i] {
            continue;
        }
        keep.push(all[i].0);
        let (a, ta) = all[i];
        let ab = [a[0], a[1], a[2], a[3]];
        for j in 0..n {
            let (b, tb) = all[j];
            if ta == tb {
                continue;
            }
            let dist = (a[0] - b[0]).hypot(a[1] - b[1]);
            if dist <= center_distance || box_iou(&ab, &[b[0], b[1], b[2], b[3]]) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }
    keep
}

/// Cut a `tile_w` x `tile_h` window at `(x0, y0)`, zero-padding the part that
/// falls outside the image (the `copyMakeBorder(..., value=0)` of
/// `ScaledTileCropDataset`). Returns the window and its valid extent.
pub fn crop_tile(img: &Image, x0: usize, y0: usize, tile_w: usize, tile_h: usize) -> (Image, usize, usize) {
    let c = img.channels;
    let vw = img.width.saturating_sub(x0).min(tile_w);
    let vh = img.height.saturating_sub(y0).min(tile_h);
    let mut out = Image::zeros(tile_w, tile_h, c);
    for y in 0..vh {
        let src = ((y0 + y) * img.width + x0) * c;
        let dst = y * tile_w * c;
        out.data[dst..dst + vw * c].copy_from_slice(&img.data[src..src + vw * c]);
    }
    (out, vw, vh)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    // ---- test_tiling.py ----

    #[test]
    fn tile_starts_cover_with_overlap() {
        assert_eq!(tile_starts(1360, 512, 64), vec![0, 424, 848]);
        assert_eq!(tile_starts(765, 512, 64), vec![0, 253]);
        assert_eq!(tile_starts(400, 512, 64), vec![0]);
        for length in [513u32, 765, 1020, 1360, 2000] {
            let s = tile_starts(length, 512, 64);
            assert_eq!(s[0], 0);
            assert_eq!(s[s.len() - 1] + 512, length);
            assert!(s.windows(2).all(|p| p[1] - p[0] <= 512 - 64));
        }
        assert_eq!(tile_grid(1360, 765, TILE_SIZE, MIN_OVERLAP).len(), 6);
    }

    #[test]
    fn scale_to_short_side_shape_scale_and_boxes() {
        let img = Image::zeros(960, 540, 3);
        let (out, boxes, s) = scale_to_short_side(&img, Some(&[[96.0f32, 54.0, 192.0, 108.0]][..]), 765);
        assert_eq!((out.height, out.width), (765, 1360));
        assert!((s - 765.0 / 540.0).abs() < 1e-9);
        let b = boxes.unwrap()[0];
        for (got, want) in b.iter().zip([136.0, 76.5, 272.0, 153.0]) {
            assert!(approx(*got, want, 0.2), "{got} vs {want}");
        }
    }

    #[test]
    fn clip_boxes_to_tile_rules() {
        let boxes = [
            [110.0, 110.0, 120.0, 130.0], // fully inside: shifted only
            [95.0, 100.0, 115.0, 110.0],  // 75% visible: clipped and kept
            [90.0, 100.0, 106.0, 110.0],  // 37.5% visible: dropped
            [611.0, 200.0, 640.0, 220.0], // 1px sliver at the right edge: dropped
            [300.0, 300.0, 301.5, 320.0], // 1.5px wide to begin with: dropped
        ];
        let out = clip_boxes_to_tile(&boxes, 100.0, 100.0, 512.0, 512.0, 0.5, 2.0);
        assert_eq!(out, vec![[10.0, 10.0, 20.0, 30.0], [0.0, 0.0, 15.0, 10.0]]);
    }

    #[test]
    fn clip_boxes_small_sliver_kept_if_mostly_visible() {
        // 3px box with 2px left inside: 66% visible, 2px wide -> kept.
        let out = clip_boxes_to_tile(&[[99.0, 0.0, 102.0, 10.0]], 100.0, 0.0, 512.0, 512.0, 0.5, 2.0);
        assert_eq!(out, vec![[0.0, 0.0, 2.0, 10.0]]);
    }

    #[test]
    fn detections_map_back_and_merge_across_tiles() {
        // The same person (full-frame 500,300) is found in the overlap of two tiles.
        let tile_a = [[500.0, 300.0, 8.0, 20.0, 0.9], [100.0, 100.0, 8.0, 20.0, 0.5]];
        let tile_b = [[76.5, 300.0, 8.0, 20.0, 0.7]]; // x0 = 424 -> x = 500.5
        let full_a = tile_detections_to_full(&tile_a, 0.0, 0.0);
        let full_b = tile_detections_to_full(&tile_b, 424.0, 0.0);
        assert_eq!([full_b[0][0], full_b[0][1]], [500.5, 300.0]);

        let merged = merge_tile_detections(&[full_a, full_b], 0.5, 2.0);
        assert_eq!(merged.len(), 2);
        assert_eq!([merged[0][4], merged[1][4]], [0.9, 0.5]); // the higher score survives
        assert_eq!([merged[0][0], merged[0][1]], [500.0, 300.0]);
    }

    #[test]
    fn merge_keeps_neighbours_within_one_tile() {
        let same = vec![[50.0, 50.0, 8.0, 20.0, 0.9], [51.0, 50.0, 8.0, 20.0, 0.8]];
        assert_eq!(merge_tile_detections(&[same], 0.5, 2.0).len(), 2);
        assert!(merge_tile_detections(&[], 0.5, 2.0).is_empty());
    }

    // ---- cv2.resize emulation ----

    fn row(values: &[f32]) -> Image {
        Image::new(values.len(), 1, 1, values.to_vec())
    }

    #[test]
    fn area_integer_shrink_is_block_average() {
        let out = resize(&row(&[1.0, 2.0, 3.0, 4.0]), 2, 1, Interpolation::Area);
        assert_eq!(out.data, vec![1.5, 3.5]);
        let sq = Image::new(2, 2, 1, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(resize(&sq, 1, 1, Interpolation::Area).data, vec![2.5]);
    }

    #[test]
    fn area_fractional_shrink_uses_partial_coverage() {
        // 3 -> 2: [2/3 a0 + 1/3 a1, 1/3 a1 + 2/3 a2]
        let out = resize(&row(&[0.0, 3.0, 6.0]), 2, 1, Interpolation::Area);
        assert!(approx(out.data[0], 1.0, 1e-6) && approx(out.data[1], 5.0, 1e-6), "{:?}", out.data);
    }

    #[test]
    fn linear_enlarge_uses_half_pixel_centres_and_clamps() {
        let out = resize(&row(&[0.0, 1.0]), 4, 1, Interpolation::Linear);
        assert_eq!(out.data, vec![0.0, 0.25, 0.75, 1.0]);
    }

    #[test]
    fn resize_keeps_constant_images_and_channels() {
        let img = Image::new(5, 3, 2, (0..30).map(|i| if i % 2 == 0 { 0.25 } else { 0.75 }).collect());
        for (w, h, m) in [(2, 2, Interpolation::Area), (9, 7, Interpolation::Linear), (9, 2, Interpolation::Area)] {
            let out = resize(&img, w, h, m);
            assert_eq!((out.width, out.height, out.channels), (w, h, 2));
            for px in out.data.chunks(2) {
                assert!(approx(px[0], 0.25, 1e-6) && approx(px[1], 0.75, 1e-6));
            }
        }
        assert_eq!(resize(&img, 5, 3, Interpolation::Area), img);
    }

    #[test]
    fn crop_tile_zero_pads_outside_the_image() {
        let img = Image::new(3, 2, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let (t, vw, vh) = crop_tile(&img, 1, 0, 4, 3);
        assert_eq!((vw, vh), (2, 2));
        assert_eq!(t.data, vec![2.0, 3.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }
}
