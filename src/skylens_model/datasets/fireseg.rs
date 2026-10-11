//! Rust twin of `datasets/fireseg.py`.
//!
//! Fire Segmentation: YOLO-seg polygon annotations for aerial/ground fire (1,348 RGB frames,
//! 1,146 train / 202 val, single class `fire`). Polygons are rasterised into the unified
//! danger schema as [`DangerClass::FIRE`].
//!
//! A label file with no rows is a *negative* sample (no fire in frame) and yields an
//! all-`normal` mask rather than `None`: "there is no fire here" is a training signal.
//!
//! NOT PORTED: `download()` (HuggingFace snapshot with `HF_HUB_DISABLE_XET=1`). Fetch the data
//! from Python; a missing tree fails with the manual-download message.

use std::path::{Path, PathBuf};

use super::base::{
    ensure_exists, finish_sample, list_images, mask_tensor, read_rgb_raster, resolve_root,
    tensor_u8, verify_split, Availability, DangerClass, DatasetMeta, Raster, Result, Sample,
    SkyLensDataset, Transform, IMG_EXT,
};

pub const HF_REPO_ID: &str = "sreeharivp23/fire-segmentation-dataset";

pub const FIRESEG_META: DatasetMeta = DatasetMeta {
    name: "Fire Segmentation (YOLO-seg polygons)",
    splits: &["train", "val"],
    availability: Availability::Auto,
    homepage: "https://huggingface.co/datasets/sreeharivp23/fire-segmentation-dataset",
    download_note: "Public, non-gated HuggingFace dataset (sreeharivp23/fire-segmentation-dataset).\n\
The repo is served through the Xet backend, which throttles aggressively (HTTP 429).\n\
Set HF_HUB_DISABLE_XET=1 when fetching it from Python.",
    expected_layout: "<root>/\n  data.yaml                (names: {0: fire})\n  images/train/*.jpg       images/val/*.jpg\n  labels/train/*.txt       labels/val/*.txt\n                           (YOLO-seg: cls x1 y1 x2 y2 ... normalised;\n                            an empty file = no fire in that frame)",
};

/// Reads a YOLO-seg label file into normalised polygons (`[(x, y), ...]` in `[0, 1]`).
///
/// Each line is `<class> x1 y1 ... xn yn`. Lines with fewer than 7 tokens (class + 3 points)
/// are skipped; an odd trailing coordinate is dropped. A missing or empty file gives `[]`.
pub fn read_yolo_polygons(path: &Path) -> Vec<Vec<(f32, f32)>> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let mut polys = Vec::new();
    for line in text.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 7 {
            continue;
        }
        let coords: Vec<f32> = match parts[1..].iter().map(|v| v.parse::<f32>()).collect() {
            Ok(c) => c,
            Err(_) => continue, // numpy would raise; a bad line is skipped here
        };
        let n = coords.len() / 2 * 2;
        polys.push(coords[..n].chunks_exact(2).map(|c| (c[0], c[1])).collect());
    }
    polys
}

fn put(mask: &mut Raster<u8>, x: i32, y: i32, v: u8) {
    if x >= 0 && y >= 0 && (x as usize) < mask.w && (y as usize) < mask.h {
        mask.set(y as usize, x as usize, 0, v);
    }
}

fn draw_line(mask: &mut Raster<u8>, (mut x0, mut y0): (i32, i32), (x1, y1): (i32, i32), v: u8) {
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        put(mask, x0, y0, v);
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

/// Fills an integer polygon into a single-channel mask (even-odd rule, boundary included),
/// the stand-in for `cv2.fillPoly`.
pub fn fill_polygon(mask: &mut Raster<u8>, pts: &[(i32, i32)], value: u8) {
    if pts.len() < 3 {
        return;
    }
    let ymin = pts.iter().map(|p| p.1).min().unwrap().max(0);
    let ymax = pts.iter().map(|p| p.1).max().unwrap().min(mask.h as i32 - 1);
    for y in ymin..=ymax {
        let mut xs: Vec<f64> = Vec::new();
        for i in 0..pts.len() {
            let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
            if a.1 == b.1 {
                continue;
            }
            let (lo, hi) = if a.1 < b.1 { (a, b) } else { (b, a) };
            if y >= lo.1 && y < hi.1 {
                let t = f64::from(y - lo.1) / f64::from(hi.1 - lo.1);
                xs.push(f64::from(lo.0) + t * f64::from(hi.0 - lo.0));
            }
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for pair in xs.chunks_exact(2) {
            let (x0, x1) = (pair[0].ceil() as i32, pair[1].floor() as i32);
            for x in x0..=x1 {
                put(mask, x, y, value);
            }
        }
    }
    for i in 0..pts.len() {
        draw_line(mask, pts[i], pts[(i + 1) % pts.len()], value);
    }
}

/// Rasterises normalised polygons into an `(h, w)` mask of `normal` with `fire` inside.
pub fn rasterize_fire_mask(polys: &[Vec<(f32, f32)>], h: usize, w: usize) -> Raster<u8> {
    let mut mask = Raster::<u8>::from_vec(h, w, 1, vec![DangerClass::NORMAL; h * w]);
    for poly in polys {
        // np.round is round-half-to-even
        let pts: Vec<(i32, i32)> = poly
            .iter()
            .map(|&(x, y)| ((x * w as f32).round_ties_even() as i32, (y * h as f32).round_ties_even() as i32))
            .collect();
        fill_polygon(&mut mask, &pts, DangerClass::FIRE);
    }
    mask
}

/// YOLO-seg fire polygons to `danger_mask` in `{0 normal, 1 fire}`; `person_boxes` is `None`.
pub struct FireSegmentation {
    pub root: PathBuf,
    pub split: String,
    transforms: Option<Transform>,
    index: Vec<(PathBuf, PathBuf)>,
}

impl FireSegmentation {
    pub fn new(root: impl AsRef<Path>, split: &str, transforms: Option<Transform>) -> Result<Self> {
        let split = verify_split(&FIRESEG_META, split)?;
        let root = resolve_root(root);
        let (img_dir, lab_dir) = (root.join("images").join(&split), root.join("labels").join(&split));
        let exists = img_dir.is_dir() && !list_images(&img_dir, &IMG_EXT).is_empty();
        ensure_exists(&FIRESEG_META, &root, exists)?;
        let index = list_images(&img_dir, &IMG_EXT)
            .into_iter()
            .map(|p| {
                let stem = super::base::file_stem(&p);
                (p, lab_dir.join(format!("{stem}.txt")))
            })
            .collect();
        Ok(FireSegmentation { root, split, transforms, index })
    }
}

impl SkyLensDataset for FireSegmentation {
    fn len(&self) -> usize {
        self.index.len()
    }

    fn get(&self, idx: usize) -> Result<Sample> {
        let (img_path, lab_path) = &self.index[idx];
        let image = read_rgb_raster(img_path)?;
        let mask = rasterize_fire_mask(&read_yolo_polygons(lab_path), image.h, image.w);
        let mut s = Sample::new(tensor_u8(&image), true, false);
        s.danger_mask = Some(mask_tensor(&mask));
        finish_sample(s, self.transforms.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_polygon_is_filled() {
        // 4x4 image, polygon covering the middle 2x2 (normalised 0.25..0.75)
        let poly = vec![vec![(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]];
        let m = rasterize_fire_mask(&poly, 4, 4);
        assert_eq!(m.at(1, 1, 0), DangerClass::FIRE);
        assert_eq!(m.at(2, 2, 0), DangerClass::FIRE);
        assert_eq!(m.at(0, 0, 0), DangerClass::NORMAL);
        assert_eq!(m.at(3, 3, 0), DangerClass::NORMAL);
    }

    #[test]
    fn empty_label_gives_all_normal() {
        let m = rasterize_fire_mask(&[], 3, 3);
        assert!(m.data.iter().all(|&v| v == DangerClass::NORMAL));
    }

    #[test]
    fn yolo_polygon_parsing() {
        let dir = std::env::temp_dir().join("skylens_fireseg_test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.txt");
        // 3 points + one dangling coordinate; a too-short line is skipped
        std::fs::write(&p, "0 0.1 0.1 0.9 0.1 0.5 0.9 0.3\n0 0.1 0.1\n").unwrap();
        let polys = read_yolo_polygons(&p);
        assert_eq!(polys.len(), 1);
        assert_eq!(polys[0].len(), 3);
        assert!(read_yolo_polygons(&dir.join("missing.txt")).is_empty());
    }

    #[test]
    fn negative_rounding_is_half_to_even() {
        // 0.5 * 5 = 2.5 rounds to 2, like numpy
        let poly = vec![vec![(0.5, 0.0), (1.0, 0.0), (1.0, 1.0)]];
        let m = rasterize_fire_mask(&poly, 5, 5);
        assert_eq!(m.at(0, 2, 0), DangerClass::FIRE);
    }
}
