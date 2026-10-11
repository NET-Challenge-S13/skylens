//! Rust twin of `datasets/sard.py`.
//!
//! SARD: Search And Rescue Dataset (UAV person detection), 1,980 drone frames and 6,525 person
//! instances including exhausted / injured postures. Person-only: `danger_mask` is always
//! `None`. YOLO-format mirrors are accepted in the Roboflow per-split layout or flat
//! `images/` + `labels/`; the loader auto-detects.
//!
//! The Python dataset class has no notion of the project's own frame-block split; that lives in
//! `data/sard/SPLIT.json` (SPEC §8). [`SARD::with_split_file`] reads it, and [`block_split`] is
//! the pure rule (50-frame blocks, 10-frame buffer at block boundaries) for generating one.
//!
//! NOT PORTED: `download()` (Kaggle CLI). Fetch the data from Python.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use super::base::{
    boxes_tensor, ensure_exists, file_stem, finish_sample, list_images, read_rgb_raster,
    resolve_root, tensor_u8, verify_split, Availability, DatasetError, DatasetMeta, Result,
    Sample, SkyLensDataset, Transform, IMG_EXT,
};

pub const SARD_KAGGLE_SLUG: &str = "julienmeine/search-and-rescue-dataset";

pub const SARD_META: DatasetMeta = DatasetMeta {
    name: "SARD (Search And Rescue Dataset)",
    splits: &["train", "val", "test"],
    availability: Availability::Account,
    homepage: "https://universe.roboflow.com/search?q=SARD+search+and+rescue",
    download_note: "SARD has no official direct URL. Two automatable routes exist, both\n\
needing credentials:\n\
  1) Kaggle:   kaggle datasets download -d <owner>/<slug> --unzip -p <root>\n\
  2) Roboflow: export a YOLOv8 version with your API key (search 'SARD').\n\
The original release is described in Sambolek & Ivasic-Kos, IEEE Access 2021.",
    expected_layout: "<root>/\n  train/images/*.jpg   train/labels/*.txt   (YOLO: cls cx cy w h, normalised)\n  valid/images/...     valid/labels/...\n  test/images/...      test/labels/...\nor flat:  <root>/images/*.jpg  +  <root>/labels/*.txt",
};

/// Parses a YOLO `.txt` label file into xyxy pixel boxes.
///
/// Each line is `cls cx cy w h`, geometry normalised to `[0, 1]`. `keep_classes = None` keeps
/// every class (SARD is single-class). Malformed lines and degenerate boxes are skipped.
pub fn read_yolo_labels(
    path: &Path,
    width: usize,
    height: usize,
    keep_classes: Option<&HashSet<i64>>,
) -> Vec<[f32; 4]> {
    let Ok(bytes) = std::fs::read(path) else {
        return Vec::new();
    };
    // errors="ignore" in Python
    let text = String::from_utf8_lossy(&bytes);
    parse_yolo_labels(&text, width, height, keep_classes)
}

/// Text form of [`read_yolo_labels`] (also used by tests).
pub fn parse_yolo_labels(
    text: &str,
    width: usize,
    height: usize,
    keep_classes: Option<&HashSet<i64>>,
) -> Vec<[f32; 4]> {
    let (width, height) = (width as f32, height as f32);
    let mut boxes = Vec::new();
    for line in text.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 5 {
            continue;
        }
        let Ok(cls) = parts[0].parse::<f64>().map(|v| v as i64) else { continue };
        let nums: Vec<f32> = match parts[1..5].iter().map(|v| v.parse::<f32>()).collect() {
            Ok(n) => n,
            Err(_) => continue,
        };
        let (cx, cy, bw, bh) = (nums[0], nums[1], nums[2], nums[3]);
        if let Some(keep) = keep_classes {
            if !keep.contains(&cls) {
                continue;
            }
        }
        let x1 = (cx - bw / 2.0) * width;
        let y1 = (cy - bh / 2.0) * height;
        let x2 = (cx + bw / 2.0) * width;
        let y2 = (cy + bh / 2.0) * height;
        if x2 > x1 && y2 > y1 {
            boxes.push([x1, y1, x2, y2]);
        }
    }
    boxes
}

/// Frame-block split rule: frames are grouped in blocks of `block` consecutive frame numbers;
/// `is_val_block(block_index)` decides each block's side, and frames within `buffer` frames of
/// a boundary that separates a train block from a val block are dropped.
///
/// Returns `(train, val, dropped)` as frame numbers, each in input order.
///
/// ASSUMPTION: Python has no implementation of this rule (only its output, `SPLIT.json`), so
/// the assignment of blocks to sides is supplied by the caller rather than guessed.
pub fn block_split(
    frames: &[u32],
    block: u32,
    buffer: u32,
    is_val_block: impl Fn(u32) -> bool,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let (mut train, mut val, mut dropped) = (Vec::new(), Vec::new(), Vec::new());
    for &f in frames {
        let b = f / block;
        let side = is_val_block(b);
        let pos = f % block;
        let near_start = pos < buffer && b > 0 && is_val_block(b - 1) != side;
        let near_end = pos >= block - buffer && is_val_block(b + 1) != side;
        if near_start || near_end {
            dropped.push(f);
        } else if side {
            val.push(f);
        } else {
            train.push(f);
        }
    }
    (train, val, dropped)
}

/// SARD person detection. Returns xyxy `person_boxes`, `danger_mask = None`.
pub struct SARD {
    pub root: PathBuf,
    pub split: String,
    transforms: Option<Transform>,
    index: Vec<(PathBuf, PathBuf)>,
}

impl SARD {
    pub fn new(root: impl AsRef<Path>, split: &str, transforms: Option<Transform>) -> Result<Self> {
        let split = verify_split(&SARD_META, split)?;
        let root = resolve_root(root);
        let (img, lab) = Self::dirs(&root, &split);
        ensure_exists(&SARD_META, &root, img.is_dir() && !list_images(&img, &IMG_EXT).is_empty())?;
        let index = Self::build_index(&img, &lab);
        Ok(SARD { root, split, transforms, index })
    }

    /// Like [`SARD::new`], but the sample set comes from a `SPLIT.json` of the form
    /// `{"train": ["stem", ...], "val": [...]}` (entries may carry an extension). Images are
    /// looked up in the layout's `train` directory, or the flat `images/` directory.
    ///
    /// ASSUMPTION: the on-disk format of `data/sard/SPLIT.json` is not in the repository;
    /// this shape (split name to list of file stems) is what the loader expects.
    pub fn with_split_file(
        root: impl AsRef<Path>,
        split: &str,
        split_json: impl AsRef<Path>,
        transforms: Option<Transform>,
    ) -> Result<Self> {
        let mut ds = SARD::new(&root, "train", None)?;
        let text = std::fs::read_to_string(split_json)?;
        let map: HashMap<String, Vec<String>> = serde_json::from_str(&text)?;
        let wanted = map.get(split).ok_or_else(|| DatasetError::InvalidSplit {
            dataset: SARD_META.name.to_string(),
            split: split.to_string(),
            expected: map.keys().cloned().collect(),
        })?;
        let stems: HashSet<String> = wanted
            .iter()
            .map(|s| file_stem(Path::new(s)))
            .collect();
        ds.index.retain(|(img, _)| stems.contains(&file_stem(img)));
        ds.split = split.to_string();
        ds.transforms = transforms;
        Ok(ds)
    }

    fn dirs(root: &Path, split: &str) -> (PathBuf, PathBuf) {
        let aliases: &[&str] = match split {
            "train" => &["train"],
            "val" => &["valid", "val"],
            _ => &["test"],
        };
        for alias in aliases {
            let (img, lab) = (root.join(alias).join("images"), root.join(alias).join("labels"));
            if img.is_dir() {
                return (img, lab);
            }
        }
        (root.join("images"), root.join("labels"))
    }

    fn build_index(img_dir: &Path, lab_dir: &Path) -> Vec<(PathBuf, PathBuf)> {
        list_images(img_dir, &IMG_EXT)
            .into_iter()
            .map(|p| {
                let stem = file_stem(&p);
                (p, lab_dir.join(format!("{stem}.txt")))
            })
            .collect()
    }
}

impl SkyLensDataset for SARD {
    fn len(&self) -> usize {
        self.index.len()
    }

    fn get(&self, idx: usize) -> Result<Sample> {
        let (img_path, lab_path) = &self.index[idx];
        let image = read_rgb_raster(img_path)?;
        let boxes = read_yolo_labels(lab_path, image.w, image.h, None);
        let mut s = Sample::new(tensor_u8(&image), true, false);
        s.person_boxes = Some(boxes_tensor(&boxes));
        finish_sample(s, self.transforms.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yolo_boxes_to_xyxy_pixels() {
        let b = parse_yolo_labels("0 0.5 0.5 0.2 0.4\n", 100, 200, None);
        assert_eq!(b.len(), 1);
        let want = [40.0, 60.0, 60.0, 140.0];
        for (g, w) in b[0].iter().zip(want) {
            assert!((g - w).abs() < 1e-3, "{g} vs {w}");
        }
    }

    #[test]
    fn malformed_and_degenerate_lines_skipped() {
        let text = "0 0.5 0.5\nx 0.5 0.5 0.1 0.1\n0 0.5 0.5 0 0.1\n0 0.5 0.5 0.1 0.1\n";
        assert_eq!(parse_yolo_labels(text, 10, 10, None).len(), 1);
    }

    #[test]
    fn class_filter() {
        let keep: HashSet<i64> = [1].into_iter().collect();
        let text = "0 0.5 0.5 0.1 0.1\n1 0.5 0.5 0.1 0.1\n";
        assert_eq!(parse_yolo_labels(text, 10, 10, Some(&keep)).len(), 1);
    }

    #[test]
    fn block_split_drops_boundary_frames() {
        // blocks of 10 frames: block 0 train, block 1 val, block 2 train; buffer 2
        let frames: Vec<u32> = (0..30).collect();
        let (train, val, dropped) = block_split(&frames, 10, 2, |b| b == 1);
        // frames 8,9 (end of train block 0), 10,11 (start of val), 18,19, 20,21 are dropped
        assert_eq!(dropped, vec![8, 9, 10, 11, 18, 19, 20, 21]);
        assert_eq!(val, (12..18).collect::<Vec<_>>());
        assert_eq!(train.len() + val.len() + dropped.len(), 30);
        assert!(!train.contains(&9) && train.contains(&7) && train.contains(&22));
    }

    #[test]
    fn block_split_same_side_neighbours_keep_everything() {
        let frames: Vec<u32> = (0..20).collect();
        let (train, val, dropped) = block_split(&frames, 10, 3, |_| false);
        assert_eq!(train.len(), 20);
        assert!(val.is_empty() && dropped.is_empty());
    }
}
