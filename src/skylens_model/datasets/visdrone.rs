//! Rust twin of `datasets/visdrone.py`.
//!
//! VisDrone2019-DET, drone-view object detection. Used as the backbone warm-up stage
//! (README §6.1). Only `pedestrian` (1) and `people` (2) are kept and merged into one `person`
//! class; `people` covers humans in other poses, which is the prone/exhausted case SkyLens
//! cares about.
//!
//! Annotation line: `<x>,<y>,<w>,<h>,<score>,<category>,<truncation>,<occlusion>` with `x, y`
//! the top-left corner in pixels. Rows with `score == 0` are ignored regions and dropped.
//!
//! The 765-px short side plus 512 random crops used for training live in `utils/tiling.rs`
//! (they are a wrapper over this dataset, not part of it).
//!
//! NOT PORTED: `download()` (HuggingFace mirror `Voxel51/VisDrone2019-DET`). The FiftyOne
//! export layout (`samples.json` + `data/`) is still read.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use super::base::{
    boxes_tensor, dir_has_entries, ensure_exists, file_stem, finish_sample, list_images,
    read_rgb_raster, resolve_root, tensor_u8, verify_split, Availability, DatasetError,
    DatasetMeta, Result, Sample, SkyLensDataset, Transform, IMG_EXT,
};

pub const VISDRONE_CATEGORIES: [&str; 12] = [
    "ignored", "pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle",
    "awning-tricycle", "bus", "motor", "others",
];

/// Categories folded into the single SkyLens `person` class.
pub const VISDRONE_PERSON_IDS: [i64; 2] = [1, 2];

pub const HF_REPO_ID: &str = "Voxel51/VisDrone2019-DET";

pub const VISDRONE_META: DatasetMeta = DatasetMeta {
    name: "VisDrone2019-DET (person subset)",
    splits: &["train", "val", "test"],
    availability: Availability::Account,
    homepage: "https://github.com/VisDrone/VisDrone-Dataset",
    download_note: "The official VisDrone release is hosted on Google Drive / OneDrive links\n\
from https://github.com/VisDrone/VisDrone-Dataset, which are not reliably\n\
wget-able. Options:\n\
  1) HuggingFace mirror (Voxel51/VisDrone2019-DET, FiftyOne export).\n\
  2) pip install gdown && gdown <drive-file-id>\n\
  3) Download the four DET zips manually in a browser.",
    expected_layout: "<root>/\n  VisDrone2019-DET-train/images/*.jpg\n  VisDrone2019-DET-train/annotations/*.txt\n  VisDrone2019-DET-val/{images,annotations}/\n  VisDrone2019-DET-test-dev/{images,annotations}/",
};

const FIFTYONE_MANIFEST: &str = "samples.json";
const FIFTYONE_PERSON_LABELS: [&str; 2] = ["pedestrians", "people"];

/// Parses a VisDrone annotation text into xyxy pixel boxes.
pub fn parse_visdrone_annotation(
    text: &str,
    keep: &[i64],
    drop_ignored: bool,
) -> Vec<[f32; 4]> {
    let mut boxes = Vec::new();
    for line in text.lines() {
        let parts: Vec<&str> = line.split([',', ' ', '\t']).filter(|p| !p.is_empty()).collect();
        if parts.len() < 6 {
            continue;
        }
        let geom: std::result::Result<Vec<f32>, _> = parts[..4].iter().map(|v| v.parse::<f32>()).collect();
        let (Ok(geom), Ok(score), Ok(category)) =
            (geom, parts[4].parse::<f64>(), parts[5].parse::<f64>())
        else {
            continue;
        };
        let (score, category) = (score as i64, category as i64);
        if drop_ignored && score == 0 {
            continue;
        }
        if !keep.contains(&category) {
            continue;
        }
        let (x, y, w, h) = (geom[0], geom[1], geom[2], geom[3]);
        if w > 0.0 && h > 0.0 {
            boxes.push([x, y, x + w, y + h]);
        }
    }
    boxes
}

/// Reads a VisDrone `.txt` annotation file. A missing file gives no boxes.
pub fn read_visdrone_annotation(path: &Path, keep: &[i64], drop_ignored: bool) -> Vec<[f32; 4]> {
    match std::fs::read(path) {
        Ok(bytes) => parse_visdrone_annotation(&String::from_utf8_lossy(&bytes), keep, drop_ignored),
        Err(_) => Vec::new(),
    }
}

/// Per-sample annotation: a file to parse lazily, or boxes already in memory (FiftyOne).
#[derive(Debug, Clone)]
enum Ann {
    File(PathBuf),
    Boxes(Vec<[f32; 4]>),
}

/// VisDrone2019-DET restricted to pedestrian/people boxes. `danger_mask` is always `None`.
pub struct VisDronePerson {
    pub root: PathBuf,
    pub split: String,
    transforms: Option<Transform>,
    index: Vec<(PathBuf, Ann)>,
}

impl VisDronePerson {
    pub fn new(root: impl AsRef<Path>, split: &str, transforms: Option<Transform>) -> Result<Self> {
        let split = verify_split(&VISDRONE_META, split)?;
        let root = resolve_root(root);
        let manifest = Self::fiftyone_manifest(&root);
        let (img, ann) = Self::dirs(&root, &split);
        let exists = manifest.is_some()
            || (img.is_dir() && !list_images(&img, &IMG_EXT).is_empty() && dir_has_entries(&img));
        ensure_exists(&VISDRONE_META, &root, exists)?;
        let index = match manifest {
            Some(m) => Self::build_index_fiftyone(&root, &m, &split)?,
            None => list_images(&img, &IMG_EXT)
                .into_iter()
                .map(|p| {
                    let stem = file_stem(&p);
                    (p, Ann::File(ann.join(format!("{stem}.txt"))))
                })
                .collect(),
        };
        Ok(VisDronePerson { root, split, transforms, index })
    }

    fn split_dirs(split: &str) -> &'static [&'static str] {
        match split {
            "train" => &["VisDrone2019-DET-train", "train"],
            "val" => &["VisDrone2019-DET-val", "val"],
            _ => &["VisDrone2019-DET-test-dev", "VisDrone2019-DET-test-challenge", "test"],
        }
    }

    fn dirs(root: &Path, split: &str) -> (PathBuf, PathBuf) {
        let aliases = Self::split_dirs(split);
        for alias in aliases {
            let base = root.join(alias);
            if base.join("images").is_dir() {
                return (base.join("images"), base.join("annotations"));
            }
        }
        let base = root.join(aliases[0]);
        (base.join("images"), base.join("annotations"))
    }

    /// Path of the FiftyOne manifest when the root is a FiftyOne export.
    fn fiftyone_manifest(root: &Path) -> Option<PathBuf> {
        let p = root.join(FIFTYONE_MANIFEST);
        (p.is_file() && root.join("data").is_dir()).then_some(p)
    }

    /// Flattens a FiftyOne export into `(image path, xyxy boxes)` records.
    ///
    /// `bounding_box` is `[x, y, w, h]` normalised from the top-left corner, so it is scaled by
    /// the sample metadata's width/height. The split is selected by the sample `tags`.
    fn build_index_fiftyone(root: &Path, manifest: &Path, split: &str) -> Result<Vec<(PathBuf, Ann)>> {
        let doc: serde_json::Value = serde_json::from_slice(&std::fs::read(manifest)?)?;
        let samples = doc
            .get("samples")
            .and_then(|s| s.as_array())
            .ok_or_else(|| DatasetError::InvalidSample("samples.json has no `samples` array".into()))?;
        let person: HashSet<&str> = FIFTYONE_PERSON_LABELS.into_iter().collect();

        let mut records = Vec::new();
        for s in samples {
            let tagged = s
                .get("tags")
                .and_then(|t| t.as_array())
                .map_or(false, |t| t.iter().any(|v| v.as_str() == Some(split)));
            if !tagged {
                continue;
            }
            let meta = s.get("metadata");
            let dim = |k: &str| meta.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
            let (w, h) = (dim("width"), dim("height"));
            let mut boxes = Vec::new();
            let dets = s
                .get("ground_truth")
                .and_then(|g| g.get("detections"))
                .and_then(|d| d.as_array());
            for det in dets.into_iter().flatten() {
                let label = det.get("label").and_then(|l| l.as_str()).unwrap_or("");
                if !person.contains(label) {
                    continue;
                }
                let bb: Vec<f64> = match det.get("bounding_box").and_then(|b| b.as_array()) {
                    Some(a) if a.len() == 4 => a.iter().filter_map(|v| v.as_f64()).collect(),
                    _ => continue,
                };
                let (Some(w), Some(h)) = (w.filter(|v| *v != 0.0), h.filter(|v| *v != 0.0)) else {
                    continue;
                };
                if bb.len() != 4 {
                    continue;
                }
                let (x, y, bw, bh) = (bb[0], bb[1], bb[2], bb[3]);
                boxes.push([
                    (x * w) as f32,
                    (y * h) as f32,
                    ((x + bw) * w) as f32,
                    ((y + bh) * h) as f32,
                ]);
            }
            let filepath = s.get("filepath").and_then(|f| f.as_str()).unwrap_or("");
            records.push((root.join(filepath), Ann::Boxes(boxes)));
        }
        Ok(records)
    }
}

impl SkyLensDataset for VisDronePerson {
    fn len(&self) -> usize {
        self.index.len()
    }

    fn get(&self, idx: usize) -> Result<Sample> {
        let (img_path, ann) = &self.index[idx];
        let boxes = match ann {
            Ann::Boxes(b) => b.clone(),
            Ann::File(p) => read_visdrone_annotation(p, &VISDRONE_PERSON_IDS, true),
        };
        let image = read_rgb_raster(img_path)?;
        let mut s = Sample::new(tensor_u8(&image), true, false);
        s.person_boxes = Some(boxes_tensor(&boxes));
        finish_sample(s, self.transforms.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_only_pedestrian_and_people() {
        let text = "10,20,30,40,1,1,0,0\n\
                    10,20,30,40,1,2,0,0\n\
                    10,20,30,40,1,4,0,0\n";
        let b = parse_visdrone_annotation(text, &VISDRONE_PERSON_IDS, true);
        assert_eq!(b, vec![[10.0, 20.0, 40.0, 60.0], [10.0, 20.0, 40.0, 60.0]]);
    }

    #[test]
    fn ignored_regions_are_dropped() {
        let text = "0,0,10,10,0,1,0,0\n0,0,10,10,1,1,0,0\n";
        assert_eq!(parse_visdrone_annotation(text, &VISDRONE_PERSON_IDS, true).len(), 1);
        assert_eq!(parse_visdrone_annotation(text, &VISDRONE_PERSON_IDS, false).len(), 2);
    }

    #[test]
    fn zero_size_and_short_rows_skipped() {
        let text = "0,0,0,10,1,1,0,0\n1,2,3\n";
        assert!(parse_visdrone_annotation(text, &VISDRONE_PERSON_IDS, true).is_empty());
    }

    #[test]
    fn category_table() {
        assert_eq!(VISDRONE_CATEGORIES[1], "pedestrian");
        assert_eq!(VISDRONE_CATEGORIES[2], "people");
    }
}
