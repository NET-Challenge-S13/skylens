//! Rust twin of `datasets/airesq.py`.
//!
//! AIResQ: thermal-only UAV search-and-rescue dataset, 9,788 drone-view thermal frames (up to
//! 2048x1536) including accident-scene postures. Samples come back with `has_rgb = false`,
//! `has_thermal = true` and a single-channel `image` in `[0.1, 1.0]`; the collator places it in
//! channel 3 and zeroes the RGB planes. `danger_mask` is always `None`.
//!
//! NOT PORTED: `download()` via `direct_url` (needs torchvision's download/extract helpers).

use std::path::{Path, PathBuf};

use super::base::{
    boxes_tensor, ensure_exists, file_stem, finish_sample, list_images, read_thermal_raster,
    resolve_root, stack_rgb_thermal, tensor_f32, verify_split, Availability, DatasetMeta, Result,
    Sample, SkyLensDataset, Transform,
};
use super::sard::read_yolo_labels;

/// Extensions of the thermal frames (exact-case, like the Python tuple).
pub const AIRESQ_IMG_EXT: [&str; 7] = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPG", ".PNG"];

pub const AIRESQ_META: DatasetMeta = DatasetMeta {
    name: "AIResQ (thermal UAV SAR)",
    splits: &["train", "val", "test"],
    availability: Availability::Account,
    homepage: "https://www.nature.com/articles/s41597-026-07663-9",
    download_note: "AIResQ is a Scientific Data descriptor; the data itself sits in the\n\
repository named in the paper's 'Data Records' section (typically Figshare\n\
or Zenodo). Follow the DOI in Data Records; if it resolves to Zenodo/Figshare\n\
the record is directly downloadable.",
    expected_layout: "<root>/\n  <split>/images/*.jpg   (thermal frames)\n  <split>/labels/*.txt   (YOLO: cls cx cy w h, normalised)",
};

/// AIResQ thermal person detection.
pub struct AIResQ {
    pub root: PathBuf,
    pub split: String,
    transforms: Option<Transform>,
    index: Vec<(PathBuf, PathBuf)>,
}

impl AIResQ {
    pub fn new(root: impl AsRef<Path>, split: &str, transforms: Option<Transform>) -> Result<Self> {
        let split = verify_split(&AIRESQ_META, split)?;
        let root = resolve_root(root);
        let (img, lab) = Self::dirs(&root, &split);
        let frames = list_images(&img, &AIRESQ_IMG_EXT);
        ensure_exists(&AIRESQ_META, &root, img.is_dir() && !frames.is_empty())?;
        let index = frames
            .into_iter()
            .map(|p| {
                let stem = file_stem(&p);
                (p, lab.join(format!("{stem}.txt")))
            })
            .collect();
        Ok(AIResQ { root, split, transforms, index })
    }

    fn dirs(root: &Path, split: &str) -> (PathBuf, PathBuf) {
        let aliases: &[&str] = match split {
            "train" => &["train"],
            "val" => &["valid", "val"],
            _ => &["test"],
        };
        for alias in aliases {
            let img = root.join(alias).join("images");
            if img.is_dir() {
                return (img, root.join(alias).join("labels"));
            }
        }
        (root.join("images"), root.join("labels"))
    }
}

impl SkyLensDataset for AIResQ {
    fn len(&self) -> usize {
        self.index.len()
    }

    fn get(&self, idx: usize) -> Result<Sample> {
        let (img_path, lab_path) = &self.index[idx];
        let thermal = read_thermal_raster(img_path)?;
        let boxes = read_yolo_labels(lab_path, thermal.w, thermal.h, None);
        // (H, W, 1) in [0.1, 1]
        let image = stack_rgb_thermal(None, Some(&thermal))?;
        let mut s = Sample::new(tensor_f32(&image), false, true);
        s.person_boxes = Some(boxes_tensor(&boxes));
        finish_sample(s, self.transforms.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thermal_extensions_include_tiff() {
        assert!(AIRESQ_IMG_EXT.contains(&".tif"));
        assert!(AIRESQ_IMG_EXT.contains(&".tiff"));
        assert!(!AIRESQ_IMG_EXT.contains(&".TIF"));
    }

    #[test]
    fn thermal_only_stack_is_single_channel_in_live_range() {
        let th = super::super::base::Raster::from_vec(1, 3, 1, vec![0.0f32, 5.0, 10.0]);
        let s = stack_rgb_thermal(None, Some(&th)).unwrap();
        assert_eq!(s.c, 1);
        assert!((s.at(0, 0, 0) - 0.1).abs() < 1e-6);
        assert!((s.at(0, 2, 0) - 1.0).abs() < 1e-6);
    }
}
