//! Rust twin of `datasets/rescuenet.py`.
//!
//! RescueNet: post-disaster UAV semantic segmentation (Hurricane Michael), 4,494 frames with
//! 11-class masks. The only public source that grades building damage and flags blocked roads
//! from a low-altitude aerial viewpoint.
//!
//! Class mapping to the unified SkyLens schema:
//!
//! | id | RescueNet class            | SkyLens         |
//! |----|----------------------------|-----------------|
//! | 0  | unlabeled                  | 255 ignore      |
//! | 1  | water                      | 0 normal        |
//! | 2  | building no damage         | 0 normal        |
//! | 3  | building medium damage     | 0 normal        |
//! | 4  | building major damage      | 2 collapse      |
//! | 5  | building total destruction | 2 collapse      |
//! | 6  | vehicle                    | 0 normal        |
//! | 7  | road clear                 | 0 normal        |
//! | 8  | road blocked               | 3 road_blocked  |
//! | 9  | tree                       | 0 normal        |
//! | 10 | pool                       | 0 normal        |
//!
//! Only major damage and total destruction are a collapse hazard; folding medium damage in
//! would flood the positive set.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::base::{
    ensure_exists, file_stem, finish_sample, has_ext, list_dir, mask_tensor, read_mask_raster,
    read_rgb_raster, resize_nearest, resolve_root, tensor_u8, verify_split, Availability,
    DangerClass, DatasetMeta, Raster, Result, Sample, SkyLensDataset, Transform, IMG_EXT,
};

pub const RESCUENET_CLASSES: [&str; 11] = [
    "unlabeled",
    "water",
    "building-no-damage",
    "building-medium-damage",
    "building-major-damage",
    "building-total-destruction",
    "vehicle",
    "road-clear",
    "road-blocked",
    "tree",
    "pool",
];

const fn build_lut() -> [u8; 256] {
    let mut lut = [DangerClass::NORMAL; 256];
    lut[0] = DangerClass::IGNORE; // unlabeled
    lut[4] = DangerClass::COLLAPSE; // building major damage
    lut[5] = DangerClass::COLLAPSE; // building total destruction
    lut[8] = DangerClass::ROAD_BLOCKED; // road blocked
    // ids >= 11 are not part of the label set, so treat them as ignore
    let mut i = RESCUENET_CLASSES.len();
    while i < 256 {
        lut[i] = DangerClass::IGNORE;
        i += 1;
    }
    lut
}

/// Lookup table indexed by the raw RescueNet label id.
pub const RESCUENET_TO_SKYLENS: [u8; 256] = build_lut();

pub const RESCUENET_META: DatasetMeta = DatasetMeta {
    name: "RescueNet",
    splits: &["train", "val", "test"],
    availability: Availability::Manual,
    homepage: "https://github.com/BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation",
    download_note: "The authors distribute RescueNet through a Google Drive / Dropbox link\n\
published in the GitHub README (and mirrored on the paper's Figshare record).\n\
Large Drive files hit the virus-scan interstitial, so download in a browser,\n\
or try:  pip install gdown && gdown --folder <drive-folder-url>",
    expected_layout: "<root>/\n  train/train-org-img/*.jpg\n  train/train-label-img/*_lab.png\n  val/val-org-img/*.jpg      val/val-label-img/*_lab.png\n  test/test-org-img/*.jpg    test/test-label-img/*_lab.png",
};

/// Maps a raw RescueNet label raster into the unified schema.
pub fn map_rescuenet_labels(raw: &Raster<u8>) -> Raster<u8> {
    Raster::from_vec(
        raw.h,
        raw.w,
        1,
        raw.data.iter().map(|&v| RESCUENET_TO_SKYLENS[v as usize]).collect(),
    )
}

/// RescueNet danger-zone segmentation. `person_boxes` is always `None`.
pub struct RescueNetSegmentation {
    pub root: PathBuf,
    pub split: String,
    transforms: Option<Transform>,
    index: Vec<(PathBuf, PathBuf)>,
}

impl RescueNetSegmentation {
    pub fn new(root: impl AsRef<Path>, split: &str, transforms: Option<Transform>) -> Result<Self> {
        let split = verify_split(&RESCUENET_META, split)?;
        let root = resolve_root(root);
        let (img, lab) = Self::dirs(&root, &split);
        ensure_exists(&RESCUENET_META, &root, img.is_dir() && lab.is_dir())?;
        let index = Self::build_index(&img, &lab);
        Ok(RescueNetSegmentation { root, split, transforms, index })
    }

    fn dirs(root: &Path, s: &str) -> (PathBuf, PathBuf) {
        let base = root.join(s);
        let (img, lab) = (base.join(format!("{s}-org-img")), base.join(format!("{s}-label-img")));
        if img.is_dir() && lab.is_dir() {
            return (img, lab);
        }
        // flat variant: <root>/<split>-org-img
        (root.join(format!("{s}-org-img")), root.join(format!("{s}-label-img")))
    }

    fn build_index(img_dir: &Path, lab_dir: &Path) -> Vec<(PathBuf, PathBuf)> {
        let labels: HashMap<String, PathBuf> = list_dir(lab_dir)
            .into_iter()
            .filter(|p| has_ext(p, &IMG_EXT))
            .map(|p| (file_stem(&p), p))
            .collect();
        let mut pairs = Vec::new();
        for img in list_dir(img_dir).into_iter().filter(|p| has_ext(p, &IMG_EXT)) {
            let stem = file_stem(&img);
            let lab = labels.get(&format!("{stem}_lab")).or_else(|| labels.get(&stem));
            if let Some(lab) = lab {
                pairs.push((img, lab.clone()));
            }
        }
        pairs
    }
}

impl SkyLensDataset for RescueNetSegmentation {
    fn len(&self) -> usize {
        self.index.len()
    }

    fn get(&self, idx: usize) -> Result<Sample> {
        let (img_path, lab_path) = &self.index[idx];
        let image = read_rgb_raster(img_path)?;
        let mut mask = map_rescuenet_labels(&read_mask_raster(lab_path)?);
        if (mask.h, mask.w) != (image.h, image.w) {
            mask = resize_nearest(&mask, image.h, image.w);
        }
        let mut s = Sample::new(tensor_u8(&image), true, false);
        s.danger_mask = Some(mask_tensor(&mask));
        finish_sample(s, self.transforms.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eleven_to_four_mapping() {
        let expect = [255u8, 0, 0, 0, 2, 2, 0, 0, 3, 0, 0];
        for (id, want) in expect.iter().enumerate() {
            assert_eq!(RESCUENET_TO_SKYLENS[id], *want, "raw id {id}");
        }
    }

    #[test]
    fn out_of_range_ids_are_ignored() {
        for id in 11..256 {
            assert_eq!(RESCUENET_TO_SKYLENS[id], DangerClass::IGNORE);
        }
    }

    #[test]
    fn medium_damage_is_not_collapse() {
        assert_eq!(RESCUENET_TO_SKYLENS[3], DangerClass::NORMAL);
    }

    #[test]
    fn raster_mapping() {
        let raw = Raster::from_vec(1, 4, 1, vec![0u8, 4, 8, 7]);
        assert_eq!(map_rescuenet_labels(&raw).data, vec![255, 2, 3, 0]);
    }
}
