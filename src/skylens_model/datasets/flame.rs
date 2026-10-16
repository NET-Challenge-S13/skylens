//! Rust twin of `datasets/flame.py`.
//!
//! * [`FlameSegmentation`]: the FLAME "Fire Segmentation" subset, RGB frames with binary fire
//!   masks. Feeds the danger-zone head with `fire` (class 1).
//! * [`Flame3Pairs`]: FLAME 3 image quartets (FOV-corrected RGB + radiometric thermal TIFF),
//!   the reference set for validating 4-channel early fusion.
//!
//! Both live behind IEEE DataPort (free account, interactive download), so a missing dataset
//! fails with the manual-download message instead of pretending to download.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::base::{
    Availability, DangerClass, DatasetMeta, IMG_EXT, Raster, Result, Sample, SkyLensDataset,
    Transform, dir_has_entries, ensure_exists, file_stem, find_with_stem, finish_sample,
    list_images, mask_tensor, read_mask_raster, read_rgb_raster, read_thermal_raster,
    resize_bilinear, resolve_root, stack_rgb_thermal, tensor_f32, tensor_u8, verify_split,
    walk_files,
};

/// Deterministic train/val partition for datasets shipped without splits: the first
/// `max(1, int(n * val_frac))` items are validation, the rest train.
pub fn split_slice<T>(items: Vec<T>, split: &str, val_frac: f64) -> Vec<T> {
    let n_val = if items.is_empty() {
        0
    } else {
        ((items.len() as f64 * val_frac) as usize).max(1)
    };
    if split == "train" {
        items.into_iter().skip(n_val).collect()
    } else {
        items.into_iter().take(n_val).collect()
    }
}

// --------------------------------------------------------------------------- //
// FLAME fire segmentation
// --------------------------------------------------------------------------- //

pub const FLAME_META: DatasetMeta = DatasetMeta {
    name: "FLAME (fire segmentation)",
    splits: &["train", "val"],
    availability: Availability::Account,
    homepage: "https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs",
    download_note: "IEEE DataPort requires a free account and an interactive (browser) download;\n\
there is no stable direct URL. Grab the 'Images for fire segmentation' and\n\
'Masks annotation' items (items 9 and 10 of the FLAME dataset).",
    expected_layout: "<root>/\n  Images/   frame_0001.jpg ...\n  Masks/    frame_0001.png ...   (binary: nonzero = fire)",
};

/// FLAME fire-segmentation subset. Yields `danger_mask` in `{0 normal, 1 fire}` and
/// `person_boxes = None` (segmentation-only dataset).
pub struct FlameSegmentation {
    pub root: PathBuf,
    pub split: String,
    transforms: Option<Transform>,
    index: Vec<(PathBuf, PathBuf)>,
}

impl FlameSegmentation {
    pub fn new(root: impl AsRef<Path>, split: &str, transforms: Option<Transform>) -> Result<Self> {
        let split = verify_split(&FLAME_META, split)?;
        let root = resolve_root(root);
        let (img, msk) = Self::dirs(&root);
        ensure_exists(
            &FLAME_META,
            &root,
            img.is_dir() && msk.is_dir() && dir_has_entries(&img),
        )?;
        let index = Self::build_index(&img, &msk, &split);
        Ok(FlameSegmentation {
            root,
            split,
            transforms,
            index,
        })
    }

    fn dirs(root: &Path) -> (PathBuf, PathBuf) {
        for (i, m) in [("Images", "Masks"), ("images", "masks")] {
            let (img, msk) = (root.join(i), root.join(m));
            if img.is_dir() && msk.is_dir() {
                return (img, msk);
            }
        }
        (root.join("Images"), root.join("Masks"))
    }

    fn build_index(img_dir: &Path, msk_dir: &Path, split: &str) -> Vec<(PathBuf, PathBuf)> {
        let mut pairs = Vec::new();
        for img in list_images(img_dir, &IMG_EXT) {
            if let Some(mask) = find_with_stem(msk_dir, &file_stem(&img), &IMG_EXT) {
                pairs.push((img, mask));
            }
        }
        split_slice(pairs, split, 0.1)
    }
}

/// Any nonzero mask pixel is fire.
pub fn flame_mask_to_danger(raw: &Raster<u8>) -> Raster<u8> {
    Raster::from_vec(
        raw.h,
        raw.w,
        1,
        raw.data
            .iter()
            .map(|&v| {
                if v > 0 {
                    DangerClass::FIRE
                } else {
                    DangerClass::NORMAL
                }
            })
            .collect(),
    )
}

impl SkyLensDataset for FlameSegmentation {
    fn len(&self) -> usize {
        self.index.len()
    }

    fn get(&self, idx: usize) -> Result<Sample> {
        let (img_path, mask_path) = &self.index[idx];
        let image = read_rgb_raster(img_path)?;
        let mask = flame_mask_to_danger(&read_mask_raster(mask_path)?);
        let mut s = Sample::new(tensor_u8(&image), true, false);
        s.danger_mask = Some(mask_tensor(&mask));
        finish_sample(s, self.transforms.as_ref())
    }
}

// --------------------------------------------------------------------------- //
// FLAME 3 pairs
// --------------------------------------------------------------------------- //

pub const FLAME3_META: DatasetMeta = DatasetMeta {
    name: "FLAME 3 (RGB + radiometric thermal)",
    splits: &["train", "val"],
    availability: Availability::Account,
    homepage: "https://ieee-dataport.org/open-access/flame-3-radiometric-thermal-uav-imagery-wildfire-management",
    download_note: "IEEE DataPort requires a free account and an interactive download.\n\
Use the 'Fire Imagery -- CV Subset' package, which contains the image\n\
quartets (raw RGB / raw thermal / FOV-corrected RGB / thermal TIFF).",
    expected_layout: "<root>/\n  Corrected FOV Images/   <stem>.jpg ...   (FOV-matched RGB)\n  Thermal TIFFs/          <stem>.tiff ...  (radiometric, per-pixel degC)",
};

const FLAME3_RGB_DIRS: [&str; 4] = ["Corrected FOV Images", "corrected_fov", "rgb", "RGB"];
const FLAME3_THERMAL_DIRS: [&str; 4] = ["Thermal TIFFs", "thermal_tiff", "thermal", "Thermal"];

/// FLAME 3: FOV-corrected RGB paired with radiometric thermal. Produces a 4-channel `image`
/// (RGB in `[0,1]` + thermal in `[0.1, 1.0]`) with `has_rgb = has_thermal = true`.
/// `danger_mask` is `None`: FLAME 3 ships frame-level labels only, exposed as `frame_label`.
pub struct Flame3Pairs {
    pub root: PathBuf,
    pub split: String,
    transforms: Option<Transform>,
    index: Vec<(PathBuf, PathBuf)>,
}

impl Flame3Pairs {
    pub fn new(root: impl AsRef<Path>, split: &str, transforms: Option<Transform>) -> Result<Self> {
        let split = verify_split(&FLAME3_META, split)?;
        let root = resolve_root(root);
        let (rgb, th) = Self::dirs(&root);
        ensure_exists(&FLAME3_META, &root, rgb.is_some() && th.is_some())?;
        let index = Self::build_index(&rgb.unwrap(), &th.unwrap(), &split);
        Ok(Flame3Pairs {
            root,
            split,
            transforms,
            index,
        })
    }

    fn dirs(root: &Path) -> (Option<PathBuf>, Option<PathBuf>) {
        let find = |names: &[&str]| names.iter().map(|n| root.join(n)).find(|p| p.is_dir());
        (find(&FLAME3_RGB_DIRS), find(&FLAME3_THERMAL_DIRS))
    }

    fn build_index(rgb_dir: &Path, th_dir: &Path, split: &str) -> Vec<(PathBuf, PathBuf)> {
        let mut thermal_by_stem: HashMap<String, PathBuf> = HashMap::new();
        for p in walk_files(th_dir) {
            let ext = p
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();
            if ext == "tif" || ext == "tiff" {
                thermal_by_stem.insert(file_stem(&p), p);
            }
        }
        let mut pairs = Vec::new();
        for img in walk_files(rgb_dir) {
            if !super::base::has_ext(&img, &IMG_EXT) {
                continue;
            }
            if let Some(th) = thermal_by_stem.get(&file_stem(&img)) {
                pairs.push((img, th.clone()));
            }
        }
        split_slice(pairs, split, 0.1)
    }
}

/// Frame label from the RGB file's parent folder name: "no fire" -> 0, "fire" -> 1.
pub fn frame_label_from_parent(parent: &str) -> Option<i64> {
    let p = parent.to_lowercase();
    if p.contains("no") && p.contains("fire") {
        Some(0)
    } else if p.contains("fire") {
        Some(1)
    } else {
        None
    }
}

impl SkyLensDataset for Flame3Pairs {
    fn len(&self) -> usize {
        self.index.len()
    }

    fn get(&self, idx: usize) -> Result<Sample> {
        let (rgb_path, th_path) = &self.index[idx];
        let rgb = read_rgb_raster(rgb_path)?;
        let mut thermal = read_thermal_raster(th_path)?;
        if (thermal.h, thermal.w) != (rgb.h, rgb.w) {
            thermal = resize_bilinear(&thermal, rgb.h, rgb.w);
        }
        let stacked = stack_rgb_thermal(Some(&rgb), Some(&thermal))?;
        let mut s = Sample::new(tensor_f32(&stacked), true, true);
        let parent = rgb_path
            .parent()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        s.frame_label = frame_label_from_parent(&parent);
        finish_sample(s, self.transforms.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_slice_takes_first_tenth_as_val() {
        let items: Vec<i32> = (0..20).collect();
        let val = split_slice(items.clone(), "val", 0.1);
        let train = split_slice(items, "train", 0.1);
        assert_eq!(val, vec![0, 1]);
        assert_eq!(train.len(), 18);
        assert_eq!(train[0], 2);
    }

    #[test]
    fn split_slice_keeps_at_least_one_val() {
        let val = split_slice(vec![1, 2, 3], "val", 0.1);
        assert_eq!(val, vec![1]);
        assert!(split_slice(Vec::<i32>::new(), "val", 0.1).is_empty());
    }

    #[test]
    fn mask_is_binarised_to_fire() {
        let raw = Raster::from_vec(1, 3, 1, vec![0u8, 1, 255]);
        let m = flame_mask_to_danger(&raw);
        assert_eq!(m.data, vec![0, 1, 1]);
    }

    #[test]
    fn frame_label_rules() {
        assert_eq!(frame_label_from_parent("No_Fire"), Some(0));
        assert_eq!(frame_label_from_parent("Fire"), Some(1));
        assert_eq!(frame_label_from_parent("images"), None);
    }
}
