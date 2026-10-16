//! Rust twin of `datasets/llvip.py`.
//!
//! LLVIP: visible/infrared paired pedestrian dataset (all night-time), 16,836 strictly
//! registered RGB/IR pairs from fixed cameras with VOC-XML person boxes. Used to validate that
//! 4-channel early fusion trains and as a night-time person prior. Fixed CCTV viewpoint, not
//! aerial: structural pre-training only.
//!
//! NOT PORTED: `download()` (HuggingFace snapshot of one ~4 GB zip) and `xml.etree` (VOC XML is
//! read with a small tag scanner, enough for the flat `<object><name/><bndbox/></object>`
//! layout LLVIP uses; a document that does not parse yields no boxes, like `ParseError`).

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use super::base::{
    Availability, DatasetError, DatasetMeta, IMG_EXT, Result, Sample, SkyLensDataset, Transform,
    boxes_tensor, ensure_exists, file_stem, find_with_stem, finish_sample, list_images,
    read_rgb_raster, read_thermal_raster, resize_bilinear, resolve_root, stack_rgb_thermal,
    tensor_f32, tensor_u8, verify_split,
};

pub const HF_REPO_ID: &str = "jsonhash/LLVIP";

pub const LLVIP_META: DatasetMeta = DatasetMeta {
    name: "LLVIP",
    splits: &["train", "test"],
    availability: Availability::Auto,
    homepage: "https://bupt-ai-cz.github.io/LLVIP/",
    download_note: "The project page links Google Drive / Baidu / OneDrive copies, none of\n\
which are wget-able. The practical route is the HuggingFace mirror:\n\
    huggingface-cli download jsonhash/LLVIP --repo-type dataset --local-dir <root>\n\
    unzip <root>/LLVIP.zip -d <root>\n\
The repo ships one big zip, so the unzip is manual.",
    expected_layout: "<root>/\n  visible/train/*.jpg     visible/test/*.jpg\n  infrared/train/*.jpg    infrared/test/*.jpg\n  Annotations/*.xml       (Pascal VOC, class 'person')",
};

pub const PERSON_NAMES: [&str; 3] = ["person", "people", "pedestrian"];

/// Text between `<tag>` and `</tag>` inside `block`, trimmed.
fn tag_text<'a>(block: &'a str, tag: &str) -> Option<&'a str> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = block.find(&open)? + open.len();
    let end = block[start..].find(&close)? + start;
    Some(block[start..end].trim())
}

/// Parses Pascal-VOC XML text into xyxy pixel boxes for the given person class names.
/// An empty `names` set keeps every object.
pub fn parse_voc_person_boxes(xml: &str, names: &HashSet<&str>) -> Vec<[f32; 4]> {
    let mut boxes = Vec::new();
    let mut rest = xml;
    while let Some(i) = rest.find("<object>") {
        let after = &rest[i + "<object>".len()..];
        let Some(j) = after.find("</object>") else {
            break;
        };
        let obj = &after[..j];
        rest = &after[j + "</object>".len()..];

        let name = tag_text(obj, "name").unwrap_or("").to_lowercase();
        if !names.is_empty() && !names.contains(name.as_str()) {
            continue;
        }
        let Some(bb) = tag_text(obj, "bndbox") else {
            continue;
        };
        let coord = |t: &str| tag_text(bb, t).and_then(|v| v.parse::<f32>().ok());
        let (Some(x1), Some(y1), Some(x2), Some(y2)) =
            (coord("xmin"), coord("ymin"), coord("xmax"), coord("ymax"))
        else {
            continue;
        };
        if x2 > x1 && y2 > y1 {
            boxes.push([x1, y1, x2, y2]);
        }
    }
    boxes
}

/// Reads a VOC XML file. Missing file gives no boxes.
pub fn read_voc_person_boxes(path: &Path, names: &HashSet<&str>) -> Vec<[f32; 4]> {
    match std::fs::read_to_string(path) {
        Ok(text) => parse_voc_person_boxes(&text, names),
        Err(_) => Vec::new(),
    }
}

/// What ends up in `image`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Modality {
    /// 4-channel stack, `has_rgb = has_thermal = true`.
    Rgbt,
    Rgb,
    Thermal,
}

impl Modality {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "rgbt" => Ok(Modality::Rgbt),
            "rgb" => Ok(Modality::Rgb),
            "thermal" => Ok(Modality::Thermal),
            other => Err(DatasetError::InvalidSample(format!(
                "modality must be rgbt|rgb|thermal, got {other:?}"
            ))),
        }
    }

    fn needs_vis(self) -> bool {
        matches!(self, Modality::Rgbt | Modality::Rgb)
    }

    fn needs_ir(self) -> bool {
        matches!(self, Modality::Rgbt | Modality::Thermal)
    }
}

type Record = (Option<PathBuf>, Option<PathBuf>, PathBuf);

/// LLVIP RGB+IR pairs with person boxes.
pub struct LLVIP {
    pub root: PathBuf,
    pub split: String,
    pub modality: Modality,
    transforms: Option<Transform>,
    index: Vec<Record>,
}

impl LLVIP {
    pub fn new(
        root: impl AsRef<Path>,
        split: &str,
        modality: Modality,
        transforms: Option<Transform>,
    ) -> Result<Self> {
        let split = verify_split(&LLVIP_META, split)?;
        let root = resolve_root(root);
        let (vis, ir, ann) = Self::dirs(&root, &split);
        let exists = (!modality.needs_vis() || vis.is_dir())
            && (!modality.needs_ir() || ir.is_dir())
            && ann.is_dir();
        ensure_exists(&LLVIP_META, &root, exists)?;

        let primary = if modality.needs_vis() { &vis } else { &ir };
        let mut index = Vec::new();
        for img in list_images(primary, &IMG_EXT) {
            let stem = file_stem(&img);
            let ir_path = find_with_stem(&ir, &stem, &IMG_EXT);
            let vis_path = find_with_stem(&vis, &stem, &IMG_EXT);
            if modality == Modality::Rgbt && (ir_path.is_none() || vis_path.is_none()) {
                continue;
            }
            index.push((vis_path, ir_path, ann.join(format!("{stem}.xml"))));
        }
        Ok(LLVIP {
            root,
            split,
            modality,
            transforms,
            index,
        })
    }

    fn dirs(root: &Path, split: &str) -> (PathBuf, PathBuf, PathBuf) {
        (
            root.join("visible").join(split),
            root.join("infrared").join(split),
            root.join("Annotations"),
        )
    }
}

impl SkyLensDataset for LLVIP {
    fn len(&self) -> usize {
        self.index.len()
    }

    fn get(&self, idx: usize) -> Result<Sample> {
        let (vis_path, ir_path, ann_path) = &self.index[idx];
        let missing = |what: &'static str| DatasetError::Unreadable {
            kind: what,
            path: String::new(),
        };

        let rgb = if self.modality.needs_vis() {
            Some(read_rgb_raster(
                vis_path.as_ref().ok_or_else(|| missing("image"))?,
            )?)
        } else {
            None
        };
        let mut thermal = if self.modality.needs_ir() {
            Some(read_thermal_raster(
                ir_path.as_ref().ok_or_else(|| missing("thermal frame"))?,
            )?)
        } else {
            None
        };
        if let (Some(r), Some(t)) = (&rgb, &thermal) {
            if (t.h, t.w) != (r.h, r.w) {
                thermal = Some(resize_bilinear(t, r.h, r.w));
            }
        }

        let image = match (&rgb, &thermal) {
            (Some(r), None) => tensor_u8(r),
            _ => tensor_f32(&stack_rgb_thermal(rgb.as_ref(), thermal.as_ref())?),
        };
        let mut s = Sample::new(image, rgb.is_some(), thermal.is_some());
        let names: HashSet<&str> = PERSON_NAMES.into_iter().collect();
        s.person_boxes = Some(boxes_tensor(&read_voc_person_boxes(ann_path, &names)));
        finish_sample(s, self.transforms.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const XML: &str = r#"<annotation>
        <object><name>person</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>30</xmax><ymax>60</ymax></bndbox></object>
        <object><name>car</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>5</xmax><ymax>5</ymax></bndbox></object>
        <object><name> Person </name><bndbox><xmin>5</xmin><ymin>5</ymin><xmax>5</xmax><ymax>9</ymax></bndbox></object>
        <object><name>pedestrian</name><bndbox><xmin>2</xmin><ymin>3</ymin><xmax>4</xmax><ymax>8</ymax></bndbox></object>
    </annotation>"#;

    #[test]
    fn voc_filters_by_name_and_drops_degenerate() {
        let names: HashSet<&str> = PERSON_NAMES.into_iter().collect();
        let b = parse_voc_person_boxes(XML, &names);
        assert_eq!(b, vec![[10.0, 20.0, 30.0, 60.0], [2.0, 3.0, 4.0, 8.0]]);
    }

    #[test]
    fn empty_name_set_keeps_all_classes() {
        let b = parse_voc_person_boxes(XML, &HashSet::new());
        assert_eq!(b.len(), 3); // the zero-width box is still dropped
    }

    #[test]
    fn modality_parsing() {
        assert_eq!(Modality::parse("rgbt").unwrap(), Modality::Rgbt);
        assert!(Modality::parse("depth").is_err());
    }
}
