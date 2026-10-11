//! Rust twin of `utils/training_args.py`.
//!
//! SkyLens training arguments: the SkyLens-specific fields (identical names and defaults to
//! `SkyLensTrainingArguments`) plus the subset of HuggingFace `TrainingArguments` that the
//! loop in `trainer.rs` actually uses, with HF's default values.
//!
//! NOT PORTED: the rest of `TrainingArguments` (distributed setup, DeepSpeed, hub push,
//! `torch_compile`, ...). They configure machinery the Rust loop does not have. Fields that
//! the Python training scripts pass but that are missing here are rejected by
//! `deny_unknown_fields`, so a config file that relies on them fails loudly instead of being
//! silently ignored.
//!
//! Modality dropout is a data pipeline / model concern (README §2.2) and so is not here.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

fn d_person_head_stride() -> usize {
    4
}
fn d_true() -> bool {
    true
}
fn d_eval_max_detections() -> usize {
    100
}
fn d_eval_score_threshold() -> f64 {
    0.3
}
fn d_eval_ap_score_floor() -> f64 {
    0.05
}
fn d_point_distance_threshold() -> f64 {
    8.0
}
fn d_num_danger_classes() -> usize {
    4
}
fn d_seg_ignore_index() -> i64 {
    255
}
fn d_learning_rate() -> f64 {
    5e-5
}
fn d_weight_decay() -> f64 {
    0.0
}
fn d_epochs() -> f64 {
    3.0
}
fn d_batch() -> usize {
    8
}
fn d_one_usize() -> usize {
    1
}
fn d_max_grad_norm() -> f64 {
    1.0
}
fn d_scheduler() -> String {
    "linear".to_string()
}
fn d_seed() -> u64 {
    42
}
fn d_logging_steps() -> u64 {
    500
}
fn d_save_steps() -> u64 {
    500
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SkyLensTrainingArguments {
    // ---- HuggingFace TrainingArguments subset (HF defaults) ----
    pub output_dir: PathBuf,
    #[serde(default = "d_learning_rate")]
    pub learning_rate: f64,
    #[serde(default = "d_weight_decay")]
    pub weight_decay: f64,
    #[serde(default = "d_epochs")]
    pub num_train_epochs: f64,
    #[serde(default = "d_batch")]
    pub per_device_train_batch_size: usize,
    #[serde(default = "d_batch")]
    pub per_device_eval_batch_size: usize,
    #[serde(default = "d_one_usize")]
    pub gradient_accumulation_steps: usize,
    #[serde(default = "d_max_grad_norm")]
    pub max_grad_norm: f64,
    /// Fraction of total steps used for linear warmup.
    #[serde(default)]
    pub warmup_ratio: f64,
    /// `linear` | `constant` | `cosine`.
    #[serde(default = "d_scheduler")]
    pub lr_scheduler_type: String,
    #[serde(default = "d_seed")]
    pub seed: u64,
    #[serde(default = "d_logging_steps")]
    pub logging_steps: u64,
    #[serde(default = "d_save_steps")]
    pub save_steps: u64,
    /// Evaluate every N steps; `None` evaluates once per epoch.
    #[serde(default)]
    pub eval_steps: Option<u64>,
    #[serde(default)]
    pub save_total_limit: Option<usize>,
    #[serde(default)]
    pub resume_from_checkpoint: Option<PathBuf>,
    #[serde(default)]
    pub dataloader_num_workers: usize,

    // ---- SkyLens-specific fields ----
    /// Point-head heatmap stride (input resolution / heatmap resolution); used to map heatmap
    /// coordinates back to input pixels at evaluation.
    #[serde(default = "d_person_head_stride")]
    pub person_head_stride: usize,
    /// Log each component of the loss dict (seg / heatmap / wh). Practically mandatory with
    /// per-head training (README §6.3): the total alone hides which head is learning.
    #[serde(default = "d_true")]
    pub log_loss_components: bool,
    /// Max peaks taken from the heatmap per image at evaluation (top-k).
    #[serde(default = "d_eval_max_detections")]
    pub eval_max_detections: usize,
    /// Minimum heatmap score counted as a detection at evaluation.
    #[serde(default = "d_eval_score_threshold")]
    pub eval_score_threshold: f64,
    /// Minimum score used for bbox mAP / point AP, kept separate from
    /// `eval_score_threshold` so the PR curve is not truncated.
    #[serde(default = "d_eval_ap_score_floor")]
    pub eval_ap_score_floor: f64,
    /// Epochs at the start of training during which the backbone is frozen (README §6.1). 0
    /// disables the warmup.
    #[serde(default)]
    pub freeze_backbone_epochs: f64,
    /// Point-matching tolerance in input-resolution pixels (distance, not IoU: README §1.4).
    #[serde(default = "d_point_distance_threshold")]
    pub point_distance_threshold: f64,
    /// Danger-zone class count (normal / fire / collapse / road_blocked).
    #[serde(default = "d_num_danger_classes")]
    pub num_danger_classes: usize,
    /// Ignore index of the seg labels, excluded from both loss and metrics.
    #[serde(default = "d_seg_ignore_index")]
    pub seg_ignore_index: i64,
    /// Must be false: the collator turns raw samples into the batch.
    #[serde(default)]
    pub remove_unused_columns: bool,
}

impl SkyLensTrainingArguments {
    /// Defaults for everything but `output_dir`.
    pub fn new(output_dir: impl Into<PathBuf>) -> Self {
        let json = serde_json::json!({ "output_dir": output_dir.into() });
        serde_json::from_value(json).expect("defaults deserialise")
    }

    /// `__post_init__` checks. Returns a readable message on the first violation.
    pub fn validate(&self) -> Result<(), String> {
        if self.remove_unused_columns {
            return Err("SkyLensTrainingArguments needs remove_unused_columns=false: with true \
                        the Trainer would strip raw keys such as 'image' before the \
                        SkyLensCollator runs and raise a KeyError."
                .into());
        }
        if self.person_head_stride < 1 {
            return Err("person_head_stride must be >= 1".into());
        }
        if self.eval_max_detections < 1 {
            return Err("eval_max_detections must be >= 1".into());
        }
        if !(0.0..=1.0).contains(&self.eval_score_threshold) {
            return Err("eval_score_threshold must be within [0, 1]".into());
        }
        if !(0.0..=1.0).contains(&self.eval_ap_score_floor) {
            return Err("eval_ap_score_floor must be within [0, 1]".into());
        }
        if self.freeze_backbone_epochs < 0.0 {
            return Err("freeze_backbone_epochs must be >= 0".into());
        }
        if self.gradient_accumulation_steps < 1 {
            return Err("gradient_accumulation_steps must be >= 1".into());
        }
        if self.per_device_train_batch_size < 1 {
            return Err("per_device_train_batch_size must be >= 1".into());
        }
        if !matches!(self.lr_scheduler_type.as_str(), "linear" | "constant" | "cosine") {
            return Err(format!(
                "lr_scheduler_type must be linear|constant|cosine; got {:?}",
                self.lr_scheduler_type
            ));
        }
        Ok(())
    }

    /// Parses JSON and runs [`validate`](Self::validate).
    pub fn from_json(text: &str) -> Result<Self, String> {
        let args: Self = serde_json::from_str(text).map_err(|e| e.to_string())?;
        args.validate()?;
        Ok(args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_python() {
        let a = SkyLensTrainingArguments::new("out");
        assert_eq!(a.person_head_stride, 4);
        assert!(a.log_loss_components);
        assert_eq!(a.eval_max_detections, 100);
        assert_eq!(a.eval_score_threshold, 0.3);
        assert_eq!(a.eval_ap_score_floor, 0.05);
        assert_eq!(a.freeze_backbone_epochs, 0.0);
        assert_eq!(a.point_distance_threshold, 8.0);
        assert_eq!(a.num_danger_classes, 4);
        assert_eq!(a.seg_ignore_index, 255);
        assert!(!a.remove_unused_columns);
        assert!(a.validate().is_ok());
    }

    #[test]
    fn hf_defaults() {
        let a = SkyLensTrainingArguments::new("out");
        assert_eq!(a.learning_rate, 5e-5);
        assert_eq!(a.num_train_epochs, 3.0);
        assert_eq!(a.per_device_train_batch_size, 8);
        assert_eq!(a.max_grad_norm, 1.0);
        assert_eq!(a.seed, 42);
    }

    #[test]
    fn validation_rules() {
        let mut a = SkyLensTrainingArguments::new("out");
        a.remove_unused_columns = true;
        assert!(a.validate().is_err());
        a.remove_unused_columns = false;
        a.eval_score_threshold = 1.5;
        assert!(a.validate().is_err());
        a.eval_score_threshold = 0.3;
        a.freeze_backbone_epochs = -1.0;
        assert!(a.validate().is_err());
        a.freeze_backbone_epochs = 0.0;
        a.person_head_stride = 0;
        assert!(a.validate().is_err());
    }

    #[test]
    fn json_roundtrip_and_unknown_keys() {
        let a = SkyLensTrainingArguments::from_json(
            r#"{"output_dir": "runs/x", "freeze_backbone_epochs": 2.0, "person_head_stride": 2}"#,
        )
        .unwrap();
        assert_eq!(a.freeze_backbone_epochs, 2.0);
        assert_eq!(a.person_head_stride, 2);
        assert!(SkyLensTrainingArguments::from_json(r#"{"output_dir": "x", "bogus": 1}"#).is_err());
        assert!(SkyLensTrainingArguments::from_json(r#"{"output_dir": "x", "remove_unused_columns": true}"#).is_err());
    }
}
