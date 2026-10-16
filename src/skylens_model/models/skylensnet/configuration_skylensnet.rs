//! Rust twin of `models/skylensnet/configuration_skylensnet.py`.
//!
//! [`SkyLensConfig`] mirrors the HuggingFace `PretrainedConfig` subclass field for
//! field, with the same defaults, so the `config.json` that `save_pretrained`
//! writes deserialises directly. Keys that HuggingFace adds on its own
//! (`architectures`, `transformers_version`, `dtype`, `use_cache`, ...) are
//! ignored, and keys missing from older checkpoints fall back to the Python
//! defaults (for example a config saved before the offset head existed loads
//! with `use_offset_head = false`, exactly like `SkyLensConfig.from_dict`).
//!
//! Training-only fields (loss weights, modality dropout probabilities) are kept
//! so a round trip does not lose them, even though inference never reads them.

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

/// HuggingFace `model_type` of the SkyLens checkpoints.
pub const MODEL_TYPE: &str = "skylens";

/// Configuration of `SkyLensModel` / `SkyLensForDisasterPerception`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SkyLensConfig {
    /// Encoder name handed to `AutoBackbone` (e.g. `"microsoft/resnet-50"`).
    pub backbone: String,
    /// True when the encoder came from the timm ecosystem (`TimmBackbone`).
    pub use_timm_backbone: bool,
    /// True when ImageNet weights were loaded and the first conv was inflated.
    pub use_pretrained_backbone: bool,
    /// Indices into the backbone `stage_names` (`["stem", "stage1", ...]`)
    /// whose outputs feed the UNet skip connections. Negative values count from
    /// the end, as in `BackboneMixin`.
    pub backbone_out_indices: Vec<i64>,
    /// Input channels without the validity plane: RGB (3) + thermal (1).
    pub in_channels: i64,
    /// Append a 0/1 thermal validity plane after the thermal channel.
    pub use_validity_channel: bool,
    /// Output channels of each UNet decoder block.
    pub decoder_channels: Vec<i64>,
    /// Hazard classes: 0 normal, 1 fire, 2 collapse, 3 road_blocked.
    pub num_danger_classes: i64,
    /// Output stride of the point-detection head.
    pub person_head_stride: i64,
    /// Training-only: probability of dropping thermal.
    pub modality_dropout_rgb_only: f64,
    /// Training-only: probability of dropping RGB.
    pub modality_dropout_thermal_only: f64,
    pub seg_loss_weight: f64,
    pub heatmap_loss_weight: f64,
    pub wh_loss_weight: f64,
    pub dice_loss_weight: f64,
    /// Training-only: the box branch reads the shared feature without
    /// back-propagating into it. A no-op at inference.
    pub wh_detach: bool,
    /// CenterNet sub-pixel offset head on top of the heatmap stem.
    pub use_offset_head: bool,
    pub offset_loss_weight: f64,
    pub danger_ignore_index: i64,
    /// HuggingFace bookkeeping, kept so a re-serialised config stays loadable.
    pub model_type: String,
}

impl Default for SkyLensConfig {
    fn default() -> Self {
        Self {
            backbone: "microsoft/resnet-50".to_string(),
            use_timm_backbone: false,
            use_pretrained_backbone: true,
            backbone_out_indices: vec![0, 1, 2, 3],
            in_channels: 4,
            use_validity_channel: false,
            decoder_channels: vec![256, 128, 64, 32],
            num_danger_classes: 4,
            person_head_stride: 4,
            modality_dropout_rgb_only: 0.25,
            modality_dropout_thermal_only: 0.25,
            seg_loss_weight: 1.0,
            heatmap_loss_weight: 1.0,
            wh_loss_weight: 0.1,
            dice_loss_weight: 0.0,
            wh_detach: false,
            use_offset_head: false,
            offset_loss_weight: 1.0,
            danger_ignore_index: 255,
            model_type: MODEL_TYPE.to_string(),
        }
    }
}

impl SkyLensConfig {
    /// Parses a `config.json` body and applies the same checks as the Python
    /// constructor.
    pub fn from_json_str(json: &str) -> Result<Self> {
        let config: Self = serde_json::from_str(json).context("parsing SkyLens config.json")?;
        config.validate()?;
        Ok(config)
    }

    /// Reads `config.json` from a file path.
    pub fn from_json_file(path: &Path) -> Result<Self> {
        let body =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Self::from_json_str(&body).with_context(|| format!("loading {}", path.display()))
    }

    /// Reads `<dir>/config.json`, the layout `save_pretrained` writes.
    pub fn from_pretrained(dir: &Path) -> Result<Self> {
        Self::from_json_file(&dir.join("config.json"))
    }

    /// Serialises to the HF-style JSON layout.
    pub fn to_json_string(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// The same invariants `SkyLensConfig.__init__` enforces, plus the ones the
    /// Rust model needs to build at all.
    pub fn validate(&self) -> Result<()> {
        if self.modality_dropout_rgb_only + self.modality_dropout_thermal_only > 1.0 {
            bail!(
                "modality_dropout_rgb_only + modality_dropout_thermal_only must be <= 1.0 \
                 (got {} + {})",
                self.modality_dropout_rgb_only,
                self.modality_dropout_thermal_only
            );
        }
        if self.in_channels < 4 {
            bail!("in_channels must be at least 4 (RGB 3 + thermal 1)");
        }
        if self.person_head_stride < 1 {
            bail!("person_head_stride must be >= 1");
        }
        if self.num_danger_classes < 1 {
            bail!("num_danger_classes must be >= 1");
        }
        if self.decoder_channels.is_empty() || self.decoder_channels.iter().any(|&c| c < 1) {
            bail!("decoder_channels must be a non-empty list of positive channel counts");
        }
        if self.backbone_out_indices.is_empty() {
            bail!("backbone_out_indices must not be empty");
        }
        Ok(())
    }

    /// Channels the first backbone conv actually receives (validity included).
    pub fn num_input_channels(&self) -> i64 {
        self.in_channels + i64::from(self.use_validity_channel)
    }

    /// `backbone_out_indices` resolved against `num_stage_names` stage names
    /// (`stem` + one per encoder stage), negative indices wrapped like
    /// `BackboneMixin.verify_out_features_out_indices`. Must be strictly
    /// increasing, as HF requires.
    pub fn resolved_out_indices(&self, num_stage_names: usize) -> Result<Vec<usize>> {
        let n = num_stage_names as i64;
        let mut out = Vec::with_capacity(self.backbone_out_indices.len());
        for &idx in &self.backbone_out_indices {
            let pos = if idx < 0 { idx.rem_euclid(n) } else { idx };
            if !(0..n).contains(&pos) {
                bail!("backbone_out_indices entry {idx} is outside {num_stage_names} stage names");
            }
            out.push(pos as usize);
        }
        if out.windows(2).any(|w| w[0] >= w[1]) {
            bail!(
                "backbone_out_indices must be unique and in stage order, got {:?}",
                out
            );
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `config.json` of a real run (runs/skylens/final), saved by
    /// transformers 5.15 before the offset head and the Dice term existed.
    const SAVED: &str = r#"{
      "architectures": ["SkyLensForDisasterPerception"],
      "backbone": "microsoft/resnet-50",
      "backbone_out_indices": [0, 1, 2, 3],
      "danger_ignore_index": 255,
      "decoder_channels": [256, 128, 64, 32],
      "dtype": "float32",
      "heatmap_loss_weight": 1.0,
      "in_channels": 4,
      "modality_dropout_rgb_only": 0.25,
      "modality_dropout_thermal_only": 0.25,
      "model_type": "skylens",
      "num_danger_classes": 4,
      "person_head_stride": 4,
      "seg_loss_weight": 1.0,
      "transformers_version": "5.15.0",
      "use_cache": false,
      "use_pretrained_backbone": true,
      "use_timm_backbone": false,
      "use_validity_channel": false,
      "wh_loss_weight": 0.1
    }"#;

    #[test]
    fn defaults_match_python() {
        let c = SkyLensConfig::default();
        assert_eq!(c.backbone, "microsoft/resnet-50");
        assert_eq!(c.backbone_out_indices, vec![0, 1, 2, 3]);
        assert_eq!(c.decoder_channels, vec![256, 128, 64, 32]);
        assert_eq!(c.in_channels, 4);
        assert_eq!(c.num_danger_classes, 4);
        assert_eq!(c.person_head_stride, 4);
        assert_eq!(c.wh_loss_weight, 0.1);
        assert_eq!(c.danger_ignore_index, 255);
        assert!(!c.use_offset_head && !c.wh_detach && !c.use_validity_channel);
        assert_eq!(c.num_input_channels(), 4);
    }

    #[test]
    fn saved_config_loads_and_ignores_hf_keys() {
        let c = SkyLensConfig::from_json_str(SAVED).unwrap();
        assert_eq!(c.backbone, "microsoft/resnet-50");
        assert_eq!(c.decoder_channels, vec![256, 128, 64, 32]);
        assert!(c.use_pretrained_backbone);
    }

    /// Port of `test_offset_head.py::test_old_config_loads_without_head`:
    /// a config without the offset keys loads with the head disabled.
    #[test]
    fn old_config_loads_without_offset_head() {
        let c = SkyLensConfig::from_json_str(SAVED).unwrap();
        assert!(!c.use_offset_head);
        assert_eq!(c.offset_loss_weight, 1.0);
        assert_eq!(c.dice_loss_weight, 0.0);
    }

    #[test]
    fn validity_channel_adds_an_input() {
        let c = SkyLensConfig {
            use_validity_channel: true,
            ..Default::default()
        };
        assert_eq!(c.num_input_channels(), 5);
    }

    #[test]
    fn invalid_configs_are_rejected() {
        let too_much_dropout =
            r#"{"modality_dropout_rgb_only": 0.6, "modality_dropout_thermal_only": 0.5}"#;
        assert!(SkyLensConfig::from_json_str(too_much_dropout).is_err());
        assert!(SkyLensConfig::from_json_str(r#"{"in_channels": 3}"#).is_err());
    }

    #[test]
    fn json_round_trip() {
        let c = SkyLensConfig {
            use_offset_head: true,
            ..Default::default()
        };
        let back = SkyLensConfig::from_json_str(&c.to_json_string().unwrap()).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn out_indices_resolve_like_backbone_mixin() {
        let c = SkyLensConfig {
            backbone_out_indices: vec![1, -1],
            ..Default::default()
        };
        assert_eq!(c.resolved_out_indices(5).unwrap(), vec![1, 4]);
        let bad = SkyLensConfig {
            backbone_out_indices: vec![2, 1],
            ..Default::default()
        };
        assert!(bad.resolved_out_indices(5).is_err());
        let oob = SkyLensConfig {
            backbone_out_indices: vec![0, 5],
            ..Default::default()
        };
        assert!(oob.resolved_out_indices(5).is_err());
    }
}
