//! Rust twin of `models/skylensnet/modeling_skylensnet.py` (inference only).
//!
//! ```text
//! pixel_values (B, C, H, W)   C = RGB(3) + thermal(1) [+ validity(1)]
//!     |  modality mask: only the explicit one (no sampling at inference)
//!     v
//! HF ResNet backbone  -> multi-scale features (backbone_out_indices)
//!     v
//! UNet decoder (skip connections) -> high-resolution feature map
//!     |-> danger_head : per-pixel hazard logits, bilinear to (H, W)
//!     |-> person_stem -> heatmap_head (sigmoid) [-> offset_head]
//!     '-> wh_stem     -> wh_head
//! ```
//!
//! Every parameter is created under the same dotted path as the PyTorch
//! `state_dict` key (`skylens.backbone.embedder.embedder.convolution.weight`,
//! `skylens.decoder.blocks.0.conv.block.1.running_mean`, `person_stem.block.3.weight`,
//! ...), so `VarStore::load` reads the `model.safetensors` that `save_pretrained`
//! writes. Training-side pieces (losses, weight init, 4-channel inflation,
//! modality dropout sampling, `wh_detach`) have no effect at inference and are
//! not ported. BatchNorm always runs with its running statistics (`eval()`).
//!
//! [`SkyLensNetDetector`] wraps the network as a [`Detector`] for the frame
//! hook, reproducing the deployment path of `scripts/eval_deploy.py`.
//!
//! # Where this cannot match PyTorch exactly
//!
//! - **Backbone.** Only the transformers `ResNetBackbone` (`use_timm_backbone =
//!   false`, the `microsoft/resnet-*` family) is implemented. `config.json`
//!   only names the backbone, so the stage layout is read from the
//!   safetensors header (embedding width, stage depths, hidden sizes, basic vs
//!   bottleneck). `downsample_in_bottleneck` and, for bottleneck nets,
//!   `downsample_in_first_stage` cannot be seen in the weights and are taken as
//!   the HF defaults (`false`), which is what every `microsoft/resnet-*` uses.
//! - **Unused stages are not built.** HF runs every encoder stage even when
//!   `backbone_out_indices` stops earlier (the default `(0, 1, 2, 3)` never reads
//!   `stage4`). Their outputs are discarded, so skipping them changes nothing
//!   but time; their weights stay in the file and are ignored.
//! - **Legacy checkpoints** saved before the `wh` branch got its own stem (no
//!   `wh_stem.*` keys) run `wh_head` on the `person_stem` output, which is how
//!   they were trained. The current Python class would instead randomly
//!   initialise `wh_stem` for them.
//! - **Resizing.** OpenCV is not linked; `utils::tiling::resize` reimplements
//!   `cv2.resize` (`INTER_AREA` / `INTER_LINEAR`) and agrees to float rounding.
//! - **Thermal.** Python normalises thermal per frame into `[0.1, 1.0]` at
//!   native resolution (`_stack_rgb_thermal`), then the training cache resizes
//!   it and stores it as 8-bit PNG. Here the same normalisation is applied, then
//!   the frame is resized in `f32` without the 8-bit quantisation.
//! - **Hazard map.** Python has no frame-level segmentation inference; the head
//!   was trained and evaluated on frames squashed to 512x512 with area
//!   interpolation (`cache.resize_sample`). [`HazardMode::Squash`] (default)
//!   runs that view once per frame, upsamples the logits bilinearly
//!   (`align_corners = false`, as the model itself does) to the frame size and
//!   applies softmax. [`HazardMode::Tiles`] instead averages the per-tile
//!   softmax of the person tiles, then resizes bilinearly.
//! - **Precision.** On CUDA the weights, inputs and activations are bf16
//!   ([`Precision::Bf16`], default), which halves the GPU memory of the weights
//!   and the caching allocator. [`Precision::Fp16Autocast`] runs as
//!   `eval_deploy.py` does (fp32 weights under fp16 autocast), for comparing
//!   against its numbers. CPU always runs in fp32. Outputs are fp32 either way.
//! - **Decoding** runs on CPU in `utils::metrics::decode_heatmap_peaks`; score
//!   ties are ordered by index where torch `topk` leaves them unspecified.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result, anyhow, bail, ensure};
use tch::nn::{self, Module, ModuleT};
use tch::{Device, Kind, Tensor};

use super::configuration_skylensnet::SkyLensConfig;
use crate::detect::{Detector, FrameDetections, HAZARD_CLASSES, PersonPoint};
use crate::utils::metrics::{Detection, decode_heatmap_peaks};
use crate::utils::tiling::{self, Image, Interpolation};

/// Weight file written by `save_pretrained`.
pub const WEIGHTS_NAME: &str = "model.safetensors";
/// Config file written by `save_pretrained`.
pub const CONFIG_NAME: &str = "config.json";
/// State-dict prefix of the backbone inside `SkyLensForDisasterPerception`.
pub const BACKBONE_PREFIX: &str = "skylens.backbone.";

// ---------------------------------------------------------------------------
// safetensors header
// ---------------------------------------------------------------------------

/// dtype and shape of one tensor in a safetensors file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorMeta {
    pub dtype: String,
    pub shape: Vec<i64>,
}

/// Reads only the JSON header of a safetensors file (8-byte little-endian
/// length, then the header), without loading any tensor data.
pub fn read_safetensors_header(path: &Path) -> Result<BTreeMap<String, TensorMeta>> {
    let mut f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len).context("reading the safetensors header length")?;
    let n = u64::from_le_bytes(len);
    ensure!(n <= 100 * 1024 * 1024, "safetensors header of {n} bytes is implausibly large");
    let mut buf = vec![0u8; n as usize];
    f.read_exact(&mut buf).context("reading the safetensors header")?;
    let raw: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&buf).context("parsing the safetensors header")?;
    let mut out = BTreeMap::new();
    for (name, v) in raw {
        if name == "__metadata__" {
            continue;
        }
        let dtype = v
            .get("dtype")
            .and_then(|d| d.as_str())
            .ok_or_else(|| anyhow!("tensor {name} has no dtype"))?
            .to_string();
        let shape = v
            .get("shape")
            .and_then(|s| s.as_array())
            .ok_or_else(|| anyhow!("tensor {name} has no shape"))?
            .iter()
            .map(|d| d.as_i64().ok_or_else(|| anyhow!("tensor {name} has a bad shape")))
            .collect::<Result<Vec<_>>>()?;
        out.insert(name, TensorMeta { dtype, shape });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// building blocks
// ---------------------------------------------------------------------------

fn conv(p: &nn::Path, i: i64, o: i64, k: i64, stride: i64, bias: bool) -> nn::Conv2D {
    nn::conv2d(
        p,
        i,
        o,
        k,
        nn::ConvConfig { stride, padding: k / 2, bias, ..Default::default() },
    )
}

fn bn(p: &nn::Path, c: i64) -> nn::BatchNorm {
    // PyTorch BatchNorm2d defaults: eps 1e-5, momentum 0.1, affine.
    nn::batch_norm2d(p, c, Default::default())
}

/// `SkyLensConvBlock`: (Conv3x3 no-bias, BN, ReLU) x2 under `block.{0,1,3,4}`.
#[derive(Debug)]
pub struct SkyLensConvBlock {
    conv1: nn::Conv2D,
    bn1: nn::BatchNorm,
    conv2: nn::Conv2D,
    bn2: nn::BatchNorm,
}

impl SkyLensConvBlock {
    pub fn new(p: &nn::Path, in_channels: i64, out_channels: i64) -> Self {
        let b = p.sub("block");
        Self {
            conv1: conv(&b.sub("0"), in_channels, out_channels, 3, 1, false),
            bn1: bn(&b.sub("1"), out_channels),
            conv2: conv(&b.sub("3"), out_channels, out_channels, 3, 1, false),
            bn2: bn(&b.sub("4"), out_channels),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x = self.bn1.forward_t(&self.conv1.forward(x), false).relu();
        self.bn2.forward_t(&self.conv2.forward(&x), false).relu()
    }
}

/// `SkyLensDecoderBlock`: upsample (to the skip size, or x2), concat, conv block.
#[derive(Debug)]
struct SkyLensDecoderBlock {
    conv: SkyLensConvBlock,
}

impl SkyLensDecoderBlock {
    fn forward(&self, x: &Tensor, skip: Option<&Tensor>) -> Tensor {
        let x = match skip {
            Some(s) => {
                let sz = s.size();
                let up = x.upsample_nearest2d([sz[2], sz[3]], None::<f64>, None::<f64>);
                Tensor::cat(&[up, s.shallow_clone()], 1)
            }
            None => {
                // F.interpolate(scale_factor=2.0): size floor(2 * in), scale 2.
                let sz = x.size();
                x.upsample_nearest2d([sz[2] * 2, sz[3] * 2], 2.0, 2.0)
            }
        };
        self.conv.forward(&x)
    }
}

/// `SkyLensUNetDecoder` under `skylens.decoder.blocks.{i}.conv`.
#[derive(Debug)]
struct SkyLensUNetDecoder {
    blocks: Vec<SkyLensDecoderBlock>,
    out_channels: i64,
}

impl SkyLensUNetDecoder {
    /// `encoder_channels` is shallow to deep; the deepest feeds the decoder.
    fn new(p: &nn::Path, encoder_channels: &[i64], decoder_channels: &[i64]) -> Self {
        let skips: Vec<i64> = encoder_channels[..encoder_channels.len() - 1].iter().rev().copied().collect();
        let mut in_ch = encoder_channels[encoder_channels.len() - 1];
        let bp = p.sub("blocks");
        let mut blocks = Vec::with_capacity(decoder_channels.len());
        for (i, &out_ch) in decoder_channels.iter().enumerate() {
            let skip_ch = skips.get(i).copied().unwrap_or(0);
            blocks.push(SkyLensDecoderBlock {
                conv: SkyLensConvBlock::new(&bp.sub(i).sub("conv"), in_ch + skip_ch, out_ch),
            });
            in_ch = out_ch;
        }
        Self { blocks, out_channels: in_ch }
    }

    fn forward(&self, features: &[Tensor]) -> Tensor {
        let skips: Vec<&Tensor> = features[..features.len() - 1].iter().rev().collect();
        let mut x = features[features.len() - 1].shallow_clone();
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, skips.get(i).copied());
        }
        x
    }
}

// ---------------------------------------------------------------------------
// transformers ResNetBackbone
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResNetLayerType {
    Basic,
    Bottleneck,
}

/// The parts of a transformers `ResNetConfig` that shape the weights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResNetSpec {
    pub embedding_size: i64,
    pub hidden_sizes: Vec<i64>,
    pub depths: Vec<i64>,
    pub layer_type: ResNetLayerType,
    pub downsample_in_first_stage: bool,
    pub downsample_in_bottleneck: bool,
}

impl ResNetSpec {
    /// Published `microsoft/resnet-*` configurations.
    pub fn for_name(name: &str) -> Option<Self> {
        use ResNetLayerType::*;
        let (layer_type, depths): (_, &[i64]) = match name {
            "microsoft/resnet-18" => (Basic, &[2, 2, 2, 2][..]),
            "microsoft/resnet-34" => (Basic, &[3, 4, 6, 3][..]),
            "microsoft/resnet-50" => (Bottleneck, &[3, 4, 6, 3][..]),
            "microsoft/resnet-101" => (Bottleneck, &[3, 4, 23, 3][..]),
            "microsoft/resnet-152" => (Bottleneck, &[3, 8, 36, 3][..]),
            _ => return None,
        };
        let hidden_sizes = match layer_type {
            Basic => vec![64, 128, 256, 512],
            Bottleneck => vec![256, 512, 1024, 2048],
        };
        Some(Self {
            embedding_size: 64,
            hidden_sizes,
            depths: depths.to_vec(),
            layer_type,
            downsample_in_first_stage: false,
            downsample_in_bottleneck: false,
        })
    }

    /// Reads the stage layout from checkpoint keys under `prefix`
    /// (`skylens.backbone.` for a SkyLens checkpoint).
    pub fn from_header(header: &BTreeMap<String, TensorMeta>, prefix: &str) -> Result<Self> {
        let shape = |key: &str| header.get(&format!("{prefix}{key}")).map(|m| m.shape.clone());
        let emb = shape("embedder.embedder.convolution.weight")
            .ok_or_else(|| anyhow!("no {prefix}embedder.embedder.convolution.weight: not a transformers ResNet backbone"))?;
        ensure!(emb.len() == 4, "stem conv weight must be 4-D, got {emb:?}");

        let stages_prefix = format!("{prefix}encoder.stages.");
        let mut depths: BTreeMap<usize, usize> = BTreeMap::new();
        for key in header.keys() {
            let Some(rest) = key.strip_prefix(&stages_prefix) else { continue };
            let mut parts = rest.split('.');
            let (Some(s), Some("layers"), Some(l)) = (parts.next(), parts.next(), parts.next()) else {
                continue;
            };
            let (Ok(s), Ok(l)) = (s.parse::<usize>(), l.parse::<usize>()) else { continue };
            let d = depths.entry(s).or_insert(0);
            *d = (*d).max(l + 1);
        }
        ensure!(!depths.is_empty(), "no encoder stages under {stages_prefix}");
        ensure!(
            depths.keys().copied().eq(0..depths.len()),
            "encoder stages are not numbered 0..n: {:?}",
            depths.keys().collect::<Vec<_>>()
        );

        let layer_type = if shape("encoder.stages.0.layers.0.layer.2.convolution.weight").is_some() {
            ResNetLayerType::Bottleneck
        } else {
            ResNetLayerType::Basic
        };
        let last = match layer_type {
            ResNetLayerType::Bottleneck => 2,
            ResNetLayerType::Basic => 1,
        };
        let hidden_sizes = (0..depths.len())
            .map(|s| {
                shape(&format!("encoder.stages.{s}.layers.0.layer.{last}.convolution.weight"))
                    .map(|w| w[0])
                    .ok_or_else(|| anyhow!("stage {s} has no layer.{last} conv"))
            })
            .collect::<Result<Vec<_>>>()?;
        let embedding_size = emb[0];
        // A basic first stage only grows a shortcut when it downsamples.
        let downsample_in_first_stage = layer_type == ResNetLayerType::Basic
            && embedding_size == hidden_sizes[0]
            && shape("encoder.stages.0.layers.0.shortcut.convolution.weight").is_some();
        Ok(Self {
            embedding_size,
            hidden_sizes,
            depths: depths.values().map(|&d| d as i64).collect(),
            layer_type,
            downsample_in_first_stage,
            downsample_in_bottleneck: false,
        })
    }

    /// `["stem", "stage1", ...]` length.
    pub fn num_stage_names(&self) -> usize {
        self.depths.len() + 1
    }

    /// Output channels per stage name (`ResNetBackbone.num_features`).
    pub fn num_features(&self) -> Vec<i64> {
        std::iter::once(self.embedding_size).chain(self.hidden_sizes.iter().copied()).collect()
    }
}

/// `ResNetConvLayer` (`convolution` + `normalization` [+ ReLU]); also serves
/// as `ResNetShortCut` (1x1, no activation).
#[derive(Debug)]
struct ResNetConvLayer {
    conv: nn::Conv2D,
    norm: nn::BatchNorm,
    relu: bool,
}

impl ResNetConvLayer {
    fn new(p: &nn::Path, i: i64, o: i64, k: i64, stride: i64, relu: bool) -> Self {
        Self {
            conv: conv(&p.sub("convolution"), i, o, k, stride, false),
            norm: bn(&p.sub("normalization"), o),
            relu,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let y = self.norm.forward_t(&self.conv.forward(x), false);
        if self.relu { y.relu() } else { y }
    }
}

/// `ResNetBasicLayer` / `ResNetBottleNeckLayer`.
#[derive(Debug)]
struct ResNetResidualLayer {
    shortcut: Option<ResNetConvLayer>,
    layer: Vec<ResNetConvLayer>,
}

impl ResNetResidualLayer {
    fn new(p: &nn::Path, spec: &ResNetSpec, i: i64, o: i64, stride: i64) -> Self {
        let shortcut = (i != o || stride != 1)
            .then(|| ResNetConvLayer::new(&p.sub("shortcut"), i, o, 1, stride, false));
        let l = p.sub("layer");
        let layer = match spec.layer_type {
            ResNetLayerType::Basic => vec![
                ResNetConvLayer::new(&l.sub("0"), i, o, 3, stride, true),
                ResNetConvLayer::new(&l.sub("1"), o, o, 3, 1, false),
            ],
            ResNetLayerType::Bottleneck => {
                let r = o / 4;
                let (s1, s2) = if spec.downsample_in_bottleneck { (stride, 1) } else { (1, stride) };
                vec![
                    ResNetConvLayer::new(&l.sub("0"), i, r, 1, s1, true),
                    ResNetConvLayer::new(&l.sub("1"), r, r, 3, s2, true),
                    ResNetConvLayer::new(&l.sub("2"), r, o, 1, 1, false),
                ]
            }
        };
        Self { shortcut, layer }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let residual = match &self.shortcut {
            Some(s) => s.forward(x),
            None => x.shallow_clone(),
        };
        let mut h = x.shallow_clone();
        for l in &self.layer {
            h = l.forward(&h);
        }
        (h + residual).relu()
    }
}

/// transformers `ResNetBackbone` under `skylens.backbone`.
#[derive(Debug)]
struct ResNetBackbone {
    embedder: ResNetConvLayer,
    stages: Vec<Vec<ResNetResidualLayer>>,
    out_indices: Vec<usize>,
    channels: Vec<i64>,
}

impl ResNetBackbone {
    fn new(p: &nn::Path, spec: &ResNetSpec, in_channels: i64, out_indices: Vec<usize>) -> Self {
        let embedder = ResNetConvLayer::new(
            &p.sub("embedder").sub("embedder"),
            in_channels,
            spec.embedding_size,
            7,
            2,
            true,
        );
        // hidden_states[k] is the output of stage k - 1 (k = 0 is the stem), so
        // only stages 0..max_index are ever read.
        let needed = out_indices.iter().copied().max().unwrap_or(0);
        let sp = p.sub("encoder").sub("stages");
        let mut stages = Vec::with_capacity(needed);
        let mut in_ch = spec.embedding_size;
        for s in 0..needed {
            let out_ch = spec.hidden_sizes[s];
            let stride = if s == 0 { if spec.downsample_in_first_stage { 2 } else { 1 } } else { 2 };
            let lp = sp.sub(s).sub("layers");
            let layers = (0..spec.depths[s])
                .map(|j| {
                    let (i, st) = if j == 0 { (in_ch, stride) } else { (out_ch, 1) };
                    ResNetResidualLayer::new(&lp.sub(j), spec, i, out_ch, st)
                })
                .collect();
            stages.push(layers);
            in_ch = out_ch;
        }
        let features = spec.num_features();
        let channels = out_indices.iter().map(|&i| features[i]).collect();
        Self { embedder, stages, out_indices, channels }
    }

    fn forward(&self, x: &Tensor) -> Vec<Tensor> {
        let stem = self.embedder.forward(x).max_pool2d([3, 3], [2, 2], [1, 1], [1, 1], false);
        let mut hidden = vec![stem];
        for stage in &self.stages {
            let mut h = hidden[hidden.len() - 1].shallow_clone();
            for layer in stage {
                h = layer.forward(&h);
            }
            hidden.push(h);
        }
        self.out_indices.iter().map(|&i| hidden[i].shallow_clone()).collect()
    }
}

// ---------------------------------------------------------------------------
// SkyLensModel / SkyLensForDisasterPerception
// ---------------------------------------------------------------------------

/// Shape choices that `config.json` does not record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchSpec {
    pub backbone: ResNetSpec,
    /// `wh_stem` exists (current layout). `false` reproduces checkpoints from
    /// before the wh branch got its own stem: `wh_head` reads `person_stem`.
    pub wh_stem: bool,
}

impl ArchSpec {
    /// Infers the architecture from a SkyLens checkpoint header.
    pub fn from_header(header: &BTreeMap<String, TensorMeta>) -> Result<Self> {
        Ok(Self {
            backbone: ResNetSpec::from_header(header, BACKBONE_PREFIX)?,
            wh_stem: header.contains_key("wh_stem.block.0.weight"),
        })
    }
}

/// Encoder + UNet decoder (`SkyLensModel`).
#[derive(Debug)]
pub struct SkyLensModel {
    backbone: ResNetBackbone,
    decoder: SkyLensUNetDecoder,
}

impl SkyLensModel {
    pub fn new(p: &nn::Path, config: &SkyLensConfig, spec: &ResNetSpec) -> Result<Self> {
        let out_indices = config.resolved_out_indices(spec.num_stage_names())?;
        let backbone =
            ResNetBackbone::new(&p.sub("backbone"), spec, config.num_input_channels(), out_indices);
        let decoder =
            SkyLensUNetDecoder::new(&p.sub("decoder"), &backbone.channels, &config.decoder_channels);
        Ok(Self { backbone, decoder })
    }

    pub fn out_channels(&self) -> i64 {
        self.decoder.out_channels
    }

    /// Encoder channels feeding the decoder, shallow to deep.
    pub fn encoder_channels(&self) -> &[i64] {
        &self.backbone.channels
    }

    pub fn forward(&self, pixel_values: &Tensor) -> Tensor {
        self.decoder.forward(&self.backbone.forward(pixel_values))
    }
}

/// Model outputs (`SkyLensOutput` without the training losses).
#[derive(Debug)]
pub struct SkyLensOutput {
    /// `(B, num_danger_classes, H, W)` hazard logits at input resolution.
    pub danger_logits: Tensor,
    /// `(B, 1, H/s, W/s)` person heatmap, sigmoid already applied.
    pub person_heatmap: Tensor,
    /// `(B, 2, H/s, W/s)` box size in grid units.
    pub person_wh: Tensor,
    /// `(B, 2, H/s, W/s)` sub-cell centre offset, when the head exists.
    pub person_offset: Option<Tensor>,
}

/// `SkyLensForDisasterPerception`: shared encoder, hazard and person heads.
#[derive(Debug)]
pub struct SkyLensForDisasterPerception {
    config: SkyLensConfig,
    skylens: SkyLensModel,
    danger_head: nn::Conv2D,
    person_stem: SkyLensConvBlock,
    wh_stem: Option<SkyLensConvBlock>,
    heatmap_head: nn::Conv2D,
    wh_head: nn::Conv2D,
    offset_head: Option<nn::Conv2D>,
}

impl SkyLensForDisasterPerception {
    /// Creates every parameter under its PyTorch `state_dict` name below `p`
    /// (pass `vs.root()`).
    pub fn new(p: &nn::Path, config: &SkyLensConfig, arch: &ArchSpec) -> Result<Self> {
        config.validate()?;
        ensure!(!config.use_timm_backbone, "timm backbones are not supported by the Rust model");
        let skylens = SkyLensModel::new(&p.sub("skylens"), config, &arch.backbone)?;
        let f = skylens.out_channels();
        Ok(Self {
            config: config.clone(),
            danger_head: conv(&p.sub("danger_head"), f, config.num_danger_classes, 1, 1, true),
            person_stem: SkyLensConvBlock::new(&p.sub("person_stem"), f, f),
            wh_stem: arch.wh_stem.then(|| SkyLensConvBlock::new(&p.sub("wh_stem"), f, f)),
            heatmap_head: conv(&p.sub("heatmap_head"), f, 1, 1, 1, true),
            wh_head: conv(&p.sub("wh_head"), f, 2, 1, 1, true),
            offset_head: config
                .use_offset_head
                .then(|| conv(&p.sub("offset_head"), f, 2, 1, 1, true)),
            skylens,
        })
    }

    pub fn config(&self) -> &SkyLensConfig {
        &self.config
    }

    pub fn encoder_channels(&self) -> &[i64] {
        self.skylens.encoder_channels()
    }

    /// Zeroes absent modalities and writes the validity plane
    /// (`_apply_modality_mask`). `modality_mask` is `(B, 2)` bool
    /// `[rgb_present, thermal_present]`.
    pub fn apply_modality_mask(&self, pixel_values: &Tensor, modality_mask: &Tensor) -> Tensor {
        let size = pixel_values.size();
        let (c, h, w) = (size[1], size[2], size[3]);
        let kind = pixel_values.kind();
        let rgb = modality_mask.select(1, 0).to_kind(kind).view([-1, 1, 1, 1]);
        let thermal = modality_mask.select(1, 1).to_kind(kind).view([-1, 1, 1, 1]);
        let n_in = self.config.in_channels;
        let mut parts = vec![
            pixel_values.narrow(1, 0, 3) * &rgb,
            pixel_values.narrow(1, 3, n_in - 3) * &thermal,
        ];
        if c > n_in {
            if self.config.use_validity_channel {
                parts.push(thermal.expand([-1, c - n_in, h, w], false));
            } else {
                parts.push(pixel_values.narrow(1, n_in, c - n_in));
            }
        }
        Tensor::cat(&parts, 1)
    }

    /// Inference forward. Call inside `tch::no_grad`. Modality dropout is never
    /// sampled here (Python samples it only in training mode); an explicit
    /// `modality_mask` is always applied, as in Python.
    pub fn forward(&self, pixel_values: &Tensor, modality_mask: Option<&Tensor>) -> SkyLensOutput {
        let size = pixel_values.size();
        let (height, width) = (size[2], size[3]);
        let pixel_values = match modality_mask {
            Some(m) => self.apply_modality_mask(pixel_values, m),
            None => pixel_values.shallow_clone(),
        };

        let features = self.skylens.forward(&pixel_values);

        let mut danger_logits = self.danger_head.forward(&features);
        let ds = danger_logits.size();
        if (ds[2], ds[3]) != (height, width) {
            danger_logits =
                danger_logits.upsample_bilinear2d([height, width], false, None::<f64>, None::<f64>);
        }

        let stride = self.config.person_head_stride;
        let target = [(height / stride).max(1), (width / stride).max(1)];
        let fs = features.size();
        let person_feat = if [fs[2], fs[3]] != target {
            features.upsample_bilinear2d(target, false, None::<f64>, None::<f64>)
        } else {
            features
        };
        let person_feat_hm = self.person_stem.forward(&person_feat);
        let person_heatmap = self.heatmap_head.forward(&person_feat_hm).sigmoid();
        let person_wh = match &self.wh_stem {
            Some(stem) => self.wh_head.forward(&stem.forward(&person_feat)),
            None => self.wh_head.forward(&person_feat_hm),
        };
        let person_offset = self.offset_head.as_ref().map(|h| h.forward(&person_feat_hm));

        SkyLensOutput { danger_logits, person_heatmap, person_wh, person_offset }
    }
}

// ---------------------------------------------------------------------------
// Detector
// ---------------------------------------------------------------------------

/// How the per-pixel hazard map is produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HazardMode {
    /// One extra forward on the frame squashed to `size x size` with area
    /// interpolation, the resolution the segmentation head was trained and
    /// evaluated at (`cache.resize_sample`).
    Squash { size: u32 },
    /// Average the softmax of the native-scale person tiles.
    Tiles,
}

/// Inference settings. Defaults are the deployment protocol of
/// `scripts/eval_deploy.py` (`SIZE, K, SHORT, OVERLAP = 512, 300, 765, 64`,
/// `--ap-floor 0.05`, tile merge IoU 0.5 / centre distance 2 px).
#[derive(Debug, Clone, PartialEq)]
pub struct DetectorOptions {
    /// Short side of the native-scale frame.
    pub short_side: u32,
    /// Tile size.
    pub tile: u32,
    /// Minimum overlap between tiles.
    pub min_overlap: u32,
    /// Tiles per forward pass (`--bs`); does not change results.
    pub tile_batch: usize,
    /// Peaks below this score are dropped. 0.05 is the decode floor of
    /// `eval_deploy.py`; 0.3 is the operating point its F1 is reported at.
    pub score_threshold: f32,
    /// Maximum detections per tile.
    pub top_k: usize,
    /// Cross-tile duplicate rule.
    pub merge_iou: f64,
    pub merge_center_distance: f64,
    pub hazard: HazardMode,
    /// Arithmetic on CUDA (CPU is always fp32).
    pub precision: Precision,
}

/// Arithmetic of a CUDA forward pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// fp32 weights and activations.
    Fp32,
    /// fp32 weights under fp16 autocast, as `eval_deploy.py` runs.
    Fp16Autocast,
    /// bf16 weights, inputs and activations.
    Bf16,
}

impl Default for DetectorOptions {
    fn default() -> Self {
        Self {
            short_side: tiling::SHORT_SIDE,
            tile: tiling::TILE_SIZE,
            min_overlap: tiling::MIN_OVERLAP,
            tile_batch: 4,
            score_threshold: 0.05,
            top_k: 300,
            merge_iou: 0.5,
            merge_center_distance: 2.0,
            hazard: HazardMode::Squash { size: tiling::TILE_SIZE },
            precision: Precision::Bf16,
        }
    }
}

/// The loaded network and the store that owns its weights.
struct Loaded {
    _vs: nn::VarStore,
    net: SkyLensForDisasterPerception,
}

/// libtorch SkyLensNet behind the [`Detector`] contract.
///
/// tch tensors are `Send` but not `Sync`, so the network sits behind a
/// `Mutex`; concurrent `detect` calls run one forward at a time.
pub struct SkyLensNetDetector {
    config: SkyLensConfig,
    options: DetectorOptions,
    device: Device,
    model: Mutex<Loaded>,
}

#[allow(dead_code)]
fn assert_detector_is_send_sync() {
    fn check<T: Send + Sync>() {}
    check::<SkyLensNetDetector>();
}

impl std::fmt::Debug for SkyLensNetDetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SkyLensNetDetector")
            .field("config", &self.config)
            .field("options", &self.options)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

impl SkyLensNetDetector {
    /// Loads a `save_pretrained` directory (`config.json` + `model.safetensors`)
    /// with the default deployment options.
    pub fn load(checkpoint: &Path, device: Device) -> Result<Self> {
        Self::load_with_options(checkpoint, device, DetectorOptions::default())
    }

    pub fn load_with_options(checkpoint: &Path, device: Device, options: DetectorOptions) -> Result<Self> {
        let config = SkyLensConfig::from_pretrained(checkpoint)?;
        ensure!(!config.use_timm_backbone, "timm backbones are not supported by the Rust model");
        ensure!(
            config.in_channels == 4,
            "the inference channel layout is RGB + thermal; in_channels = {} is not supported",
            config.in_channels
        );
        ensure!(
            config.num_danger_classes as usize == HAZARD_CLASSES,
            "checkpoint has {} hazard classes, the frame hook expects {HAZARD_CLASSES}",
            config.num_danger_classes
        );
        ensure!(options.tile > 0 && options.tile_batch > 0, "tile and tile_batch must be positive");

        let weights = checkpoint.join(WEIGHTS_NAME);
        let header = read_safetensors_header(&weights)?;
        let arch = ArchSpec::from_header(&header)?;
        if let Some(named) = ResNetSpec::for_name(&config.backbone)
            && named != arch.backbone
        {
            tracing::warn!(
                backbone = %config.backbone,
                ?named,
                inferred = ?arch.backbone,
                "checkpoint backbone layout differs from the published config; using the checkpoint"
            );
        }

        let mut vs = nn::VarStore::new(device);
        let net = SkyLensForDisasterPerception::new(&vs.root(), &config, &arch)?;
        check_against_header(&vs, &header)?;
        vs.load(&weights).with_context(|| format!("loading {}", weights.display()))?;
        // Loading adopts the file dtype; cast to the run precision.
        if device.is_cuda() && options.precision == Precision::Bf16 {
            vs.bfloat16();
        } else {
            vs.float();
        }
        vs.freeze();

        Ok(Self { config, options, device, model: Mutex::new(Loaded { _vs: vs, net }) })
    }

    pub fn config(&self) -> &SkyLensConfig {
        &self.config
    }

    pub fn options(&self) -> &DetectorOptions {
        &self.options
    }

    fn run(&self, pixel_values: &Tensor, modality_mask: &Tensor) -> Result<SkyLensOutput> {
        let loaded = self.model.lock().map_err(|_| anyhow!("SkyLensNet model lock poisoned"))?;
        let cuda = self.device.is_cuda();
        let autocast = cuda && self.options.precision == Precision::Fp16Autocast;
        let bf16 = cuda && self.options.precision == Precision::Bf16;
        Ok(tch::no_grad(|| {
            tch::autocast(autocast, || {
                let x = if bf16 { pixel_values.to_kind(Kind::BFloat16) } else { pixel_values.shallow_clone() };
                let out = loaded.net.forward(&x, Some(modality_mask));
                SkyLensOutput {
                    danger_logits: out.danger_logits.to_kind(Kind::Float),
                    person_heatmap: out.person_heatmap.to_kind(Kind::Float),
                    person_wh: out.person_wh.to_kind(Kind::Float),
                    person_offset: out.person_offset.map(|o| o.to_kind(Kind::Float)),
                }
            })
        }))
    }

    /// Builds `(B, C, T, T)` pixel values and the `(B, 2)` modality mask from
    /// HWC tiles holding RGB [+ thermal].
    fn batch_tensors(&self, tiles: &[Image], has_thermal: bool) -> (Tensor, Tensor) {
        let c = self.config.num_input_channels() as usize;
        let (w, h) = (tiles[0].width, tiles[0].height);
        let plane = w * h;
        let mut buf = vec![0.0f32; tiles.len() * c * plane];
        for (b, t) in tiles.iter().enumerate() {
            let base = b * c * plane;
            for ch in 0..t.channels.min(4) {
                let dst = &mut buf[base + ch * plane..base + (ch + 1) * plane];
                for (i, d) in dst.iter_mut().enumerate() {
                    *d = t.data[i * t.channels + ch];
                }
            }
            if c > 4 {
                // validity plane(s): 1.0 where thermal is real (collator layout)
                let v = if has_thermal { 1.0 } else { 0.0 };
                buf[base + 4 * plane..base + c * plane].iter_mut().for_each(|d| *d = v);
            }
        }
        let px = Tensor::from_slice(buf.as_slice())
            .view([tiles.len() as i64, c as i64, h as i64, w as i64])
            .to_device(self.device);
        let mask: Vec<bool> = (0..tiles.len()).flat_map(|_| [true, has_thermal]).collect();
        let mask = Tensor::from_slice(mask.as_slice()).view([tiles.len() as i64, 2]).to_device(self.device);
        (px, mask)
    }
}

/// Every variable the model created must exist in the file with the same shape.
fn check_against_header(vs: &nn::VarStore, header: &BTreeMap<String, TensorMeta>) -> Result<()> {
    let mut missing = Vec::new();
    let mut mismatched = Vec::new();
    for (name, t) in vs.variables() {
        match header.get(&name) {
            None => missing.push(name),
            Some(meta) if meta.shape != t.size() => {
                mismatched.push(format!("{name}: file {:?} vs model {:?}", meta.shape, t.size()))
            }
            Some(_) => {}
        }
    }
    missing.sort();
    mismatched.sort();
    if !missing.is_empty() || !mismatched.is_empty() {
        bail!(
            "checkpoint does not match the SkyLensNet layout: missing {:?}, shape mismatches {:?}",
            missing,
            mismatched
        );
    }
    Ok(())
}

/// Copies a tensor to a flat CPU `f32` vector.
fn to_vec_f32(t: &Tensor) -> Result<Vec<f32>> {
    let flat = t.to_device(Device::Cpu).to_kind(Kind::Float).contiguous().reshape([-1]);
    Ok(Vec::<f32>::try_from(&flat)?)
}

/// RGB8 [+ thermal8] to an HWC `f32` image: RGB / 255 (collator) and thermal
/// min-max normalised into `[0.1, 1.0]` (`_stack_rgb_thermal`).
pub fn frame_image(rgb: &[u8], thermal: Option<&[u8]>, width: usize, height: usize) -> Image {
    let n = width * height;
    let c = if thermal.is_some() { 4 } else { 3 };
    let mut data = vec![0.0f32; n * c];
    let (lo, hi) = thermal.map_or((0.0, 0.0), |t| {
        let lo = t.iter().copied().min().unwrap_or(0);
        let hi = t.iter().copied().max().unwrap_or(0);
        (f32::from(lo), f32::from(hi))
    });
    for i in 0..n {
        for ch in 0..3 {
            data[i * c + ch] = f32::from(rgb[i * 3 + ch]) / 255.0;
        }
        if let Some(t) = thermal {
            let norm = if hi <= lo { 0.0 } else { (f32::from(t[i]) - lo) / (hi - lo) };
            data[i * c + 3] = 0.1 + 0.9 * norm;
        }
    }
    Image::new(width, height, c, data)
}

impl Detector for SkyLensNetDetector {
    fn detect(
        &self,
        rgb: &[u8],
        thermal: Option<&[u8]>,
        width: u32,
        height: u32,
    ) -> Result<FrameDetections> {
        let (w, h) = (width as usize, height as usize);
        ensure!(w > 0 && h > 0, "empty frame");
        ensure!(rgb.len() == w * h * 3, "rgb must be width * height * 3 bytes");
        if let Some(t) = thermal {
            ensure!(t.len() == w * h, "thermal must be width * height bytes");
        }
        let has_thermal = thermal.is_some();
        let opts = &self.options;
        let image = frame_image(rgb, thermal, w, h);

        // --- people: native-scale tiles (eval_deploy.tile_one) -------------
        let (scaled, _, _) = tiling::scale_to_short_side(&image, None, opts.short_side);
        let (sw, sh) = (scaled.width, scaled.height);
        let t = opts.tile as usize;
        let grid = tiling::tile_grid(sw as u32, sh as u32, opts.tile, opts.min_overlap);
        let stride = self.config.person_head_stride as f32;
        let classes = self.config.num_danger_classes;
        let tiled_hazard = opts.hazard == HazardMode::Tiles;
        let (acc, cnt) = if tiled_hazard {
            let opt = (Kind::Float, self.device);
            (
                Some(Tensor::zeros([classes, sh as i64, sw as i64], opt)),
                Some(Tensor::zeros([1, sh as i64, sw as i64], opt)),
            )
        } else {
            (None, None)
        };

        let mut per_tile: Vec<Vec<Detection>> = Vec::with_capacity(grid.len());
        for chunk in grid.chunks(opts.tile_batch) {
            let mut images = Vec::with_capacity(chunk.len());
            let mut valid = Vec::with_capacity(chunk.len());
            for &(x, y) in chunk {
                let (img, vw, vh) = tiling::crop_tile(&scaled, x as usize, y as usize, t, t);
                images.push(img);
                valid.push((vw as i64, vh as i64));
            }
            let (px, mask) = self.batch_tensors(&images, has_thermal);
            let out = self.run(&px, &mask)?;

            let hs = out.person_heatmap.size();
            let (fh, fw) = (hs[2] as usize, hs[3] as usize);
            let hm = to_vec_f32(&out.person_heatmap)?;
            let wh = to_vec_f32(&out.person_wh)?;
            let off = out.person_offset.as_ref().map(to_vec_f32).transpose()?;
            let n = fh * fw;
            for (b, &(x0, y0)) in chunk.iter().enumerate() {
                let det = decode_heatmap_peaks(
                    &hm[b * n..(b + 1) * n],
                    Some(&wh[b * 2 * n..(b + 1) * 2 * n]),
                    off.as_ref().map(|o| &o[b * 2 * n..(b + 1) * 2 * n]),
                    fh,
                    fw,
                    opts.top_k,
                    opts.score_threshold,
                    stride,
                    false,
                );
                per_tile.push(tiling::tile_detections_to_full(&det, f64::from(x0), f64::from(y0)));
            }

            if let (Some(acc), Some(cnt)) = (acc.as_ref(), cnt.as_ref()) {
                for (b, (&(x0, y0), &(vw, vh))) in chunk.iter().zip(&valid).enumerate() {
                    let p = out
                        .danger_logits
                        .get(b as i64)
                        .softmax(0, Kind::Float)
                        .narrow(1, 0, vh)
                        .narrow(2, 0, vw);
                    let mut a = acc.narrow(1, y0 as i64, vh).narrow(2, x0 as i64, vw);
                    a += &p;
                    let mut c = cnt.narrow(1, y0 as i64, vh).narrow(2, x0 as i64, vw);
                    c += 1.0;
                }
            }
        }

        let merged = tiling::merge_tile_detections(&per_tile, opts.merge_iou, opts.merge_center_distance);
        // Scaled frame -> original frame, per axis (boxes were scaled by new / old).
        let (fx, fy) = (sw as f64 / w as f64, sh as f64 / h as f64);
        let people = merged
            .iter()
            .map(|d| PersonPoint { x: (d[0] / fx) as f32, y: (d[1] / fy) as f32, score: d[4] as f32 })
            .collect();

        // --- hazard map ----------------------------------------------------
        let prob = match (opts.hazard, acc, cnt) {
            (HazardMode::Tiles, Some(acc), Some(cnt)) => {
                let p = (acc / cnt.clamp_min(1.0)).unsqueeze(0);
                if (sw, sh) == (w, h) {
                    p
                } else {
                    tch::no_grad(|| {
                        p.upsample_bilinear2d([h as i64, w as i64], false, None::<f64>, None::<f64>)
                    })
                }
            }
            (HazardMode::Squash { size }, _, _) => {
                let s = size as usize;
                let squashed = tiling::resize(&image, s, s, Interpolation::Area);
                let (px, mask) = self.batch_tensors(std::slice::from_ref(&squashed), has_thermal);
                let logits = self.run(&px, &mask)?.danger_logits;
                tch::no_grad(|| {
                    let logits = if (s, s) == (w, h) {
                        logits
                    } else {
                        logits.upsample_bilinear2d([h as i64, w as i64], false, None::<f64>, None::<f64>)
                    };
                    logits.softmax(1, Kind::Float)
                })
            }
            _ => bail!("hazard accumulation missing"),
        };
        let hazard_prob = to_vec_f32(&prob)?;
        ensure!(
            hazard_prob.len() == HAZARD_CLASSES * w * h,
            "hazard map has {} values, expected {}",
            hazard_prob.len(),
            HAZARD_CLASSES * w * h
        );

        Ok(FrameDetections { width, height, hazard_prob, people })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ResNet small enough to run on CPU in tests.
    fn tiny_spec() -> ResNetSpec {
        ResNetSpec {
            embedding_size: 8,
            hidden_sizes: vec![8, 16, 32, 64],
            depths: vec![1, 2, 1, 1],
            layer_type: ResNetLayerType::Basic,
            downsample_in_first_stage: false,
            downsample_in_bottleneck: false,
        }
    }

    fn tiny_config(offset: bool) -> SkyLensConfig {
        SkyLensConfig {
            backbone: "tiny".into(),
            use_pretrained_backbone: false,
            decoder_channels: vec![16, 8, 8, 4],
            use_offset_head: offset,
            ..Default::default()
        }
    }

    fn names(vs: &nn::VarStore) -> std::collections::BTreeSet<String> {
        vs.variables().into_keys().collect()
    }

    #[test]
    fn parameter_names_follow_the_state_dict() {
        let vs = nn::VarStore::new(Device::Cpu);
        let arch = ArchSpec { backbone: tiny_spec(), wh_stem: true };
        SkyLensForDisasterPerception::new(&vs.root(), &tiny_config(true), &arch).unwrap();
        let n = names(&vs);
        for key in [
            "skylens.backbone.embedder.embedder.convolution.weight",
            "skylens.backbone.embedder.embedder.normalization.running_var",
            "skylens.backbone.encoder.stages.0.layers.0.layer.0.convolution.weight",
            "skylens.backbone.encoder.stages.1.layers.0.shortcut.convolution.weight",
            "skylens.backbone.encoder.stages.1.layers.1.layer.1.normalization.bias",
            "skylens.backbone.encoder.stages.2.layers.0.shortcut.normalization.running_mean",
            "skylens.decoder.blocks.0.conv.block.0.weight",
            "skylens.decoder.blocks.3.conv.block.4.weight",
            "danger_head.weight",
            "danger_head.bias",
            "person_stem.block.1.running_mean",
            "person_stem.block.3.weight",
            "wh_stem.block.4.running_var",
            "heatmap_head.bias",
            "wh_head.weight",
            "offset_head.weight",
        ] {
            assert!(n.contains(key), "missing {key}");
        }
        // the first stage keeps 8 channels at stride 1, so it has no shortcut
        assert!(!n.contains("skylens.backbone.encoder.stages.0.layers.0.shortcut.convolution.weight"));
        // out_indices (0,1,2,3) never read the 4th stage, so it is not built
        assert!(!n.iter().any(|k| k.contains("encoder.stages.3.")));
        assert!(!n.iter().any(|k| k.contains("num_batches_tracked")));
    }

    #[test]
    fn resnet50_layout_matches_a_real_checkpoint() {
        // Shapes taken from runs/skylens/final/model.safetensors.
        let vs = nn::VarStore::new(Device::Cpu);
        let arch = ArchSpec { backbone: ResNetSpec::for_name("microsoft/resnet-50").unwrap(), wh_stem: true };
        let net = SkyLensForDisasterPerception::new(&vs.root(), &SkyLensConfig::default(), &arch).unwrap();
        assert_eq!(net.encoder_channels(), &[64, 256, 512, 1024]);
        let vars = vs.variables();
        let shape = |k: &str| vars[k].size();
        assert_eq!(shape("skylens.backbone.embedder.embedder.convolution.weight"), vec![64, 4, 7, 7]);
        assert_eq!(shape("skylens.backbone.encoder.stages.0.layers.0.shortcut.convolution.weight"), vec![256, 64, 1, 1]);
        assert_eq!(shape("skylens.backbone.encoder.stages.0.layers.1.layer.0.convolution.weight"), vec![64, 256, 1, 1]);
        assert_eq!(shape("skylens.decoder.blocks.0.conv.block.0.weight"), vec![256, 1536, 3, 3]);
        assert_eq!(shape("skylens.decoder.blocks.1.conv.block.0.weight"), vec![128, 512, 3, 3]);
        assert_eq!(shape("skylens.decoder.blocks.2.conv.block.0.weight"), vec![64, 192, 3, 3]);
        assert_eq!(shape("skylens.decoder.blocks.3.conv.block.0.weight"), vec![32, 64, 3, 3]);
        assert_eq!(shape("danger_head.weight"), vec![4, 32, 1, 1]);
        assert_eq!(shape("person_stem.block.0.weight"), vec![32, 32, 3, 3]);
    }

    #[test]
    fn forward_shapes() {
        let vs = nn::VarStore::new(Device::Cpu);
        let arch = ArchSpec { backbone: tiny_spec(), wh_stem: true };
        let net = SkyLensForDisasterPerception::new(&vs.root(), &tiny_config(true), &arch).unwrap();
        let x = Tensor::rand([2, 4, 64, 96], (Kind::Float, Device::Cpu));
        let mask = Tensor::from_slice(&[true, false, true, true]).view([2, 2]);
        let out = tch::no_grad(|| net.forward(&x, Some(&mask)));
        assert_eq!(out.danger_logits.size(), vec![2, 4, 64, 96]);
        assert_eq!(out.person_heatmap.size(), vec![2, 1, 16, 24]);
        assert_eq!(out.person_wh.size(), vec![2, 2, 16, 24]);
        assert_eq!(out.person_offset.unwrap().size(), vec![2, 2, 16, 24]);
        let hm = to_vec_f32(&out.person_heatmap).unwrap();
        assert!(hm.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn modality_mask_zeroes_missing_thermal_and_sets_validity() {
        let vs = nn::VarStore::new(Device::Cpu);
        let cfg = SkyLensConfig { use_validity_channel: true, ..tiny_config(false) };
        let arch = ArchSpec { backbone: tiny_spec(), wh_stem: true };
        let net = SkyLensForDisasterPerception::new(&vs.root(), &cfg, &arch).unwrap();
        let x = Tensor::ones([1, 5, 2, 2], (Kind::Float, Device::Cpu));
        let mask = Tensor::from_slice(&[true, false]).view([1, 2]);
        let y = to_vec_f32(&net.apply_modality_mask(&x, &mask)).unwrap();
        assert_eq!(&y[..12], &[1.0; 12]); // RGB kept
        assert_eq!(&y[12..], &[0.0; 8]); // thermal and validity zero
    }

    #[test]
    fn arch_is_inferred_from_the_header() {
        let meta = |s: &[i64]| TensorMeta { dtype: "F32".into(), shape: s.to_vec() };
        let mut h = BTreeMap::new();
        h.insert("skylens.backbone.embedder.embedder.convolution.weight".to_string(), meta(&[64, 4, 7, 7]));
        for (s, (out, depth)) in [(256, 3), (512, 4), (1024, 6), (2048, 3)].into_iter().enumerate() {
            for l in 0..depth {
                for (k, o) in [(0, out / 4), (1, out / 4), (2, out)] {
                    h.insert(
                        format!("skylens.backbone.encoder.stages.{s}.layers.{l}.layer.{k}.convolution.weight"),
                        meta(&[o, 1, 1, 1]),
                    );
                }
            }
        }
        let arch = ArchSpec::from_header(&h).unwrap();
        assert_eq!(arch.backbone, ResNetSpec::for_name("microsoft/resnet-50").unwrap());
        assert!(!arch.wh_stem); // legacy layout: no wh_stem keys
        h.insert("wh_stem.block.0.weight".into(), meta(&[32, 32, 3, 3]));
        assert!(ArchSpec::from_header(&h).unwrap().wh_stem);
    }

    #[test]
    fn frame_image_scales_rgb_and_reserves_zero_for_absent_thermal() {
        let rgb = [0u8, 255, 51, 255, 0, 102];
        let img = frame_image(&rgb, Some(&[10, 30]), 2, 1);
        assert_eq!(img.channels, 4);
        assert_eq!(img.at(0, 0, 1), 1.0);
        assert!((img.at(0, 0, 2) - 0.2).abs() < 1e-7);
        assert!((img.at(0, 0, 3) - 0.1).abs() < 1e-7 && (img.at(1, 0, 3) - 1.0).abs() < 1e-7);
        let flat = frame_image(&rgb, Some(&[7, 7]), 2, 1);
        assert!((flat.at(1, 0, 3) - 0.1).abs() < 1e-7);
        assert_eq!(frame_image(&rgb, None, 2, 1).channels, 3);
    }

    /// Saves a randomly initialised tiny model as a `save_pretrained`
    /// directory and runs it through the detector end to end.
    fn tiny_checkpoint(tag: &str, offset: bool, wh_stem: bool) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("skylensnet-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let vs = nn::VarStore::new(Device::Cpu);
        let cfg = tiny_config(offset);
        SkyLensForDisasterPerception::new(&vs.root(), &cfg, &ArchSpec { backbone: tiny_spec(), wh_stem })
            .unwrap();
        vs.save(dir.join(WEIGHTS_NAME)).unwrap();
        std::fs::write(dir.join(CONFIG_NAME), cfg.to_json_string().unwrap()).unwrap();
        dir
    }

    fn frame(w: usize, h: usize) -> (Vec<u8>, Vec<u8>) {
        let rgb = (0..w * h * 3).map(|i| (i * 37 % 251) as u8).collect();
        let th = (0..w * h).map(|i| (i * 13 % 199) as u8).collect();
        (rgb, th)
    }

    #[test]
    fn detector_end_to_end_on_cpu() {
        let dir = tiny_checkpoint("e2e", true, true);
        let det = SkyLensNetDetector::load(&dir, Device::Cpu).unwrap();
        let (w, h) = (100usize, 60usize);
        let (rgb, th) = frame(w, h);
        for thermal in [None, Some(th.as_slice())] {
            let out = det.detect(&rgb, thermal, w as u32, h as u32).unwrap();
            assert_eq!((out.width, out.height), (w as u32, h as u32));
            assert_eq!(out.hazard_prob.len(), HAZARD_CLASSES * w * h);
            for i in [0, w * h / 2, w * h - 1] {
                let s: f32 = (0..HAZARD_CLASSES).map(|c| out.hazard_prob[c * w * h + i]).sum();
                assert!((s - 1.0).abs() < 1e-4, "probabilities sum to {s}");
            }
            for p in &out.people {
                assert!(p.score >= 0.05 && p.score <= 1.0);
                assert!(p.x > -10.0 && p.x < w as f32 + 10.0 && p.y > -10.0 && p.y < h as f32 + 10.0);
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn detector_tiled_hazard_and_legacy_wh_layout() {
        let dir = tiny_checkpoint("legacy", false, false);
        let opts = DetectorOptions { hazard: HazardMode::Tiles, ..Default::default() };
        let det = SkyLensNetDetector::load_with_options(&dir, Device::Cpu, opts).unwrap();
        let (w, h) = (80usize, 50usize);
        let (rgb, _) = frame(w, h);
        let out = det.detect(&rgb, None, w as u32, h as u32).unwrap();
        let s: f32 = (0..HAZARD_CLASSES).map(|c| out.hazard_prob[c * w * h]).sum();
        assert!((s - 1.0).abs() < 1e-4);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn detector_rejects_bad_buffers_and_missing_weights() {
        let dir = tiny_checkpoint("bad", false, true);
        let det = SkyLensNetDetector::load(&dir, Device::Cpu).unwrap();
        assert!(det.detect(&[0u8; 5], None, 2, 2).is_err());
        assert!(det.detect(&[0u8; 12], Some(&[0u8; 3]), 2, 2).is_err());
        // A checkpoint whose config asks for an offset head the weights lack.
        let cfg = tiny_config(true);
        std::fs::write(dir.join(CONFIG_NAME), cfg.to_json_string().unwrap()).unwrap();
        let err = SkyLensNetDetector::load(&dir, Device::Cpu).unwrap_err().to_string();
        assert!(err.contains("offset_head"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// bf16 on CUDA keeps bf16 weights and stays close to the fp32 result. Skipped without CUDA.
    #[test]
    fn cuda_bf16_tracks_fp32() {
        #[cfg(windows)]
        // SAFETY: a NUL-terminated library name; a failed load leaves tch on CPU.
        unsafe {
            windows_sys::Win32::System::LibraryLoader::LoadLibraryA(c"torch_cuda.dll".as_ptr().cast());
        }
        if !tch::Cuda::is_available() {
            eprintln!("no CUDA; skipped");
            return;
        }
        let dir = tiny_checkpoint("bf16", true, true);
        let load = |precision| {
            let opts = DetectorOptions { precision, ..Default::default() };
            SkyLensNetDetector::load_with_options(&dir, Device::Cuda(0), opts).unwrap()
        };
        let (fp32, bf16) = (load(Precision::Fp32), load(Precision::Bf16));
        let kinds: Vec<Kind> = bf16.model.lock().unwrap()._vs.variables().values().map(|t| t.kind()).collect();
        assert!(kinds.iter().all(|k| *k == Kind::BFloat16), "{kinds:?}");
        let (w, h) = (100usize, 60usize);
        let (rgb, th) = frame(w, h);
        let a = fp32.detect(&rgb, Some(&th), w as u32, h as u32).unwrap();
        let b = bf16.detect(&rgb, Some(&th), w as u32, h as u32).unwrap();
        let worst = a.hazard_prob.iter().zip(&b.hazard_prob).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(worst < 0.05, "hazard probability differs by {worst}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
