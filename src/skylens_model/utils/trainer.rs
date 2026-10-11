//! Rust twin of `utils/trainer.py`.
//!
//! The training-loop structure of `SkyLensTrainer`: optimizer, loss composition, head-wise
//! training from GT presence, modality dropout, backbone-freeze warmup, loss-component logging,
//! checkpointing and the evaluation step. Training itself still happens in Python; this is
//! the same loop expressed over `tch`, driven through the [`DualHeadModel`] trait so it does
//! not import `models/` (same rule as the Python module: `utils` never depends on `models`).
//!
//! # Loss composition
//!
//! In Python the *model* computes the loss from the batch (README §6.3): a head's loss is on
//! only when its GT key is present. The trainer only records the components. A Rust model does
//! not have to carry that logic, so [`compute_losses`] does it here with the same formulas
//! and default weights as `SkyLensNetConfig`:
//!
//! * seg: `seg_weight * CE(ignore 255)` + `seg_weight * dice_weight * soft_dice(ignore 255)`
//!   with `dice_weight = 1.0`, i.e. CE and Dice 1:1. (The Python config default is 0.0, which
//!   switches Dice off; the training recipe uses 1.0, hence the default here.)
//! * person heatmap: CenterNet penalty-reduced focal loss, weight 1.0
//! * person wh: masked L1, weight 0.1
//! * person offset: masked L1, weight 1.0 (only when the model outputs an offset map)
//!
//! # Not ported
//!
//! * `transformers.Trainer` machinery: mixed precision (`fp16`/`bf16`), DDP, `torch.compile`,
//!   gradient checkpointing, optimizer-state checkpoints. A checkpoint here holds weights and a
//!   small `trainer_state.json`; resuming restores weights and the step counter only.
//! * The decoding half of `prediction_step` (`decode_heatmap_peaks` / `decode_gt_boxes`). It
//!   belongs to `utils/metrics.rs`; [`PredictionStep`] hands back the raw head outputs and the
//!   GT tensors so the caller can decode with that module.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use anyhow::Result;
use tch::nn::{self, OptimizerConfig};
use tch::{Device, Kind, Reduction, Tensor};

use super::callbacks::{GracefulInterruptCallback, TrainerCallback, TrainerControl, TrainerState};
use super::collate::{Batch, SkyLensCollator};
use super::training_args::SkyLensTrainingArguments;
use crate::datasets::base::{SimpleRng, SkyLensDataset, IGNORE_INDEX};

// --------------------------------------------------------------------------- //
// model contract
// --------------------------------------------------------------------------- //

/// Head outputs, named like Python's `SkyLensOutput`.
#[derive(Debug, Default)]
pub struct ModelOutput {
    /// `(B, num_classes, H, W)` logits, before softmax.
    pub danger_logits: Option<Tensor>,
    /// `(B, 1, h, w)` heatmap, sigmoid already applied.
    pub person_heatmap: Option<Tensor>,
    /// `(B, 2, h, w)` width/height regression.
    pub person_wh: Option<Tensor>,
    /// `(B, 2, h, w)` sub-cell offset, if the model has that head.
    pub person_offset: Option<Tensor>,
}

/// What the loop needs from a model.
pub trait DualHeadModel {
    /// Forward pass. `train` switches dropout / batch-norm behaviour.
    fn forward(&self, pixel_values: &Tensor, train: bool) -> ModelOutput;

    /// The variable store that holds every parameter.
    fn var_store(&self) -> &nn::VarStore;

    /// Variable-name prefix of the backbone (`FreezeBackboneCallback` freezes these).
    /// Python tries the attribute names `backbone`, `encoder`, `feature_extractor` in order.
    fn backbone_prefixes(&self) -> Vec<String> {
        vec!["backbone".into(), "encoder".into(), "feature_extractor".into()]
    }
}

// --------------------------------------------------------------------------- //
// loss functions (modeling_skylensnet.py)
// --------------------------------------------------------------------------- //

/// CenterNet penalty-reduced pixel-wise focal loss. `pred` is a probability (sigmoid already
/// applied), `target` the gaussian GT heatmap; only `target == 1` pixels are positives.
pub fn centernet_focal_loss(pred: &Tensor, target: &Tensor, alpha: f64, beta: f64) -> Tensor {
    let eps = 1e-4;
    let pred = pred.clamp(eps, 1.0 - eps);

    let pos_mask = target.eq(1.0).to_kind(Kind::Float);
    let neg_mask = 1.0f64 - &pos_mask;

    let pos_loss = pred.log() * (1.0f64 - &pred).pow_tensor_scalar(alpha) * &pos_mask;
    let neg_weights = (1.0f64 - target).pow_tensor_scalar(beta);
    let neg_loss = (1.0f64 - &pred).log() * pred.pow_tensor_scalar(alpha) * neg_weights * &neg_mask;

    let num_pos = pos_mask.sum(Kind::Float);
    let pos_loss = pos_loss.sum(Kind::Float);
    let neg_loss = neg_loss.sum(Kind::Float);

    if num_pos.double_value(&[]) == 0.0 {
        neg_loss.neg()
    } else {
        (pos_loss + neg_loss).neg() / num_pos
    }
}

/// L1 regression evaluated only at valid centres (division by zero guarded).
pub fn masked_l1_loss(pred: &Tensor, target: &Tensor, mask: &Tensor) -> Tensor {
    let mask = mask.expand_as(pred).to_kind(Kind::Float);
    (pred * &mask).l1_loss(&(target * &mask), Reduction::Sum) / mask.sum(Kind::Float).clamp_min(1.0)
}

/// Multi-class soft Dice loss (class mean) with ignore pixels removed from numerator and
/// denominator. `logits` is `(B, C, H, W)` before softmax, `labels` is `(B, H, W)` integer.
/// Absent classes give dice = 1 thanks to smoothing, so no NaN.
pub fn masked_soft_dice_loss(logits: &Tensor, labels: &Tensor, ignore_index: i64, smooth: f64) -> Tensor {
    let num_classes = logits.size()[1];
    // float32 so sums do not overflow under half precision
    let probs = logits.to_kind(Kind::Float).softmax(1, Kind::Float);
    let labels = labels.to_kind(Kind::Int64);
    let valid = labels.ne(ignore_index);
    let safe_labels = labels.where_self(&valid, &labels.zeros_like());
    let one_hot = safe_labels.one_hot(num_classes).permute([0, 3, 1, 2]).to_kind(Kind::Float);
    let valid_f = valid.unsqueeze(1).to_kind(Kind::Float);
    let probs = probs * &valid_f;
    let one_hot = one_hot * &valid_f;

    let dims: &[i64] = &[0, 2, 3];
    let intersection = (&probs * &one_hot).sum_dim_intlist(dims, false, Kind::Float);
    let denominator = probs.sum_dim_intlist(dims, false, Kind::Float)
        + one_hot.sum_dim_intlist(dims, false, Kind::Float);
    let dice = (intersection * 2.0 + smooth) / (denominator + smooth);
    1.0 - dice.mean(Kind::Float)
}

/// Loss weights; defaults follow `SkyLensNetConfig`, except `dice_weight` (see module docs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LossConfig {
    pub seg_weight: f64,
    /// Dice relative to CE inside the seg loss. 1.0 is CE : Dice = 1 : 1; 0 disables Dice.
    pub dice_weight: f64,
    pub heatmap_weight: f64,
    pub wh_weight: f64,
    pub offset_weight: f64,
    pub ignore_index: i64,
}

impl Default for LossConfig {
    fn default() -> Self {
        LossConfig {
            seg_weight: 1.0,
            dice_weight: 1.0,
            heatmap_weight: 1.0,
            wh_weight: 0.1,
            offset_weight: 1.0,
            ignore_index: i64::from(IGNORE_INDEX),
        }
    }
}

/// Weighted total plus the raw (unweighted) components, keyed like Python's `loss_dict`.
#[derive(Debug)]
pub struct LossOutput {
    pub total: Option<Tensor>,
    pub components: BTreeMap<String, Tensor>,
}

/// Head-wise loss: a head contributes only when both its output and its GT key exist.
/// With no GT at all, `total` is `None` (Python: `loss is None`).
pub fn compute_losses(out: &ModelOutput, batch: &Batch, cfg: &LossConfig) -> LossOutput {
    let mut comps: BTreeMap<String, Tensor> = BTreeMap::new();

    if let (Some(logits), Some(labels)) = (&out.danger_logits, &batch.danger_labels) {
        let labels = labels.to_kind(Kind::Int64);
        let ce = logits.cross_entropy_loss::<Tensor>(&labels, None, Reduction::Mean, cfg.ignore_index, 0.0);
        comps.insert("danger_seg".into(), ce);
        if cfg.dice_weight > 0.0 {
            comps.insert(
                "danger_dice".into(),
                masked_soft_dice_loss(logits, &labels, cfg.ignore_index, 1.0),
            );
        }
    }
    if let (Some(pred), Some(gt)) = (&out.person_heatmap, &batch.person_heatmap) {
        comps.insert("person_heatmap".into(), centernet_focal_loss(pred, gt, 2.0, 4.0));
    }
    if let (Some(pred), Some(gt), Some(mask)) = (&out.person_wh, &batch.person_wh, &batch.person_reg_mask) {
        comps.insert("person_wh".into(), masked_l1_loss(pred, gt, mask));
    }
    if let (Some(pred), Some(gt), Some(mask)) =
        (&out.person_offset, &batch.person_offset, &batch.person_reg_mask)
    {
        comps.insert("person_offset".into(), masked_l1_loss(pred, gt, mask));
    }

    if comps.is_empty() {
        return LossOutput { total: None, components: comps };
    }
    let weight = |k: &str| match k {
        "danger_seg" => cfg.seg_weight,
        "danger_dice" => cfg.seg_weight * cfg.dice_weight,
        "person_heatmap" => cfg.heatmap_weight,
        "person_wh" => cfg.wh_weight,
        _ => cfg.offset_weight,
    };
    let mut total: Option<Tensor> = None;
    for (k, v) in &comps {
        let term = v * weight(k);
        total = Some(match total {
            Some(t) => t + term,
            None => term,
        });
    }
    LossOutput { total, components: comps }
}

/// Shape of a 0-dim tensor.
const SCALAR: &[i64] = &[];

/// A zero connected to the graph (or a fresh trainable scalar), so `backward` never breaks on
/// a batch without any GT.
fn zero_loss(out: &ModelOutput, device: Device) -> Tensor {
    let reference = [&out.danger_logits, &out.person_heatmap, &out.person_wh]
        .into_iter()
        .flatten()
        .next();
    match reference {
        Some(t) if t.requires_grad() => t.sum(Kind::Float) * 0.0,
        _ => Tensor::zeros(SCALAR, (Kind::Float, device)).set_requires_grad(true),
    }
}

// --------------------------------------------------------------------------- //
// modality dropout (README §2.2, modeling_skylensnet._sample_modality_mask)
// --------------------------------------------------------------------------- //

/// Per-sample `[rgb_present, thermal_present]` draws: with probability `p_rgb_only` the
/// thermal plane is removed, with `p_thermal_only` the RGB planes are, otherwise both stay.
pub fn sample_modality_flags(
    batch_size: usize,
    p_rgb_only: f64,
    p_thermal_only: f64,
    rng: &mut SimpleRng,
) -> Vec<[bool; 2]> {
    (0..batch_size)
        .map(|_| {
            let u = rng.random();
            let thermal_present = !(u < p_rgb_only);
            let rgb_present = !(u >= p_rgb_only && u < p_rgb_only + p_thermal_only);
            [rgb_present, thermal_present]
        })
        .collect()
}

/// `(B, 2)` bool tensor from [`sample_modality_flags`].
pub fn modality_mask_tensor(flags: &[[bool; 2]], device: Device) -> Tensor {
    let flat: Vec<bool> = flags.iter().flatten().copied().collect();
    Tensor::from_slice(&flat).reshape([flags.len() as i64, 2]).to_device(device)
}

/// Zeroes the channels of an absent modality and records thermal validity (README §2.3).
///
/// `in_channels` is the model's data channel count (4); a validity channel beyond it is set
/// to the thermal-present flag. Out-of-place (the Python version clones first).
pub fn apply_modality_mask(
    pixel_values: &Tensor,
    modality_mask: &Tensor,
    in_channels: i64,
    use_validity_channel: bool,
) -> Tensor {
    let size = pixel_values.size();
    let (b, c, h, w) = (size[0], size[1], size[2], size[3]);
    let kind = pixel_values.kind();
    let rgb_present = modality_mask.select(1, 0).to_kind(kind).reshape([-1, 1, 1, 1]);
    let thermal_present = modality_mask.select(1, 1).to_kind(kind).reshape([-1, 1, 1, 1]);

    let mut parts = vec![
        pixel_values.narrow(1, 0, 3) * &rgb_present,
        pixel_values.narrow(1, 3, in_channels - 3) * &thermal_present,
    ];
    if c > in_channels {
        if use_validity_channel {
            parts.push(thermal_present.expand([b, c - in_channels, h, w], true));
        } else {
            parts.push(pixel_values.narrow(1, in_channels, c - in_channels));
        }
    }
    Tensor::cat(&parts, 1)
}

// --------------------------------------------------------------------------- //
// backbone freeze warmup (FreezeBackboneCallback)
// --------------------------------------------------------------------------- //

/// Freezes the backbone for the first `freeze_epochs` epochs (README §6.1) so the gradients of
/// the randomly initialised dual heads do not wreck the pretrained backbone early on.
pub struct FreezeBackboneCallback {
    freeze_epochs: f64,
    prefixes: Vec<String>,
    frozen: bool,
}

impl FreezeBackboneCallback {
    pub fn new(freeze_epochs: f64, backbone_prefixes: Vec<String>) -> Self {
        FreezeBackboneCallback { freeze_epochs, prefixes: backbone_prefixes, frozen: false }
    }

    /// Sets `requires_grad` on every variable under the first prefix that matches anything.
    /// Returns false when no backbone is found.
    fn set_requires_grad(&self, vs: &nn::VarStore, flag: bool) -> bool {
        let vars = vs.variables();
        for prefix in &self.prefixes {
            let dotted = format!("{prefix}.");
            let hits: Vec<&Tensor> = vars
                .iter()
                .filter(|(name, _)| name.as_str() == prefix.as_str() || name.starts_with(dotted.as_str()))
                .map(|(_, t)| t)
                .collect();
            if !hits.is_empty() {
                for t in hits {
                    let _ = t.set_requires_grad(flag);
                }
                return true;
            }
        }
        false
    }

    pub fn is_frozen(&self) -> bool {
        self.frozen
    }
}

impl TrainerCallback for FreezeBackboneCallback {
    fn on_epoch_begin(&mut self, state: &TrainerState, _c: &mut TrainerControl, vs: &nn::VarStore) {
        if self.freeze_epochs <= 0.0 {
            return;
        }
        let should_freeze = state.epoch < self.freeze_epochs;
        if should_freeze && !self.frozen {
            if self.set_requires_grad(vs, false) {
                self.frozen = true;
            }
        } else if !should_freeze && self.frozen {
            self.set_requires_grad(vs, true);
            self.frozen = false;
        }
    }

    fn on_train_end(&mut self, _s: &TrainerState, _c: &mut TrainerControl, vs: &nn::VarStore) {
        // leave the model in a normal state even if training ended while frozen
        if self.frozen {
            self.set_requires_grad(vs, true);
            self.frozen = false;
        }
    }
}

// --------------------------------------------------------------------------- //
// learning-rate schedule
// --------------------------------------------------------------------------- //

/// Learning rate at optimizer step `step` (0-based) of `total` steps, HF schedule semantics:
/// linear warmup over `warmup_ratio * total` steps, then linear decay to 0, cosine decay, or
/// constant.
pub fn lr_at(kind: &str, base: f64, step: u64, total: u64, warmup_ratio: f64) -> f64 {
    let total = total.max(1) as f64;
    let warmup = (warmup_ratio * total).ceil();
    let s = step as f64;
    if warmup > 0.0 && s < warmup {
        return base * s / warmup;
    }
    let progress = if total > warmup { (s - warmup) / (total - warmup) } else { 1.0 };
    match kind {
        "constant" => base,
        "cosine" => base * 0.5 * (1.0 + (std::f64::consts::PI * progress.min(1.0)).cos()),
        _ => base * (1.0 - progress).max(0.0),
    }
}

// --------------------------------------------------------------------------- //
// dataset composition (scripts/train_experiment.py)
// --------------------------------------------------------------------------- //

/// Concatenation of datasets (`ConcatDataset`).
pub struct ConcatDataset {
    parts: Vec<Box<dyn SkyLensDataset>>,
}

impl ConcatDataset {
    pub fn new(parts: Vec<Box<dyn SkyLensDataset>>) -> Self {
        ConcatDataset { parts }
    }
}

impl SkyLensDataset for ConcatDataset {
    fn len(&self) -> usize {
        self.parts.iter().map(|p| p.len()).sum()
    }

    fn get(&self, mut idx: usize) -> crate::datasets::base::Result<crate::datasets::base::Sample> {
        for p in &self.parts {
            if idx < p.len() {
                return p.get(idx);
            }
            idx -= p.len();
        }
        Err(crate::datasets::base::DatasetError::InvalidSample("concat index out of range".into()))
    }
}

/// A dataset viewed through an index list (`Subset`, also used for oversampling).
pub struct IndexedDataset {
    inner: Box<dyn SkyLensDataset>,
    indices: Vec<usize>,
}

impl IndexedDataset {
    pub fn new(inner: Box<dyn SkyLensDataset>, indices: Vec<usize>) -> Self {
        IndexedDataset { inner, indices }
    }
}

impl SkyLensDataset for IndexedDataset {
    fn len(&self) -> usize {
        self.indices.len()
    }

    fn get(&self, idx: usize) -> crate::datasets::base::Result<crate::datasets::base::Sample> {
        self.inner.get(self.indices[idx])
    }
}

/// Index plan for oversampling images that contain a rare class (SPEC §8: `road_blocked`
/// images x4): every image once, plus `flagged` images `factor - 1` more times.
pub fn oversample_indices(len: usize, flagged: &[usize], factor: usize) -> Vec<usize> {
    let mut out: Vec<usize> = (0..len).collect();
    for _ in 1..factor.max(1) {
        out.extend_from_slice(flagged);
    }
    out
}

/// Index plan that repeats a whole dataset `factor` times. SPEC §8: fire_seg is scaled by the
/// same factor as the RescueNet oversampling so the seg-set composition (fire share) is kept.
pub fn repeat_indices(len: usize, factor: usize) -> Vec<usize> {
    (0..factor.max(1)).flat_map(|_| 0..len).collect()
}

/// Indices of samples whose mask contains `class` (cached as `has_class_<class>.json`).
pub fn find_class_indices(ds: &dyn SkyLensDataset, cache_dir: &Path, class: u8) -> Result<Vec<usize>> {
    let marker = cache_dir.join(format!("has_class_{class}.json"));
    if marker.is_file() {
        return Ok(serde_json::from_slice(&std::fs::read(&marker)?)?);
    }
    let mut found = Vec::new();
    for i in 0..ds.len() {
        let s = ds.get(i)?;
        if let Some(mask) = &s.danger_mask {
            if mask.eq(i64::from(class)).sum(Kind::Int64).int64_value(&[]) > 0 {
                found.push(i);
            }
        }
    }
    std::fs::write(&marker, serde_json::to_vec(&found)?)?;
    Ok(found)
}

// --------------------------------------------------------------------------- //
// trainer
// --------------------------------------------------------------------------- //

/// One logged row (`Trainer.log`).
#[derive(Debug, Clone, Default)]
pub struct LogEntry {
    pub step: u64,
    pub epoch: f64,
    pub values: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Default)]
pub struct TrainResult {
    pub global_step: u64,
    pub history: Vec<LogEntry>,
    pub interrupted: bool,
}

/// Output of [`SkyLensTrainer::prediction_step`].
#[derive(Debug)]
pub struct PredictionStep {
    pub loss: Option<Tensor>,
    /// `(B, H, W)` int16 argmax class map (logits are not accumulated, README memory note),
    /// or an all-ignore placeholder when the model has no seg output.
    pub seg_pred: Tensor,
    /// `(B, H, W)` int16 GT, ignore-filled when the batch has no seg GT.
    pub seg_gt: Tensor,
    /// Raw point heads, for `metrics::decode_heatmap_peaks`. NOT PORTED: decoding to `(B,K,5)`.
    pub outputs: ModelOutput,
}

pub struct SkyLensTrainer<M: DualHeadModel> {
    pub model: M,
    pub args: SkyLensTrainingArguments,
    pub loss_cfg: LossConfig,
    /// `(p_rgb_only, p_thermal_only)` applied on top of the collator's own dropout; `None`
    /// leaves modality masks exactly as the collator produced them.
    pub modality_dropout: Option<(f64, f64)>,
    /// Channel count of the data planes, 4 for RGB+thermal.
    pub in_channels: i64,
    pub use_validity_channel: bool,
    callbacks: Vec<Box<dyn TrainerCallback>>,
    loss_log_buffer: HashMap<String, Vec<f64>>,
    rng: SimpleRng,
}

impl<M: DualHeadModel> SkyLensTrainer<M> {
    pub fn new(model: M, args: SkyLensTrainingArguments) -> Result<Self> {
        args.validate().map_err(anyhow::Error::msg)?;
        let rng = SimpleRng::new(args.seed);
        let mut t = SkyLensTrainer {
            model,
            args,
            loss_cfg: LossConfig::default(),
            modality_dropout: None,
            in_channels: 4,
            use_validity_channel: false,
            callbacks: Vec::new(),
            loss_log_buffer: HashMap::new(),
            rng,
        };
        if t.args.freeze_backbone_epochs > 0.0 {
            let prefixes = t.model.backbone_prefixes();
            t.add_callback(Box::new(FreezeBackboneCallback::new(t.args.freeze_backbone_epochs, prefixes)));
        }
        Ok(t)
    }

    pub fn add_callback(&mut self, cb: Box<dyn TrainerCallback>) {
        self.callbacks.push(cb);
    }

    /// Adds [`GracefulInterruptCallback`] (stop and save on SIGINT/SIGTERM).
    pub fn with_graceful_interrupt(mut self) -> Self {
        self.add_callback(Box::new(GracefulInterruptCallback::new()));
        self
    }

    // ---- loss ----

    /// Runs the model and the head-wise loss. Components are accumulated for logging.
    pub fn compute_loss(&mut self, batch: &Batch, train: bool) -> (Tensor, ModelOutput) {
        let mut pixels = batch.pixel_values.shallow_clone();
        if train {
            if let Some((p_rgb_only, p_thermal_only)) = self.modality_dropout {
                let b = pixels.size()[0] as usize;
                let flags = sample_modality_flags(b, p_rgb_only, p_thermal_only, &mut self.rng);
                let mask = modality_mask_tensor(&flags, pixels.device());
                pixels = apply_modality_mask(&pixels, &mask, self.in_channels, self.use_validity_channel);
            }
        }
        let out = self.model.forward(&pixels, train);
        let losses = compute_losses(&out, batch, &self.loss_cfg);
        match losses.total {
            Some(total) => {
                self.accumulate_components(&losses.components);
                (total, out)
            }
            None => {
                // no GT at all: a zero connected to the graph so backward does not break
                let z = zero_loss(&out, batch.pixel_values.device());
                (z, out)
            }
        }
    }

    fn accumulate_components(&mut self, comps: &BTreeMap<String, Tensor>) {
        if !self.args.log_loss_components {
            return;
        }
        for (k, v) in comps {
            if v.numel() == 0 {
                continue;
            }
            let val = v.detach().to_kind(Kind::Float).mean(Kind::Float).double_value(&[]);
            self.loss_log_buffer.entry(format!("loss_{k}")).or_default().push(val);
        }
    }

    /// Means of the buffered components, rounded to 6 digits, clearing the buffer
    /// (the first half of `Trainer.log`).
    pub fn drain_loss_logs(&mut self) -> BTreeMap<String, f64> {
        let mut out = BTreeMap::new();
        for (k, vals) in self.loss_log_buffer.drain() {
            if !vals.is_empty() {
                let mean = vals.iter().sum::<f64>() / vals.len() as f64;
                out.insert(k, (mean * 1e6).round() / 1e6);
            }
        }
        out
    }

    // ---- callbacks ----

    fn fire(
        &mut self,
        state: &TrainerState,
        control: &mut TrainerControl,
        hook: impl Fn(&mut dyn TrainerCallback, &TrainerState, &mut TrainerControl, &nn::VarStore),
    ) {
        for cb in self.callbacks.iter_mut() {
            hook(cb.as_mut(), state, control, self.model.var_store());
        }
    }

    // ---- train ----

    /// Optimizer steps per epoch and in total.
    pub fn plan_steps(&self, dataset_len: usize) -> (u64, u64) {
        let micro = dataset_len.div_ceil(self.args.per_device_train_batch_size.max(1));
        let per_epoch = micro.div_ceil(self.args.gradient_accumulation_steps.max(1)).max(1) as u64;
        let total = (per_epoch as f64 * self.args.num_train_epochs).ceil() as u64;
        (per_epoch, total)
    }

    pub fn train(
        &mut self,
        train_ds: &dyn SkyLensDataset,
        collator: &mut SkyLensCollator,
        device: Device,
    ) -> Result<TrainResult> {
        let bs = self.args.per_device_train_batch_size;
        let accum = self.args.gradient_accumulation_steps.max(1);
        let (steps_per_epoch, total_steps) = self.plan_steps(train_ds.len());

        let mut opt = nn::AdamW::default()
            .wd(self.args.weight_decay)
            .build(self.model.var_store(), self.args.learning_rate)?;

        let mut state = TrainerState::default();
        let mut control = TrainerControl::default();
        let mut result = TrainResult::default();

        if let Some(ckpt) = self.args.resume_from_checkpoint.clone() {
            state.global_step = self.load_checkpoint(&ckpt)?;
        }

        self.fire(&state, &mut control, |cb, s, c, vs| cb.on_train_begin(s, c, vs));

        let n_epochs = self.args.num_train_epochs.ceil() as u64;
        let first_epoch = state.global_step / steps_per_epoch;
        let mut skip_steps = state.global_step % steps_per_epoch;

        'epochs: for epoch in first_epoch..n_epochs {
            state.epoch = epoch as f64;
            self.fire(&state, &mut control, |cb, s, c, vs| cb.on_epoch_begin(s, c, vs));

            let mut order: Vec<usize> = (0..train_ds.len()).collect();
            let mut epoch_rng = SimpleRng::new(self.args.seed.wrapping_add(epoch));
            epoch_rng.shuffle(&mut order);

            let mut step_in_epoch = 0u64;
            let micro_batches: Vec<&[usize]> = order.chunks(bs).collect();
            // one group of `accum` micro-batches is one optimizer step (the last may be short)
            for group in micro_batches.chunks(accum) {
                // fast-forward over steps already done before a resume
                if skip_steps > 0 {
                    skip_steps -= 1;
                    step_in_epoch += 1;
                    continue;
                }

                for chunk in group {
                    let samples = chunk
                        .iter()
                        .map(|&i| train_ds.get(i))
                        .collect::<std::result::Result<Vec<_>, _>>()?;
                    let batch = collator.collate(&samples)?.to_device(device);
                    let (loss, _out) = self.compute_loss(&batch, true);
                    (loss / group.len() as f64).backward();
                }

                if self.args.max_grad_norm > 0.0 {
                    opt.clip_grad_norm(self.args.max_grad_norm);
                }
                let lr = lr_at(
                    &self.args.lr_scheduler_type,
                    self.args.learning_rate,
                    state.global_step,
                    total_steps,
                    self.args.warmup_ratio,
                );
                opt.set_lr(lr);
                opt.step();
                opt.zero_grad();

                state.global_step += 1;
                step_in_epoch += 1;
                state.epoch = epoch as f64 + step_in_epoch as f64 / steps_per_epoch as f64;
                control = TrainerControl::default();
                self.fire(&state, &mut control, |cb, s, c, vs| cb.on_step_end(s, c, vs));

                if self.args.logging_steps > 0 && state.global_step % self.args.logging_steps == 0 {
                    let mut entry = LogEntry { step: state.global_step, epoch: state.epoch, ..Default::default() };
                    entry.values.insert("learning_rate".into(), lr);
                    entry.values.extend(self.drain_loss_logs());
                    tracing::info!(step = entry.step, epoch = entry.epoch, values = ?entry.values, "train");
                    result.history.push(entry);
                }
                if control.should_save
                    || (self.args.save_steps > 0 && state.global_step % self.args.save_steps == 0)
                {
                    self.save_checkpoint(&state)?;
                }
                if control.should_training_stop || state.global_step >= total_steps {
                    result.interrupted = control.should_training_stop;
                    break 'epochs;
                }
            }
        }

        let final_state = state;
        self.fire(&final_state, &mut control, |cb, s, c, vs| cb.on_train_end(s, c, vs));
        // flush components that never reached a logging boundary
        let rest = self.drain_loss_logs();
        if !rest.is_empty() {
            result.history.push(LogEntry { step: state.global_step, epoch: state.epoch, values: rest });
        }
        result.global_step = state.global_step;
        Ok(result)
    }

    // ---- evaluation ----

    /// One no-grad forward on a batch. The seg head is reduced to an int16 argmax map; point
    /// heads are returned raw (see the module docs for what is not ported).
    pub fn prediction_step(&mut self, batch: &Batch) -> PredictionStep {
        let _guard = tch::no_grad_guard();
        let out = self.model.forward(&batch.pixel_values, false);
        let losses = compute_losses(&out, batch, &self.loss_cfg);
        if losses.total.is_some() {
            self.accumulate_components(&losses.components);
        }
        let loss = losses.total.map(|t| t.detach());

        let ignore = self.args.seg_ignore_index;
        let (seg_pred, seg_gt) = match (&out.danger_logits, &batch.danger_labels) {
            (Some(logits), Some(gt)) => (
                logits.argmax(1, false).to_kind(Kind::Int16),
                gt.to_kind(Kind::Int16),
            ),
            _ => {
                let reference = out.danger_logits.as_ref().or(batch.danger_labels.as_ref());
                let (shape, device) = match reference {
                    Some(r) => {
                        let s = r.size();
                        (vec![s[0], s[s.len() - 2], s[s.len() - 1]], r.device())
                    }
                    None => (vec![1, 1, 1], batch.pixel_values.device()),
                };
                let filler = Tensor::full(shape, ignore, (Kind::Int16, device));
                (filler.shallow_clone(), filler)
            }
        };
        PredictionStep { loss, seg_pred, seg_gt, outputs: out }
    }

    /// Mean loss and mean loss components over a dataset.
    pub fn evaluate(
        &mut self,
        ds: &dyn SkyLensDataset,
        collator: &mut SkyLensCollator,
        device: Device,
    ) -> Result<BTreeMap<String, f64>> {
        let bs = self.args.per_device_eval_batch_size.max(1);
        let mut losses = Vec::new();
        let indices: Vec<usize> = (0..ds.len()).collect();
        for chunk in indices.chunks(bs) {
            let samples = chunk.iter().map(|&i| ds.get(i)).collect::<std::result::Result<Vec<_>, _>>()?;
            let batch = collator.collate(&samples)?.to_device(device);
            let step = self.prediction_step(&batch);
            if let Some(l) = step.loss {
                losses.push(l.double_value(&[]));
            }
        }
        let mut out = self.drain_loss_logs();
        if !losses.is_empty() {
            out.insert("eval_loss".into(), losses.iter().sum::<f64>() / losses.len() as f64);
        }
        let state = TrainerState::default();
        let mut control = TrainerControl::default();
        self.fire(&state, &mut control, |cb, s, c, vs| cb.on_evaluate(s, c, vs));
        Ok(out)
    }

    // ---- checkpoints ----

    fn checkpoint_dir(&self, step: u64) -> PathBuf {
        self.args.output_dir.join(format!("checkpoint-{step}"))
    }

    /// Saves weights and `trainer_state.json`, then enforces `save_total_limit`.
    pub fn save_checkpoint(&self, state: &TrainerState) -> Result<PathBuf> {
        let dir = self.checkpoint_dir(state.global_step);
        std::fs::create_dir_all(&dir)?;
        self.model.var_store().save(dir.join("model.safetensors"))?;
        std::fs::write(
            dir.join("trainer_state.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "global_step": state.global_step,
                "epoch": state.epoch,
            }))?,
        )?;
        if let Some(limit) = self.args.save_total_limit {
            prune_checkpoints(&self.args.output_dir, limit.max(1))?;
        }
        Ok(dir)
    }

    /// Restores weights and returns the checkpoint's global step.
    pub fn load_checkpoint(&mut self, dir: &Path) -> Result<u64> {
        // `VarStore::load` needs `&mut`; the store is owned by the model, so go through a
        // partial load into the shared tensors instead.
        let weights = Tensor::read_safetensors(dir.join("model.safetensors"))?;
        let vars = self.model.var_store().variables();
        tch::no_grad(|| {
            for (name, tensor) in weights {
                if let Some(dst) = vars.get(&name) {
                    let mut dst = dst.shallow_clone();
                    dst.copy_(&tensor);
                }
            }
        });
        let state: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("trainer_state.json"))?)?;
        Ok(state["global_step"].as_u64().unwrap_or(0))
    }
}

/// Keeps the `limit` newest `checkpoint-<step>` directories under `output_dir`.
pub fn prune_checkpoints(output_dir: &Path, limit: usize) -> Result<()> {
    let mut found: Vec<(u64, PathBuf)> = std::fs::read_dir(output_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .filter_map(|e| {
            let step = e.file_name().to_string_lossy().strip_prefix("checkpoint-")?.parse().ok()?;
            Some((step, e.path()))
        })
        .collect();
    found.sort();
    let excess = found.len().saturating_sub(limit);
    for (_, path) in found.into_iter().take(excess) {
        std::fs::remove_dir_all(path)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() -> (Kind, Device) {
        (Kind::Float, Device::Cpu)
    }

    #[test]
    fn linear_schedule_warms_up_then_decays() {
        // 100 steps, 10% warmup
        assert_eq!(lr_at("linear", 1.0, 0, 100, 0.1), 0.0);
        assert!((lr_at("linear", 1.0, 5, 100, 0.1) - 0.5).abs() < 1e-9);
        assert!((lr_at("linear", 1.0, 10, 100, 0.1) - 1.0).abs() < 1e-9);
        assert!((lr_at("linear", 1.0, 55, 100, 0.1) - 0.5).abs() < 1e-9);
        assert!(lr_at("linear", 1.0, 100, 100, 0.1).abs() < 1e-9);
        assert_eq!(lr_at("constant", 0.3, 50, 100, 0.0), 0.3);
        assert!((lr_at("cosine", 1.0, 50, 100, 0.0) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn oversampling_plan_matches_spec() {
        // 10 images, images 2 and 7 hold road_blocked, factor 4 -> 10 + 2 * 3
        let plan = oversample_indices(10, &[2, 7], 4);
        assert_eq!(plan.len(), 16);
        assert_eq!(plan.iter().filter(|&&i| i == 2).count(), 4);
        assert_eq!(plan.iter().filter(|&&i| i == 7).count(), 4);
        assert_eq!(plan.iter().filter(|&&i| i == 0).count(), 1);
        // factor 1 changes nothing
        assert_eq!(oversample_indices(3, &[1], 1), vec![0, 1, 2]);
    }

    #[test]
    fn fire_seg_scales_by_the_same_factor() {
        let fire = repeat_indices(5, 4);
        assert_eq!(fire.len(), 20);
        assert_eq!(fire.iter().filter(|&&i| i == 3).count(), 4);
        // composition is preserved: both sets scale together only if the whole set is repeated
        assert_eq!(repeat_indices(5, 1), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn modality_flags_follow_probabilities() {
        let mut rng = SimpleRng::new(11);
        let flags = sample_modality_flags(20_000, 0.25, 0.25, &mut rng);
        let n = flags.len() as f64;
        let rgb_only = flags.iter().filter(|f| f[0] && !f[1]).count() as f64 / n;
        let th_only = flags.iter().filter(|f| !f[0] && f[1]).count() as f64 / n;
        let both = flags.iter().filter(|f| f[0] && f[1]).count() as f64 / n;
        assert!((rgb_only - 0.25).abs() < 0.02, "{rgb_only}");
        assert!((th_only - 0.25).abs() < 0.02, "{th_only}");
        assert!((both - 0.5).abs() < 0.02, "{both}");
        assert!(flags.iter().all(|f| f[0] || f[1]));
    }

    #[test]
    fn modality_mask_zeroes_absent_planes_and_sets_validity() {
        let pv = Tensor::ones([2, 5, 2, 2], cpu());
        let mask = modality_mask_tensor(&[[true, false], [false, true]], Device::Cpu);
        let out = apply_modality_mask(&pv, &mask, 4, true);
        assert_eq!(out.size(), vec![2, 5, 2, 2]);
        // sample 0: thermal absent -> channel 3 and validity 0, RGB kept
        assert_eq!(out.get(0).get(3).sum(Kind::Float).double_value(&[]), 0.0);
        assert_eq!(out.get(0).get(4).sum(Kind::Float).double_value(&[]), 0.0);
        assert_eq!(out.get(0).get(0).sum(Kind::Float).double_value(&[]), 4.0);
        // sample 1: RGB absent -> channels 0..3 zero, thermal and validity kept
        assert_eq!(out.get(1).narrow(0, 0, 3).sum(Kind::Float).double_value(&[]), 0.0);
        assert_eq!(out.get(1).get(3).sum(Kind::Float).double_value(&[]), 4.0);
        assert_eq!(out.get(1).get(4).sum(Kind::Float).double_value(&[]), 4.0);
    }

    #[test]
    fn dice_is_near_zero_for_a_confident_correct_prediction_and_ignores_255() {
        // 2 classes, 1x2x2 image; labels [[0,1],[1,255]]
        let labels = Tensor::from_slice(&[0i64, 1, 1, 255]).reshape([1, 2, 2]);
        // logits favour the right class strongly; the ignored pixel is deliberately wrong
        let logits = Tensor::from_slice(&[
            20.0f32, -20.0, -20.0, 20.0, // class 0 plane: px0 high; px1,px2 low; px3 high (wrong)
            -20.0, 20.0, 20.0, -20.0, // class 1 plane
        ])
        .reshape([1, 2, 2, 2]);
        let dice = masked_soft_dice_loss(&logits, &labels, 255, 1.0).double_value(&[]);
        assert!(dice < 0.05, "{dice}");
        let ce = logits
            .cross_entropy_loss::<Tensor>(&labels, None, Reduction::Mean, 255, 0.0)
            .double_value(&[]);
        assert!(ce < 1e-6, "{ce}");
    }

    #[test]
    fn dice_with_no_gt_for_a_class_is_not_nan() {
        let labels = Tensor::zeros([1, 2, 2], (Kind::Int64, Device::Cpu));
        let logits = Tensor::zeros([1, 3, 2, 2], cpu());
        let d = masked_soft_dice_loss(&logits, &labels, 255, 1.0).double_value(&[]);
        assert!(d.is_finite());
    }

    #[test]
    fn focal_loss_falls_back_to_negatives_without_positives() {
        let pred = Tensor::full([1, 1, 2, 2], 0.5, cpu());
        let target = Tensor::zeros([1, 1, 2, 2], cpu());
        let l = centernet_focal_loss(&pred, &target, 2.0, 4.0).double_value(&[]);
        assert!(l > 0.0 && l.is_finite());
        // with a positive, normalised by the positive count
        let t = target.shallow_clone();
        let _ = t.get(0).get(0).get(0).get(0).fill_(1.0);
        let l2 = centernet_focal_loss(&pred, &t, 2.0, 4.0).double_value(&[]);
        assert!(l2 > 0.0 && l2.is_finite());
    }

    #[test]
    fn masked_l1_only_counts_valid_cells() {
        let pred = Tensor::from_slice(&[1.0f32, 5.0]).reshape([1, 1, 1, 2]);
        let target = Tensor::from_slice(&[3.0f32, 100.0]).reshape([1, 1, 1, 2]);
        let mask = Tensor::from_slice(&[1.0f32, 0.0]).reshape([1, 1, 1, 2]);
        assert!((masked_l1_loss(&pred, &target, &mask).double_value(&[]) - 2.0).abs() < 1e-6);
        // empty mask must not divide by zero
        let zero = mask.zeros_like();
        assert_eq!(masked_l1_loss(&pred, &target, &zero).double_value(&[]), 0.0);
    }

    #[test]
    fn loss_dict_is_head_wise() {
        let batch = Batch {
            pixel_values: Tensor::zeros([1, 4, 4, 4], cpu()),
            modality_mask: Tensor::from_slice(&[true, false]).reshape([1, 2]),
            danger_labels: Some(Tensor::zeros([1, 4, 4], (Kind::Int64, Device::Cpu))),
            person_heatmap: None,
            person_wh: None,
            person_reg_mask: None,
            person_offset: None,
        };
        let out = ModelOutput {
            danger_logits: Some(Tensor::zeros([1, 4, 4, 4], cpu())),
            person_heatmap: Some(Tensor::full([1, 1, 1, 1], 0.5, cpu())),
            ..Default::default()
        };
        let l = compute_losses(&out, &batch, &LossConfig::default());
        // seg GT present -> CE and Dice; no point GT -> no point losses
        assert!(l.components.contains_key("danger_seg"));
        assert!(l.components.contains_key("danger_dice"));
        assert!(!l.components.contains_key("person_heatmap"));
        // CE:Dice weights are 1:1
        let want = l.components["danger_seg"].double_value(&[]) + l.components["danger_dice"].double_value(&[]);
        assert!((l.total.unwrap().double_value(&[]) - want).abs() < 1e-6);
    }

    #[test]
    fn no_gt_means_no_loss() {
        let batch = Batch {
            pixel_values: Tensor::zeros([1, 4, 4, 4], cpu()),
            modality_mask: Tensor::from_slice(&[true, false]).reshape([1, 2]),
            danger_labels: None,
            person_heatmap: None,
            person_wh: None,
            person_reg_mask: None,
            person_offset: None,
        };
        let out = ModelOutput { danger_logits: Some(Tensor::zeros([1, 4, 4, 4], cpu())), ..Default::default() };
        assert!(compute_losses(&out, &batch, &LossConfig::default()).total.is_none());
        let z = zero_loss(&out, Device::Cpu);
        assert_eq!(z.double_value(&[]), 0.0);
        assert!(z.requires_grad());
    }

    #[test]
    fn freeze_callback_toggles_backbone_only() {
        let vs = nn::VarStore::new(Device::Cpu);
        let root = vs.root();
        let _ = (&root / "backbone").zeros("w", &[2]);
        let _ = (&root / "head").zeros("w", &[2]);

        let mut cb = FreezeBackboneCallback::new(2.0, vec!["backbone".into()]);
        let mut ctl = TrainerControl::default();
        let at = |epoch: f64| TrainerState { epoch, global_step: 0 };
        let grad = |name: &str| vs.variables()[name].requires_grad();

        cb.on_epoch_begin(&at(0.0), &mut ctl, &vs);
        assert!(cb.is_frozen());
        assert!(!grad("backbone.w") && grad("head.w"));
        cb.on_epoch_begin(&at(1.0), &mut ctl, &vs);
        assert!(!grad("backbone.w"));
        cb.on_epoch_begin(&at(2.0), &mut ctl, &vs);
        assert!(!cb.is_frozen() && grad("backbone.w"));
        // ending while frozen restores the model
        cb.on_epoch_begin(&at(0.0), &mut ctl, &vs);
        cb.on_train_end(&at(0.0), &mut ctl, &vs);
        assert!(grad("backbone.w"));
    }

    #[test]
    fn prune_keeps_newest_checkpoints() {
        let dir = std::env::temp_dir().join("skylens_prune_test");
        let _ = std::fs::remove_dir_all(&dir);
        for n in [10, 20, 30, 40] {
            std::fs::create_dir_all(dir.join(format!("checkpoint-{n}"))).unwrap();
        }
        prune_checkpoints(&dir, 2).unwrap();
        assert!(!dir.join("checkpoint-10").exists() && !dir.join("checkpoint-20").exists());
        assert!(dir.join("checkpoint-30").exists() && dir.join("checkpoint-40").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
