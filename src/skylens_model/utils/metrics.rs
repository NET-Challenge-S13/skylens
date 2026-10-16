//! Rust twin of `utils/metrics.py`.
//!
//! Evaluation metrics and the CenterNet peak decoder used at inference:
//!
//! - [`decode_heatmap_peaks`] / [`decode_gt_boxes`]: CenterNet decoding. Python
//!   runs these as torch ops; here they run on CPU slices (one image at a time)
//!   because the detector copies the small stride-4 maps back anyway.
//! - [`SegmentationMetrics`]: confusion-matrix mIoU.
//! - [`PointDetectionMetrics`], [`PointAveragePrecision`], [`point_match_flags`],
//!   [`size_relative_radius`], [`recall_sweep`]: point matching by pixel
//!   distance. The deployment radius is `max(8px, 0.15 * GT box height)`.
//! - [`BoxDetectionMetrics`] / [`box_iou_matrix`]: COCO-style box mAP
//!   (101-point interpolation), reported for comparison with the literature.
//!
//! Row layouts follow the Python arrays: detections are `[x, y, w, h, score]`
//! (centre format, [`Detection`]), point predictions `[x, y, score]`, point GTs
//! `[x, y]`, box GTs `[x, y, w, h]`.
//!
//! Differences from the Python implementation:
//!
//! - `decode_heatmap_peaks` returns only the surviving detections (score above
//!   the threshold and > 0), sorted by score, instead of a zero-padded
//!   `(B, k, 5)` tensor. Every Python consumer filters `score > 0` first.
//! - Ties are broken deterministically: equal scores keep ascending flat-index
//!   order in the decoder (torch `topk` leaves tie order unspecified), and
//!   `PointDetectionMetrics::update` uses a stable sort where numpy used the
//!   default (non-stable) quicksort. Results only differ on exact score ties.
//! - `build_compute_metrics` (the HF `Trainer` hook) is training-side and is not
//!   ported.

use std::collections::BTreeMap;

/// `[x, y, w, h, score]` in input-image pixels, centre format.
pub type Detection = [f64; 5];

/// COCO IoU thresholds 0.50:0.05:0.95 (`COCO_IOU_THRESHOLDS`).
pub fn coco_iou_thresholds() -> Vec<f64> {
    (0..10).map(|i| (50 + 5 * i) as f64 / 100.0).collect()
}

// --------------------------------------------------------------------------
// CenterNet heatmap decoding
// --------------------------------------------------------------------------

/// CenterNet peak decoding of one image (`decode_heatmap_peaks`).
///
/// `heatmap` is the sigmoid probability map `(h, w)` row-major. `wh` and
/// `offset` are `(2, h, w)` planar maps in grid units. A pixel survives the 3x3
/// max-pool NMS when it equals its neighbourhood maximum; the top `k` of those
/// with score `>= threshold` are returned as `[x, y, w, h, score]` in input
/// pixels. The centre is `(cell + offset) * stride`, or `(cell + 0.5) * stride`
/// without an offset map, or `cell * stride` with `legacy_decode`.
///
/// Coordinates are computed in `f32`, as the torch implementation does, and
/// widened to `f64` at the end (the `.numpy().astype(np.float64)` step).
#[allow(clippy::too_many_arguments)]
pub fn decode_heatmap_peaks(
    heatmap: &[f32],
    wh: Option<&[f32]>,
    offset: Option<&[f32]>,
    height: usize,
    width: usize,
    k: usize,
    threshold: f32,
    stride: f32,
    legacy_decode: bool,
) -> Vec<Detection> {
    let n = height * width;
    assert_eq!(heatmap.len(), n, "heatmap must be height * width");
    if let Some(w) = wh {
        assert_eq!(w.len(), 2 * n, "wh must be 2 * height * width");
    }
    if let Some(o) = offset {
        assert_eq!(o.len(), 2 * n, "offset must be 2 * height * width");
    }

    // 3x3 max-pool NMS (stride 1, padding 1, padding never wins).
    let mut peaks: Vec<(f32, usize)> = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let v = heatmap[y * width + x];
            let mut m = f32::NEG_INFINITY;
            for yy in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                for xx in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                    let u = heatmap[yy * width + xx];
                    if u > m || u.is_nan() {
                        m = u;
                    }
                }
            }
            // `torch.where(pooled == hm, hm, 0)`: non-peaks score 0.
            let score = if m == v { v } else { 0.0 };
            // `keep = scores >= threshold`, then consumers drop score == 0.
            if score >= threshold && score > 0.0 {
                peaks.push((score, y * width + x));
            }
        }
    }
    // topk(sorted=True): descending score; ties by ascending index.
    peaks.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    peaks.truncate(k);

    peaks
        .into_iter()
        .map(|(score, idx)| {
            let mut xs = (idx % width) as f32;
            let mut ys = (idx / width) as f32;
            if let (Some(o), false) = (offset, legacy_decode) {
                xs += o[idx];
                ys += o[n + idx];
            } else if !legacy_decode {
                xs += 0.5;
                ys += 0.5;
            }
            let (w, h) = match wh {
                Some(m) => (m[idx] * stride, m[n + idx] * stride),
                None => (0.0, 0.0),
            };
            [
                f64::from(xs * stride),
                f64::from(ys * stride),
                f64::from(w),
                f64::from(h),
                f64::from(score),
            ]
        })
        .collect()
}

/// Rebuilds GT boxes from the collator's CenterNet targets (`decode_gt_boxes`)
/// for one image. Returns `[x, y, w, h]` per centre where `reg_mask > 0`, in
/// flat-index order, at most `k` of them.
#[allow(clippy::too_many_arguments)]
pub fn decode_gt_boxes(
    reg_mask: &[f32],
    wh: &[f32],
    offset: Option<&[f32]>,
    height: usize,
    width: usize,
    k: usize,
    stride: f32,
    legacy_decode: bool,
) -> Vec<[f64; 4]> {
    let n = height * width;
    assert_eq!(reg_mask.len(), n);
    assert_eq!(wh.len(), 2 * n);
    (0..n)
        .filter(|&i| reg_mask[i] > 0.0)
        .take(k)
        .map(|idx| {
            let mut xs = (idx % width) as f32;
            let mut ys = (idx / width) as f32;
            if let Some(o) = offset {
                xs += o[idx];
                ys += o[n + idx];
            } else if !legacy_decode {
                xs += 0.5;
                ys += 0.5;
            }
            [
                f64::from(xs * stride),
                f64::from(ys * stride),
                f64::from(wh[idx] * stride),
                f64::from(wh[n + idx] * stride),
            ]
        })
        .collect()
}

// --------------------------------------------------------------------------
// Segmentation
// --------------------------------------------------------------------------

/// Accumulates a confusion matrix (rows GT, columns prediction) and reports
/// mIoU, per-class IoU and pixel accuracy. `ignore_index` never enters it.
#[derive(Debug, Clone)]
pub struct SegmentationMetrics {
    pub num_classes: usize,
    pub ignore_index: i64,
    pub confusion: Vec<u64>,
}

impl SegmentationMetrics {
    pub fn new(num_classes: usize, ignore_index: i64) -> Self {
        assert!(num_classes >= 1, "num_classes >= 1");
        Self {
            num_classes,
            ignore_index,
            confusion: vec![0; num_classes * num_classes],
        }
    }

    pub fn reset(&mut self) {
        self.confusion.iter_mut().for_each(|c| *c = 0);
    }

    /// `preds` and `targets` are class-index maps of the same length.
    pub fn update(&mut self, preds: &[i64], targets: &[i64]) {
        assert_eq!(preds.len(), targets.len(), "preds/targets size mismatch");
        let c = self.num_classes as i64;
        for (&p, &t) in preds.iter().zip(targets) {
            let valid = t != self.ignore_index && (0..c).contains(&t) && (0..c).contains(&p);
            if valid {
                self.confusion[(t * c + p) as usize] += 1;
            }
        }
    }

    /// Keys: `miou`, `pixel_accuracy`, `iou_class_{c}` (NaN when the class
    /// never appears in GT or prediction).
    pub fn compute(&self) -> BTreeMap<String, f64> {
        let c = self.num_classes;
        let cm = |r: usize, k: usize| self.confusion[r * c + k] as f64;
        let mut ious = Vec::with_capacity(c);
        let (mut tp_sum, mut total) = (0.0, 0.0);
        for i in 0..c {
            let tp = cm(i, i);
            let gt: f64 = (0..c).map(|k| cm(i, k)).sum();
            let pred: f64 = (0..c).map(|r| cm(r, i)).sum();
            let union = gt + pred - tp;
            tp_sum += tp;
            total += gt;
            ious.push(if union > 0.0 {
                Some(tp / union.max(1.0))
            } else {
                None
            });
        }
        let present: Vec<f64> = ious.iter().flatten().copied().collect();
        let mut out = BTreeMap::new();
        out.insert(
            "miou".to_string(),
            if present.is_empty() {
                0.0
            } else {
                present.iter().sum::<f64>() / present.len() as f64
            },
        );
        out.insert(
            "pixel_accuracy".to_string(),
            if total > 0.0 { tp_sum / total } else { 0.0 },
        );
        for (i, iou) in ious.iter().enumerate() {
            out.insert(format!("iou_class_{i}"), iou.unwrap_or(f64::NAN));
        }
        out
    }
}

// --------------------------------------------------------------------------
// Point detection
// --------------------------------------------------------------------------

/// Indices of `scores` in descending order, stable on ties
/// (`np.argsort(-scores, kind="stable")`).
fn argsort_desc_stable(scores: impl Iterator<Item = f64>) -> Vec<usize> {
    let s: Vec<f64> = scores.collect();
    let mut order: Vec<usize> = (0..s.len()).collect();
    order.sort_by(|&a, &b| s[b].total_cmp(&s[a]));
    order
}

/// Nearest unused GT for prediction `p`, with its distance
/// (`np.argmin` over distances with used GTs set to inf, first minimum wins).
fn nearest_unused(p: &[f64; 3], gts: &[[f64; 2]], used: &[bool]) -> Option<(usize, f64)> {
    let mut best: Option<(usize, f64)> = None;
    for (j, g) in gts.iter().enumerate() {
        if used[j] {
            continue;
        }
        let d = (g[0] - p[0]).hypot(g[1] - p[1]);
        if best.is_none_or(|(_, bd)| d < bd) {
            best = Some((j, d));
        }
    }
    best.filter(|(_, d)| d.is_finite())
}

/// Greedy point P/R/F1 at a fixed pixel distance.
///
/// Predictions in descending score each take the nearest unused GT; within
/// `distance_threshold` it is a TP, otherwise a FP. Unmatched GTs are FNs.
#[derive(Debug, Clone, Default)]
pub struct PointDetectionMetrics {
    pub distance_threshold: f64,
    pub tp: u64,
    pub fp: u64,
    pub fn_: u64,
}

impl PointDetectionMetrics {
    pub fn new(distance_threshold: f64) -> Self {
        Self {
            distance_threshold,
            ..Default::default()
        }
    }

    pub fn reset(&mut self) {
        self.tp = 0;
        self.fp = 0;
        self.fn_ = 0;
    }

    /// One image: predictions `[x, y, score]`, GTs `[x, y]`.
    pub fn update(&mut self, preds: &[[f64; 3]], gts: &[[f64; 2]]) {
        if preds.is_empty() {
            self.fn_ += gts.len() as u64;
            return;
        }
        if gts.is_empty() {
            self.fp += preds.len() as u64;
            return;
        }
        let flags = point_match_flags(preds, gts, &Radius::Fixed(self.distance_threshold));
        let tp = flags.iter().filter(|&&f| f).count() as u64;
        self.tp += tp;
        self.fp += preds.len() as u64 - tp;
        self.fn_ += gts.len() as u64 - tp;
    }

    /// Keys: `point_precision`, `point_recall`, `point_f1`, `point_tp`,
    /// `point_fp`, `point_fn`.
    pub fn compute(&self) -> BTreeMap<String, f64> {
        let (tp, fp, fn_) = (self.tp as f64, self.fp as f64, self.fn_ as f64);
        let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        let recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        BTreeMap::from([
            ("point_precision".to_string(), precision),
            ("point_recall".to_string(), recall),
            ("point_f1".to_string(), f1),
            ("point_tp".to_string(), tp),
            ("point_fp".to_string(), fp),
            ("point_fn".to_string(), fn_),
        ])
    }
}

/// Match radius: one value for every GT, or one per GT.
#[derive(Debug, Clone, PartialEq)]
pub enum Radius {
    Fixed(f64),
    PerGt(Vec<f64>),
}

impl Radius {
    fn at(&self, j: usize) -> f64 {
        match self {
            Radius::Fixed(r) => *r,
            Radius::PerGt(r) => r[j],
        }
    }
}

/// Per-GT radius `max(min_px, frac * height)` (`size_relative_radius`).
/// The deployment protocol uses `frac = 0.15`, `min_px = 8`.
pub fn size_relative_radius(heights: &[f64], frac: f64, min_px: f64) -> Vec<f64> {
    heights.iter().map(|&h| min_px.max(frac * h)).collect()
}

/// TP flag per prediction under the greedy rule (`point_match_flags`).
///
/// Predictions are visited in descending score (stable), each takes the nearest
/// unused GT and is a TP iff that distance is within *that GT's* radius.
/// Flags align with `preds`.
pub fn point_match_flags(preds: &[[f64; 3]], gts: &[[f64; 2]], radius: &Radius) -> Vec<bool> {
    let mut flags = vec![false; preds.len()];
    if preds.is_empty() || gts.is_empty() {
        return flags;
    }
    if let Radius::PerGt(r) = radius {
        assert_eq!(r.len(), gts.len(), "one radius per GT");
    }
    let mut used = vec![false; gts.len()];
    for i in argsort_desc_stable(preds.iter().map(|p| p[2])) {
        if let Some((j, d)) = nearest_unused(&preds[i], gts, &used)
            && d <= radius.at(j)
        {
            used[j] = true;
            flags[i] = true;
        }
    }
    flags
}

// --------------------------------------------------------------------------
// Average precision
// --------------------------------------------------------------------------

/// 101-point interpolated AP, COCO convention (`_average_precision`).
pub fn average_precision(scores: &[f64], is_tp: &[bool], num_gt: usize) -> f64 {
    assert_eq!(scores.len(), is_tp.len());
    if num_gt == 0 || scores.is_empty() {
        return 0.0;
    }
    let order = argsort_desc_stable(scores.iter().copied());
    let mut recall = Vec::with_capacity(order.len());
    let mut precision = Vec::with_capacity(order.len());
    let (mut tp_cum, mut fp_cum) = (0.0f64, 0.0f64);
    for &i in &order {
        if is_tp[i] {
            tp_cum += 1.0;
        } else {
            fp_cum += 1.0;
        }
        recall.push(tp_cum / num_gt as f64);
        precision.push(tp_cum / (tp_cum + fp_cum).max(f64::EPSILON));
    }
    // Precision envelope: running max from the back.
    for i in (0..precision.len().saturating_sub(1)).rev() {
        precision[i] = precision[i].max(precision[i + 1]);
    }
    // np.linspace(0, 1, 101): i * (1/100), last point exactly 1.0.
    let step = 1.0 / 100.0;
    let mut total = 0.0;
    for i in 0..=100 {
        let rp = if i == 100 { 1.0 } else { i as f64 * step };
        // searchsorted(side="left"): first index with recall >= rp.
        let idx = recall.partition_point(|&r| r < rp);
        if idx < precision.len() {
            total += precision[idx];
        }
    }
    total / 101.0
}

/// IoU of two centre-format boxes `[x, y, w, h]` (one cell of
/// [`box_iou_matrix`]).
pub fn box_iou(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let xyxy = |t: &[f64; 4]| {
        [
            t[0] - t[2] / 2.0,
            t[1] - t[3] / 2.0,
            t[0] + t[2] / 2.0,
            t[1] + t[3] / 2.0,
        ]
    };
    let area = |t: &[f64; 4]| (t[2] - t[0]).max(0.0) * (t[3] - t[1]).max(0.0);
    let (aa, bb) = (xyxy(a), xyxy(b));
    let iw = (aa[2].min(bb[2]) - aa[0].max(bb[0])).max(0.0);
    let ih = (aa[3].min(bb[3]) - aa[1].max(bb[1])).max(0.0);
    let inter = iw * ih;
    let union = area(&aa) + area(&bb) - inter;
    if union > 0.0 {
        inter / union.max(f64::EPSILON)
    } else {
        0.0
    }
}

/// IoU matrix of two sets of centre-format boxes `[x, y, w, h]`.
pub fn box_iou_matrix(a: &[[f64; 4]], b: &[[f64; 4]]) -> Vec<Vec<f64>> {
    a.iter()
        .map(|ra| b.iter().map(|rb| box_iou(ra, rb)).collect())
        .collect()
}

/// Score-descending greedy matching (`_greedy_match`). `fitness[i][j]` is
/// larger-is-better; a pair matches only where `keep[i][j]`.
fn greedy_match(
    fitness: &[Vec<f64>],
    order: &[usize],
    keep: &[Vec<bool>],
    n_gt: usize,
) -> Vec<bool> {
    let mut tp = vec![false; fitness.len()];
    let mut used = vec![false; n_gt];
    if n_gt == 0 {
        return tp;
    }
    for &i in order {
        // np.argmax over the row with used GTs at -inf: first maximum wins.
        let mut j = 0;
        let mut best = f64::NEG_INFINITY;
        for (g, &v) in fitness[i].iter().enumerate() {
            let v = if used[g] { f64::NEG_INFINITY } else { v };
            if v > best {
                best = v;
                j = g;
            }
        }
        if best.is_finite() && keep[i][j] {
            used[j] = true;
            tp[i] = true;
        }
    }
    tp
}

/// Collects (score, TP flags per threshold) over images (`_APAccumulator`).
#[derive(Debug, Clone)]
struct ApAccumulator {
    scores: Vec<f64>,
    tp: Vec<Vec<bool>>,
    num_gt: usize,
}

impl ApAccumulator {
    fn new(num_thresholds: usize) -> Self {
        Self {
            scores: Vec::new(),
            tp: vec![Vec::new(); num_thresholds],
            num_gt: 0,
        }
    }

    fn reset(&mut self) {
        self.scores.clear();
        self.tp.iter_mut().for_each(Vec::clear);
        self.num_gt = 0;
    }

    fn add(&mut self, scores: &[f64], tp_per_threshold: &[Vec<bool>], n_gt: usize) {
        self.scores.extend_from_slice(scores);
        for (acc, tp) in self.tp.iter_mut().zip(tp_per_threshold) {
            acc.extend_from_slice(tp);
        }
        self.num_gt += n_gt;
    }

    fn ap_per_threshold(&self) -> Vec<f64> {
        self.tp
            .iter()
            .map(|tp| average_precision(&self.scores, tp, self.num_gt))
            .collect()
    }
}

/// COCO-style single-class (person) box mAP, 101-point interpolation.
#[derive(Debug, Clone)]
pub struct BoxDetectionMetrics {
    pub iou_thresholds: Vec<f64>,
    acc: ApAccumulator,
}

impl Default for BoxDetectionMetrics {
    fn default() -> Self {
        Self::new(coco_iou_thresholds())
    }
}

impl BoxDetectionMetrics {
    pub fn new(iou_thresholds: Vec<f64>) -> Self {
        assert!(
            !iou_thresholds.is_empty(),
            "iou_thresholds must not be empty"
        );
        let acc = ApAccumulator::new(iou_thresholds.len());
        Self {
            iou_thresholds,
            acc,
        }
    }

    pub fn reset(&mut self) {
        self.acc.reset();
    }

    /// One image: detections `[x, y, w, h, score]`, GTs `[x, y, w, h]`.
    pub fn update(&mut self, preds: &[Detection], gts: &[[f64; 4]]) {
        let n_gt = gts.len();
        let nt = self.iou_thresholds.len();
        if preds.is_empty() {
            self.acc.add(&[], &vec![Vec::new(); nt], n_gt);
            return;
        }
        let scores: Vec<f64> = preds.iter().map(|p| p[4]).collect();
        let order = argsort_desc_stable(scores.iter().copied());
        let boxes: Vec<[f64; 4]> = preds.iter().map(|p| [p[0], p[1], p[2], p[3]]).collect();
        let iou = box_iou_matrix(&boxes, gts);
        let tps: Vec<Vec<bool>> = self
            .iou_thresholds
            .iter()
            .map(|&thr| {
                let keep: Vec<Vec<bool>> = iou
                    .iter()
                    .map(|row| row.iter().map(|&v| v >= thr).collect())
                    .collect();
                greedy_match(&iou, &order, &keep, n_gt)
            })
            .collect();
        self.acc.add(&scores, &tps, n_gt);
    }

    /// Keys: `map_50_95`, `map_50`, `map_75` (when 0.75 is a threshold).
    pub fn compute(&self) -> BTreeMap<String, f64> {
        let aps = self.acc.ap_per_threshold();
        let mut out = BTreeMap::new();
        out.insert(
            "map_50_95".to_string(),
            aps.iter().sum::<f64>() / aps.len() as f64,
        );
        for (thr, ap) in self.iou_thresholds.iter().zip(&aps) {
            if (thr - 0.5).abs() < 1e-9 {
                out.insert("map_50".to_string(), *ap);
            } else if (thr - 0.75).abs() < 1e-9 {
                out.insert("map_75".to_string(), *ap);
            }
        }
        out.entry("map_50".to_string()).or_insert(aps[0]);
        out
    }
}

/// Distance-matched AP: [`PointDetectionMetrics`] without a single score
/// threshold, integrated over the whole PR curve (101-point).
#[derive(Debug, Clone)]
pub struct PointAveragePrecision {
    pub radius: Radius,
    acc: ApAccumulator,
}

impl PointAveragePrecision {
    pub fn new(distance_threshold: f64) -> Self {
        Self {
            radius: Radius::Fixed(distance_threshold),
            acc: ApAccumulator::new(1),
        }
    }

    pub fn reset(&mut self) {
        self.acc.reset();
    }

    /// One image: predictions `[x, y, score]`, GTs `[x, y]`.
    pub fn update(&mut self, preds: &[[f64; 3]], gts: &[[f64; 2]]) {
        let radius = self.radius.clone();
        self.update_with_radius(preds, gts, &radius);
    }

    /// Same as [`Self::update`] with an explicit (possibly per-GT) radius, the
    /// size-relative protocol of `eval_deploy.py`.
    pub fn update_with_radius(&mut self, preds: &[[f64; 3]], gts: &[[f64; 2]], radius: &Radius) {
        let n_gt = gts.len();
        let scores: Vec<f64> = preds.iter().map(|p| p[2]).collect();
        if preds.is_empty() || n_gt == 0 {
            self.acc.add(&scores, &[vec![false; preds.len()]], n_gt);
            return;
        }
        // Greedy rule of `_greedy_match(-dist, order, dist <= r)`; identical to
        // `point_match_flags` (nearest unused GT, then its radius).
        let tp = point_match_flags(preds, gts, radius);
        self.acc.add(&scores, &[tp], n_gt);
    }

    /// Key: `point_ap`.
    pub fn compute(&self) -> BTreeMap<String, f64> {
        BTreeMap::from([("point_ap".to_string(), self.acc.ap_per_threshold()[0])])
    }
}

/// Recall summaries of a point PR sweep (`recall_sweep`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecallSweep {
    /// Highest recall over the thresholds with precision `>= min_precision`.
    pub recall_at_p50: f64,
    /// Threshold where it was reached (NaN when none qualified).
    pub recall_at_p50_thr: f64,
    /// Recall at the lowest threshold of the sweep.
    pub recall_max: f64,
}

/// Default sweep 0.05..0.95 step 0.01 (`np.round(np.arange(5, 96) / 100, 2)`).
pub fn default_recall_thresholds() -> Vec<f64> {
    (5..96).map(|i| i as f64 / 100.0).collect()
}

pub fn recall_sweep(
    scores: &[f64],
    is_tp: &[bool],
    num_gt: usize,
    thresholds: &[f64],
    min_precision: f64,
) -> RecallSweep {
    assert_eq!(scores.len(), is_tp.len());
    let (mut best, mut best_thr) = (0.0, f64::NAN);
    let mut recalls = Vec::with_capacity(thresholds.len());
    for &th in thresholds {
        let mut tp = 0usize;
        let mut n = 0usize;
        for (&s, &t) in scores.iter().zip(is_tp) {
            if s >= th - 1e-9 {
                n += 1;
                tp += usize::from(t);
            }
        }
        let r = if num_gt > 0 {
            tp as f64 / num_gt as f64
        } else {
            0.0
        };
        let prec = if n > 0 { tp as f64 / n as f64 } else { 0.0 };
        recalls.push(r);
        if prec >= min_precision && r > best {
            best = r;
            best_thr = th;
        }
    }
    // np.argmin: first minimum threshold.
    let recall_max = thresholds
        .iter()
        .enumerate()
        .fold(None::<(usize, f64)>, |acc, (i, &t)| match acc {
            Some((_, bt)) if t >= bt => acc,
            _ => Some((i, t)),
        })
        .map_or(0.0, |(i, _)| recalls[i]);
    RecallSweep {
        recall_at_p50: best,
        recall_at_p50_thr: best_thr,
        recall_max,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Three predictions (0.9 / 0.8 / 0.7); only #1 and #3 hit the two GTs.
    // PR: (r=0.5, p=1) -> (0.5, 0.5) -> (1.0, 2/3); envelope 1.0 for 51 points,
    // 2/3 for 50 points.
    const AP_MIXED: f64 = (51.0 * 1.0 + 50.0 * (2.0 / 3.0)) / 101.0;
    const PREDS: [Detection; 3] = [
        [10.0, 10.0, 10.0, 10.0, 0.9],
        [200.0, 200.0, 10.0, 10.0, 0.8],
        [50.0, 50.0, 10.0, 10.0, 0.7],
    ];
    const GTS: [[f64; 4]; 2] = [[10.0, 10.0, 10.0, 10.0], [50.0, 50.0, 10.0, 10.0]];

    fn pts(d: &[Detection]) -> Vec<[f64; 3]> {
        d.iter().map(|p| [p[0], p[1], p[4]]).collect()
    }

    fn gpts(g: &[[f64; 4]]) -> Vec<[f64; 2]> {
        g.iter().map(|p| [p[0], p[1]]).collect()
    }

    // ---- test_metrics.py ----

    #[test]
    fn box_iou_matrix_hand_values() {
        let a = [[0.0, 0.0, 10.0, 10.0]];
        assert_eq!(box_iou_matrix(&a, &a)[0][0], 1.0);
        assert_eq!(box_iou_matrix(&a, &[[100.0, 100.0, 10.0, 10.0]])[0][0], 0.0);
        assert_eq!(
            box_iou_matrix(&a, &[[5.0, 0.0, 10.0, 10.0]])[0][0],
            1.0 / 3.0
        );
        assert_eq!(box_iou_matrix(&a, &[[0.0, 0.0, 20.0, 10.0]])[0][0], 0.5);
    }

    #[test]
    fn box_map_matches_hand_calculation() {
        let mut m = BoxDetectionMetrics::default();
        m.update(&PREDS, &GTS);
        let out = m.compute();
        assert!((out["map_50"] - AP_MIXED).abs() < 1e-12);
        assert!((out["map_75"] - AP_MIXED).abs() < 1e-12);
        assert!((out["map_50_95"] - AP_MIXED).abs() < 1e-12);
    }

    #[test]
    fn box_map_perfect_and_zero() {
        let mut perfect = BoxDetectionMetrics::default();
        perfect.update(&[PREDS[0], PREDS[2]], &GTS);
        assert!((perfect.compute()["map_50_95"] - 1.0).abs() < 1e-12);

        let mut miss = BoxDetectionMetrics::default();
        miss.update(&[[900.0, 900.0, 10.0, 10.0, 0.9]], &GTS);
        assert_eq!(miss.compute()["map_50_95"], 0.0);
    }

    #[test]
    fn box_map_iou_thresholds_discriminate() {
        // IoU = 2/3 -> TP only at 0.50..0.65 -> mAP@0.5:0.95 = 0.4
        let mut m = BoxDetectionMetrics::default();
        m.update(&[[2.0, 0.0, 10.0, 10.0, 0.9]], &[[0.0, 0.0, 10.0, 10.0]]);
        let out = m.compute();
        assert_eq!(out["map_50"], 1.0);
        assert_eq!(out["map_75"], 0.0);
        assert!((out["map_50_95"] - 0.4).abs() < 1e-12);
    }

    #[test]
    fn point_average_precision() {
        let mut p = PointAveragePrecision::new(8.0);
        p.update(&pts(&PREDS), &gpts(&GTS));
        assert!((p.compute()["point_ap"] - AP_MIXED).abs() < 1e-12);
    }

    #[test]
    fn decode_gt_boxes_cell_centre_and_legacy() {
        let (h, w) = (8, 8);
        let mut reg = vec![0.0f32; h * w];
        let mut wh = vec![0.0f32; 2 * h * w];
        let i = 3 * w + 2;
        reg[i] = 1.0;
        wh[i] = 5.0;
        wh[h * w + i] = 7.0;
        let boxes = decode_gt_boxes(&reg, &wh, None, h, w, 4, 4.0, false);
        assert_eq!(boxes, vec![[10.0, 14.0, 20.0, 28.0]]);
        let legacy = decode_gt_boxes(&reg, &wh, None, h, w, 4, 4.0, true);
        assert_eq!(legacy, vec![[8.0, 12.0, 20.0, 28.0]]);
    }

    #[test]
    fn decode_heatmap_single_peak_at_cell_centre() {
        let (h, w) = (8, 8);
        let mut hm = vec![0.0f32; h * w];
        hm[5 * w + 3] = 0.9; // y=5, x=3
        let det = decode_heatmap_peaks(&hm, None, None, h, w, 3, 0.3, 4.0, false);
        assert_eq!(det.len(), 1);
        assert_eq!([det[0][0], det[0][1]], [14.0, 22.0]);
        assert!((det[0][4] - 0.9).abs() < 1e-6);
        // Round trip with the int(c / stride) encoding: error <= stride / 2.
        for c in [12.0f64, 13.9, 15.99] {
            assert_eq!((c / 4.0) as i64, 3);
            assert!((c - det[0][0]).abs() <= 2.0);
        }
        let legacy = decode_heatmap_peaks(&hm, None, None, h, w, 3, 0.3, 4.0, true);
        assert_eq!([legacy[0][0], legacy[0][1]], [12.0, 20.0]);
    }

    #[test]
    fn decode_heatmap_offset_overrides_half_cell() {
        let (h, w) = (8, 8);
        let n = h * w;
        let mut hm = vec![0.0f32; n];
        let i = 5 * w + 3;
        hm[i] = 0.9;
        let mut off = vec![0.0f32; 2 * n];
        off[i] = 0.25;
        off[n + i] = 0.75;
        let det = decode_heatmap_peaks(&hm, None, Some(&off), h, w, 1, 0.3, 4.0, false);
        assert_eq!([det[0][0], det[0][1]], [13.0, 23.0]); // (3.25*4, 5.75*4)
    }

    #[test]
    fn decode_heatmap_nms_threshold_and_topk() {
        let (h, w) = (6, 6);
        let mut hm = vec![0.0f32; h * w];
        hm[2 * w + 2] = 0.8; // peak
        hm[2 * w + 3] = 0.7; // suppressed by its neighbour
        hm[5 * w + 5] = 0.6; // peak in the corner
        hm[0] = 0.2; // under the threshold
        let det = decode_heatmap_peaks(&hm, None, None, h, w, 10, 0.3, 4.0, false);
        let scores: Vec<f64> = det.iter().map(|d| d[4]).collect();
        assert_eq!(det.len(), 2);
        assert!((scores[0] - 0.8).abs() < 1e-6 && (scores[1] - 0.6).abs() < 1e-6);
        assert_eq!(
            decode_heatmap_peaks(&hm, None, None, h, w, 1, 0.3, 4.0, false).len(),
            1
        );
    }

    #[test]
    fn segmentation_miou_ignores_and_skips_absent_classes() {
        let mut m = SegmentationMetrics::new(4, 255);
        m.update(&[0, 0, 1, 1, 3], &[0, 1, 1, 1, 255]);
        let out = m.compute();
        // class 0: tp 1, gt 1, pred 2 -> 1/2; class 1: tp 2, gt 3, pred 2 -> 2/3
        assert!((out["iou_class_0"] - 0.5).abs() < 1e-12);
        assert!((out["iou_class_1"] - 2.0 / 3.0).abs() < 1e-12);
        assert!(out["iou_class_2"].is_nan() && out["iou_class_3"].is_nan());
        assert!((out["miou"] - (0.5 + 2.0 / 3.0) / 2.0).abs() < 1e-12);
        assert!((out["pixel_accuracy"] - 0.75).abs() < 1e-12);
    }

    #[test]
    fn point_detection_metrics_counts() {
        let mut m = PointDetectionMetrics::new(8.0);
        m.update(&pts(&PREDS), &gpts(&GTS));
        let out = m.compute();
        assert_eq!(
            (out["point_tp"], out["point_fp"], out["point_fn"]),
            (2.0, 1.0, 0.0)
        );
        m.update(&[], &[[0.0, 0.0]]);
        assert_eq!(m.fn_, 1);
    }

    // ---- test_size_relative_match.py ----

    /// Small deterministic generator for the randomised parity test.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    #[test]
    fn per_gt_radius_of_8_equals_scalar_matching() {
        let mut rng = Lcg(1);
        for _ in 0..20 {
            let preds: Vec<[f64; 3]> = (0..60)
                .map(|_| {
                    [
                        rng.next() * 120.0,
                        rng.next() * 120.0,
                        0.05 + rng.next() * 0.95,
                    ]
                })
                .collect();
            let gts: Vec<[f64; 2]> = (0..35)
                .map(|_| [rng.next() * 120.0, rng.next() * 120.0])
                .collect();
            let old = point_match_flags(&preds, &gts, &Radius::Fixed(8.0));
            let new = point_match_flags(&preds, &gts, &Radius::PerGt(vec![8.0; gts.len()]));
            assert_eq!(old, new);
        }
    }

    #[test]
    fn large_gt_matched_at_12px_passes_rel15_only() {
        let gt = [[100.0, 100.0]];
        let heights = [120.0]; // r = max(8, 18) = 18
        let pred = [[112.0, 100.0, 0.9]];
        assert!(!point_match_flags(&pred, &gt, &Radius::Fixed(8.0))[0]);
        let rel = Radius::PerGt(size_relative_radius(&heights, 0.15, 8.0));
        assert!(point_match_flags(&pred, &gt, &rel)[0]);
        // small boxes keep the 8px floor
        assert_eq!(size_relative_radius(&[20.0], 0.15, 8.0)[0], 8.0);
    }

    #[test]
    fn recall_sweep_summaries() {
        let s = [0.9, 0.8, 0.5, 0.35, 0.2, 0.1];
        let f = [true, true, false, true, false, true];
        let r = recall_sweep(&s, &f, 5, &[0.1, 0.3], 0.5);
        assert_eq!(r.recall_max, 0.8);
        assert_eq!(r.recall_at_p50, 0.8);
        assert_eq!(r.recall_at_p50_thr, 0.1);
        assert_eq!(default_recall_thresholds().len(), 91);
    }

    // ---- test_offset_head.py::test_decode_with_offset_recovers_subcell_centre ----

    #[test]
    fn decode_with_offset_recovers_subcell_centre() {
        // true centre (9.0, 14.5) px, w=6 h=9 at stride 4
        let (h, w) = (8, 8);
        let n = h * w;
        let i = 3 * w + 2;
        let mut hm = vec![0.0f32; n];
        hm[i] = 0.9;
        let mut wh = vec![0.0f32; 2 * n];
        wh[i] = 1.5;
        wh[n + i] = 2.25;
        let mut off = vec![0.0f32; 2 * n];
        off[i] = 0.25;
        off[n + i] = 0.625;
        let det = decode_heatmap_peaks(&hm, Some(&wh), Some(&off), h, w, 2, 0.3, 4.0, false);
        assert_eq!(det.len(), 1);
        assert_eq!(&det[0][..4], &[9.0, 14.5, 6.0, 9.0]);
        // the GT decode with the collator target yields the same continuous box
        let mut reg = vec![0.0f32; n];
        reg[i] = 1.0;
        let gt = decode_gt_boxes(&reg, &wh, Some(&off), h, w, 2, 4.0, false);
        assert_eq!(gt, vec![[9.0, 14.5, 6.0, 9.0]]);
    }
}
