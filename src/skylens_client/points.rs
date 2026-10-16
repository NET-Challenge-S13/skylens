//! What the board reads off the point chunks it holds (pointblitz chunk v2): sampled positions,
//! height percentiles, a point-size estimate, and the height-ramp recolouring of the point panel
//! (legacy `pointLayer.ts`). The same recolouring for point slot messages (decision 0052).
//!
//! Positions are quantised to each chunk's box (`origin + bbox_min + q / 65535 * extent`), so they
//! are read without touching anything but the 6 position bytes of each record.

use crate::glam::DVec3;
use crate::pointblitz::io::chunk::{ChunkHeader, HEADER_LEN, decode_header, rgb565};
use crate::pointblitz::io::slots;

/// Absolute ENU position of record `i` of a chunk.
fn position(h: &ChunkHeader, body: &[u8], i: usize) -> DVec3 {
    let at = i * usize::from(h.stride);
    let q = |k: usize| u16::from_le_bytes([body[at + 2 * k], body[at + 2 * k + 1]]);
    let rel = h.dequantise([q(0), q(1), q(2)]);
    DVec3::new(
        h.origin[0] + f64::from(rel[0]),
        h.origin[1] + f64::from(rel[1]),
        h.origin[2] + f64::from(rel[2]),
    )
}

/// The header and the records of a chunk, if it is valid and complete.
fn records(bytes: &[u8]) -> Option<(ChunkHeader, &[u8])> {
    let h = decode_header(bytes).ok()?;
    let body = bytes.get(HEADER_LEN..)?;
    let n = h.point_count as usize;
    (body.len() >= n * usize::from(h.stride)).then_some((h, body))
}

/// Up to `max` positions spread evenly over a chunk's records.
pub fn sample_chunk(bytes: &[u8], max: usize) -> Vec<DVec3> {
    let Some((h, body)) = records(bytes) else { return Vec::new() };
    let n = h.point_count as usize;
    if n == 0 || max == 0 {
        return Vec::new();
    }
    let take = n.min(max);
    (0..take).map(|k| position(&h, body, k * n / take)).collect()
}

/// Up to `max` positions spread evenly over several chunks (in proportion to their size).
pub fn sample_chunks<'a>(chunks: impl IntoIterator<Item = &'a [u8]>, max: usize) -> Vec<DVec3> {
    let chunks: Vec<&[u8]> = chunks.into_iter().collect();
    let total: u64 = chunks.iter().filter_map(|c| decode_header(c).ok()).map(|h| u64::from(h.point_count)).sum();
    if total == 0 || max == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for c in chunks {
        let Ok(h) = decode_header(c) else { continue };
        let share = ((u64::from(h.point_count) * max as u64).div_ceil(total)) as usize;
        out.extend(sample_chunk(c, share.max(1)));
    }
    out
}

/// The value at quantile `q` (0..1) of an ascending slice.
pub fn quantile(sorted: &[f64], q: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let i = ((q * sorted.len() as f64).floor() as usize).min(sorted.len() - 1);
    Some(sorted[i])
}

/// 1st and 99th percentile of the heights (legacy `zLow` / `zHigh`): neither a hard-coded ground
/// level nor a few stray points set the colour scale.
pub fn height_range(samples: &[DVec3]) -> Option<(f64, f64)> {
    let mut z: Vec<f64> = samples.iter().map(|p| p.z).filter(|v| v.is_finite()).collect();
    z.sort_by(f64::total_cmp);
    Some((quantile(&z, 0.01)?, quantile(&z, 0.99)?))
}

/// Box of the 1st..99th percentile core per axis (legacy `cloud.core`), for framing.
pub fn core_box(samples: &[DVec3]) -> Option<(DVec3, DVec3)> {
    let axis = |k: usize| {
        let mut v: Vec<f64> = samples.iter().map(|p| p[k]).filter(|v| v.is_finite()).collect();
        v.sort_by(f64::total_cmp);
        Some((quantile(&v, 0.01)?, quantile(&v, 0.99)?))
    };
    let (x, y, z) = (axis(0)?, axis(1)?, axis(2)?);
    Some((DVec3::new(x.0, y.0, z.0), DVec3::new(x.1, y.1, z.1)))
}

/// Point diameter in metres for a delivery: 1.5 x the typical spacing, the legacy `voxel * 1.5`
/// rule with the spacing measured from the chunks (median over chunks of
/// `sqrt(footprint / points)`), within 5 cm .. 1 m. 0.15 m (the legacy default) when unknown.
pub fn point_size_m<'a>(chunks: impl IntoIterator<Item = &'a [u8]>) -> f32 {
    point_size_from_spacings(chunk_spacings(chunks))
}

/// `sqrt(footprint / points)` of every chunk with 16 points or more.
pub fn chunk_spacings<'a>(chunks: impl IntoIterator<Item = &'a [u8]>) -> Vec<f64> {
    chunks
        .into_iter()
        .filter_map(|c| decode_header(c).ok())
        .filter_map(|h| {
            let lo = DVec3::new(f64::from(h.bbox_min[0]), f64::from(h.bbox_min[1]), 0.0);
            let hi = DVec3::new(f64::from(h.bbox_max[0]), f64::from(h.bbox_max[1]), 0.0);
            spacing((lo, hi), u64::from(h.point_count))
        })
        .collect()
}

/// Typical spacing of `points` points over a box's footprint (x, y): `sqrt(area / points)`, for
/// 16 points or more over a non-empty footprint.
pub fn spacing((lo, hi): (DVec3, DVec3), points: u64) -> Option<f64> {
    let area = (hi.x - lo.x) * (hi.y - lo.y);
    (points >= 16 && area > 0.0).then(|| (area / points as f64).sqrt())
}

/// The point diameter from per-delivery spacings (see [`point_size_m`]).
pub fn point_size_from_spacings(mut spacing: Vec<f64>) -> f32 {
    if spacing.is_empty() {
        return 0.15;
    }
    spacing.sort_by(f64::total_cmp);
    let median = spacing[spacing.len() / 2];
    (median * 1.5).clamp(0.05, 1.0) as f32
}

/// The legacy height ramp (blue - cyan - green - yellow - red), `t` in 0..1.
pub fn ramp(t: f64) -> [f64; 3] {
    let t = t.clamp(0.0, 1.0);
    let c = |v: f64| v.clamp(0.0, 1.0);
    [c(1.5 - (4.0 * t - 3.0).abs()), c(1.5 - (4.0 * t - 2.0).abs()), c(1.5 - (4.0 * t - 1.0).abs())]
}

/// Shading factor of the legacy point shader for a point without a normal:
/// `0.55 + 0.45 * 0.8`.
pub const UNLIT_SHADE: f64 = 0.55 + 0.45 * 0.8;

/// Colour bytes of a point at height `z` over the range `(low, high)`.
pub fn height_color(z: f64, low: f64, high: f64) -> [u8; 3] {
    let h = (z - low) / (high - low).max(0.01);
    ramp(h).map(|v| (v * UNLIT_SHADE * 255.0).round().clamp(0.0, 255.0) as u8)
}

/// The legacy legend bar: `linear-gradient(to top, rgb(r,g,b) 0%, ... rgb(r,g,b) 100%)` over nine
/// stops of the ramp (`pointInset.ts` `legendGradient`).
pub fn legend_gradient() -> String {
    let stops: Vec<String> = (0..=8)
        .map(|i| {
            let [r, g, b] = ramp(f64::from(i) / 8.0).map(|v| (v * 255.0).round() as u8);
            format!("rgb({r},{g},{b}) {}%", js_number(f64::from(i) / 8.0 * 100.0))
        })
        .collect();
    format!("linear-gradient(to top, {})", stops.join(", "))
}

/// A number as JavaScript prints it in a template string (no trailing `.0`).
pub fn js_number(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 { format!("{}", v as i64) } else { format!("{v}") }
}

/// A copy of a point slot message with every add and move coloured by height (the slot positions
/// are ENU metres about the anchor, so `z` is the height). Header, ids, positions and removes are
/// unchanged; an invalid message comes back as it is.
pub fn recolor_slots_by_height(msg: &[u8], low: f64, high: f64) -> Vec<u8> {
    let mut out = msg.to_vec();
    let Ok(h) = slots::validate(msg) else { return out };
    let records = (h.n_add + h.n_move) as usize;
    for k in 0..records {
        let at = slots::HEADER_LEN + k * slots::ADD_BYTES;
        let z = f32::from_le_bytes(msg[at + 12..at + 16].try_into().expect("4 bytes"));
        out[at + 16..at + 19].copy_from_slice(&height_color(f64::from(z), low, high));
    }
    out
}

/// A copy of a chunk with every point coloured by height. Header and positions are unchanged.
pub fn recolor_by_height(bytes: &[u8], low: f64, high: f64) -> Vec<u8> {
    let mut out = bytes.to_vec();
    let Some((h, body)) = records(bytes) else { return out };
    let stride = usize::from(h.stride);
    for i in 0..h.point_count as usize {
        let z = position(&h, body, i).z;
        let [r, g, b] = height_color(z, low, high);
        let at = HEADER_LEN + i * stride;
        if stride == 12 {
            out[at + 8..at + 12].copy_from_slice(&[r, g, b, 255]);
        } else {
            out[at + 6..at + 8].copy_from_slice(&rgb565(r, g, b).to_le_bytes());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pointblitz::io::Point;
    use crate::pointblitz::io::chunk::encode_with;

    fn chunk(points: &[[f32; 3]], stride: usize) -> Vec<u8> {
        let pts: Vec<Point> = points.iter().map(|p| Point { position: *p, color: [10, 20, 30] }).collect();
        encode_with(0, 0, 0, &pts, stride)
    }

    #[test]
    fn samples_decode_absolute_positions() {
        let c = chunk(&[[100.0, 200.0, 5.0], [110.0, 190.0, 15.0], [105.0, 195.0, 10.0]], 12);
        let s = sample_chunk(&c, 10);
        assert_eq!(s.len(), 3);
        assert!(s[0].distance(DVec3::new(100.0, 200.0, 5.0)) < 1e-3);
        assert!(s[1].distance(DVec3::new(110.0, 190.0, 15.0)) < 1e-3);
        assert_eq!(sample_chunk(&c, 2).len(), 2);
        assert!(sample_chunk(b"nope", 5).is_empty());
        let both = sample_chunks([c.as_slice(), c.as_slice()], 4);
        assert_eq!(both.len(), 4);
    }

    #[test]
    fn percentiles_ignore_outliers() {
        let mut pts: Vec<DVec3> = (0..1000).map(|i| DVec3::new(0.0, 0.0, f64::from(i % 10))).collect();
        pts.push(DVec3::new(0.0, 0.0, 500.0));
        let (lo, hi) = height_range(&pts).unwrap();
        assert_eq!(lo, 0.0);
        assert_eq!(hi, 9.0);
        assert!(height_range(&[]).is_none());
        let (a, b) = core_box(&pts).unwrap();
        assert_eq!((a.z, b.z), (0.0, 9.0));
    }

    #[test]
    fn ramp_and_legend_match_the_legacy_values() {
        assert_eq!(ramp(0.0), [0.0, 0.0, 0.5]);
        assert_eq!(ramp(0.5), [0.5, 1.0, 0.5]);
        assert_eq!(ramp(1.0), [0.5, 0.0, 0.0]);
        assert_eq!(
            legend_gradient(),
            "linear-gradient(to top, rgb(0,0,128) 0%, rgb(0,0,255) 12.5%, rgb(0,128,255) 25%, \
             rgb(0,255,255) 37.5%, rgb(128,255,128) 50%, rgb(255,255,0) 62.5%, rgb(255,128,0) 75%, \
             rgb(255,0,0) 87.5%, rgb(128,0,0) 100%)"
        );
        assert_eq!(js_number(37.5), "37.5");
        assert_eq!(js_number(100.0), "100");
    }

    #[test]
    fn recolouring_keeps_positions_and_paints_by_height() {
        for stride in [12, 8] {
            let c = chunk(&[[0.0, 0.0, 0.0], [1.0, 1.0, 10.0]], stride);
            let out = recolor_by_height(&c, 0.0, 10.0);
            assert_eq!(out.len(), c.len());
            assert_eq!(&out[..HEADER_LEN], &c[..HEADER_LEN]);
            assert_eq!(sample_chunk(&out, 2), sample_chunk(&c, 2));
            if stride == 12 {
                let low = height_color(0.0, 0.0, 10.0);
                let high = height_color(10.0, 0.0, 10.0);
                assert_eq!(&out[HEADER_LEN + 8..HEADER_LEN + 12], &[low[0], low[1], low[2], 255]);
                assert_eq!(&out[HEADER_LEN + 20..HEADER_LEN + 24], &[high[0], high[1], high[2], 255]);
                // Low is blue, high is red.
                assert!(low[2] > low[0] && high[0] > high[2]);
            }
        }
    }

    #[test]
    fn slot_messages_are_recoloured_by_height_only() {
        let mut pack = Vec::new();
        slots::Record { id: 0, position: [1.0, 2.0, 0.0], color: [9, 9, 9], zone: 4 }.write(&mut pack);
        slots::Record { id: 1, position: [1.0, 2.0, 10.0], color: [9, 9, 9], zone: 0 }.write(&mut pack);
        pack.extend(7u32.to_le_bytes());
        let header = slots::SlotHeader { seq: 3, n_add: 1, n_move: 1, n_remove: 1, capacity: 8 };
        let msg = slots::encode(&header, &pack).unwrap();
        let out = recolor_slots_by_height(&msg, 0.0, 10.0);
        let d = slots::Diff::parse(&out).unwrap();
        assert_eq!(d.header, header);
        let add = d.add_records().next().unwrap();
        let mv = d.move_records().next().unwrap();
        assert_eq!((add.id, add.position, add.zone), (0, [1.0, 2.0, 0.0], 4));
        assert_eq!(add.color, height_color(0.0, 0.0, 10.0));
        assert_eq!(mv.color, height_color(10.0, 0.0, 10.0));
        assert_eq!(d.remove_ids().collect::<Vec<_>>(), vec![7]);
        assert_eq!(recolor_slots_by_height(b"nope", 0.0, 1.0), b"nope".to_vec());
    }

    #[test]
    fn spacing_needs_points_and_a_footprint() {
        let b = (DVec3::ZERO, DVec3::new(10.0, 10.0, 3.0));
        assert_eq!(spacing(b, 100), Some(1.0));
        assert_eq!(spacing(b, 15), None);
        assert_eq!(spacing((DVec3::ZERO, DVec3::new(0.0, 10.0, 0.0)), 100), None);
        assert_eq!(point_size_from_spacings(vec![0.1, 0.2, 0.3]), 0.3);
        assert_eq!(point_size_from_spacings(Vec::new()), 0.15);
    }

    #[test]
    fn point_size_follows_the_spacing() {
        // 100 points over 10 x 10 m: 1 m spacing -> 1.5 m, clamped to 1 m.
        let pts: Vec<[f32; 3]> = (0..100).map(|i| [(i % 10) as f32 * 10.0 / 9.0, (i / 10) as f32 * 10.0 / 9.0, 0.0]).collect();
        let c = chunk(&pts, 12);
        assert_eq!(point_size_m([c.as_slice()]), 1.0);
        let dense: Vec<[f32; 3]> = (0..10_000).map(|i| [(i % 100) as f32 * 0.1, (i / 100) as f32 * 0.1, 0.0]).collect();
        let c = chunk(&dense, 12);
        let s = point_size_m([c.as_slice()]);
        assert!((s - 0.15).abs() < 0.01, "{s}");
        assert_eq!(point_size_m(std::iter::empty()), 0.15);
    }
}
