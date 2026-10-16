//! The board's slot buffer (`DeltaCloud`) must end up exactly where cumulus3d's `DisplayMirror`
//! does for the same sequence of `DisplayDelta`s, and so must a late joiner that starts from a
//! snapshot of the mirror. Crosses cumulus3d -> skylens-shared encoding -> skylens-viewer.

use cumulus3d_cli::live::{DisplayDelta, DisplayMirror, KeyedDiff};
use cumulus3d_core::io::PointCloud;
use skylens_shared::delta::{AddRec, MoveRec, pack_adds, pack_moves, pack_removes};
use skylens_shared::proto::payload::v1::{DeltaKind, PointsDelta};
use skylens_viewer::delta_cloud::DeltaCloud;

/// xorshift, so the test needs no rand crate and is repeatable.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() % 10_000) as f32 / 10_000.0
    }
}

fn wire(d: &DisplayDelta, snapshot: bool) -> PointsDelta {
    let adds: Vec<AddRec> = d.adds.iter().map(|a| AddRec { id: a.id, xyz: a.xyz, rgb: a.rgb, zone: a.zone }).collect();
    let moves: Vec<MoveRec> = d.moves.iter().map(|m| MoveRec { id: m.id, xyz: m.xyz, rgb: m.rgb }).collect();
    PointsDelta {
        seq: d.seq as u32,
        kind: DeltaKind::Refined as i32,
        capacity: d.capacity,
        total_points: d.total_points as u32,
        snapshot,
        adds: pack_adds(&adds),
        moves: pack_moves(&moves),
        removes: pack_removes(&d.removes),
        ..Default::default()
    }
}

fn snapshot_of(m: &DisplayMirror, seq: u64, capacity: u32) -> PointsDelta {
    let adds: Vec<AddRec> = m.points().map(|(id, p)| AddRec { id, xyz: p.xyz, rgb: p.rgb, zone: p.zone }).collect();
    PointsDelta {
        seq: seq as u32,
        kind: DeltaKind::Replace as i32,
        capacity,
        total_points: adds.len() as u32,
        snapshot: true,
        adds: pack_adds(&adds),
        ..Default::default()
    }
}

fn assert_same(mirror: &DisplayMirror, cloud: &DeltaCloud) {
    assert_eq!(cloud.alive_count(), mirror.len());
    for (id, p) in mirror.points() {
        let s = cloud.slot(id).expect("slot");
        assert!(cloud.is_alive(id), "slot {id} alive");
        assert_eq!(s.pos, p.xyz, "slot {id} position");
        assert_eq!(s.rgbz, [p.rgb[0], p.rgb[1], p.rgb[2], p.zone], "slot {id} colour/zone");
    }
}

#[test]
fn slot_buffer_equals_the_display_mirror_and_a_snapshot_joiner() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut diff = KeyedDiff::new();
    let mut mirror = DisplayMirror::new();
    let mut cloud = DeltaCloud::new();
    let mut now = 0.0f32;
    let mut late: Option<DeltaCloud> = None;

    for step in 0..12u64 {
        // A cloud whose keys drift in and out; zones change for some keys (remove + add).
        let mut c = PointCloud::default();
        let mut zones = Vec::new();
        for k in 0..400u64 {
            if rng.next() % 5 == 0 && step > 0 {
                continue; // key leaves
            }
            c.positions.push([k as f32 * 0.1 + rng.unit() * 0.2 * (step % 3) as f32, rng.unit(), rng.unit()]);
            c.colors.push([(k % 255) as u8, (step * 20) as u8, 7]);
            c.keys.push(k + 1000 * (step % 2 == 1 && k > 380) as u64);
            zones.push(if rng.next() % 40 == 0 { 1 } else { (k % 3) as usize });
        }
        let d = diff.apply(&c, &zones).unwrap();
        mirror.apply(&d).unwrap();
        cloud.ingest(wire(&d, false));
        now += 5.0;
        cloud.advance(now);
        // A joiner appears after step 5 and starts from a snapshot of the mirror.
        if step == 5 {
            let mut j = DeltaCloud::new();
            j.ingest(snapshot_of(&mirror, d.seq, d.capacity));
            late = Some(j);
        } else if let Some(j) = late.as_mut() {
            j.ingest(wire(&d, false));
            j.advance(now);
        }
        assert_same(&mirror, &cloud);
        if let Some(j) = &late {
            assert_same(&mirror, j);
        }
    }
    assert_eq!(cloud.gaps, 0);
}
