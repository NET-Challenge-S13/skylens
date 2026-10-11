//! Real-data check against the flight-01 dataset (P1.1 completion criterion).
//!
//! The PLY files are not in the repository (SPEC §8.3). Set `PB_FLIGHT01=<dir>` to run;
//! without it the test passes after printing that it was skipped, so CI stays green.
//!
//! `cargo test -p pointblitz-io --release --test flight01 -- --nocapture`

use pointblitz_io::chunk::{self, FLAG_LAST_IN_GENERATION};
use pointblitz_io::{PlyStream, parse_header, points};
use std::time::Instant;

/// (file, points) from SPEC §2.2.
const FLIGHT01: [(&str, usize); 14] = [
    ("event_01_0097.0s_preview_0.ply", 151_510),
    ("event_02_0127.1s_refined_0.ply", 336_118),
    ("event_03_0298.9s_preview_1.ply", 371_585),
    ("event_04_0327.4s_refined_1.ply", 749_216),
    ("event_05_0570.7s_preview_2.ply", 798_744),
    ("event_06_0623.5s_refined_2.ply", 1_192_254),
    ("event_07_0934.8s_preview_3.ply", 1_236_878),
    ("event_08_0994.2s_refined_3.ply", 1_640_611),
    ("event_09_1333.1s_preview_4.ply", 1_676_080),
    ("event_10_1388.6s_refined_4.ply", 2_059_581),
    ("event_11_1760.7s_preview_5.ply", 2_113_986),
    ("event_12_1805.1s_refined_5.ply", 2_404_412),
    ("event_13_2005.6s_preview_6.ply", 2_426_023),
    ("event_14_2053.0s_refined_6.ply", 2_502_015),
];

#[test]
fn flight01_counts_stream_and_chunks() {
    let Ok(dir) = std::env::var("PB_FLIGHT01") else {
        eprintln!("PB_FLIGHT01 not set — skipped");
        return;
    };
    for (name, expected) in FLIGHT01 {
        let bytes = std::fs::read(format!("{dir}/{name}")).unwrap();
        let h = parse_header(&bytes).unwrap();
        assert_eq!(h.vertex_count, expected, "{name}");
        assert_eq!(h.stride, 27, "{name}");

        // Whole-file parse.
        let t = Instant::now();
        let whole: Vec<_> = points(&bytes).unwrap().collect();
        let t_parse = t.elapsed();
        assert_eq!(whole.len(), expected);

        // Streaming in 64 KiB pieces (network-sized) gives the same points.
        let t = Instant::now();
        let mut streamed = Vec::with_capacity(expected);
        let mut s = PlyStream::new();
        for piece in bytes.chunks(64 * 1024) {
            s.push(piece, |p| streamed.push(p)).unwrap();
        }
        let t_stream = t.elapsed();
        assert!(s.is_complete());
        assert!(streamed == whole, "{name}: stream differs");

        // One generation, 256 Ki points per chunk: sizes and flags (process stride, decision 0051).
        let t = Instant::now();
        let chunks = chunk::ply_to_chunks(&bytes, 0, 256 * 1024).unwrap();
        let t_chunk = t.elapsed();
        let mut total = 0usize;
        let mut max_err = 0.0f64;
        let mut max_step = 0.0f64;
        for (i, c) in chunks.iter().enumerate() {
            let ch = chunk::decode_header(c).unwrap();
            let stride = usize::from(ch.stride);
            assert_eq!(
                c.len(),
                chunk::HEADER_LEN + stride * ch.point_count as usize
            );
            assert_eq!(ch.flags == FLAG_LAST_IN_GENERATION, i + 1 == chunks.len());
            // Round trip: colour exact (stride 12), position within half a quantisation step.
            let step = [0, 1, 2].map(|a| f64::from(ch.bbox_max[a] - ch.bbox_min[a]) / 65535.0);
            max_step = max_step.max(step.into_iter().fold(0.0, f64::max));
            for (k, rec) in chunk::vertex_bytes(c).chunks_exact(stride).enumerate() {
                let orig = &whole[total + k];
                let q = [0, 1, 2].map(|a| u16::from_le_bytes([rec[2 * a], rec[2 * a + 1]]));
                let back = ch.dequantise(q);
                for a in 0..3 {
                    let err =
                        (f64::from(back[a]) + ch.origin[a] - f64::from(orig.position[a])).abs();
                    assert!(
                        err <= step[a] / 2.0 + 1e-4,
                        "{name}: axis {a} error {err} m"
                    );
                    max_err = max_err.max(err);
                }
                if stride == 12 {
                    assert_eq!(&rec[8..11], &orig.color, "{name}: colour");
                    assert_eq!(rec[11], 255);
                }
            }
            total += ch.point_count as usize;
        }
        assert_eq!(total, expected);
        let chunk_bytes: usize = chunks.iter().map(Vec::len).sum();
        println!(
            "{name}: {expected} pts, PLY {} B → chunks {chunk_bytes} B ({} chunks); parse {:.1} ms, stream {:.1} ms, to-chunks {:.1} ms, max round-trip error {:.2} mm (largest step {:.2} mm)",
            bytes.len(),
            chunks.len(),
            t_parse.as_secs_f64() * 1e3,
            t_stream.as_secs_f64() * 1e3,
            t_chunk.as_secs_f64() * 1e3,
            max_err * 1e3,
            max_step * 1e3,
        );
    }
}
