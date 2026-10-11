//! CPU side of the mesh path: payload decoding, block version bookkeeping, smooth normals and
//! vertex packing. Nothing here touches the GPU, so it is all unit tested.

use std::collections::HashMap;

use anyhow::{bail, Context};
use prost::Message;
use skylens_shared::proto::payload::v1::MeshBlock;

/// `[x, y, z]` block coordinate from deltamesh.
pub type BlockId = [i32; 3];

/// Colour used when a block arrives without per-vertex colours.
pub const DEFAULT_COLOR: [u8; 3] = [180, 180, 180];

/// One vertex as uploaded: position (f32×3), normal (f32×3), colour (unorm8×4).
pub const VERTEX_STRIDE: usize = 28;

/// A validated `MeshBlock`. Zero triangles means "remove this block".
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedBlock {
    pub id: BlockId,
    pub version: u32,
    /// Absolute ENU metres (x east, y north, z up).
    pub positions: Vec<[f32; 3]>,
    pub colors: Vec<[u8; 3]>,
    pub indices: Vec<u32>,
}

impl DecodedBlock {
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn is_removal(&self) -> bool {
        self.indices.is_empty()
    }
}

/// Decodes and validates a `CHUNK_KIND_MESH_BLOCK` payload.
pub fn decode_block(payload: &[u8]) -> anyhow::Result<DecodedBlock> {
    let msg = MeshBlock::decode(payload).context("mesh block: protobuf decode")?;
    let id: BlockId = match msg.id.as_slice() {
        [x, y, z] => [*x, *y, *z],
        other => bail!("mesh block: id has {} components, expected 3", other.len()),
    };
    if msg.positions.len() % 3 != 0 {
        bail!("mesh block {id:?}: {} position floats is not a multiple of 3", msg.positions.len());
    }
    if msg.indices.len() % 3 != 0 {
        bail!("mesh block {id:?}: {} indices is not a multiple of 3", msg.indices.len());
    }
    let vertex_count = msg.positions.len() / 3;
    let positions: Vec<[f32; 3]> = msg.positions.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect();
    if let Some(bad) = positions.iter().find(|p| p.iter().any(|c| !c.is_finite())) {
        bail!("mesh block {id:?}: non-finite position {bad:?}");
    }
    let colors: Vec<[u8; 3]> = if msg.colors.is_empty() {
        vec![DEFAULT_COLOR; vertex_count]
    } else if msg.colors.len() == vertex_count * 3 {
        msg.colors.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
    } else {
        bail!(
            "mesh block {id:?}: {} colour bytes for {vertex_count} vertices (expected {})",
            msg.colors.len(),
            vertex_count * 3
        );
    };
    if let Some(&bad) = msg.indices.iter().find(|&&i| i as usize >= vertex_count) {
        bail!("mesh block {id:?}: index {bad} out of range for {vertex_count} vertices");
    }
    Ok(DecodedBlock { id, version: msg.version, positions, colors, indices: msg.indices })
}

/// Last version seen per block id, removals included (a removed block keeps its version as a
/// tombstone so a late, older copy cannot bring it back).
#[derive(Debug, Default)]
pub struct BlockVersions {
    seen: HashMap<BlockId, u32>,
}

impl BlockVersions {
    /// True (and records it) if `version` is strictly newer than anything seen for `id`.
    /// The first version seen for an id is always admitted.
    pub fn admit(&mut self, id: BlockId, version: u32) -> bool {
        match self.seen.get(&id) {
            Some(&have) if version <= have => false,
            _ => {
                self.seen.insert(id, version);
                true
            }
        }
    }

    pub fn clear(&mut self) {
        self.seen.clear();
    }
}

/// Smooth per-vertex normals: the area-weighted sum of the adjacent faces' normals, normalised.
/// Faces are counter-clockwise from outside, so normals point outwards. Vertices with no
/// non-degenerate face get +z (up). Computed in f64 because positions are absolute ENU metres
/// and edge vectors would otherwise lose precision far from the origin.
pub fn smooth_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut acc = vec![[0.0f64; 3]; positions.len()];
    let p = |i: u32| positions[i as usize].map(f64::from);
    for tri in indices.chunks_exact(3) {
        let (a, b, c) = (p(tri[0]), p(tri[1]), p(tri[2]));
        let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        // |e1 × e2| = 2 × area, so summing the raw cross product weights by area.
        let n = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        for &v in tri {
            let s = &mut acc[v as usize];
            s[0] += n[0];
            s[1] += n[1];
            s[2] += n[2];
        }
    }
    acc.into_iter()
        .map(|n| {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if len > 1e-12 {
                [(n[0] / len) as f32, (n[1] / len) as f32, (n[2] / len) as f32]
            } else {
                [0.0, 0.0, 1.0]
            }
        })
        .collect()
}

/// Interleaves position, normal and colour into the vertex buffer layout (little endian, which
/// every wgpu backend including wasm uses).
pub fn pack_vertices(positions: &[[f32; 3]], normals: &[[f32; 3]], colors: &[[u8; 3]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(positions.len() * VERTEX_STRIDE);
    for ((p, n), c) in positions.iter().zip(normals).zip(colors) {
        for v in p.iter().chain(n) {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&[c[0], c[1], c[2], 255]);
    }
    out
}

pub fn pack_indices(indices: &[u32]) -> Vec<u8> {
    indices.iter().flat_map(|i| i.to_le_bytes()).collect()
}

/// Turns a camera-relative view-projection (pointblitz's `Camera::view_proj`, eye at the origin)
/// into one that takes absolute ENU metres: `vp × translate(−eye)`, composed in f64.
pub fn absolute_view_proj(camera_relative: [[f32; 4]; 4], eye: [f64; 3]) -> [[f32; 4]; 4] {
    let m = camera_relative.map(|col| col.map(f64::from));
    let mut out = camera_relative;
    for row in 0..4 {
        let t = m[3][row] - m[0][row] * eye[0] - m[1][row] * eye[1] - m[2][row] * eye[2];
        out[3][row] = t as f32;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode(block: MeshBlock) -> Vec<u8> {
        block.encode_to_vec()
    }

    fn quad() -> MeshBlock {
        // Unit square in the z = 0 plane, counter-clockwise seen from +z.
        MeshBlock {
            id: vec![1, -2, 3],
            version: 7,
            positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            colors: vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 10, 20, 30],
            indices: vec![0, 1, 2, 0, 2, 3],
        }
    }

    #[test]
    fn decodes_a_valid_block() {
        let b = decode_block(&encode(quad())).unwrap();
        assert_eq!(b.id, [1, -2, 3]);
        assert_eq!(b.version, 7);
        assert_eq!(b.positions.len(), 4);
        assert_eq!(b.positions[2], [1.0, 1.0, 0.0]);
        assert_eq!(b.colors[3], [10, 20, 30]);
        assert_eq!(b.triangle_count(), 2);
        assert!(!b.is_removal());
    }

    #[test]
    fn zero_triangles_is_a_removal() {
        let msg = MeshBlock { id: vec![0, 0, 0], version: 2, ..Default::default() };
        let b = decode_block(&encode(msg)).unwrap();
        assert!(b.is_removal());
        assert!(b.positions.is_empty());
    }

    #[test]
    fn missing_colours_fall_back_to_grey() {
        let b = decode_block(&encode(MeshBlock { colors: vec![], ..quad() })).unwrap();
        assert_eq!(b.colors, vec![DEFAULT_COLOR; 4]);
    }

    #[test]
    fn rejects_malformed_blocks() {
        assert!(decode_block(&[0xff, 0xff, 0xff]).is_err());
        assert!(decode_block(&encode(MeshBlock { id: vec![1, 2], ..quad() })).is_err());
        let mut short = quad();
        short.positions.pop();
        assert!(decode_block(&encode(short)).is_err());
        assert!(decode_block(&encode(MeshBlock { indices: vec![0, 1], ..quad() })).is_err());
        assert!(decode_block(&encode(MeshBlock { indices: vec![0, 1, 4], ..quad() })).is_err());
        assert!(decode_block(&encode(MeshBlock { colors: vec![1, 2, 3], ..quad() })).is_err());
        let mut nan = quad();
        nan.positions[0] = f32::NAN;
        assert!(decode_block(&encode(nan)).is_err());
    }

    #[test]
    fn versions_admit_only_newer() {
        let mut v = BlockVersions::default();
        assert!(v.admit([0, 0, 0], 0), "first sight is admitted, even version 0");
        assert!(v.admit([0, 0, 0], 3));
        assert!(!v.admit([0, 0, 0], 3), "same version is stale");
        assert!(!v.admit([0, 0, 0], 2), "older version is stale");
        assert!(v.admit([1, 0, 0], 1), "ids are independent");
        assert!(v.admit([0, 0, 0], 4));
    }

    #[test]
    fn removal_tombstone_blocks_older_copies() {
        let mut v = BlockVersions::default();
        assert!(v.admit([5, 5, 5], 3));
        assert!(v.admit([5, 5, 5], 4)); // removal at version 4
        assert!(!v.admit([5, 5, 5], 3), "late older copy must not resurrect it");
        v.clear();
        assert!(v.admit([5, 5, 5], 1), "clear forgets everything");
    }

    #[test]
    fn flat_quad_normals_point_up() {
        let b = decode_block(&encode(quad())).unwrap();
        for n in smooth_normals(&b.positions, &b.indices) {
            assert!((n[0]).abs() < 1e-6 && (n[1]).abs() < 1e-6 && (n[2] - 1.0).abs() < 1e-6, "{n:?}");
        }
    }

    #[test]
    fn shared_edge_normal_is_averaged() {
        // Two faces folded along the y axis at 90 degrees: one facing +z, one facing +x.
        let positions = [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ];
        // Face A (z = 0 plane, x <= 0): 2, 0, 1 is CCW from +z. Face B (x = 0 plane): 0, 3, 1
        // is CCW from +x.
        let indices = [2, 0, 1, 0, 3, 1];
        let n = smooth_normals(&positions, &indices);
        let h = std::f32::consts::FRAC_1_SQRT_2;
        let close = |a: [f32; 3], b: [f32; 3]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5);
        assert!(close(n[2], [0.0, 0.0, 1.0]), "{:?}", n[2]);
        assert!(close(n[3], [1.0, 0.0, 0.0]), "{:?}", n[3]);
        assert!(close(n[0], [h, 0.0, h]), "{:?}", n[0]);
        assert!(close(n[1], [h, 0.0, h]), "{:?}", n[1]);
    }

    #[test]
    fn far_from_origin_normals_stay_accurate() {
        let o = [5000.0f32, -5000.0, 120.0];
        let positions = [
            o,
            [o[0] + 0.2, o[1], o[2]],
            [o[0] + 0.2, o[1] + 0.2, o[2]],
        ];
        let n = smooth_normals(&positions, &[0, 1, 2]);
        assert!((n[0][2] - 1.0).abs() < 1e-4, "{:?}", n[0]);
    }

    #[test]
    fn isolated_and_degenerate_vertices_default_up() {
        let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [9.0, 9.0, 9.0]];
        let n = smooth_normals(&positions, &[0, 1, 2]);
        assert!(n.iter().all(|v| *v == [0.0, 0.0, 1.0]));
    }

    #[test]
    fn packs_interleaved_vertices() {
        let bytes = pack_vertices(&[[1.0, 2.0, 3.0]], &[[0.0, 0.0, 1.0]], &[[9, 8, 7]]);
        assert_eq!(bytes.len(), VERTEX_STRIDE);
        assert_eq!(&bytes[0..4], &1.0f32.to_le_bytes());
        assert_eq!(&bytes[8..12], &3.0f32.to_le_bytes());
        assert_eq!(&bytes[20..24], &1.0f32.to_le_bytes());
        assert_eq!(&bytes[24..28], &[9, 8, 7, 255]);
        assert_eq!(pack_indices(&[1, 256]), vec![1, 0, 0, 0, 0, 1, 0, 0]);
    }

    #[test]
    fn absolute_view_proj_matches_camera_relative() {
        // A column-major matrix with a non-trivial translation column.
        let vp = [
            [0.5, 0.1, 0.0, 0.0],
            [0.0, 0.8, 0.2, -0.3],
            [0.1, 0.0, 0.0, -1.0],
            [0.0, 0.0, 0.05, 0.0],
        ];
        let eye = [1000.0, -2000.0, 50.0];
        let abs = absolute_view_proj(vp, eye);
        let mul = |m: [[f32; 4]; 4], p: [f64; 3]| -> [f64; 4] {
            let mut r = [0.0; 4];
            for (row, out) in r.iter_mut().enumerate() {
                *out = f64::from(m[0][row]) * p[0]
                    + f64::from(m[1][row]) * p[1]
                    + f64::from(m[2][row]) * p[2]
                    + f64::from(m[3][row]);
            }
            r
        };
        let p = [1003.0, -1990.0, 52.0];
        let want = mul(vp, [p[0] - eye[0], p[1] - eye[1], p[2] - eye[2]]);
        let got = mul(abs, p);
        for (a, b) in want.iter().zip(got) {
            assert!((a - b).abs() < 1e-3, "{want:?} vs {got:?}");
        }
    }
}
