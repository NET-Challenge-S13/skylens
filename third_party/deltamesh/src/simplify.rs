//! Block mesh simplification with meshoptimizer edge collapse.
//!
//! A block mesh covers local cells `-1..=dim-1`. Vertices in cell layers `-1` and `dim-1` are also produced by the
//! neighbouring block, so every vertex in those two layers is locked. That keeps boundary vertices bitwise equal
//! between neighbours after simplification.
//!
//! Edge collapse merges a vertex into one of the remaining vertices, so it never creates new positions; surviving
//! vertices keep their original position and colour. meshoptimizer runs single-threaded and is deterministic, so equal
//! input gives equal output and preview replacement stays bitwise exact.

use meshopt::{SimplifyOptions, simplify_with_locks_decoder};

/// Reduces the triangle count within `max_error` and compacts the unused vertices away.
///
/// Surviving vertices are renumbered in their original order.
///
/// # Arguments
///
/// * `max_error` - Absolute error bound in metres. Zero or negative disables simplification.
/// * `pos`, `col` - Vertex positions (`x y z`) and colours (`r g b`).
/// * `tri` - Triangle indices.
/// * `lock` - Per-vertex seam flag. `build_block` derives it from integer cell coordinates, so it is exact regardless
///   of coordinate magnitude.
///
/// # Returns
///
/// The simplified `(positions, colours, indices)`, or the input unchanged when simplification is disabled or the mesh
/// is empty.
pub(crate) fn simplify_block(
    max_error: f32,
    pos: Vec<f32>,
    col: Vec<u8>,
    tri: Vec<u32>,
    lock: &[bool],
) -> (Vec<f32>, Vec<u8>, Vec<u32>) {
    if max_error <= 0.0 || tri.is_empty() {
        return (pos, col, tri);
    }
    let verts: Vec<[f32; 3]> = pos.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect();
    let out = simplify_with_locks_decoder(
        &tri,
        &verts,
        lock,
        0,
        max_error,
        SimplifyOptions::ErrorAbsolute,
        None,
    );

    let mut remap = vec![u32::MAX; verts.len()];
    for &i in &out {
        remap[i as usize] = 0;
    }
    let mut npos = Vec::new();
    let mut ncol = Vec::new();
    let mut n = 0u32;
    for (i, r) in remap.iter_mut().enumerate() {
        if *r != u32::MAX {
            *r = n;
            n += 1;
            npos.extend_from_slice(&pos[i * 3..i * 3 + 3]);
            ncol.extend_from_slice(&col[i * 3..i * 3 + 3]);
        }
    }
    let ntri = out.iter().map(|&i| remap[i as usize]).collect();
    (npos, ncol, ntri)
}
