//! The CPU side of the layers, kept free of GPU types so it can be tested without an adapter:
//! colour conversion, fog packing, normal generation, vertex packing, uniform packing, mip chains
//! and the per-frame draw lists.

use super::{Blend, Fog, LineSet, Material, MeshData, Shading, TextureImage, Topology};
use bytemuck::{Pod, Zeroable};
use glam::{DVec3, Vec3};
use std::f32::consts::PI;

// ---- colour ---------------------------------------------------------------------------------

/// sRGB transfer function, display value to linear (same curve as three.js `SRGBToLinear`).
pub(crate) fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Inverse of `srgb_to_linear`.
pub(crate) fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

fn lin3(c: [f32; 3]) -> Vec3 {
    Vec3::new(
        srgb_to_linear(c[0]),
        srgb_to_linear(c[1]),
        srgb_to_linear(c[2]),
    )
}

/// Display-encoded RGBA (alpha untouched) to linear RGBA.
pub(crate) fn linear_rgba(c: [f32; 4]) -> [f32; 4] {
    let l = lin3([c[0], c[1], c[2]]);
    [l.x, l.y, l.z, c[3]]
}

// ---- fog ------------------------------------------------------------------------------------

/// Fog as the shaders take it. `t = depth * scale + bias` is 0 at `near` and 1 at `far`; the
/// shaders then apply `smoothstep(0, 1, t) * color.a`, which equals three.js
/// `smoothstep(fogNear, fogFar, depth)` for linear `Fog`. `color.a` is 0 when fog is off.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FogParams {
    pub color: [f32; 4],
    pub scale: f32,
    pub bias: f32,
}

pub(crate) fn fog_params(fog: Option<Fog>) -> FogParams {
    match fog {
        None => FogParams {
            color: [0.0; 4],
            scale: 0.0,
            bias: 0.0,
        },
        Some(f) => {
            let span = (f.far - f.near).max(1e-6);
            FogParams {
                color: [f.color[0], f.color[1], f.color[2], 1.0],
                scale: 1.0 / span,
                bias: -f.near / span,
            }
        }
    }
}

fn smoothstep01(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// CPU mirror of the shaders' fog weight for a view depth in metres.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn fog_amount(p: &FogParams, depth: f32) -> f32 {
    smoothstep01(depth * p.scale + p.bias) * p.color[3]
}

/// CPU mirror of the shaders' final fog mix (in the colour target's space).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn apply_fog(p: &FogParams, rgb: [f32; 3], depth: f32) -> [f32; 3] {
    let k = fog_amount(p, depth);
    [0, 1, 2].map(|i| rgb[i] + (p.color[i] - rgb[i]) * k)
}

/// Uniform shared by the mesh and line shaders (group 0, vertex stage only).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct FrameUniform {
    pub view_proj: [[f32; 4]; 4],
    pub fog_color: [f32; 4],
    /// x: fog scale, y: fog bias, z: 1 when the shader must sRGB-encode its output.
    pub params: [f32; 4],
}

pub(crate) fn frame_uniform(
    view_proj: [[f32; 4]; 4],
    fog: Option<Fog>,
    encode_srgb: bool,
) -> FrameUniform {
    let f = fog_params(fog);
    FrameUniform {
        view_proj,
        fog_color: f.color,
        params: [f.scale, f.bias, if encode_srgb { 1.0 } else { 0.0 }, 0.0],
    }
}

// ---- normals --------------------------------------------------------------------------------

/// Unnormalised face normal (length = twice the area), counter-clockwise front.
fn face_normal(a: DVec3, b: DVec3, c: DVec3) -> DVec3 {
    (b - a).cross(c - a)
}

fn unit_or_up(n: DVec3) -> [f32; 3] {
    let len = n.length();
    if len > 0.0 && len.is_finite() {
        (n / len).as_vec3().to_array()
    } else {
        [0.0, 0.0, 1.0]
    }
}

/// Area-weighted vertex normals (like three.js `computeVertexNormals`). Vertices touched by no
/// triangle, or only by degenerate ones, get +Z.
pub(crate) fn smooth_normals(positions: &[[f64; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut acc = vec![DVec3::ZERO; positions.len()];
    for t in indices.as_chunks::<3>().0 {
        let [a, b, c] = t.map(|i| i as usize);
        let n = face_normal(
            positions[a].into(),
            positions[b].into(),
            positions[c].into(),
        );
        acc[a] += n;
        acc[b] += n;
        acc[c] += n;
    }
    acc.into_iter().map(unit_or_up).collect()
}

// ---- mesh packing ---------------------------------------------------------------------------

/// Interleaved mesh vertex: 36 bytes, positions relative to `PackedMesh::origin`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub(crate) struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub color: [u8; 4],
}

pub(crate) struct PackedMesh {
    /// Centre of the bounding box in absolute ENU metres; vertices are stored relative to it so
    /// f32 only has to cover the mesh extent.
    pub origin: DVec3,
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
}

fn bbox_centre(positions: &[[f64; 3]]) -> DVec3 {
    let Some(first) = positions.first() else {
        return DVec3::ZERO;
    };
    let (mut lo, mut hi) = (DVec3::from(*first), DVec3::from(*first));
    for p in positions {
        let p = DVec3::from(*p);
        lo = lo.min(p);
        hi = hi.max(p);
    }
    (lo + hi) * 0.5
}

/// Validates a mesh and packs it for upload. Missing normals are generated: one normal per face
/// (vertices are unshared) when `flat`, area-weighted smooth normals otherwise.
pub(crate) fn pack_mesh(mesh: &MeshData) -> PackedMesh {
    let n = mesh.positions.len();
    assert!(
        mesh.indices.len().is_multiple_of(3),
        "mesh indices must be whole triangles"
    );
    assert!(
        mesh.indices.iter().all(|&i| (i as usize) < n),
        "mesh index out of range"
    );
    if let Some(v) = &mesh.normals {
        assert_eq!(v.len(), n, "one normal per position");
    }
    if let Some(v) = &mesh.colors {
        assert_eq!(v.len(), n, "one colour per position");
    }
    if let Some(v) = &mesh.uvs {
        assert_eq!(v.len(), n, "one uv per position");
    }
    let origin = bbox_centre(&mesh.positions);
    let vertex = |i: usize, normal: [f32; 3]| MeshVertex {
        position: (DVec3::from(mesh.positions[i]) - origin)
            .as_vec3()
            .to_array(),
        normal,
        uv: mesh.uvs.as_ref().map_or([0.0; 2], |v| v[i]),
        color: mesh.colors.as_ref().map_or([255; 4], |v| v[i]),
    };

    match (&mesh.normals, mesh.flat) {
        (Some(normals), _) => PackedMesh {
            origin,
            vertices: (0..n).map(|i| vertex(i, normals[i])).collect(),
            indices: mesh.indices.clone(),
        },
        (None, false) => {
            let normals = smooth_normals(&mesh.positions, &mesh.indices);
            PackedMesh {
                origin,
                vertices: (0..n).map(|i| vertex(i, normals[i])).collect(),
                indices: mesh.indices.clone(),
            }
        }
        (None, true) => {
            let mut vertices = Vec::with_capacity(mesh.indices.len());
            for t in mesh.indices.as_chunks::<3>().0 {
                let [a, b, c] = t.map(|i| i as usize);
                let normal = unit_or_up(face_normal(
                    mesh.positions[a].into(),
                    mesh.positions[b].into(),
                    mesh.positions[c].into(),
                ));
                vertices.extend([vertex(a, normal), vertex(b, normal), vertex(c, normal)]);
            }
            let indices = (0..vertices.len() as u32).collect();
            PackedMesh {
                origin,
                vertices,
                indices,
            }
        }
    }
}

// ---- mesh material uniform ------------------------------------------------------------------

/// Per-mesh uniform slot (group 1, dynamic offset, vertex stage only). Colours are linear;
/// light colours already carry intensity and the Lambert 1/pi.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub(crate) struct MeshObject {
    pub offset: [f32; 4],
    pub tint: [f32; 4],
    pub sky: [f32; 4],
    pub ground: [f32; 4],
    pub sun_dir: [f32; 4],
    pub sun: [f32; 4],
    pub emissive: [f32; 4],
}

fn v4(v: Vec3) -> [f32; 4] {
    [v.x, v.y, v.z, 0.0]
}

/// Packs a material. Unlit is expressed as a Lambert with irradiance 1 and no emissive, so one
/// shader serves both: `out = base * irradiance(n) + emissive`.
///
/// Lambert follows three.js (r155+, physical light units): hemisphere irradiance
/// `mix(ground, sky, 0.5 * n.z + 0.5)` with +Z as up (ENU), plus `sun * max(dot(n, l), 0)`, both
/// times `BRDF_Lambert = diffuse / pi`.
pub(crate) fn mesh_object(material: &Material, offset: DVec3) -> MeshObject {
    let offset = offset.as_vec3();
    let tint = linear_rgba(material.tint);
    match material.shading {
        Shading::Unlit => MeshObject {
            offset: [offset.x, offset.y, offset.z, 0.0],
            tint,
            sky: [1.0, 1.0, 1.0, 0.0],
            ground: [1.0, 1.0, 1.0, 0.0],
            sun_dir: [0.0, 0.0, 1.0, 0.0],
            sun: [0.0; 4],
            emissive: [0.0; 4],
        },
        Shading::Lambert {
            sky,
            ground,
            hemi_intensity,
            sun_dir,
            sun_color,
            sun_intensity,
            emissive,
        } => MeshObject {
            offset: [offset.x, offset.y, offset.z, 0.0],
            tint,
            sky: v4(lin3(sky) * hemi_intensity / PI),
            ground: v4(lin3(ground) * hemi_intensity / PI),
            sun_dir: v4(Vec3::from(sun_dir).normalize_or_zero()),
            sun: v4(lin3(sun_color) * sun_intensity / PI),
            emissive: v4(lin3(emissive)),
        },
    }
}

/// CPU mirror of the mesh vertex shader's `irradiance`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn irradiance(o: &MeshObject, n: [f32; 3]) -> [f32; 3] {
    let n = Vec3::from(n);
    let sky = Vec3::from_slice(&o.sky[..3]);
    let ground = Vec3::from_slice(&o.ground[..3]);
    let sun = Vec3::from_slice(&o.sun[..3]);
    let l = Vec3::from_slice(&o.sun_dir[..3]);
    let hemi = ground.lerp(sky, 0.5 * n.z + 0.5);
    (hemi + sun * n.dot(l).max(0.0)).to_array()
}

// ---- lines ----------------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub(crate) struct LineVertex {
    pub position: [f32; 3],
    pub color: [u8; 4],
}

/// Per-line-set uniform slot (group 1, dynamic offset, vertex stage only).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub(crate) struct LineObject {
    pub offset: [f32; 4],
    /// Linear RGBA multiplied with the (decoded) vertex colour.
    pub tint: [f32; 4],
}

pub(crate) struct PackedLines {
    pub origin: DVec3,
    pub vertices: Vec<LineVertex>,
    /// Vertices to draw: even for `List` (a trailing odd vertex is dropped), 0 for a `Strip` of
    /// fewer than two.
    pub draw_count: u32,
    pub tint: [f32; 4],
}

pub(crate) fn pack_lines(lines: &LineSet) -> PackedLines {
    let n = lines.positions.len();
    if let Some(c) = &lines.colors {
        assert_eq!(c.len(), n, "one colour per line vertex");
    }
    let origin = bbox_centre(&lines.positions);
    let vertices = (0..n)
        .map(|i| LineVertex {
            position: (DVec3::from(lines.positions[i]) - origin)
                .as_vec3()
                .to_array(),
            color: lines.colors.as_ref().map_or([255; 4], |c| c[i]),
        })
        .collect();
    let draw_count = match lines.topology {
        Topology::List => n & !1,
        Topology::Strip if n >= 2 => n,
        Topology::Strip => 0,
    } as u32;
    let tint = if lines.colors.is_some() {
        [1.0; 4]
    } else {
        linear_rgba(lines.color)
    };
    PackedLines {
        origin,
        vertices,
        draw_count,
        tint,
    }
}

// ---- textures -------------------------------------------------------------------------------

/// A texture ready for `create_texture_with_data`: every mip level, tightly packed, level 0
/// first. Sizes follow wgpu (`max(1, size >> level)`).
pub(crate) struct MipChain {
    pub width: u32,
    pub height: u32,
    pub levels: u32,
    pub data: Vec<u8>,
}

/// 2x2 box filter averaged in linear light (alpha linearly), edge texels clamped.
fn halve(w: u32, h: u32, src: &[u8], lut: &[f32; 256]) -> (u32, u32, Vec<u8>) {
    let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
    let mut out = Vec::with_capacity((nw * nh * 4) as usize);
    for y in 0..nh {
        for x in 0..nw {
            let mut sum = [0.0f32; 4];
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let sx = (2 * x + dx).min(w - 1);
                let sy = (2 * y + dy).min(h - 1);
                let i = ((sy * w + sx) * 4) as usize;
                for (c, s) in sum.iter_mut().take(3).enumerate() {
                    *s += lut[src[i + c] as usize];
                }
                sum[3] += f32::from(src[i + 3]) / 255.0;
            }
            for s in &sum[..3] {
                out.push((linear_to_srgb(s / 4.0) * 255.0).round() as u8);
            }
            out.push((sum[3] / 4.0 * 255.0).round() as u8);
        }
    }
    (nw, nh, out)
}

/// Shrinks the image by halving until it fits `max_dim` (WebGL2 guarantees only 2048), then
/// builds the full mip chain down to 1x1 (three.js mipmaps textures by default).
pub(crate) fn mip_chain(img: &TextureImage, max_dim: u32) -> MipChain {
    assert!(img.width > 0 && img.height > 0, "empty texture");
    assert_eq!(
        img.rgba.len(),
        (img.width * img.height * 4) as usize,
        "texture size does not match its pixels"
    );
    let lut: [f32; 256] = std::array::from_fn(|i| srgb_to_linear(i as f32 / 255.0));
    let (mut w, mut h) = (img.width, img.height);
    let mut base = std::borrow::Cow::Borrowed(img.rgba.as_slice());
    while w > max_dim.max(1) || h > max_dim.max(1) {
        let (nw, nh, next) = halve(w, h, &base, &lut);
        (w, h, base) = (nw, nh, std::borrow::Cow::Owned(next));
    }
    let levels = 32 - w.max(h).leading_zeros();
    let (width, height) = (w, h);
    let mut data = base.to_vec();
    let mut level = base.into_owned();
    for _ in 1..levels {
        let (nw, nh, next) = halve(w, h, &level, &lut);
        data.extend_from_slice(&next);
        (w, h, level) = (nw, nh, next);
    }
    MipChain {
        width,
        height,
        levels,
        data,
    }
}

// ---- draw lists -----------------------------------------------------------------------------

/// What `prepare` decided to draw for one object this frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Planned {
    pub id: u64,
    /// Uniform slot written by `prepare`.
    pub slot: u32,
    /// Geometry version the slot was written for; a later `upsert` makes the entry stale.
    pub version: u64,
    pub pipeline: usize,
}

/// One candidate for the draw list.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Candidate {
    pub id: u64,
    pub version: u64,
    pub visible: bool,
    pub blend: Blend,
    pub pipeline: usize,
    /// Distance from the eye, for back-to-front sorting of blended objects.
    pub distance: f64,
}

/// Splits visible objects into the opaque list (input order) and the blended list sorted far to
/// near (three.js sorts transparent objects back to front); slots are assigned in input order.
pub(crate) fn plan(candidates: &[Candidate]) -> (Vec<Planned>, Vec<Planned>) {
    let mut opaque = Vec::new();
    let mut blended = Vec::new();
    for (slot, c) in candidates.iter().filter(|c| c.visible).enumerate() {
        let p = Planned {
            id: c.id,
            slot: slot as u32,
            version: c.version,
            pipeline: c.pipeline,
        };
        if c.blend == Blend::Opaque {
            opaque.push(p);
        } else {
            blended.push((c.distance, p));
        }
    }
    // Stable sort keeps input order for equal distances.
    blended.sort_by(|a, b| b.0.total_cmp(&a.0));
    (opaque, blended.into_iter().map(|(_, p)| p).collect())
}

pub(crate) fn blend_index(b: Blend) -> usize {
    match b {
        Blend::Opaque => 0,
        Blend::Alpha => 1,
        Blend::Additive => 2,
    }
}

/// Mesh pipeline table index: blend x double-sided.
pub(crate) fn mesh_pipeline(blend: Blend, double_sided: bool) -> usize {
    blend_index(blend) * 2 + usize::from(double_sided)
}

/// Line pipeline table index: blend x depth test x topology.
pub(crate) fn line_pipeline(blend: Blend, depth_test: bool, topology: Topology) -> usize {
    blend_index(blend) * 4 + usize::from(depth_test) * 2 + usize::from(topology == Topology::Strip)
}

/// Bytes of `n` uniform slots of `stride`, `items[i]` at `i * stride`.
pub(crate) fn pack_slots<T: Pod>(items: &[T], stride: usize) -> Vec<u8> {
    let size = size_of::<T>();
    assert!(size <= stride);
    let mut out = vec![0u8; stride * items.len()];
    for (i, item) in items.iter().enumerate() {
        out[i * stride..i * stride + size].copy_from_slice(bytemuck::bytes_of(item));
    }
    out
}

/// Slot stride for a uniform of `size` bytes under the device's dynamic-offset alignment.
pub(crate) fn slot_stride(size: u64, align: u64) -> u64 {
    size.div_ceil(align) * align
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    // ---- colour

    #[test]
    fn srgb_round_trips_and_matches_reference_values() {
        assert!(close(srgb_to_linear(0.5), 0.214_041));
        assert!(close(srgb_to_linear(1.0), 1.0));
        assert_eq!(srgb_to_linear(0.0), 0.0);
        for i in 0..=255 {
            let c = i as f32 / 255.0;
            assert!(close(linear_to_srgb(srgb_to_linear(c)), c), "{i}");
        }
    }

    // ---- fog

    #[test]
    fn no_fog_never_tints() {
        let p = fog_params(None);
        assert_eq!(fog_amount(&p, 0.0), 0.0);
        assert_eq!(fog_amount(&p, 1e6), 0.0);
        assert_eq!(apply_fog(&p, [0.2, 0.3, 0.4], 1e6), [0.2, 0.3, 0.4]);
    }

    #[test]
    fn linear_fog_is_three_js_smoothstep_between_near_and_far() {
        let fog = Fog {
            color: [0.039, 0.067, 0.157],
            near: 10.0,
            far: 110.0,
        };
        let p = fog_params(Some(fog));
        assert!(close(p.scale, 0.01) && close(p.bias, -0.1));
        assert_eq!(fog_amount(&p, 0.0), 0.0);
        assert_eq!(fog_amount(&p, 10.0), 0.0);
        assert!(close(fog_amount(&p, 60.0), 0.5));
        // smoothstep(0, 1, 0.25) = 0.15625
        assert!(close(fog_amount(&p, 35.0), 0.156_25));
        assert!(close(fog_amount(&p, 110.0), 1.0));
        assert!(close(fog_amount(&p, 500.0), 1.0));
        let c = apply_fog(&p, [1.0, 1.0, 1.0], 1000.0);
        assert!((0..3).all(|i| close(c[i], fog.color[i])));
    }

    #[test]
    fn degenerate_fog_range_does_not_divide_by_zero() {
        let p = fog_params(Some(Fog {
            color: [1.0; 3],
            near: 50.0,
            far: 50.0,
        }));
        assert!(p.scale.is_finite() && p.bias.is_finite());
        assert_eq!(fog_amount(&p, 49.0), 0.0);
        assert_eq!(fog_amount(&p, 51.0), 1.0);
    }

    #[test]
    fn frame_uniform_layout_matches_the_wgsl_struct() {
        assert_eq!(size_of::<FrameUniform>(), 96);
        let u = frame_uniform([[0.0; 4]; 4], None, true);
        assert_eq!(u.params[2], 1.0);
        assert_eq!(u.fog_color[3], 0.0);
    }

    // ---- normals

    fn quad() -> MeshData {
        // Unit square in the z = 0 plane, counter-clockwise seen from +Z.
        MeshData {
            positions: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            indices: vec![0, 1, 2, 0, 2, 3],
            ..Default::default()
        }
    }

    #[test]
    fn smooth_normals_of_a_ccw_quad_point_up() {
        let n = smooth_normals(&quad().positions, &quad().indices);
        assert_eq!(n, vec![[0.0, 0.0, 1.0]; 4]);
    }

    #[test]
    fn smooth_normals_are_area_weighted() {
        // Vertex 0 is shared by a big triangle facing +Z and a small one facing +X.
        let positions = [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let n = smooth_normals(&positions, &[0, 1, 2, 0, 3, 4]);
        let v = Vec3::from(n[0]);
        assert!(close(v.length(), 1.0));
        // Areas 50 and 0.5: the +Z face dominates.
        assert!(v.z > 0.99 && v.x > 0.0);
        // Unused vertex gets +Z rather than NaN.
        let n = smooth_normals(&positions, &[0, 1, 2]);
        assert_eq!(n[4], [0.0, 0.0, 1.0]);
    }

    #[test]
    fn degenerate_triangles_do_not_produce_nan() {
        let positions = [[0.0; 3], [0.0; 3], [0.0; 3]];
        let n = smooth_normals(&positions, &[0, 1, 2]);
        assert_eq!(n, vec![[0.0, 0.0, 1.0]; 3]);
    }

    #[test]
    fn flat_meshes_unshare_vertices_with_face_normals() {
        // A box corner: one face up (+Z), one face south (-Y), sharing the edge 0-1.
        let mesh = MeshData {
            positions: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, -1.0],
            ],
            colors: Some(vec![
                [10, 20, 30, 255],
                [40, 50, 60, 255],
                [1, 2, 3, 4],
                [5, 6, 7, 8],
            ]),
            indices: vec![0, 1, 2, 1, 0, 3],
            flat: true,
            ..Default::default()
        };
        let p = pack_mesh(&mesh);
        assert_eq!(p.vertices.len(), 6);
        assert_eq!(p.indices, (0..6).collect::<Vec<u32>>());
        for v in &p.vertices[..3] {
            assert_eq!(v.normal, [0.0, 0.0, 1.0]);
        }
        for v in &p.vertices[3..] {
            assert_eq!(v.normal, [0.0, -1.0, 0.0]);
        }
        // Attributes travel with their vertex.
        assert_eq!(p.vertices[3].color, [40, 50, 60, 255]);
        assert_eq!(p.vertices[5].color, [5, 6, 7, 8]);
    }

    #[test]
    fn given_normals_are_kept_and_geometry_stays_indexed() {
        let mut m = quad();
        m.normals = Some(vec![[1.0, 0.0, 0.0]; 4]);
        m.flat = true;
        let p = pack_mesh(&m);
        assert_eq!(p.vertices.len(), 4);
        assert_eq!(p.indices, m.indices);
        assert!(p.vertices.iter().all(|v| v.normal == [1.0, 0.0, 0.0]));
    }

    // ---- packing

    #[test]
    fn mesh_vertices_are_36_bytes_relative_to_the_bbox_centre() {
        assert_eq!(size_of::<MeshVertex>(), 36);
        let mut m = quad();
        m.uvs = Some(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        let p = pack_mesh(&m);
        assert_eq!(p.origin, DVec3::new(0.5, 0.5, 0.0));
        assert_eq!(p.vertices[0].position, [-0.5, -0.5, 0.0]);
        assert_eq!(p.vertices[2].uv, [1.0, 1.0]);
        // White when no vertex colours are given.
        assert_eq!(p.vertices[1].color, [255; 4]);
    }

    #[test]
    fn far_meshes_keep_centimetre_detail() {
        // 500 km from the ENU origin, 1 cm apart: f32 alone would merge these.
        let base = 500_000.0;
        let m = MeshData {
            positions: vec![
                [base, base, 10.0],
                [base + 0.01, base, 10.0],
                [base, base + 0.01, 10.0],
            ],
            indices: vec![0, 1, 2],
            ..Default::default()
        };
        let p = pack_mesh(&m);
        let dx = p.vertices[1].position[0] - p.vertices[0].position[0];
        assert!((dx - 0.01).abs() < 1e-6, "{dx}");
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn bad_indices_are_rejected() {
        let mut m = quad();
        m.indices = vec![0, 1, 9];
        pack_mesh(&m);
    }

    #[test]
    fn lines_pack_colours_and_counts() {
        assert_eq!(size_of::<LineVertex>(), 16);
        let set = |n: usize, topology, colors: Option<Vec<[u8; 4]>>| LineSet {
            positions: (0..n).map(|i| [i as f64, 0.0, 0.0]).collect(),
            topology,
            colors,
            color: [1.0, 0.5, 0.0, 0.9],
            blend: Blend::Alpha,
            depth_test: true,
        };
        let p = pack_lines(&set(5, Topology::List, None));
        assert_eq!(p.draw_count, 4);
        assert_eq!(p.origin, DVec3::new(2.0, 0.0, 0.0));
        assert_eq!(p.vertices[0].color, [255; 4]);
        assert!(close(p.tint[1], 0.214_041) && p.tint[3] == 0.9);
        assert_eq!(pack_lines(&set(1, Topology::Strip, None)).draw_count, 0);
        assert_eq!(pack_lines(&set(3, Topology::Strip, None)).draw_count, 3);
        let c = pack_lines(&set(
            2,
            Topology::List,
            Some(vec![[1, 2, 3, 4], [5, 6, 7, 8]]),
        ));
        assert_eq!(c.tint, [1.0; 4]);
        assert_eq!(c.vertices[1].color, [5, 6, 7, 8]);
    }

    // ---- materials

    #[test]
    fn mesh_object_layout_matches_the_wgsl_struct() {
        assert_eq!(size_of::<MeshObject>(), 112);
        assert_eq!(size_of::<LineObject>(), 32);
    }

    #[test]
    fn unlit_is_irradiance_one_without_emissive() {
        let m = Material {
            shading: Shading::Unlit,
            tint: [1.0, 1.0, 1.0, 0.5],
            blend: Blend::Alpha,
            double_sided: true,
        };
        let o = mesh_object(&m, DVec3::new(1.0, 2.0, 3.0));
        assert_eq!(o.offset, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(o.tint, [1.0, 1.0, 1.0, 0.5]);
        for n in [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]] {
            assert_eq!(irradiance(&o, n), [1.0, 1.0, 1.0]);
        }
        assert_eq!(o.emissive, [0.0; 4]);
    }

    #[test]
    fn lambert_follows_three_js_hemisphere_plus_sun() {
        let m = Material {
            shading: Shading::Lambert {
                sky: [1.0, 1.0, 1.0],
                ground: [0.0, 0.0, 0.0],
                hemi_intensity: PI,
                sun_dir: [0.0, 0.0, 2.0],
                sun_color: [1.0, 1.0, 1.0],
                sun_intensity: PI,
                emissive: [0.5, 0.5, 0.5],
            },
            tint: [1.0; 4],
            blend: Blend::Opaque,
            double_sided: false,
        };
        let o = mesh_object(&m, DVec3::ZERO);
        // Facing up: full sky + full sun = 2 (intensity pi cancels the Lambert 1/pi).
        assert!(
            irradiance(&o, [0.0, 0.0, 1.0])
                .iter()
                .all(|&c| close(c, 2.0))
        );
        // Facing down: ground only.
        assert!(
            irradiance(&o, [0.0, 0.0, -1.0])
                .iter()
                .all(|&c| close(c, 0.0))
        );
        // Sideways: half sky, sun grazing.
        assert!(
            irradiance(&o, [1.0, 0.0, 0.0])
                .iter()
                .all(|&c| close(c, 0.5))
        );
        // Sun direction is normalised; emissive is linearised.
        assert_eq!(o.sun_dir, [0.0, 0.0, 1.0, 0.0]);
        assert!(close(o.emissive[0], 0.214_041));
    }

    // ---- textures

    #[test]
    fn mip_chain_follows_wgpu_level_sizes() {
        let img = TextureImage {
            width: 5,
            height: 2,
            rgba: vec![200; 5 * 2 * 4],
        };
        let m = mip_chain(&img, 2048);
        assert_eq!((m.width, m.height, m.levels), (5, 2, 3));
        // 5x2 + 2x1 + 1x1 texels.
        assert_eq!(m.data.len(), (10 + 2 + 1) * 4);
        // A uniform image stays uniform at every level.
        assert!(m.data.iter().all(|&b| b == 200));
    }

    #[test]
    fn mips_average_in_linear_light() {
        // Black and white texels average to linear 0.5, which is sRGB 188, not 128.
        let img = TextureImage {
            width: 2,
            height: 1,
            rgba: vec![0, 0, 0, 255, 255, 255, 255, 255],
        };
        let m = mip_chain(&img, 2048);
        assert_eq!(m.levels, 2);
        assert_eq!(&m.data[8..], &[188, 188, 188, 255]);
    }

    #[test]
    fn oversized_textures_shrink_to_the_device_limit() {
        let img = TextureImage {
            width: 4096,
            height: 8,
            rgba: vec![7; 4096 * 8 * 4],
        };
        let m = mip_chain(&img, 2048);
        assert_eq!((m.width, m.height, m.levels), (2048, 4, 12));
    }

    // ---- draw lists

    fn cand(id: u64, blend: Blend, visible: bool, distance: f64) -> Candidate {
        Candidate {
            id,
            version: id * 10,
            visible,
            blend,
            pipeline: blend_index(blend),
            distance,
        }
    }

    #[test]
    fn plan_skips_hidden_and_sorts_blended_far_to_near() {
        let (opaque, blended) = plan(&[
            cand(1, Blend::Opaque, true, 5.0),
            cand(2, Blend::Alpha, true, 10.0),
            cand(3, Blend::Additive, false, 99.0),
            cand(4, Blend::Additive, true, 50.0),
            cand(5, Blend::Opaque, true, 1.0),
        ]);
        assert_eq!(opaque.iter().map(|p| p.id).collect::<Vec<_>>(), [1, 5]);
        assert_eq!(blended.iter().map(|p| p.id).collect::<Vec<_>>(), [4, 2]);
        // Slots are dense over visible objects and carry their version.
        let mut slots: Vec<u32> = opaque.iter().chain(&blended).map(|p| p.slot).collect();
        slots.sort();
        assert_eq!(slots, [0, 1, 2, 3]);
        assert_eq!(blended[0].version, 40);
    }

    #[test]
    fn pipeline_tables_are_dense() {
        let mut seen = std::collections::BTreeSet::new();
        for b in [Blend::Opaque, Blend::Alpha, Blend::Additive] {
            for d in [false, true] {
                assert!(seen.insert(mesh_pipeline(b, d)));
            }
        }
        assert_eq!(
            seen.into_iter().collect::<Vec<_>>(),
            (0..6).collect::<Vec<_>>()
        );
        let mut seen = std::collections::BTreeSet::new();
        for b in [Blend::Opaque, Blend::Alpha, Blend::Additive] {
            for d in [false, true] {
                for t in [Topology::List, Topology::Strip] {
                    assert!(seen.insert(line_pipeline(b, d, t)));
                }
            }
        }
        assert_eq!(
            seen.into_iter().collect::<Vec<_>>(),
            (0..12).collect::<Vec<_>>()
        );
    }

    #[test]
    fn slots_respect_dynamic_offset_alignment() {
        assert_eq!(slot_stride(112, 256), 256);
        assert_eq!(slot_stride(32, 64), 64);
        assert_eq!(slot_stride(256, 256), 256);
        let bytes = pack_slots(&[[1.0f32; 4], [2.0f32; 4]], 256);
        assert_eq!(bytes.len(), 512);
        assert_eq!(&bytes[256..260], bytemuck::bytes_of(&2.0f32));
        assert_eq!(&bytes[16..20], &[0; 4]);
    }
}
