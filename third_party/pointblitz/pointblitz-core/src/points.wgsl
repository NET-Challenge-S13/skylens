// Opaque round points of a fixed screen size (decisions 0006, 0007).
// One instance per point (the chunk body), four vertices per instance (triangle strip).
// Chunk format v2 (decision 0051): positions are u16×3 quantised to the chunk's bounding box; the
// colour is RGBA8 (stride 12, entry vs12) or RGB565 in the fourth u16 (stride 8, entry vs8).

struct Camera {
    // ENU → clip, with the camera at the origin (camera-relative, decision 0013).
    view_proj: mat4x4<f32>,
    viewport_px: vec2<f32>,
    point_px: f32,
    _pad: f32,
};

struct Chunk {
    // Chunk origin + bounding-box minimum − camera position (computed in f64 on the CPU).
    offset: vec4<f32>,
    // Bounding-box size ÷ 65535: metres per quantisation step, per axis.
    scale: vec4<f32>,
};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(1) @binding(0) var<uniform> chunk: Chunk;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) corner: vec2<f32>,
};

fn place(vi: u32, q: vec3<u32>, color: vec4<f32>) -> VsOut {
    // Strip order: (-1,-1) (1,-1) (-1,1) (1,1).
    let corner = vec2<f32>(f32(vi & 1u) * 2.0 - 1.0, f32(vi >> 1u) * 2.0 - 1.0);
    let position = chunk.offset.xyz + vec3<f32>(q) * chunk.scale.xyz;
    var clip = cam.view_proj * vec4<f32>(position, 1.0);
    // Expand in clip space so the quad is point_px wide on screen regardless of depth.
    clip = vec4<f32>(clip.xy + corner * (cam.point_px / cam.viewport_px) * clip.w, clip.zw);
    var out: VsOut;
    out.clip = clip;
    out.color = color;
    out.corner = corner;
    return out;
}

@vertex
fn vs12(
    @builtin(vertex_index) vi: u32,
    @location(0) q: vec4<u32>,
    @location(1) color: vec4<f32>,
) -> VsOut {
    return place(vi, q.xyz, color);
}

@vertex
fn vs8(
    @builtin(vertex_index) vi: u32,
    @location(0) q: vec4<u32>,
) -> VsOut {
    let c = q.w;
    let color = vec4<f32>(
        f32((c >> 11u) & 31u) / 31.0,
        f32((c >> 5u) & 63u) / 63.0,
        f32(c & 31u) / 31.0,
        1.0,
    );
    return place(vi, q.xyz, color);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    if dot(in.corner, in.corner) > 1.0 {
        discard;
    }
    return in.color;
}
