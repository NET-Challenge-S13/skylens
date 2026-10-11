// Opaque round points of a fixed screen size (decisions 0006, 0007).
// One instance per point (the chunk body), four vertices per instance (triangle strip).
// Chunk format v2 (decision 0051): positions are u16×3 quantised to the chunk's bounding box; the
// colour is RGBA8 (stride 12, entry vs12) or RGB565 in the fourth u16 (stride 8, entry vs8).
// SkyLens: color.wgsl is prepended (fog); sizes may be world metres with perspective attenuation.

struct Camera {
    // ENU → clip, with the camera at the origin (camera-relative, decision 0013).
    view_proj: mat4x4<f32>,
    viewport_px: vec2<f32>,
    // Diameter in pixels, used when point_m is 0.
    point_px: f32,
    // Diameter in metres (> 0 selects perspective attenuation: px = point_m * focal_px / depth).
    point_m: f32,
    // rgb in the colour target's space, a = 1 when fog is on.
    fog_color: vec4<f32>,
    // x, y: fog t = view depth * x + y; z: focal length in pixels; w: unused.
    params: vec4<f32>,
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
    // x: fog t.
    @location(2) misc: vec4<f32>,
    @location(3) fog: vec4<f32>,
};

fn place(vi: u32, q: vec3<u32>, color: vec4<f32>) -> VsOut {
    // Strip order: (-1,-1) (1,-1) (-1,1) (1,1).
    let corner = vec2<f32>(f32(vi & 1u) * 2.0 - 1.0, f32(vi >> 1u) * 2.0 - 1.0);
    let position = chunk.offset.xyz + vec3<f32>(q) * chunk.scale.xyz;
    var clip = cam.view_proj * vec4<f32>(position, 1.0);
    // clip.w is the view depth. World-sized points shrink with distance (three.js
    // sizeAttenuation); clamped so a point next to the eye cannot cover the screen.
    let world_px = clamp(cam.point_m * cam.params.z / max(clip.w, 1e-6), 1.0, 512.0);
    let px = select(cam.point_px, world_px, cam.point_m > 0.0);
    // Expand in clip space so the quad is px wide on screen.
    let fog_t = clip.w * cam.params.x + cam.params.y;
    clip = vec4<f32>(clip.xy + corner * (px / cam.viewport_px) * clip.w, clip.zw);
    var out: VsOut;
    out.clip = clip;
    out.color = color;
    out.corner = corner;
    out.misc = vec4<f32>(fog_t, 0.0, 0.0, 0.0);
    out.fog = cam.fog_color;
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
    return vec4<f32>(apply_fog(in.color.rgb, in.misc.x, in.fog), in.color.a);
}
