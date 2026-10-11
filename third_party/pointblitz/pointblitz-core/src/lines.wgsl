// 1-pixel lines (layers.rs): camera-relative positions, per-vertex colour x tint, linear fog.
// color.wgsl is prepended.

struct Frame {
    view_proj: mat4x4<f32>,
    fog_color: vec4<f32>,
    // x, y: fog t = view depth * x + y; z: 1 when the output must be sRGB-encoded.
    params: vec4<f32>,
};

struct Object {
    // Line set origin - camera position (computed in f64 on the CPU).
    offset: vec4<f32>,
    // Linear RGBA multiplied with the decoded vertex colour.
    tint: vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(0) var<uniform> object: Object;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    // x: fog t, y: sRGB-encode flag.
    @location(1) misc: vec4<f32>,
    @location(2) fog: vec4<f32>,
};

@vertex
fn vs(@location(0) position: vec3<f32>, @location(1) color: vec4<f32>) -> VsOut {
    let clip = frame.view_proj * vec4<f32>(object.offset.xyz + position, 1.0);
    var out: VsOut;
    out.clip = clip;
    out.color = vec4<f32>(srgb_to_linear(color.rgb), color.a) * object.tint;
    out.misc = vec4<f32>(clip.w * frame.params.x + frame.params.y, frame.params.z, 0.0, 0.0);
    out.fog = frame.fog_color;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    var rgb = clamp(in.color.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    if in.misc.y > 0.5 {
        rgb = linear_to_srgb(rgb);
    }
    return vec4<f32>(apply_fog(rgb, in.misc.x, in.fog), in.color.a);
}
