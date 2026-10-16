// Meshes (layers.rs): camera-relative positions, Lambert (hemisphere + sun + emissive) or unlit,
// optional sRGB texture, linear fog. Unlit is a Lambert with irradiance 1 and no emissive.
// color.wgsl is prepended.

struct Frame {
    // ENU -> clip, with the camera at the origin.
    view_proj: mat4x4<f32>,
    // rgb in the colour target's space, a = 1 when fog is on.
    fog_color: vec4<f32>,
    // x, y: fog t = view depth * x + y; z: 1 when the output must be sRGB-encoded.
    params: vec4<f32>,
};

struct Object {
    // Mesh origin - camera position (computed in f64 on the CPU).
    offset: vec4<f32>,
    // Linear RGBA multiplied with the vertex colour and the texture.
    tint: vec4<f32>,
    // Hemisphere colours and the sun, linear, with intensity and the Lambert 1/pi folded in.
    sky: vec4<f32>,
    ground: vec4<f32>,
    // Unit vector towards the sun.
    sun_dir: vec4<f32>,
    sun: vec4<f32>,
    emissive: vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(0) var<uniform> object: Object;
@group(2) @binding(0) var tex: texture_2d<f32>;
@group(2) @binding(1) var samp: sampler;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    // uv, fog t, sRGB-encode flag.
    @location(1) uv_fog: vec4<f32>,
    @location(2) irr_front: vec3<f32>,
    @location(3) irr_back: vec3<f32>,
    @location(4) emissive: vec3<f32>,
    @location(5) fog: vec4<f32>,
};

fn irradiance(n: vec3<f32>) -> vec3<f32> {
    // ENU: +Z is up for the hemisphere light.
    let hemi = mix(object.ground.rgb, object.sky.rgb, 0.5 * n.z + 0.5);
    return hemi + object.sun.rgb * max(dot(n, object.sun_dir.xyz), 0.0);
}

@vertex
fn vs(v: VsIn) -> VsOut {
    let clip = frame.view_proj * vec4<f32>(object.offset.xyz + v.position, 1.0);
    let n = normalize(v.normal);
    var out: VsOut;
    out.clip = clip;
    out.color = vec4<f32>(srgb_to_linear(v.color.rgb), v.color.a) * object.tint;
    // clip.w is the view depth (distance along the view axis), what three.js fog uses.
    out.uv_fog = vec4<f32>(v.uv, clip.w * frame.params.x + frame.params.y, frame.params.z);
    // Both sides are lit per vertex; the fragment picks one (three.js flips back-face normals).
    out.irr_front = irradiance(n);
    out.irr_back = irradiance(-n);
    out.emissive = object.emissive.rgb;
    out.fog = frame.fog_color;
    return out;
}

@fragment
fn fs(in: VsOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let base = in.color * textureSample(tex, samp, in.uv_fog.xy);
    let irr = select(in.irr_back, in.irr_front, front);
    var rgb = clamp(base.rgb * irr + in.emissive, vec3<f32>(0.0), vec3<f32>(1.0));
    if in.uv_fog.w > 0.5 {
        rgb = linear_to_srgb(rgb);
    }
    return vec4<f32>(apply_fog(rgb, in.uv_fog.z, in.fog), base.a);
}
