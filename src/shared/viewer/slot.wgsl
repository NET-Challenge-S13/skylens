// Slot-indexed point cloud (SkyLens): one instance per server slot, per-point glide and fade
// driven by timestamps. Port of the three.js ShaderMaterial in
// the legacy status board: position = mix(from, pos, ease(u)), alpha from tm.y / tm.z, a
// screen-door dither instead of blending (opaque, depth-correct, no sorting).
// WebGL2: the uniform is read in the vertex stage only; the fragment stage gets varyings.

// Linear fog like three.js `Fog` (same as pointblitz color.wgsl): t is 0 at fog near and 1 at
// fog far, fog.a is 0 when fog is off.
fn apply_fog(rgb: vec3<f32>, t: f32, fog: vec4<f32>) -> vec3<f32> {
    return mix(rgb, fog.rgb, smoothstep(0.0, 1.0, t) * fog.a);
}

struct U {
    // ENU (about the eye offset below) -> clip.
    view_proj: mat4x4<f32>,
    viewport_px: vec2<f32>,
    point_px: f32,
    point_m: f32,
    fog_color: vec4<f32>,
    // x, y: fog t = view depth * x + y; z: focal length in pixels.
    params: vec4<f32>,
    // -eye in f64 on the CPU, rounded once: position + offset is camera-relative.
    offset: vec4<f32>,
    // now, glide seconds, fade seconds, colour mode (0 rgb, 1 zone, 2 height).
    time: vec4<f32>,
    // height ramp low, high.
    height: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: U;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    // a = fade alpha (screen-door threshold).
    @location(0) color: vec4<f32>,
    @location(1) corner: vec2<f32>,
    @location(2) misc: vec4<f32>,
    @location(3) fog: vec4<f32>,
};

fn hsv(h: f32) -> vec3<f32> {
    return clamp(abs((vec3<f32>(h * 6.0) + vec3<f32>(0.0, 4.0, 2.0)) % vec3<f32>(6.0)
        - vec3<f32>(3.0)) - vec3<f32>(1.0), vec3<f32>(0.0), vec3<f32>(1.0));
}

// The legacy height ramp (blue - cyan - green - yellow - red), shaded like `height_color`.
fn ramp(t: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0);
    let c = vec3<f32>(
        1.5 - abs(4.0 * x - 3.0),
        1.5 - abs(4.0 * x - 2.0),
        1.5 - abs(4.0 * x - 1.0),
    );
    return clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)) * 0.91;
}

@vertex
fn vs(
    @builtin(vertex_index) vi: u32,
    @location(0) pos: vec3<f32>,
    @location(1) p0: vec3<f32>,
    @location(2) tm: vec3<f32>,
    @location(3) rgbz: u32,
) -> VsOut {
    let now = u.time.x;
    let k = clamp((now - tm.x) / max(u.time.y, 1e-4), 0.0, 1.0);
    // Ease-in-out quad.
    let q = -2.0 * k + 2.0;
    let e = select(1.0 - q * q / 2.0, 2.0 * k * k, k < 0.5);
    let p = mix(p0, pos, e);
    let fade = max(u.time.z, 1e-4);
    let a = clamp((now - tm.y) / fade, 0.0, 1.0) * clamp(1.0 - (now - tm.z) / fade, 0.0, 1.0);

    var out: VsOut;
    out.corner = vec2<f32>(0.0);
    out.misc = vec4<f32>(0.0);
    out.fog = vec4<f32>(0.0);
    if a <= 0.01 {
        // Outside the clip volume, all four vertices alike: a degenerate quad.
        out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.color = vec4<f32>(0.0);
        return out;
    }

    var rgb = unpack4x8unorm(rgbz).xyz;
    let mode = u.time.w;
    if mode > 0.5 && mode < 1.5 {
        let zone = f32(rgbz >> 24u);
        rgb = vec3<f32>(0.25) + 0.75 * hsv(fract(zone * 0.618034));
    } else if mode > 1.5 {
        rgb = ramp((p.z - u.height.x) / max(u.height.y - u.height.x, 0.01));
    }

    let corner = vec2<f32>(f32(vi & 1u) * 2.0 - 1.0, f32(vi >> 1u) * 2.0 - 1.0);
    var clip = u.view_proj * vec4<f32>(p + u.offset.xyz, 1.0);
    let world_px = clamp(u.point_m * u.params.z / max(clip.w, 1e-6), 1.0, 512.0);
    let px = select(u.point_px, world_px, u.point_m > 0.0);
    let fog_t = clip.w * u.params.x + u.params.y;
    clip = vec4<f32>(clip.xy + corner * (px / u.viewport_px) * clip.w, clip.zw);
    out.clip = clip;
    out.color = vec4<f32>(rgb, a);
    out.corner = corner;
    out.misc = vec4<f32>(fog_t, 0.0, 0.0, 0.0);
    out.fog = u.fog_color;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    if dot(in.corner, in.corner) > 1.0 {
        discard;
    }
    // Screen-door dither: opaque, depth-correct, no sorting.
    let h = fract(52.9829189 * fract(dot(in.clip.xy, vec2<f32>(0.06711056, 0.00583715))));
    if h > in.color.a {
        discard;
    }
    return vec4<f32>(apply_fog(in.color.rgb, in.misc.x, in.fog), 1.0);
}
