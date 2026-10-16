// Animated round points: every slot carries target/previous position, colour, a mode and a start
// time; the position, opacity and visibility are interpolated here from `cam.params.w` (now), so
// the CPU only writes a slot when it changes. color.wgsl is prepended (fog).
// Modes: 0 static, 1 fade in, 2 fade out, 3 glide (ease-out cubic), 4 dead (free slot).
// Opacity is screen-door dithered so the pipeline stays opaque and depth-correct, without sorting.

struct Camera {
    // ENU relative to the layer origin ... → clip, with the camera at the origin.
    view_proj: mat4x4<f32>,
    viewport_px: vec2<f32>,
    point_px: f32,
    point_m: f32,
    fog_color: vec4<f32>,
    // x, y: fog scale and bias; z: focal length in pixels; w: now (book time, seconds).
    params: vec4<f32>,
    // Layer origin − camera position (f64 on the CPU); w: origin height (for height colouring).
    origin: vec4<f32>,
    // x: colour mode (0 per-point rgb, 1 height ramp, 2 flat); y, z: ramp low/high; w: unused.
    color_mode: vec4<f32>,
    flat_color: vec4<f32>,
    // fade in, fade out, glide durations in seconds.
    durations: vec4<f32>,
};

@group(0) @binding(0) var<uniform> cam: Camera;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) corner: vec2<f32>,
    // x: fog t, y: opacity.
    @location(2) misc: vec4<f32>,
    @location(3) fog: vec4<f32>,
};

// The legacy blue - cyan - green - yellow - red ramp, shaded like the baked height colours.
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
    @location(1) prev: vec3<f32>,
    @location(2) rgb_mode: vec4<u32>,
    @location(3) start: f32,
) -> VsOut {
    let mode = rgb_mode.w;
    let age = cam.params.w - start;
    var p = pos;
    var alpha = 1.0;
    if mode == 1u {
        alpha = clamp(age / cam.durations.x, 0.0, 1.0);
    } else if mode == 2u {
        alpha = 1.0 - clamp(age / cam.durations.y, 0.0, 1.0);
    } else if mode == 3u {
        let k = 1.0 - clamp(age / cam.durations.z, 0.0, 1.0);
        p = mix(pos, prev, k * k * k);
    }
    var out: VsOut;
    if mode >= 4u || alpha <= 0.0 {
        // Outside the clip volume: culled.
        out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.color = vec4<f32>(0.0);
        out.corner = vec2<f32>(0.0);
        out.misc = vec4<f32>(0.0);
        out.fog = vec4<f32>(0.0);
        return out;
    }
    var rgb = vec3<f32>(rgb_mode.xyz) / 255.0;
    if cam.color_mode.x > 1.5 {
        rgb = cam.flat_color.rgb;
    } else if cam.color_mode.x > 0.5 {
        let z = p.z + cam.origin.w;
        rgb = ramp((z - cam.color_mode.y) / max(cam.color_mode.z - cam.color_mode.y, 0.01));
    }
    // Strip order: (-1,-1) (1,-1) (-1,1) (1,1).
    let corner = vec2<f32>(f32(vi & 1u) * 2.0 - 1.0, f32(vi >> 1u) * 2.0 - 1.0);
    var clip = cam.view_proj * vec4<f32>(cam.origin.xyz + p, 1.0);
    let world_px = clamp(cam.point_m * cam.params.z / max(clip.w, 1e-6), 1.0, 512.0);
    let px = select(cam.point_px, world_px, cam.point_m > 0.0);
    let fog_t = clip.w * cam.params.x + cam.params.y;
    clip = vec4<f32>(clip.xy + corner * (px / cam.viewport_px) * clip.w, clip.zw);
    out.clip = clip;
    out.color = vec4<f32>(rgb, 1.0);
    out.corner = corner;
    out.misc = vec4<f32>(fog_t, alpha, 0.0, 0.0);
    out.fog = cam.fog_color;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    if dot(in.corner, in.corner) > 1.0 {
        discard;
    }
    if in.misc.y < 1.0 {
        // Interleaved gradient noise as the screen-door threshold.
        let n = fract(52.9829189 * fract(dot(in.clip.xy, vec2<f32>(0.06711056, 0.00583715))));
        if in.misc.y <= n {
            discard;
        }
    }
    return vec4<f32>(apply_fog(in.color.rgb, in.misc.x, in.fog), 1.0);
}
