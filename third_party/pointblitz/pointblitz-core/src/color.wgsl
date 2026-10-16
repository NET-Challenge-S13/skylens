// Colour and fog helpers, prepended to the point, mesh and line shaders (layers.rs).
// Uniform buffers are read only in vertex stages (WebGL2 downlevel), so fragment stages get
// everything they need through varyings; values that are constant per draw interpolate to
// themselves, which avoids `@interpolate(flat)` (its default `first` sampling has no GLSL form).

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((max(c, vec3<f32>(0.0)) + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let x = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    let lo = x * 12.92;
    let hi = 1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    return select(hi, lo, x <= vec3<f32>(0.0031308));
}

// Linear fog like three.js `Fog`: `t` is 0 at fog near and 1 at fog far (computed per vertex
// from the view depth, which interpolates exactly), `fog.a` is 0 when fog is off. The mix happens
// in the colour target's space, after any sRGB encoding, as three.js does.
fn apply_fog(rgb: vec3<f32>, t: f32, fog: vec4<f32>) -> vec3<f32> {
    return mix(rgb, fog.rgb, smoothstep(0.0, 1.0, t) * fog.a);
}
