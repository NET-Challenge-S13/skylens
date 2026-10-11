// Mesh blocks (deltamesh BlockMesh) drawn next to pointblitz points.
// Positions are absolute ENU metres (x east, y north, z up); `view_proj` maps them to clip space
// with pointblitz's conventions (right-handed, reversed-Z, infinite far plane).
// Lighting: one fixed directional light plus ambient, vertex colours. Kept to WebGL2-compatible
// features: uniforms only, no storage buffers.

struct Uniforms {
    view_proj: mat4x4<f32>,
    // xyz: unit vector towards the light (ENU), w: ambient term in [0, 1].
    light: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
};

@vertex
fn vs(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip = u.view_proj * vec4<f32>(in.position, 1.0);
    out.normal = in.normal;
    out.color = in.color.rgb;
    return out;
}

@fragment
fn fs(in: VsOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    // Reconstructed surfaces can be open, so back faces are drawn too and lit from their side.
    var n = normalize(in.normal);
    if !front {
        n = -n;
    }
    let ambient = u.light.w;
    let diffuse = max(dot(n, u.light.xyz), 0.0);
    let shade = ambient + (1.0 - ambient) * diffuse;
    return vec4<f32>(in.color * shade, 1.0);
}
