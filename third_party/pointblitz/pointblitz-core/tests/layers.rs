//! GPU tests for the SkyLens layers: one shared pass with meshes, points and lines.
//! Skipped (pass with a message) when no GPU adapter is available, e.g. in CI.

use pointblitz_core::headless::CAPTURE_FORMAT;
use pointblitz_core::{
    Blend, Camera, Fog, FrameStyle, Headless, LineRenderer, LineSet, Material, MeshData,
    MeshRenderer, PointSize, Renderer, Scene, Shading, TextureImage, Topology, begin_pass,
};
use pointblitz_io::Point;
use pointblitz_io::chunk::{FLAG_LAST_IN_GENERATION, encode};
use std::f32::consts::PI;

const W: u32 = 64;
const H: u32 = 64;
const NAVY: FrameStyle = FrameStyle {
    clear: [10.0 / 255.0, 17.0 / 255.0, 40.0 / 255.0, 1.0],
    fog: None,
};
const BLACK: FrameStyle = FrameStyle {
    clear: [0.0, 0.0, 0.0, 1.0],
    fog: None,
};

fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
    match pointblitz_core::headless::device() {
        Some((d, q, info)) => {
            eprintln!("adapter: {} ({:?})", info.name, info.backend);
            Some((d, q))
        }
        None => {
            eprintln!("no GPU adapter, skipped");
            None
        }
    }
}

fn pixel(img: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * W + x) * 4) as usize;
    img[i..i + 4].try_into().unwrap()
}

fn near(a: [u8; 4], b: [u8; 4], tol: u8) -> bool {
    (0..4).all(|i| a[i].abs_diff(b[i]) <= tol)
}

fn looking_north() -> Camera {
    Camera::new([0.0, -10.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 50.0)
}

fn px(cam: &Camera, p: [f64; 3]) -> (u32, u32) {
    let [x, y] = cam.project_px(p, W, H).expect("in front of camera");
    (x as u32, y as u32)
}

/// A 2 m square in the plane y = `y`, facing the camera (-Y), counter-clockwise on screen.
fn wall(y: f64) -> MeshData {
    MeshData {
        positions: vec![
            [-1.0, y, -1.0],
            [1.0, y, -1.0],
            [1.0, y, 1.0],
            [-1.0, y, 1.0],
        ],
        uvs: Some(vec![[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]),
        indices: vec![0, 1, 2, 0, 2, 3],
        ..Default::default()
    }
}

fn unlit(tint: [f32; 4], blend: Blend) -> Material {
    Material {
        shading: Shading::Unlit,
        tint,
        blend,
        double_sided: false,
    }
}

fn scene_of(device: &wgpu::Device, points: &[([f32; 3], [u8; 3])]) -> Scene {
    let pts: Vec<Point> = points
        .iter()
        .map(|&(position, color)| Point { position, color })
        .collect();
    let mut scene = Scene::new();
    scene
        .insert(device, &encode(0, 0, FLAG_LAST_IN_GENERATION, &pts))
        .unwrap();
    scene
}

#[test]
fn begin_pass_clears_to_the_style_colour() {
    let Some((device, queue)) = gpu() else { return };
    let mut hl = Headless::new(&device, [W, H]);
    let img = hl.capture_with(&device, &queue, |enc, color, depth| {
        drop(begin_pass(enc, color, depth, &NAVY));
    });
    assert_eq!(pixel(&img, 0, 0), [10, 17, 40, 255]);
    assert_eq!(pixel(&img, W - 1, H - 1), [10, 17, 40, 255]);
}

#[test]
fn unlit_mesh_keeps_its_colour_and_culls_back_faces() {
    let Some((device, queue)) = gpu() else { return };
    let cam = looking_north();
    let mut meshes = MeshRenderer::new(&device, CAPTURE_FORMAT);
    meshes.upsert(&device, &queue, 1, &wall(0.0), None, unlit([1.0, 0.5, 0.0, 1.0], Blend::Opaque));
    meshes.prepare(&queue, [W, H], &cam, None);
    let mut hl = Headless::new(&device, [W, H]);
    let img = hl.capture_with(&device, &queue, |enc, color, depth| {
        let mut pass = begin_pass(enc, color, depth, &NAVY);
        meshes.draw_opaque(&mut pass);
    });
    // sRGB in, sRGB out: 0.5 stays 128.
    assert!(near(pixel(&img, W / 2, H / 2), [255, 128, 0, 255], 1));

    // Seen from behind, a single-sided mesh disappears.
    let behind = Camera::new([0.0, 10.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 50.0);
    meshes.prepare(&queue, [W, H], &behind, None);
    let img = hl.capture_with(&device, &queue, |enc, color, depth| {
        let mut pass = begin_pass(enc, color, depth, &NAVY);
        meshes.draw_opaque(&mut pass);
    });
    assert_eq!(pixel(&img, W / 2, H / 2), [10, 17, 40, 255]);
}

#[test]
fn fog_reaches_its_colour_past_far() {
    let Some((device, queue)) = gpu() else { return };
    let cam = looking_north();
    let fog = Fog {
        color: [1.0, 0.0, 1.0],
        near: 1.0,
        far: 5.0,
    };
    let mut meshes = MeshRenderer::new(&device, CAPTURE_FORMAT);
    meshes.upsert(&device, &queue, 1, &wall(0.0), None, unlit([0.0, 1.0, 0.0, 1.0], Blend::Opaque));
    meshes.prepare(&queue, [W, H], &cam, Some(fog));
    let style = FrameStyle {
        fog: Some(fog),
        ..NAVY
    };
    let img = Headless::new(&device, [W, H]).capture_with(&device, &queue, |enc, color, depth| {
        let mut pass = begin_pass(enc, color, depth, &style);
        meshes.draw_opaque(&mut pass);
    });
    assert!(near(pixel(&img, W / 2, H / 2), [255, 0, 255, 255], 1));
    // The background is the clear colour, not fogged.
    assert_eq!(pixel(&img, 0, 0), [10, 17, 40, 255]);
}

#[test]
fn lambert_matches_three_js_numbers() {
    let Some((device, queue)) = gpu() else { return };
    let cam = looking_north();
    // Wall normal is horizontal: hemisphere gives mix(ground, sky, 0.5) = 0.5 linear with a white
    // sky of intensity pi and a black ground; 0.5 linear is sRGB 188.
    let lambert = |sun_intensity: f32| Material {
        shading: Shading::Lambert {
            sky: [1.0; 3],
            ground: [0.0; 3],
            hemi_intensity: PI,
            sun_dir: [0.0, -1.0, 0.0],
            sun_color: [1.0; 3],
            sun_intensity,
            emissive: [0.0; 3],
        },
        tint: [1.0; 4],
        blend: Blend::Opaque,
        double_sided: false,
    };
    let mut meshes = MeshRenderer::new(&device, CAPTURE_FORMAT);
    let mut hl = Headless::new(&device, [W, H]);
    let mut shot = |m: Material| {
        meshes.upsert(&device, &queue, 1, &wall(0.0), None, m);
        meshes.prepare(&queue, [W, H], &cam, None);
        let img = hl.capture_with(&device, &queue, |enc, color, depth| {
            let mut pass = begin_pass(enc, color, depth, &BLACK);
            meshes.draw_opaque(&mut pass);
        });
        pixel(&img, W / 2, H / 2)
    };
    assert!(near(shot(lambert(0.0)), [188, 188, 188, 255], 2));
    // A sun facing the wall at intensity pi / 2 adds another 0.5: white.
    assert!(near(shot(lambert(PI / 2.0)), [255, 255, 255, 255], 1));
    // set_material switches without re-upload.
    meshes.set_material(1, unlit([1.0, 0.0, 0.0, 1.0], Blend::Opaque));
    meshes.prepare(&queue, [W, H], &cam, None);
    let img = hl.capture_with(&device, &queue, |enc, color, depth| {
        let mut pass = begin_pass(enc, color, depth, &BLACK);
        meshes.draw_opaque(&mut pass);
    });
    assert!(near(pixel(&img, W / 2, H / 2), [255, 0, 0, 255], 1));
}

#[test]
fn textures_are_sampled_with_their_uvs() {
    let Some((device, queue)) = gpu() else { return };
    let cam = looking_north();
    let texture = TextureImage {
        width: 2,
        height: 1,
        rgba: vec![255, 0, 0, 255, 0, 0, 255, 255],
    };
    let mut meshes = MeshRenderer::new(&device, CAPTURE_FORMAT);
    meshes.upsert(
        &device,
        &queue,
        7,
        &wall(0.0),
        Some(&texture),
        unlit([1.0; 4], Blend::Opaque),
    );
    meshes.prepare(&queue, [W, H], &cam, None);
    let img = Headless::new(&device, [W, H]).capture_with(&device, &queue, |enc, color, depth| {
        let mut pass = begin_pass(enc, color, depth, &BLACK);
        meshes.draw_opaque(&mut pass);
    });
    // Texel centres sit at u = 0.25 (red) and u = 0.75 (blue).
    let (lx, ly) = px(&cam, [-0.5, 0.0, 0.0]);
    let (rx, ry) = px(&cam, [0.5, 0.0, 0.0]);
    let l = pixel(&img, lx, ly);
    let r = pixel(&img, rx, ry);
    assert!(l[0] > 200 && l[2] < 60, "{l:?}");
    assert!(r[2] > 200 && r[0] < 60, "{r:?}");
}

#[test]
fn meshes_points_and_lines_share_one_depth_buffer() {
    let Some((device, queue)) = gpu() else { return };
    let cam = looking_north();
    let mut meshes = MeshRenderer::new(&device, CAPTURE_FORMAT);
    meshes.upsert(&device, &queue, 1, &wall(0.0), None, unlit([0.0, 0.0, 1.0, 1.0], Blend::Opaque));
    // Green point in front of the wall, red one behind it.
    let front = [0.5f32, -5.0, 0.3];
    let back = [-0.5f32, 5.0, -0.3];
    let scene = scene_of(&device, &[(front, [0, 255, 0]), (back, [255, 0, 0])]);
    let mut points = Renderer::new(&device, CAPTURE_FORMAT);
    points.set_point_size(PointSize::Pixels(4.0));
    // A depth-tested line behind the wall and an always-on-top one, both white.
    let line = |z: f64, depth_test: bool| LineSet {
        positions: vec![[-3.0, 2.0, z], [3.0, 2.0, z]],
        topology: Topology::List,
        colors: None,
        color: [1.0; 4],
        blend: Blend::Opaque,
        depth_test,
    };
    let mut lines = LineRenderer::new(&device, CAPTURE_FORMAT);
    lines.upsert(&device, &queue, 1, &line(0.5, true));
    lines.upsert(&device, &queue, 2, &line(-0.5, false));

    meshes.prepare(&queue, [W, H], &cam, None);
    points.prepare(&device, &queue, [W, H], &cam, &[&scene], None);
    lines.prepare(&queue, [W, H], &cam, None);
    let img = Headless::new(&device, [W, H]).capture_with(&device, &queue, |enc, color, depth| {
        let mut pass = begin_pass(enc, color, depth, &BLACK);
        meshes.draw_opaque(&mut pass);
        points.draw(&mut pass, &[&scene], None);
        meshes.draw_transparent(&mut pass);
        lines.draw(&mut pass);
    });
    let (fx, fy) = px(&cam, front.map(f64::from));
    let (bx, by) = px(&cam, back.map(f64::from));
    assert_eq!(pixel(&img, fx, fy), [0, 255, 0, 255], "front point drawn");
    assert_eq!(pixel(&img, bx, by), [0, 0, 255, 255], "back point hidden by the wall");

    // Lines are 1 px: look for them in a small column window around their projected row.
    let column_has = |p: [f64; 3], want: [u8; 4]| {
        let (x, y) = px(&cam, p);
        (y.saturating_sub(1)..=(y + 1).min(H - 1)).any(|yy| pixel(&img, x, yy) == want)
    };
    assert!(column_has([0.0, 2.0, -0.5], [255; 4]), "on-top line over the wall");
    assert!(!column_has([0.0, 2.0, 0.5], [255; 4]), "depth-tested line hidden");
    // Outside the wall the depth-tested line shows.
    assert!(column_has([2.5, 2.0, 0.5], [255; 4]), "depth-tested line beside the wall");
}

#[test]
fn per_scene_alpha_fades_points_and_several_scenes_share_a_frame() {
    let Some((device, queue)) = gpu() else { return };
    let cam = looking_north();
    let a = [0.0f32, 0.0, 0.0];
    let b = [1.0f32, 0.0, 0.0];
    let sa = scene_of(&device, &[(a, [255, 255, 255])]);
    let sb = scene_of(&device, &[(b, [255, 255, 255])]);
    let mut points = Renderer::new(&device, CAPTURE_FORMAT);
    points.set_point_size(PointSize::Pixels(6.0));
    points.prepare(&device, &queue, [W, H], &cam, &[&sa, &sb], None);
    let img = Headless::new(&device, [W, H]).capture_with(&device, &queue, |enc, color, depth| {
        let mut pass = begin_pass(enc, color, depth, &BLACK);
        points.draw(&mut pass, &[&sa, &sb], Some(&[0.5, 1.0]));
    });
    let (ax, ay) = px(&cam, a.map(f64::from));
    let (bx, by) = px(&cam, b.map(f64::from));
    let pa = pixel(&img, ax, ay);
    assert!(pa[0].abs_diff(128) <= 2, "half faded: {pa:?}");
    assert_eq!(pixel(&img, bx, by), [255, 255, 255, 255]);
}

#[test]
fn metre_points_shrink_with_distance() {
    let Some((device, queue)) = gpu() else { return };
    // Focal length is about 68.6 px (50 degrees over 64 px): 0.3 m is ~6.9 px at 3 m, ~2 px at 10 m.
    let near_p = [-1.0f32, -7.0, 0.0];
    let far_p = [1.0f32, 0.0, 0.0];
    let scene = scene_of(&device, &[(near_p, [255, 255, 255]), (far_p, [255, 255, 255])]);
    let cam = looking_north();
    let mut points = Renderer::new(&device, CAPTURE_FORMAT);
    points.set_point_size(PointSize::Metres(0.3));
    points.prepare(&device, &queue, [W, H], &cam, &[&scene], None);
    let img = Headless::new(&device, [W, H]).capture_with(&device, &queue, |enc, color, depth| {
        let mut pass = begin_pass(enc, color, depth, &BLACK);
        points.draw(&mut pass, &[&scene], None);
    });
    // Count lit pixels on each half of the image: the near point covers more than the far one.
    let lit = |xs: std::ops::Range<u32>| {
        xs.flat_map(|x| (0..H).map(move |y| (x, y)))
            .filter(|&(x, y)| pixel(&img, x, y)[0] > 128)
            .count()
    };
    let (l, r) = (lit(0..W / 2), lit(W / 2..W));
    assert!(l > r && r >= 1, "near {l} px, far {r} px");
}

#[test]
fn additive_meshes_add_up() {
    let Some((device, queue)) = gpu() else { return };
    let cam = looking_north();
    let mut meshes = MeshRenderer::new(&device, CAPTURE_FORMAT);
    let grey = unlit([0.25, 0.25, 0.25, 1.0], Blend::Additive);
    meshes.upsert(&device, &queue, 1, &wall(0.0), None, grey);
    meshes.upsert(&device, &queue, 2, &wall(1.0), None, grey);
    meshes.prepare(&queue, [W, H], &cam, None);
    let img = Headless::new(&device, [W, H]).capture_with(&device, &queue, |enc, color, depth| {
        let mut pass = begin_pass(enc, color, depth, &BLACK);
        meshes.draw_opaque(&mut pass);
        meshes.draw_transparent(&mut pass);
    });
    // Blending happens on the stored (sRGB-encoded) values, as in three.js: 64 + 64.
    assert!(near(pixel(&img, W / 2, H / 2), [128, 128, 128, 255], 2));
}
