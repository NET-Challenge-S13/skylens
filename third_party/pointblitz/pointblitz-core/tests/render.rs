//! GPU tests: render known points and check the pixels against the CPU projection.
//! Skipped (pass with a message) when no GPU adapter is available, e.g. in CI.

use pointblitz_core::{Camera, Headless, Inserted, Scene};
use pointblitz_io::Point;
use pointblitz_io::chunk::{FLAG_LAST_IN_GENERATION, encode};

const W: u32 = 320;
const H: u32 = 180;

fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
    match pointblitz_core::headless::device() {
        Some((d, q, info)) => {
            eprintln!("adapter: {} ({:?})", info.name, info.backend);
            Some((d, q))
        }
        None => {
            eprintln!("no GPU adapter — skipped");
            None
        }
    }
}

fn pt(p: [f32; 3], c: [u8; 3]) -> Point {
    Point {
        position: p,
        color: c,
    }
}

fn pixel(img: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * W + x) * 4) as usize;
    img[i..i + 4].try_into().unwrap()
}

/// Pixel under the projected point (rounded down = the pixel containing it).
fn at(cam: &Camera, p: [f64; 3]) -> (u32, u32) {
    let [x, y] = cam.project_px(p, W, H).expect("in front of camera");
    (x as u32, y as u32)
}

fn looking_north() -> Camera {
    Camera::new([0.0, -10.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 50.0)
}

#[test]
fn a_point_lands_where_the_cpu_projection_says() {
    let Some((device, queue)) = gpu() else { return };
    let mut scene = Scene::new();
    let p = [1.5f32, 0.0, -0.8];
    scene
        .insert(
            &device,
            &encode(0, 0, FLAG_LAST_IN_GENERATION, &[pt(p, [255, 0, 0])]),
        )
        .unwrap();
    let cam = looking_north();
    let mut hl = Headless::new(&device, [W, H]);
    let img = hl.capture(&device, &queue, &cam, &scene);
    let (x, y) = at(&cam, p.map(f64::from));
    assert_eq!(pixel(&img, x, y), [255, 0, 0, 255]);
    // Background stays black; a 2 px point covers at most a 2×2 block.
    assert_eq!(pixel(&img, 0, 0), [0, 0, 0, 255]);
    let lit = img.as_chunks::<4>().0.iter().filter(|p| p[0] > 0).count();
    assert!((1..=4).contains(&lit), "lit pixels: {lit}");
}

#[test]
fn nearer_point_wins_without_sorting() {
    let Some((device, queue)) = gpu() else { return };
    let mut scene = Scene::new();
    // Far red first, near green second, then the reverse order in another chunk: depth decides.
    let far = pt([0.0, 5.0, 0.0], [255, 0, 0]);
    let near = pt([0.0, -5.0, 0.0], [0, 255, 0]);
    scene
        .insert(&device, &encode(0, 0, 0, &[far, near]))
        .unwrap();
    scene
        .insert(
            &device,
            &encode(0, 1, FLAG_LAST_IN_GENERATION, &[near, far]),
        )
        .unwrap();
    let cam = looking_north();
    let img = Headless::new(&device, [W, H]).capture(&device, &queue, &cam, &scene);
    let (x, y) = at(&cam, [0.0, 0.0, 0.0]);
    assert_eq!(pixel(&img, x, y), [0, 255, 0, 255]);
}

#[test]
fn generations_swap_only_when_complete() {
    let Some((device, queue)) = gpu() else { return };
    let mut scene = Scene::new();
    let cam = looking_north();
    let mut hl = Headless::new(&device, [W, H]);
    let a = [1.0f32, 0.0, 0.0];
    let b = [-1.0f32, 0.0, 0.0];
    let (ax, ay) = at(&cam, a.map(f64::from));
    let (bx, by) = at(&cam, b.map(f64::from));

    // Generation 0 on screen.
    let r = scene.insert(
        &device,
        &encode(0, 0, FLAG_LAST_IN_GENERATION, &[pt(a, [255, 255, 255])]),
    );
    assert_eq!(r.unwrap(), Inserted::Appended);
    // Preview: same generation appends.
    let r = scene.insert(&device, &encode(0, 1, 0, &[pt(b, [0, 0, 255])]));
    assert_eq!(r.unwrap(), Inserted::Appended);
    let img = hl.capture(&device, &queue, &cam, &scene);
    assert_eq!(pixel(&img, ax, ay)[0], 255);
    assert_eq!(pixel(&img, bx, by)[2], 255);

    // Refined: generation 1 arrives in two chunks; the old one stays until the last chunk.
    let r = scene.insert(&device, &encode(1, 0, 0, &[pt(b, [0, 255, 0])]));
    assert_eq!(r.unwrap(), Inserted::Pending);
    let img = hl.capture(&device, &queue, &cam, &scene);
    assert_eq!(pixel(&img, ax, ay)[0], 255, "old generation still shown");
    assert_eq!(scene.generation(), Some(0));

    let r = scene.insert(&device, &encode(1, 1, FLAG_LAST_IN_GENERATION, &[]));
    assert_eq!(
        r.unwrap(),
        Inserted::Swapped {
            from: Some(0),
            to: 1
        }
    );
    let img = hl.capture(&device, &queue, &cam, &scene);
    assert_eq!(
        pixel(&img, ax, ay),
        [0, 0, 0, 255],
        "old generation released"
    );
    assert_eq!(pixel(&img, bx, by), [0, 255, 0, 255]);

    // Stale chunks are dropped.
    let r = scene.insert(&device, &encode(0, 9, 0, &[pt(a, [255, 0, 0])]));
    assert_eq!(r.unwrap(), Inserted::Stale);
    assert_eq!(scene.points(), 1);
}

#[test]
fn far_from_the_origin_still_renders_at_the_right_pixel() {
    let Some((device, queue)) = gpu() else { return };
    let base = [5000.0f32, 5000.0, 120.0];
    let p = [base[0] + 0.3, base[1], base[2] + 0.1];
    let mut scene = Scene::new();
    scene
        .insert(
            &device,
            &encode(0, 0, FLAG_LAST_IN_GENERATION, &[pt(p, [255, 128, 0])]),
        )
        .unwrap();
    let cam = Camera::new(
        [5000.0, 4997.0, 120.0],
        base.map(f64::from),
        [0.0, 0.0, 1.0],
        50.0,
    );
    let img = Headless::new(&device, [W, H]).capture(&device, &queue, &cam, &scene);
    let (x, y) = at(&cam, p.map(f64::from));
    assert_eq!(pixel(&img, x, y), [255, 128, 0, 255]);
}
