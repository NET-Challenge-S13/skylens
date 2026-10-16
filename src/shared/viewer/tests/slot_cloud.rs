//! GPU tests for the slot cloud: add/remove visibility, dither fade, eased glide.
//! Skipped (pass with a message) when no GPU adapter is available, e.g. in CI.

use pointblitz::core::headless::CAPTURE_FORMAT;
use pointblitz::core::{Camera, FrameStyle, Headless, PointSize, begin_pass};
use skylens_viewer::slot_cloud::{SlotCloud, SlotInstance, SlotParams, T_LONG, T_NEVER};

const W: u32 = 64;
const H: u32 = 64;
const NOW: f32 = 10.0;
const BLACK: FrameStyle = FrameStyle { clear: [0.0, 0.0, 0.0, 1.0], fog: None };

fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
    match pointblitz::core::headless::device() {
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

fn camera() -> Camera {
    Camera::new([0.0, -10.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 50.0)
}

fn lit(img: &[u8], x: u32, y: u32) -> bool {
    let i = ((y * W + x) * 4) as usize;
    img[i] > 128 // red channel; points are pure red
}

fn lit_count(img: &[u8]) -> usize {
    (0..W * H).filter(|i| lit(img, i % W, i / W)).count()
}

fn slot(pos: [f32; 3], from: [f32; 3], tm: [f32; 3]) -> SlotInstance {
    SlotInstance { pos, from, tm, rgbz: [255, 0, 0, 3] }
}

fn render(device: &wgpu::Device, queue: &wgpu::Queue, slots: &[SlotInstance], size: PointSize) -> Vec<u8> {
    let mut cloud = SlotCloud::new(device, CAPTURE_FORMAT);
    assert!(cloud.ensure_capacity(device, slots.len()));
    assert!(!cloud.ensure_capacity(device, slots.len()));
    let all = 0..slots.len();
    cloud.upload(queue, slots, std::slice::from_ref(&all));
    cloud.set_used(slots.len());
    let params = SlotParams { now: NOW, point_size: size, ..Default::default() };
    cloud.prepare(queue, [W, H], &camera(), None, &params);
    let mut hl = Headless::new(device, [W, H]);
    hl.capture_with(device, queue, |enc, color, depth| {
        let mut pass = begin_pass(enc, color, depth, &BLACK);
        cloud.draw(&mut pass);
    })
}

#[test]
fn added_slot_is_drawn_and_never_or_dead_slots_are_not() {
    let Some((device, queue)) = gpu() else { return };
    let at = |x: f32| [x, 0.0, 0.0];
    let alive = slot(at(-2.0), at(-2.0), [T_LONG, NOW - 1.0, T_NEVER]);
    let never = slot(at(0.0), at(0.0), [T_LONG, T_NEVER, T_NEVER]);
    let dead = slot(at(2.0), at(2.0), [T_LONG, T_LONG, NOW - 5.0]);
    let img = render(&device, &queue, &[alive, never, dead], PointSize::Pixels(6.0));
    let cam = camera();
    let px = |x: f64| {
        let p = cam.project_px([x, 0.0, 0.0], W, H).unwrap();
        (p[0] as u32, p[1] as u32)
    };
    let (ax, ay) = px(-2.0);
    assert!(lit(&img, ax, ay), "added slot must be drawn");
    let (nx, ny) = px(0.0);
    assert!(!lit(&img, nx, ny), "T_NEVER slot must not be drawn");
    let (dx, dy) = px(2.0);
    assert!(!lit(&img, dx, dy), "dead slot must not be drawn");
}

#[test]
fn mid_fade_is_dithered() {
    let Some((device, queue)) = gpu() else { return };
    let c = [0.0; 3];
    let full = render(&device, &queue, &[slot(c, c, [T_LONG, NOW - 1.0, T_NEVER])], PointSize::Pixels(40.0));
    let half = render(&device, &queue, &[slot(c, c, [T_LONG, NOW - 0.075, T_NEVER])], PointSize::Pixels(40.0));
    let (nf, nh) = (lit_count(&full), lit_count(&half));
    assert!(nf > 800, "full disc expected, got {nf}");
    let ratio = nh as f64 / nf as f64;
    assert!((0.25..0.75).contains(&ratio), "half-faded slot should keep about half the pixels, ratio {ratio}");
}

#[test]
fn moving_slot_sits_at_the_eased_position() {
    let Some((device, queue)) = gpu() else { return };
    // u = 0.25 of a 0.7 s glide: linear would be at x = -1, the ease puts it at -1.5.
    let s = slot([2.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [NOW - 0.175, T_LONG, T_NEVER]);
    let img = render(&device, &queue, &[s], PointSize::Pixels(4.0));
    let cam = camera();
    let at = |x: f64| {
        let p = cam.project_px([x, 0.0, 0.0], W, H).unwrap();
        (p[0] as u32, p[1] as u32)
    };
    let (ex, ey) = at(-1.5);
    assert!(lit(&img, ex, ey), "slot should be at the eased position");
    let (lx, ly) = at(-1.0);
    assert!(!lit(&img, lx, ly), "slot must not be at the linear position");
    let (sx, sy) = at(-2.0);
    assert!(!lit(&img, sx, sy), "slot must have left its start");
}
