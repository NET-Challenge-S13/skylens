//! GPU tests for the animated point layer. Skipped (pass with a message) without an adapter.

use pointblitz_core::animated::book::{Add, Move};
use pointblitz_core::headless::CAPTURE_FORMAT;
use pointblitz_core::{AnimatedPoints, Camera, ColorMode, FrameStyle, Headless, PointSize, begin_pass};

const W: u32 = 64;
const H: u32 = 64;
const BLACK: FrameStyle = FrameStyle { clear: [0.0, 0.0, 0.0, 1.0], fog: None };

fn cam() -> Camera {
    Camera::new([0.0, -10.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 50.0)
}

struct Rig {
    device: wgpu::Device,
    queue: wgpu::Queue,
    points: AnimatedPoints,
}

fn rig() -> Option<Rig> {
    let (device, queue, _) = pointblitz_core::headless::device()?;
    let mut points = AnimatedPoints::new(&device, CAPTURE_FORMAT, [0.0, 0.0, 0.0], 4);
    points.set_point_size(PointSize::Pixels(6.0));
    Some(Rig { device, queue, points })
}

impl Rig {
    fn shot(&mut self, now: f64) -> Vec<u8> {
        let c = cam();
        self.points.prepare(&self.device, &self.queue, [W, H], &c, now, None);
        let points = &self.points;
        Headless::new(&self.device, [W, H]).capture_with(&self.device, &self.queue, |enc, color, depth| {
            let mut pass = begin_pass(enc, color, depth, &BLACK);
            points.draw(&mut pass);
        })
    }
}

fn lit(img: &[u8]) -> Vec<(u32, u32)> {
    (0..H)
        .flat_map(|y| (0..W).map(move |x| (x, y)))
        .filter(|&(x, y)| img[((y * W + x) * 4) as usize..][..3].iter().any(|&v| v > 40))
        .collect()
}

fn centre_x(img: &[u8]) -> f64 {
    let l = lit(img);
    l.iter().map(|p| f64::from(p.0)).sum::<f64>() / l.len() as f64
}

fn add(key: u64, x: f64) -> Add {
    Add { key, pos: [x, 0.0, 0.0], rgb: [255, 255, 255] }
}

#[test]
fn fade_in_glide_and_fade_out_follow_the_clock() {
    let Some(mut r) = rig() else { eprintln!("no GPU adapter, skipped"); return };
    r.points.apply(&r.device, &r.queue, &[add(1, -1.0)], &[], &[], 100.0);
    // Fade in: nothing at the start, partly dithered half way, full after 150 ms.
    let n0 = lit(&r.shot(100.0)).len();
    let nh = lit(&r.shot(100.075)).len();
    let nf = lit(&r.shot(100.5)).len();
    assert_eq!(n0, 0);
    assert!(nh > 0 && nh < nf, "half {nh} full {nf}");
    // Glide from x = -1 to x = +1.
    let from = centre_x(&r.shot(101.0));
    r.points.apply(
        &r.device,
        &r.queue,
        &[],
        &[Move { key: 1, pos: [1.0, 0.0, 0.0], rgb: None }],
        &[],
        101.0,
    );
    let start = centre_x(&r.shot(101.0));
    let mid = centre_x(&r.shot(101.25));
    let end = centre_x(&r.shot(101.8));
    assert!((start - from).abs() < 0.6, "{start} {from}");
    assert!(mid > start + 1.0 && mid < end, "{start} {mid} {end}");
    assert!(end > start + 5.0, "{start} {end}");
    // Fade out: gone after 150 ms and the slot is reusable.
    r.points.apply(&r.device, &r.queue, &[], &[], &[1], 102.0);
    assert!(!lit(&r.shot(102.05)).is_empty());
    assert!(lit(&r.shot(102.3)).is_empty());
    assert_eq!(r.points.book().draw_count(), 0);
}

#[test]
fn capacity_growth_keeps_old_points_and_colour_modes_work() {
    let Some(mut r) = rig() else { eprintln!("no GPU adapter, skipped"); return };
    r.points.apply(&r.device, &r.queue, &[add(1, 0.0)], &[], &[], 0.0);
    let adds: Vec<Add> = (2..300).map(|k| Add { key: k, pos: [100.0, 0.0, 0.0], rgb: [9, 9, 9] }).collect();
    let st = r.points.apply(&r.device, &r.queue, &adds, &[], &[], 0.0);
    assert!(st.reallocated && r.points.capacity() >= 299);
    assert_eq!(st.ranges, 1);
    let img = r.shot(1.0);
    assert!(!lit(&img).is_empty());
    // Flat red overrides the per-point white.
    r.points.set_color_mode(ColorMode::Flat([1.0, 0.0, 0.0]));
    let img = r.shot(1.0);
    let (x, y) = lit(&img)[0];
    let p = &img[((y * W + x) * 4) as usize..][..3];
    assert!(p[0] > 200 && p[1] < 30 && p[2] < 30, "{p:?}");
    // Height ramp: z = 0 over 0..1 is the blue end.
    r.points.set_color_mode(ColorMode::Height { low: 0.0, high: 1.0 });
    let img = r.shot(1.0);
    let (x, y) = lit(&img)[0];
    let p = &img[((y * W + x) * 4) as usize..][..3];
    assert!(p[2] > p[0], "{p:?}");
}
