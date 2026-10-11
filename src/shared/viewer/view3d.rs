//! The 3D view: a Blitz custom widget mounted where the legacy three.js `<canvas>` was. It owns
//! the GPU (device, colour and depth targets, the pointblitz renderers) and calls back into the
//! screen to fill each frame, following Blitz's `examples/wgpu_texture`.
//!
//! One frame:
//!
//! 1. [`ViewScreen::frame`] advances the screen's animation, uploads what changed through
//!    [`Gpu3d`] (point clouds, meshes, lines) and returns a [`FrameSpec`]: camera, background, fog.
//! 2. The widget records one pass ([`pointblitz::core::begin_pass`]): opaque meshes, then every
//!    point cloud (one pointblitz `Scene` per reconstruction segment, each with its own alpha for
//!    reveal fades), then transparent meshes, then lines. All share the reversed-Z depth buffer.
//! 3. The image goes into the Blitz scene, and [`ViewScreen::paint_overlay`] may draw vector
//!    shapes over it.
//!
//! Pointer and wheel events on the view reach [`ViewScreen::input`] as [`ViewInput`], in CSS
//! pixels relative to the view, with clicks (press and release within 4 px) recognised.

use crate::camera::Projector;
use crate::chunks::{ChunkStore, MeshEntry};
use anyrender::{PaintRef, PaintScene as _, ResourceId};
use blitz_dom::Widget;
use blitz_dom::node::ComputedStyles;
use blitz_traits::events::{BlitzWheelDelta, MouseEventButton, UiEvent};
use peniko::kurbo::{Affine, Rect};
use peniko::{Fill, ImageBrush, ImageSampler};
use pointblitz::Camera;
use pointblitz::core::renderer::DEPTH_FORMAT;
use pointblitz::core::{FrameStyle, Inserted, LineRenderer, Material, MeshData, MeshRenderer, PointSize, Renderer, Scene, begin_pass};
use pointblitz::io::chunk::{ChunkError, HEADER_LEN};
use skylens_shared::proto::payload::v1 as payload;
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;
use wgpu_context::DeviceHandle;

/// Colour format of the texture handed to Blitz (as in examples/wgpu_texture). It stores the
/// shader output as is, so colours are sRGB-encoded bytes / 255 (see [`crate::config::rgb`]).
pub const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Point clouds and meshes are addressed by caller-chosen ids; these namespaces keep the ones the
/// shared sync helpers use apart from the screens' own.
pub const SEGMENT_CLOUDS: u64 = 1 << 40;
pub const MESH_BLOCKS: u64 = 1 << 62;

/// Cloud id of a reconstruction segment.
pub fn segment_cloud(segment: u32) -> u64 {
    SEGMENT_CLOUDS | u64::from(segment)
}

/// The segment of a cloud id made by [`segment_cloud`].
pub fn cloud_segment(id: u64) -> Option<u32> {
    (id & !0xffff_ffff == SEGMENT_CLOUDS).then_some((id & 0xffff_ffff) as u32)
}

/// Mesh id of a reconstruction mesh block (20 bits per axis, offset so negatives fit).
pub fn mesh_block_key(id: [i32; 3]) -> u64 {
    let pack = |v: i32| (i64::from(v) + (1 << 19)).clamp(0, (1 << 20) - 1) as u64;
    MESH_BLOCKS | (pack(id[0]) << 40) | (pack(id[1]) << 20) | pack(id[2])
}

/// A `payload.v1.MeshBlock` as pointblitz mesh data (absolute ENU, vertex colours, smooth normals).
pub fn mesh_block_data(block: &payload::MeshBlock) -> MeshData {
    let positions: Vec<[f64; 3]> = block
        .positions
        .chunks_exact(3)
        .map(|p| [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])])
        .collect();
    let colors = (block.colors.len() >= positions.len() * 3).then(|| {
        block.colors.chunks_exact(3).take(positions.len()).map(|c| [c[0], c[1], c[2], 255]).collect()
    });
    let n = positions.len() as u32;
    let indices: Vec<u32> = block
        .indices
        .chunks_exact(3)
        .filter(|t| t.iter().all(|i| *i < n))
        .flatten()
        .copied()
        .collect();
    MeshData { positions, normals: None, colors, uvs: None, indices, flat: false }
}

/// Returns the chunk with its header generation and flags replaced. Each cloud's `Scene` applies
/// its own generation rule; the chunk store already decided what is shown, so chunks are
/// renumbered into one generation before upload.
pub fn retag(bytes: &[u8], generation: u32, flags: u32) -> Vec<u8> {
    let mut out = bytes.to_vec();
    if out.len() >= HEADER_LEN {
        out[8..12].copy_from_slice(&generation.to_le_bytes());
        out[20..24].copy_from_slice(&flags.to_le_bytes());
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Point clouds
// ---------------------------------------------------------------------------------------------

struct PointCloud {
    scene: Scene,
    alpha: f32,
    visible: bool,
    /// For clouds synced from the chunk store: (segment epoch, chunks uploaded).
    source: Option<(u64, usize)>,
}

impl PointCloud {
    fn new() -> Self {
        PointCloud { scene: Scene::new(), alpha: 1.0, visible: true, source: None }
    }
}

/// What [`PointClouds::sync_segments`] did.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct SegmentSync {
    /// Segments seen for the first time.
    pub added: Vec<u32>,
    /// Segments whose chunks were replaced (refined, newer generation).
    pub replaced: Vec<u32>,
    pub removed: Vec<u32>,
}

/// Several independent pointblitz scenes drawn in one frame, each with its own alpha.
#[derive(Default)]
pub struct PointClouds {
    clouds: BTreeMap<u64, PointCloud>,
}

impl PointClouds {
    /// Appends one chunk to cloud `id` (created on demand) under pointblitz's generation rules.
    pub fn insert_chunk(&mut self, device: &wgpu::Device, id: u64, bytes: &[u8]) -> Result<Inserted, ChunkError> {
        self.clouds.entry(id).or_insert_with(PointCloud::new).scene.insert(device, bytes)
    }

    /// Replaces cloud `id` with these chunks in one step (built aside, so it never goes blank).
    /// Alpha and visibility are kept.
    pub fn replace(&mut self, device: &wgpu::Device, id: u64, chunks: &[&[u8]]) {
        let mut scene = Scene::new();
        for c in chunks {
            if let Err(e) = scene.insert(device, &retag(c, 0, 0)) {
                crate::log(&format!("[view3d] dropped point chunk: {e}"));
            }
        }
        let cloud = self.clouds.entry(id).or_insert_with(PointCloud::new);
        cloud.scene = scene;
        cloud.source = None;
    }

    pub fn remove(&mut self, id: u64) -> bool {
        self.clouds.remove(&id).is_some()
    }

    pub fn clear(&mut self) {
        self.clouds.clear();
    }

    pub fn contains(&self, id: u64) -> bool {
        self.clouds.contains_key(&id)
    }

    pub fn ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.clouds.keys().copied()
    }

    /// Opacity multiplier of a cloud (reveal fades), 0..1.
    pub fn set_alpha(&mut self, id: u64, alpha: f32) {
        if let Some(c) = self.clouds.get_mut(&id) {
            c.alpha = alpha.clamp(0.0, 1.0);
        }
    }

    pub fn alpha(&self, id: u64) -> Option<f32> {
        self.clouds.get(&id).map(|c| c.alpha)
    }

    pub fn set_visible(&mut self, id: u64, visible: bool) {
        if let Some(c) = self.clouds.get_mut(&id) {
            c.visible = visible;
        }
    }

    /// Points on screen across visible clouds.
    pub fn points(&self) -> u64 {
        self.clouds.values().filter(|c| c.visible).map(|c| c.scene.points()).sum()
    }

    pub fn cloud_points(&self, id: u64) -> u64 {
        self.clouds.get(&id).map_or(0, |c| c.scene.points())
    }

    /// Brings the segment clouds in line with the store: appends new chunks of a segment, rebuilds
    /// a segment whose shown set was replaced, drops segments the store no longer has. New
    /// segments start at `initial_alpha` (0 for a fade-in, 1 to show at once).
    pub fn sync_segments(&mut self, device: &wgpu::Device, store: &ChunkStore, initial_alpha: f32) -> SegmentSync {
        let mut out = SegmentSync::default();
        for (segment, points) in store.segments() {
            let id = segment_cloud(segment);
            let payloads = points.payloads();
            if !self.clouds.contains_key(&id) {
                out.added.push(segment);
                let mut c = PointCloud::new();
                c.alpha = initial_alpha.clamp(0.0, 1.0);
                self.clouds.insert(id, c);
            }
            let cloud = self.clouds.get_mut(&id).expect("inserted above");
            match cloud.source {
                Some((epoch, uploaded)) if epoch == points.epoch() => {
                    for p in payloads.iter().skip(uploaded) {
                        if let Err(e) = cloud.scene.insert(device, &retag(p, 0, 0)) {
                            crate::log(&format!("[view3d] dropped point chunk of segment {segment}: {e}"));
                        }
                    }
                    cloud.source = Some((epoch, payloads.len()));
                }
                previous => {
                    if previous.is_some() {
                        out.replaced.push(segment);
                    }
                    let mut scene = Scene::new();
                    for p in payloads {
                        if let Err(e) = scene.insert(device, &retag(p, 0, 0)) {
                            crate::log(&format!("[view3d] dropped point chunk of segment {segment}: {e}"));
                        }
                    }
                    cloud.scene = scene;
                    cloud.source = Some((points.epoch(), payloads.len()));
                }
            }
        }
        let stale: Vec<u64> = self
            .clouds
            .keys()
            .copied()
            .filter(|id| cloud_segment(*id).is_some_and(|s| store.segment(s).is_none()))
            .collect();
        for id in stale {
            self.clouds.remove(&id);
            out.removed.extend(cloud_segment(id));
        }
        out
    }

    fn draw_list(&self) -> (Vec<&Scene>, Vec<f32>) {
        self.clouds
            .values()
            .filter(|c| c.visible && c.alpha > 0.0 && c.scene.points() > 0)
            .map(|c| (&c.scene, c.alpha))
            .unzip()
    }
}

// ---------------------------------------------------------------------------------------------
// GPU context handed to the screen
// ---------------------------------------------------------------------------------------------

/// What [`Gpu3d::sync_mesh_blocks`] did.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct MeshSync {
    pub upserted: Vec<[i32; 3]>,
    pub removed: Vec<[i32; 3]>,
}

/// The device and the renderers, lent to the screen every frame.
pub struct Gpu3d {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// e.g. `"Vulkan"`, `"Dx12"`, `"BrowserWebGpu"`, `"Gl"`.
    pub backend: String,
    pub points: Renderer,
    pub clouds: PointClouds,
    pub meshes: MeshRenderer,
    pub lines: LineRenderer,
    mesh_uploaded: HashMap<[i32; 3], u64>,
}

impl Gpu3d {
    fn new(handle: &DeviceHandle) -> Gpu3d {
        let device = handle.device.clone();
        let queue = handle.queue.clone();
        Gpu3d {
            backend: format!("{:?}", handle.adapter.get_info().backend),
            points: Renderer::new(&device, COLOR_FORMAT),
            meshes: MeshRenderer::new(&device, COLOR_FORMAT),
            lines: LineRenderer::new(&device, COLOR_FORMAT),
            clouds: PointClouds::default(),
            mesh_uploaded: HashMap::new(),
            device,
            queue,
        }
    }

    /// [`PointClouds::sync_segments`] with this device.
    pub fn sync_segments(&mut self, store: &ChunkStore, initial_alpha: f32) -> SegmentSync {
        self.clouds.sync_segments(&self.device, store, initial_alpha)
    }

    /// Brings the reconstruction mesh blocks in line with the store (ids from
    /// [`mesh_block_key`]); `material` picks each block's look.
    pub fn sync_mesh_blocks(&mut self, store: &ChunkStore, material: impl Fn(&MeshEntry) -> Material) -> MeshSync {
        let mut out = MeshSync::default();
        for (id, entry) in store.mesh_blocks() {
            if self.mesh_uploaded.get(&id) == Some(&entry.stamp) {
                continue;
            }
            let data = mesh_block_data(&entry.block);
            self.meshes.upsert(&self.device, &self.queue, mesh_block_key(id), &data, None, material(entry));
            self.mesh_uploaded.insert(id, entry.stamp);
            out.upserted.push(id);
        }
        let gone: Vec<[i32; 3]> = self.mesh_uploaded.keys().copied().filter(|id| store.mesh_block(*id).is_none()).collect();
        for id in gone {
            self.meshes.remove(mesh_block_key(id));
            self.mesh_uploaded.remove(&id);
            out.removed.push(id);
        }
        out
    }

    /// Re-applies a material to every synced mesh block (display mode change), no re-upload.
    pub fn restyle_mesh_blocks(&mut self, store: &ChunkStore, material: impl Fn(&MeshEntry) -> Material) {
        for (id, entry) in store.mesh_blocks() {
            if self.mesh_uploaded.contains_key(&id) {
                self.meshes.set_material(mesh_block_key(id), material(entry));
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The screen side
// ---------------------------------------------------------------------------------------------

/// Timing and size of the frame being drawn.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameInfo {
    pub now_ms: f64,
    /// Seconds since the previous frame (0 on the first, capped at 0.25).
    pub dt: f64,
    /// Target size in physical pixels.
    pub width_px: u32,
    pub height_px: u32,
    /// Device pixel ratio.
    pub scale: f64,
}

impl FrameInfo {
    pub fn css_width(&self) -> f64 {
        f64::from(self.width_px) / self.scale.max(1e-6)
    }

    pub fn css_height(&self) -> f64 {
        f64::from(self.height_px) / self.scale.max(1e-6)
    }

    /// Projection to CSS pixels of the view (DOM overlays, picking with [`ViewInput`] coords).
    pub fn projector(&self, camera: &Camera) -> Projector {
        Projector::new(camera.clone(), self.css_width(), self.css_height())
    }

    /// Projection to physical pixels (for [`ViewScreen::paint_overlay`]).
    pub fn projector_px(&self, camera: &Camera) -> Projector {
        Projector::new(camera.clone(), f64::from(self.width_px), f64::from(self.height_px))
    }
}

/// What to draw this frame.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameSpec {
    pub camera: Camera,
    pub style: FrameStyle,
    /// Changes the point size from now on (None keeps the current one).
    pub point_size: Option<PointSize>,
}

/// Pointer input on the view, in CSS pixels relative to its top-left corner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ViewInput {
    /// `button`: 0 main, 1 middle, 2 right.
    PointerDown { x: f32, y: f32, button: u8, shift: bool },
    PointerMove { x: f32, y: f32, pressed: bool },
    PointerUp { x: f32, y: f32, button: u8 },
    /// Browser convention: `delta_y > 0` scrolls down (zoom out).
    Wheel { x: f32, y: f32, delta_y: f64 },
    /// Press and release within 4 px.
    Click { x: f32, y: f32, button: u8 },
}

/// A screen's 3D content. Shared between the widget and the screen app as `Rc<RefCell<S>>`.
pub trait ViewScreen: 'static {
    /// The GPU is (again) available. Pull-based sync helpers re-upload by themselves, so most
    /// screens need nothing here.
    fn gpu_ready(&mut self, gpu: &mut Gpu3d) {
        let _ = gpu;
    }

    /// The GPU went away (renderer suspended); everything uploaded is gone.
    fn gpu_lost(&mut self) {}

    /// Advance animation, upload changes, describe the frame.
    fn frame(&mut self, gpu: &mut Gpu3d, info: &FrameInfo) -> FrameSpec;

    /// Vector shapes over the 3D image, in physical pixels (use [`FrameInfo::projector_px`]).
    fn paint_overlay(&mut self, scene: &mut anyrender::Scene, info: &FrameInfo) {
        let _ = (scene, info);
    }

    fn input(&mut self, input: &ViewInput) {
        let _ = input;
    }

    /// Whether to keep drawing frames (animation, camera motion, pending uploads).
    fn needs_redraw(&self) -> bool {
        true
    }
}

struct Target {
    texture: wgpu::Texture,
    handle: ResourceId,
}

struct GpuState {
    ctx: Gpu3d,
    depth: Option<(wgpu::Texture, [u32; 2])>,
    displayed: Option<Target>,
    next: Option<Target>,
}

/// The widget. Mount it with [`crate::dom::Dom::mount_widget`] on the legacy canvas id.
pub struct View3d<S: ViewScreen> {
    screen: Rc<RefCell<S>>,
    gpu: Option<GpuState>,
    last_paint_ms: Option<f64>,
    press: Option<(f32, f32, u8)>,
}

/// Movement under which a press and release still count as a click (CSS px).
const CLICK_SLOP: f32 = 4.0;

impl<S: ViewScreen> View3d<S> {
    pub fn new(screen: Rc<RefCell<S>>) -> Self {
        View3d { screen, gpu: None, last_paint_ms: None, press: None }
    }

    pub fn boxed(screen: Rc<RefCell<S>>) -> Box<dyn Widget> {
        Box::new(Self::new(screen))
    }

    fn render(gs: &mut GpuState, ctx: &mut dyn anyrender::RenderContext, w: u32, h: u32, spec: &FrameSpec) -> Option<ResourceId> {
        // Two colour targets like examples/wgpu_texture: draw into `next`, then swap it with the
        // one Blitz is showing.
        if gs.next.as_ref().is_some_and(|t| t.texture.width() != w || t.texture.height() != h) {
            let old = gs.next.take().expect("checked above");
            ctx.unregister_resource(old.handle);
        }
        if gs.next.is_none() {
            let texture = color_texture(&gs.ctx.device, w, h);
            let handle = ctx.try_register_custom_resource(Box::new(texture.clone())).ok()?;
            gs.next = Some(Target { texture, handle });
        }
        if gs.depth.as_ref().is_none_or(|(_, size)| *size != [w, h]) {
            gs.depth = Some((depth_texture(&gs.ctx.device, w, h), [w, h]));
        }
        let target = gs.next.as_ref().expect("created above");
        let color_view = target.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = gs.depth.as_ref().expect("created above").0.create_view(&wgpu::TextureViewDescriptor::default());

        let Gpu3d { device, queue, points, clouds, meshes, lines, .. } = &mut gs.ctx;
        if let Some(size) = spec.point_size {
            points.set_point_size(size);
        }
        let (scenes, alphas) = clouds.draw_list();
        let size = [w, h];
        points.prepare(device, queue, size, &spec.camera, &scenes, spec.style.fog);
        meshes.prepare(queue, size, &spec.camera, spec.style.fog);
        lines.prepare(queue, size, &spec.camera, spec.style.fog);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("skylens view") });
        {
            let mut pass = begin_pass(&mut encoder, &color_view, &depth_view, &spec.style);
            meshes.draw_opaque(&mut pass);
            points.draw(&mut pass, &scenes, Some(alphas.as_slice()));
            meshes.draw_transparent(&mut pass);
            lines.draw(&mut pass);
        }
        queue.submit(Some(encoder.finish()));

        let handle = target.handle;
        std::mem::swap(&mut gs.next, &mut gs.displayed);
        Some(handle)
    }
}

impl<S: ViewScreen> Widget for View3d<S> {
    fn can_create_surfaces(&mut self, render_ctx: &mut dyn anyrender::RenderContext) {
        let Some(context) = render_ctx.renderer_specific_context() else {
            crate::log("[view3d] renderer exposes no context; 3D view disabled");
            return;
        };
        match context.downcast::<DeviceHandle>() {
            Ok(handle) => {
                let mut ctx = Gpu3d::new(&handle);
                self.screen.borrow_mut().gpu_ready(&mut ctx);
                self.gpu = Some(GpuState { ctx, depth: None, displayed: None, next: None });
            }
            Err(_) => crate::log("[view3d] renderer is not wgpu; 3D view disabled"),
        }
    }

    fn destroy_surfaces(&mut self) {
        self.gpu = None;
        self.screen.borrow_mut().gpu_lost();
    }

    fn requires_redraw(&self) -> bool {
        self.gpu.is_some() && self.screen.borrow().needs_redraw()
    }

    fn handle_event(&mut self, event: &UiEvent) {
        let input = match event {
            UiEvent::PointerDown(e) => {
                let (x, y) = (e.coords.client_x, e.coords.client_y);
                let button = match e.button {
                    MouseEventButton::Main => 0,
                    MouseEventButton::Auxiliary => 1,
                    MouseEventButton::Secondary => 2,
                    _ => 3,
                };
                self.press = Some((x, y, button));
                ViewInput::PointerDown { x, y, button, shift: e.mods.shift() }
            }
            UiEvent::PointerMove(e) => {
                let pressed = !e.buttons.is_empty();
                ViewInput::PointerMove { x: e.coords.client_x, y: e.coords.client_y, pressed }
            }
            UiEvent::PointerUp(e) | UiEvent::PointerCancel(e) => {
                let (x, y) = (e.coords.client_x, e.coords.client_y);
                let button = self.press.map_or(0, |(_, _, b)| b);
                let click = matches!(event, UiEvent::PointerUp(_))
                    && self.press.is_some_and(|(px, py, _)| (x - px).abs() <= CLICK_SLOP && (y - py).abs() <= CLICK_SLOP);
                self.press = None;
                let mut screen = self.screen.borrow_mut();
                screen.input(&ViewInput::PointerUp { x, y, button });
                if click {
                    screen.input(&ViewInput::Click { x, y, button });
                }
                return;
            }
            UiEvent::Wheel(e) => {
                let delta_y = match e.delta {
                    BlitzWheelDelta::Lines(_, y) => -y * 40.0,
                    BlitzWheelDelta::Pixels(_, y) => -y,
                };
                ViewInput::Wheel { x: e.coords.client_x, y: e.coords.client_y, delta_y }
            }
            _ => return,
        };
        self.screen.borrow_mut().input(&input);
    }

    fn paint(
        &mut self,
        render_ctx: &mut dyn anyrender::RenderContext,
        _styles: &ComputedStyles,
        width: u32,
        height: u32,
        scale: f64,
    ) -> anyrender::Scene {
        let mut scene = anyrender::Scene::new();
        let Some(gs) = self.gpu.as_mut() else { return scene };
        if width == 0 || height == 0 {
            return scene;
        }
        let now = crate::now_ms();
        let dt = self.last_paint_ms.map_or(0.0, |last| ((now - last) / 1000.0).clamp(0.0, 0.25));
        self.last_paint_ms = Some(now);
        let info = FrameInfo { now_ms: now, dt, width_px: width, height_px: height, scale: if scale > 0.0 { scale } else { 1.0 } };

        let spec = self.screen.borrow_mut().frame(&mut gs.ctx, &info);
        if let Some(resource) = Self::render(gs, render_ctx, width, height, &spec) {
            scene.fill(
                Fill::NonZero,
                Affine::IDENTITY,
                PaintRef::Resource(ImageBrush { image: resource, sampler: ImageSampler::default() }),
                None,
                &Rect::from_origin_size((0.0, 0.0), (f64::from(width), f64::from(height))),
            );
        }
        self.screen.borrow_mut().paint_overlay(&mut scene, &info);
        scene
    }
}

fn color_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("skylens view color"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: COLOR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn depth_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("skylens view depth"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_live_in_separate_namespaces() {
        assert_eq!(cloud_segment(segment_cloud(7)), Some(7));
        assert_eq!(cloud_segment(segment_cloud(u32::MAX)), Some(u32::MAX));
        assert_eq!(cloud_segment(42), None);
        assert_eq!(cloud_segment(mesh_block_key([0, 0, 0])), None);
        let a = mesh_block_key([1, -2, 3]);
        let b = mesh_block_key([1, -2, 4]);
        let c = mesh_block_key([-1, -2, 3]);
        assert!(a != b && a != c && b != c);
        assert_eq!(a & MESH_BLOCKS, MESH_BLOCKS);
    }

    #[test]
    fn mesh_blocks_convert_to_absolute_coloured_meshes() {
        let block = payload::MeshBlock {
            id: vec![0, 0, 0],
            version: 1,
            positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            colors: vec![255, 0, 0, 0, 255, 0, 0, 0, 255],
            // The second triangle points past the vertices and is dropped.
            indices: vec![0, 1, 2, 0, 1, 9],
        };
        let m = mesh_block_data(&block);
        assert_eq!(m.positions, vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        assert_eq!(m.colors.unwrap()[1], [0, 255, 0, 255]);
        assert_eq!(m.indices, vec![0, 1, 2]);
        assert!(m.normals.is_none() && !m.flat);
        // Missing colours are left to the material (white).
        let bare = payload::MeshBlock { colors: vec![], ..block };
        assert!(mesh_block_data(&bare).colors.is_none());
    }

    #[test]
    fn retag_rewrites_generation_and_flags_only() {
        use pointblitz::io::Point;
        use pointblitz::io::chunk::{FLAG_LAST_IN_GENERATION, decode_header, encode_with};
        let pts = vec![Point { position: [1.0, 2.0, 3.0], color: [9, 8, 7] }];
        let src = encode_with(5, 2, FLAG_LAST_IN_GENERATION, &pts, 12);
        let out = retag(&src, 0, 0);
        let h = decode_header(&out).unwrap();
        assert_eq!((h.generation, h.flags, h.index, h.point_count), (0, 0, 2, 1));
        assert_eq!(&out[HEADER_LEN..], &src[HEADER_LEN..]);
    }
}
