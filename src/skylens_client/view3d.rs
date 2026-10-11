//! The 3D view: a Blitz custom widget (`<object id="view3d">`) that renders
//! pointblitz points and the mesh pass into a wgpu texture, following Blitz's
//! `examples/wgpu_texture`, and draws markers (drones, route, detections) as
//! vector shapes on top of it in the same anyrender scene.
//!
//! The document side never touches the GPU. It pushes [`SceneOp`]s and an
//! [`Overlay`] into [`ViewShared`]; the widget applies them in `paint`, where it
//! has the device.

use crate::camera::OrbitCamera;
use crate::chunks::{Payload, SceneOp};
use crate::display::{recolor_in_place, retag, DisplayMode, Layer};
use crate::mesh::MeshPass;
use anyrender::{PaintRef, PaintScene as _, ResourceId};
use blitz_dom::node::ComputedStyles;
use blitz_dom::Widget;
use blitz_traits::events::{BlitzWheelDelta, MouseEventButton, MouseEventButtons, UiEvent};
use peniko::kurbo::{Affine, BezPath, Circle, Line, Rect, Stroke};
use peniko::{Color, Fill, ImageBrush, ImageSampler};
use pointblitz::core::renderer::DEPTH_FORMAT;
use pointblitz::io::chunk::FLAG_LAST_IN_GENERATION;
use pointblitz::{Renderer, Scene};
use std::cell::RefCell;
use std::rc::Rc;
use wgpu_context::DeviceHandle;

/// Colour format of the texture handed to Blitz (as in examples/wgpu_texture).
const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkerKind {
    Person,
    Hazard,
}

#[derive(Debug, Clone)]
pub struct OverlayDetection {
    pub id: String,
    pub kind: MarkerKind,
    /// ENU metres (x east, y north, z up).
    pub pos: [f64; 3],
    pub outline: Vec<[f64; 3]>,
    pub focused: bool,
}

#[derive(Debug, Clone)]
pub struct OverlayDrone {
    pub pos: [f64; 3],
    pub heading_deg: f64,
    pub tint: [u8; 3],
    pub active: bool,
    pub stale: bool,
}

/// What the view draws over the 3D render, already in ENU.
#[derive(Debug, Clone, Default)]
pub struct Overlay {
    pub drones: Vec<OverlayDrone>,
    pub route: Vec<[f64; 3]>,
    /// Waypoints being planned in the route modal (drawn dashed-looking, thinner).
    pub draft: Vec<[f64; 3]>,
    pub detections: Vec<OverlayDetection>,
}

/// Camera requests from the document (keyboard shortcuts, first data).
#[derive(Debug, Clone, Copy)]
pub enum CameraRequest {
    /// Centre on a point unless the operator already moved the camera.
    FrameIfIdle([f64; 3]),
    /// Centre on a point regardless.
    Focus([f64; 3]),
}

#[derive(Debug, Default)]
pub struct ViewStats {
    pub points: u64,
    pub gpu_active: bool,
    pub backend: String,
}

/// State shared between the document (UI thread logic) and the widget.
#[derive(Debug, Default)]
pub struct ViewShared {
    pub ops: Vec<SceneOp>,
    pub display: DisplayMode,
    pub overlay: Overlay,
    pub camera_request: Option<CameraRequest>,
    /// Something changed that needs a repaint.
    pub dirty: bool,
    /// The widget lost or (re)created its GPU state and needs every chunk again.
    pub needs_resync: bool,
    pub stats: ViewStats,
}

pub type SharedView = Rc<RefCell<ViewShared>>;

pub fn shared_view() -> SharedView {
    Rc::new(RefCell::new(ViewShared {
        dirty: true,
        ..Default::default()
    }))
}

impl ViewShared {
    /// Queues ops; a full rebuild makes earlier queued point ops pointless.
    pub fn push_ops(&mut self, ops: Vec<SceneOp>) {
        for op in ops {
            match &op {
                SceneOp::RebuildPoints(_) => self
                    .ops
                    .retain(|o| !matches!(o, SceneOp::InsertPoints(..) | SceneOp::RebuildPoints(_))),
                SceneOp::ResetMesh(_) => self
                    .ops
                    .retain(|o| !matches!(o, SceneOp::ApplyMesh(_) | SceneOp::ResetMesh(_))),
                _ => {}
            }
            self.ops.push(op);
        }
        self.dirty = true;
    }
}

struct Target {
    texture: wgpu::Texture,
    handle: ResourceId,
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: Renderer,
    scene: Scene,
    mesh: MeshPass,
    depth: Option<(wgpu::Texture, [u32; 2])>,
    displayed: Option<Target>,
    next: Option<Target>,
    /// Generation stamped on chunks inserted into the merged pointblitz scene.
    generation: u32,
}

enum Drag {
    Orbit { x: f32, y: f32 },
    Pan { x: f32, y: f32 },
}

pub struct View3d {
    shared: SharedView,
    gpu: Option<Gpu>,
    camera: OrbitCamera,
    drag: Option<Drag>,
}

impl View3d {
    pub fn new(shared: SharedView) -> Self {
        Self {
            shared,
            gpu: None,
            camera: OrbitCamera::default(),
            drag: None,
        }
    }

    fn mark_dirty(&self) {
        self.shared.borrow_mut().dirty = true;
    }

    fn apply_ops(&mut self) {
        let (ops, display, request) = {
            let mut s = self.shared.borrow_mut();
            (std::mem::take(&mut s.ops), s.display, s.camera_request.take())
        };
        if let Some(req) = request {
            match req {
                CameraRequest::FrameIfIdle(p) if !self.camera.user_moved => self.camera.frame(p, None),
                CameraRequest::FrameIfIdle(_) => {}
                CameraRequest::Focus(p) => self.camera.frame(p, None),
            }
        }
        let Some(gpu) = self.gpu.as_mut() else {
            // No device yet: keep the ops for later.
            if !ops.is_empty() {
                let mut s = self.shared.borrow_mut();
                let later = std::mem::take(&mut s.ops);
                s.ops = ops;
                s.ops.extend(later);
            }
            return;
        };
        for op in ops {
            match op {
                SceneOp::InsertPoints(layer, payload) => gpu.insert(layer, &payload, display),
                SceneOp::RebuildPoints(list) => gpu.rebuild(&list, display),
                SceneOp::ApplyMesh(payload) => gpu.apply_mesh(&payload),
                SceneOp::ResetMesh(list) => {
                    gpu.mesh.clear();
                    for p in &list {
                        gpu.apply_mesh(p);
                    }
                }
            }
        }
        self.shared.borrow_mut().stats.points = gpu.scene.points();
    }

    fn render(&mut self, ctx: &mut dyn anyrender::RenderContext, width: u32, height: u32) -> Option<ResourceId> {
        let gpu = self.gpu.as_mut()?;
        if width == 0 || height == 0 {
            return None;
        }

        // Size the two colour targets like examples/wgpu_texture: draw into `next`,
        // then swap it with the one Blitz is showing.
        if gpu
            .next
            .as_ref()
            .is_some_and(|t| t.texture.width() != width || t.texture.height() != height)
        {
            let old = gpu.next.take().expect("checked above");
            ctx.unregister_resource(old.handle);
        }
        if gpu.next.is_none() {
            let texture = color_texture(&gpu.device, width, height);
            let handle = ctx.try_register_custom_resource(Box::new(texture.clone())).ok()?;
            gpu.next = Some(Target { texture, handle });
        }
        if gpu.depth.as_ref().is_none_or(|(_, size)| *size != [width, height]) {
            gpu.depth = Some((depth_texture(&gpu.device, width, height), [width, height]));
        }

        let target = gpu.next.as_ref().expect("created above");
        let color_view = target.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = gpu
            .depth
            .as_ref()
            .expect("created above")
            .0
            .create_view(&wgpu::TextureViewDescriptor::default());

        let camera = self.camera.camera();
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("skylens view") });
        // Points: pointblitz clears colour and depth, then draws every visible chunk.
        gpu.renderer.render(
            &gpu.device,
            &gpu.queue,
            &mut encoder,
            &color_view,
            &depth_view,
            [width, height],
            &camera,
            &gpu.scene,
        );
        // Mesh blocks on top. pointblitz discards its depth buffer, so this pass
        // starts from a cleared (reversed-Z) depth and the mesh is not occluded by points.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("skylens mesh"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            let aspect = f64::from(width) / f64::from(height);
            gpu.mesh.draw(&gpu.queue, &mut pass, self.camera.absolute_view_proj(aspect));
        }
        gpu.queue.submit(Some(encoder.finish()));

        let handle = target.handle;
        std::mem::swap(&mut gpu.next, &mut gpu.displayed);
        Some(handle)
    }

    fn paint_overlay(&self, scene: &mut anyrender::Scene, width: u32, height: u32, scale: f64) {
        let shared = self.shared.borrow();
        let overlay = &shared.overlay;
        let cam = self.camera.camera();
        let project = |p: [f64; 3]| cam.project_px(p, width, height);
        let s = scale.max(0.5);

        // Route in force: cyan polyline with waypoint dots.
        draw_polyline(scene, &overlay.route, &project, rgba(0x5e, 0xc8, 0xff, 230), 2.0 * s);
        for p in &overlay.route {
            if let Some([x, y]) = project(*p) {
                fill_circle(scene, x, y, 4.0 * s, rgba(0x5e, 0xc8, 0xff, 255));
            }
        }
        // Route being planned: thinner, amber.
        draw_polyline(scene, &overlay.draft, &project, rgba(0xff, 0xb4, 0x54, 200), 1.5 * s);
        for p in &overlay.draft {
            if let Some([x, y]) = project(*p) {
                stroke_circle(scene, x, y, 5.0 * s, rgba(0xff, 0xb4, 0x54, 255), 1.5 * s);
            }
        }

        // Detections: hazard zones as outlined areas, people as green points.
        for d in &overlay.detections {
            match d.kind {
                MarkerKind::Hazard => {
                    let pts: Vec<[f64; 2]> = d.outline.iter().filter_map(|p| project(*p)).collect();
                    if pts.len() >= 3 && pts.len() == d.outline.len() {
                        let mut path = BezPath::new();
                        path.move_to((pts[0][0], pts[0][1]));
                        for p in &pts[1..] {
                            path.line_to((p[0], p[1]));
                        }
                        path.close_path();
                        scene.fill(Fill::NonZero, Affine::IDENTITY, rgba(0xff, 0x52, 0x52, 56), None, &path);
                        scene.stroke(
                            &Stroke::new(1.5 * s),
                            Affine::IDENTITY,
                            rgba(0xff, 0x52, 0x52, 220),
                            None,
                            &path,
                        );
                    }
                    if let Some([x, y]) = project(d.pos) {
                        fill_circle(scene, x, y, 5.0 * s, rgba(0xff, 0x52, 0x52, 255));
                        stroke_circle(scene, x, y, 9.0 * s, rgba(0xff, 0x52, 0x52, 150), 1.5 * s);
                    }
                }
                MarkerKind::Person => {
                    if let Some([x, y]) = project(d.pos) {
                        fill_circle(scene, x, y, 5.0 * s, rgba(0x39, 0xd9, 0x8a, 255));
                        stroke_circle(scene, x, y, 9.0 * s, rgba(0xff, 0xff, 0xff, 200), 1.5 * s);
                    }
                }
            }
            if d.focused
                && let Some([x, y]) = project(d.pos)
            {
                stroke_circle(scene, x, y, 15.0 * s, rgba(0xff, 0xff, 0xff, 230), 2.0 * s);
            }
        }

        // Drones: tinted dot, heading tick, ring on the active one.
        for d in &overlay.drones {
            let Some([x, y]) = project(d.pos) else { continue };
            let alpha = if d.stale { 110 } else { 255 };
            let tint = rgba(d.tint[0], d.tint[1], d.tint[2], alpha);
            let h = d.heading_deg.to_radians();
            let ahead = [d.pos[0] + h.sin() * 12.0, d.pos[1] + h.cos() * 12.0, d.pos[2]];
            if let Some([hx, hy]) = project(ahead) {
                let (dx, dy) = (hx - x, hy - y);
                let len = (dx * dx + dy * dy).sqrt().max(1e-6);
                let reach = 16.0 * s;
                scene.stroke(
                    &Stroke::new(2.0 * s),
                    Affine::IDENTITY,
                    tint,
                    None,
                    &Line::new((x, y), (x + dx / len * reach, y + dy / len * reach)),
                );
            }
            fill_circle(scene, x, y, 6.0 * s, tint);
            if d.active {
                stroke_circle(scene, x, y, 11.0 * s, rgba(0x5e, 0xc8, 0xff, 240), 2.0 * s);
            }
        }
    }
}

impl Gpu {
    fn new(handle: &DeviceHandle) -> Self {
        let device = handle.device.clone();
        let queue = handle.queue.clone();
        let renderer = Renderer::new(&device, COLOR_FORMAT);
        let mesh = MeshPass::new(&device, COLOR_FORMAT, Some(DEPTH_FORMAT));
        Self {
            device,
            queue,
            renderer,
            scene: Scene::new(),
            mesh,
            depth: None,
            displayed: None,
            next: None,
            generation: 0,
        }
    }

    fn prepared(layer: Layer, payload: &[u8], display: DisplayMode, generation: u32, flags: u32) -> Vec<u8> {
        let mut bytes = retag(payload, generation, flags);
        recolor_in_place(layer, display, &mut bytes);
        bytes
    }

    /// Appends a chunk to the merged scene under its current generation.
    fn insert(&mut self, layer: Layer, payload: &Payload, display: DisplayMode) {
        let generation = self.scene.generation().unwrap_or(self.generation);
        let bytes = Self::prepared(layer, payload, display, generation, 0);
        if let Err(e) = self.scene.insert(&self.device, &bytes) {
            crate::log(&format!("[view3d] dropped point chunk: {e}"));
        }
    }

    /// Replaces the whole merged scene: every chunk goes in as one new generation
    /// that pointblitz swaps in at its last chunk, so the screen never goes blank.
    fn rebuild(&mut self, list: &[(Layer, Payload)], display: DisplayMode) {
        if list.is_empty() {
            self.scene = Scene::new();
            return;
        }
        let next = self.scene.generation().map_or(self.generation, |g| g.wrapping_add(1));
        self.generation = next;
        let last = list.len() - 1;
        for (i, (layer, payload)) in list.iter().enumerate() {
            let flags = if i == last { FLAG_LAST_IN_GENERATION } else { 0 };
            let bytes = Self::prepared(*layer, payload, display, next, flags);
            if let Err(e) = self.scene.insert(&self.device, &bytes) {
                crate::log(&format!("[view3d] dropped point chunk in rebuild: {e}"));
            }
        }
    }

    fn apply_mesh(&mut self, payload: &Payload) {
        if let Err(e) = self.mesh.apply(&self.device, &self.queue, payload) {
            crate::log(&format!("[view3d] mesh block rejected: {e}"));
        }
    }
}

impl Widget for View3d {
    fn can_create_surfaces(&mut self, render_ctx: &mut dyn anyrender::RenderContext) {
        let Some(context) = render_ctx.renderer_specific_context() else {
            crate::log("[view3d] renderer exposes no context; 3D view disabled");
            return;
        };
        match context.downcast::<DeviceHandle>() {
            Ok(handle) => {
                let backend = format!("{:?}", handle.adapter.get_info().backend);
                self.gpu = Some(Gpu::new(&handle));
                let mut s = self.shared.borrow_mut();
                s.ops.clear();
                s.needs_resync = true;
                s.dirty = true;
                s.stats.gpu_active = true;
                s.stats.backend = backend;
            }
            Err(_) => crate::log("[view3d] renderer is not wgpu; 3D view disabled"),
        }
    }

    fn destroy_surfaces(&mut self) {
        self.gpu = None;
        let mut s = self.shared.borrow_mut();
        s.stats.gpu_active = false;
        s.stats.points = 0;
    }

    fn requires_redraw(&self) -> bool {
        let s = self.shared.borrow();
        s.dirty || (self.gpu.is_some() && !s.ops.is_empty())
    }

    fn handle_event(&mut self, event: &UiEvent) {
        match event {
            UiEvent::PointerDown(e) => {
                let pan = matches!(e.button, MouseEventButton::Secondary | MouseEventButton::Auxiliary)
                    || e.buttons.contains(MouseEventButtons::Secondary)
                    || e.mods.shift();
                let (x, y) = (e.coords.client_x, e.coords.client_y);
                self.drag = Some(if pan { Drag::Pan { x, y } } else { Drag::Orbit { x, y } });
            }
            UiEvent::PointerMove(e) => {
                if e.buttons.is_empty() {
                    self.drag = None;
                    return;
                }
                let (x, y) = (e.coords.client_x, e.coords.client_y);
                match &mut self.drag {
                    Some(Drag::Orbit { x: lx, y: ly }) => {
                        self.camera.orbit(f64::from(x - *lx), f64::from(y - *ly));
                        (*lx, *ly) = (x, y);
                    }
                    Some(Drag::Pan { x: lx, y: ly }) => {
                        self.camera.pan(f64::from(x - *lx), f64::from(y - *ly));
                        (*lx, *ly) = (x, y);
                    }
                    None => return,
                }
                self.mark_dirty();
            }
            UiEvent::PointerUp(_) | UiEvent::PointerCancel(_) => self.drag = None,
            UiEvent::Wheel(e) => {
                let steps = match e.delta {
                    BlitzWheelDelta::Lines(_, y) => -y,
                    BlitzWheelDelta::Pixels(_, y) => -y / 60.0,
                };
                self.camera.zoom(steps);
                self.mark_dirty();
            }
            _ => {}
        }
    }

    fn paint(
        &mut self,
        render_ctx: &mut dyn anyrender::RenderContext,
        _styles: &ComputedStyles,
        width: u32,
        height: u32,
        scale: f64,
    ) -> anyrender::Scene {
        self.apply_ops();
        let mut scene = anyrender::Scene::new();
        if let Some(resource) = self.render(render_ctx, width, height) {
            scene.fill(
                Fill::NonZero,
                Affine::IDENTITY,
                PaintRef::Resource(ImageBrush {
                    image: resource,
                    sampler: ImageSampler::default(),
                }),
                None,
                &Rect::from_origin_size((0.0, 0.0), (f64::from(width), f64::from(height))),
            );
        }
        self.paint_overlay(&mut scene, width, height, scale);
        self.shared.borrow_mut().dirty = false;
        scene
    }
}

fn rgba(r: u8, g: u8, b: u8, a: u8) -> Color {
    Color::from_rgba8(r, g, b, a)
}

fn fill_circle(scene: &mut anyrender::Scene, x: f64, y: f64, r: f64, color: Color) {
    scene.fill(Fill::NonZero, Affine::IDENTITY, color, None, &Circle::new((x, y), r));
}

fn stroke_circle(scene: &mut anyrender::Scene, x: f64, y: f64, r: f64, color: Color, width: f64) {
    scene.stroke(&Stroke::new(width), Affine::IDENTITY, color, None, &Circle::new((x, y), r));
}

fn draw_polyline(
    scene: &mut anyrender::Scene,
    points: &[[f64; 3]],
    project: &impl Fn([f64; 3]) -> Option<[f64; 2]>,
    color: Color,
    width: f64,
) {
    // Split at points behind the camera instead of drawing across the screen.
    let mut path = BezPath::new();
    let mut open = false;
    for p in points {
        match project(*p) {
            Some([x, y]) => {
                if open {
                    path.line_to((x, y));
                } else {
                    path.move_to((x, y));
                    open = true;
                }
            }
            None => open = false,
        }
    }
    if points.len() >= 2 {
        scene.stroke(&Stroke::new(width), Affine::IDENTITY, color, None, &path);
    }
}

fn color_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("skylens view color"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: COLOR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn depth_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("skylens view depth"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
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
    fn rebuild_drops_queued_point_inserts() {
        let mut s = ViewShared::default();
        let p: Payload = Rc::from(&[1u8, 2, 3][..]);
        s.push_ops(vec![
            SceneOp::InsertPoints(Layer::Recon, p.clone()),
            SceneOp::ApplyMesh(p.clone()),
        ]);
        s.push_ops(vec![SceneOp::RebuildPoints(vec![(Layer::Recon, p.clone())])]);
        assert_eq!(s.ops.len(), 2);
        assert!(matches!(s.ops[0], SceneOp::ApplyMesh(_)));
        assert!(matches!(s.ops[1], SceneOp::RebuildPoints(_)));
        s.push_ops(vec![SceneOp::ResetMesh(vec![])]);
        assert_eq!(s.ops.len(), 2);
        assert!(s.dirty);
    }
}
