//! SkyLens additions: everything a map-style scene needs next to the points, drawn into one
//! caller-owned pass so points, meshes and lines share a depth buffer.
//!
//! Frame recipe (all layers use the same camera-relative convention as the points: positions are
//! absolute ENU metres, the GPU sees `position - camera.eye`):
//!
//! ```ignore
//! let style = FrameStyle { clear: [0.039, 0.067, 0.157, 1.0], fog: Some(Fog { .. }) };
//! points.prepare(device, queue, size, &camera, &scene, style.fog);
//! meshes.prepare(queue, size, &camera, style.fog);
//! lines.prepare(queue, size, &camera, style.fog);
//! {
//!     let mut pass = begin_pass(encoder, color, depth, &style);
//!     meshes.draw_opaque(&mut pass);
//!     points.draw(&mut pass, &scene);
//!     meshes.draw_transparent(&mut pass);
//!     lines.draw(&mut pass);
//! }
//! ```
//!
//! Depth is reversed-Z (`Depth32Float`, cleared to 0, compare `Greater`), exactly like the point
//! pipeline. The signatures in this file are the contract the SkyLens screens code against.

use crate::camera::Camera;

/// Linear distance fog, applied by every layer (like three.js `Fog`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fog {
    pub color: [f32; 3],
    /// Metres from the eye where fog starts and where it is total.
    pub near: f32,
    pub far: f32,
}

/// Per-frame look shared by all layers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameStyle {
    /// Background colour (linear RGBA, as the colour target expects).
    pub clear: [f64; 4],
    pub fog: Option<Fog>,
}

/// Opens the one render pass of a frame: clears colour to `style.clear`, depth to 0, and KEEPS
/// depth stored so later passes (e.g. overlays) may test against it.
pub fn begin_pass<'e>(
    encoder: &'e mut wgpu::CommandEncoder,
    color: &wgpu::TextureView,
    depth: &wgpu::TextureView,
    style: &FrameStyle,
) -> wgpu::RenderPass<'e> {
    let _ = (encoder, color, depth, style);
    todo!("pointblitz layers")
}

// ---------------------------------------------------------------------------------------------
// Meshes
// ---------------------------------------------------------------------------------------------

/// How a mesh is lit. Mirrors the three.js materials the SkyLens tower used.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Shading {
    /// Colour (vertex colour x texture x tint) as is.
    Unlit,
    /// `MeshLambertMaterial` under a hemisphere light plus one directional sun, with emissive.
    Lambert {
        sky: [f32; 3],
        ground: [f32; 3],
        hemi_intensity: f32,
        sun_dir: [f32; 3],
        sun_color: [f32; 3],
        sun_intensity: f32,
        emissive: [f32; 3],
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Blend {
    Opaque,
    /// Alpha blended, depth tested, no depth write.
    Alpha,
    /// Additive (`AdditiveBlending`), depth tested, no depth write.
    Additive,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Material {
    pub shading: Shading,
    /// Multiplied with vertex colours and the texture. Alpha is the opacity.
    pub tint: [f32; 4],
    pub blend: Blend,
    /// Draw back faces too.
    pub double_sided: bool,
}

/// RGBA8 sRGB image, row-major, top row first.
#[derive(Debug, Clone)]
pub struct TextureImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// One mesh in absolute ENU metres. Missing normals are computed (smooth when `flat == false`,
/// per-face when `flat == true`).
#[derive(Debug, Clone, Default)]
pub struct MeshData {
    pub positions: Vec<[f64; 3]>,
    pub normals: Option<Vec<[f32; 3]>>,
    /// RGBA8 per vertex; white when absent.
    pub colors: Option<Vec<[u8; 4]>>,
    /// Texture coordinates; required when a texture is given.
    pub uvs: Option<Vec<[f32; 2]>>,
    pub indices: Vec<u32>,
    pub flat: bool,
}

/// Owns GPU meshes by caller-chosen id.
pub struct MeshRenderer {
    _private: (),
}

impl MeshRenderer {
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let _ = (device, color_format);
        todo!("pointblitz layers")
    }

    /// Inserts or replaces mesh `id`. The texture, if any, is uploaded once and owned by the mesh.
    pub fn upsert(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: u64,
        mesh: &MeshData,
        texture: Option<&TextureImage>,
        material: Material,
    ) {
        let _ = (device, queue, id, mesh, texture, material);
        todo!("pointblitz layers")
    }

    /// Changes only the material (e.g. a display-mode switch) without re-uploading geometry.
    pub fn set_material(&mut self, id: u64, material: Material) {
        let _ = (id, material);
        todo!("pointblitz layers")
    }

    pub fn set_visible(&mut self, id: u64, visible: bool) {
        let _ = (id, visible);
        todo!("pointblitz layers")
    }

    pub fn remove(&mut self, id: u64) {
        let _ = id;
        todo!("pointblitz layers")
    }

    pub fn clear(&mut self) {
        todo!("pointblitz layers")
    }

    pub fn len(&self) -> usize {
        todo!("pointblitz layers")
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Uploads camera, light and fog uniforms for this frame.
    pub fn prepare(&mut self, queue: &wgpu::Queue, size: [u32; 2], camera: &Camera, fog: Option<Fog>) {
        let _ = (queue, size, camera, fog);
        todo!("pointblitz layers")
    }

    /// Draws `Blend::Opaque` meshes.
    pub fn draw_opaque(&self, pass: &mut wgpu::RenderPass<'_>) {
        let _ = pass;
        todo!("pointblitz layers")
    }

    /// Draws `Blend::Alpha` and `Blend::Additive` meshes (after opaque geometry and points).
    pub fn draw_transparent(&self, pass: &mut wgpu::RenderPass<'_>) {
        let _ = pass;
        todo!("pointblitz layers")
    }
}

// ---------------------------------------------------------------------------------------------
// Lines
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Topology {
    /// Pairs of points.
    List,
    /// One connected polyline.
    Strip,
}

/// A set of 1-pixel lines (like three.js `LineBasicMaterial` / `LineSegments`).
#[derive(Debug, Clone)]
pub struct LineSet {
    pub positions: Vec<[f64; 3]>,
    pub topology: Topology,
    /// Per-vertex RGBA8; overrides `color` when present (used for fading trails).
    pub colors: Option<Vec<[u8; 4]>>,
    pub color: [f32; 4],
    pub blend: Blend,
    /// `false` draws on top of everything (no depth test).
    pub depth_test: bool,
}

/// Owns GPU line sets by caller-chosen id.
pub struct LineRenderer {
    _private: (),
}

impl LineRenderer {
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let _ = (device, color_format);
        todo!("pointblitz layers")
    }

    pub fn upsert(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, id: u64, lines: &LineSet) {
        let _ = (device, queue, id, lines);
        todo!("pointblitz layers")
    }

    pub fn remove(&mut self, id: u64) {
        let _ = id;
        todo!("pointblitz layers")
    }

    pub fn clear(&mut self) {
        todo!("pointblitz layers")
    }

    pub fn prepare(&mut self, queue: &wgpu::Queue, size: [u32; 2], camera: &Camera, fog: Option<Fog>) {
        let _ = (queue, size, camera, fog);
        todo!("pointblitz layers")
    }

    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        let _ = pass;
        todo!("pointblitz layers")
    }
}
