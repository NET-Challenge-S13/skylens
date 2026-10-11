//! Mesh render path added next to pointblitz's point pipeline (COMPONENTS.md §3.6, §9).
//!
//! pointblitz only draws points, so deltamesh `BlockMesh` blocks (`CHUNK_KIND_MESH_BLOCK`,
//! `skylens.payload.v1.MeshBlock`) get their own pipeline here. Blocks are kept per id, only a
//! strictly newer `version` replaces a block, and a block with zero triangles is removed.
//! Normals are not in the payload: smooth per-vertex normals are computed on upload.
//!
//! Conventions, matched to pointblitz-core 0.2 so meshes and points line up:
//! - World axes: ENU metres, x east, y north, z up (`camera.rs`, decision 0013). The payload is
//!   already absolute ENU metres, so positions go to the GPU unchanged.
//! - Projection: right-handed view, reversed-Z with an infinite far plane (depth 1 at the near
//!   plane, 0 at infinity). Depth test is `Greater`, depth cleared to 0.
//! - Depth format: `pointblitz::core::renderer::DEPTH_FORMAT` (`Depth32Float`).
//! - pointblitz renders camera-relative: its `Camera::view_proj` has the eye at the origin and
//!   chunks are offset by `origin − eye` on the CPU. [`MeshPass::draw`] instead takes an
//!   *absolute* ENU → clip matrix; build it with [`absolute_view_proj`] from pointblitz's matrix
//!   and `camera.eye`. Precision cost: about 0.5 mm at 5 km from the ENU origin, well under the
//!   0.2 m mesh voxel.
//! - Colours: RGB8 read as unorm, opaque, no blending; the same as pointblitz's point colours.

mod block;

use std::collections::HashMap;

use wgpu::util::DeviceExt;

pub use block::{
    absolute_view_proj, decode_block, smooth_normals, BlockId, BlockVersions, DecodedBlock,
};

/// Unit vector towards the light in ENU (from the south-east, high up) and the ambient term.
const LIGHT: [f32; 4] = [0.267_261_24, -0.534_522_5, 0.801_783_7, 0.35];

/// Uniform block: `view_proj` (64 bytes) then `light` (16 bytes). 16-byte aligned for WebGL2.
const UNIFORM_SIZE: u64 = 80;

struct GpuBlock {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
}

pub struct MeshPass {
    pipeline: wgpu::RenderPipeline,
    uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    blocks: HashMap<BlockId, GpuBlock>,
    versions: BlockVersions,
}

impl MeshPass {
    /// `depth_format` should be pointblitz's `DEPTH_FORMAT` when drawing into a pass that shares
    /// depth with the points; `None` draws without a depth test (painter's order).
    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        depth_format: Option<wgpu::TextureFormat>,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("skylens mesh"),
            source: wgpu::ShaderSource::Wgsl(include_str!("mesh.wgsl").into()),
        });
        let layout_bg = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mesh uniforms"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(UNIFORM_SIZE),
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh"),
            bind_group_layouts: &[Some(&layout_bg)],
            immediate_size: 0,
        });
        let attributes =
            wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Unorm8x4];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: block::VERTEX_STRIDE as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &attributes,
                })],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                // Open surfaces must stay visible from below; the shader flips back-face normals.
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
                format,
                depth_write_enabled: Some(true),
                // Reversed-Z like pointblitz: nearer is larger.
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh uniforms"),
            size: UNIFORM_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh uniforms"),
            layout: &layout_bg,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });
        Self {
            pipeline,
            uniform_buf,
            bind_group,
            blocks: HashMap::new(),
            versions: BlockVersions::default(),
        }
    }

    /// Decodes a CHUNK_KIND_MESH_BLOCK payload; ignores stale versions; zero triangles removes the block.
    ///
    /// A malformed payload returns an error and leaves every block (and its version) unchanged.
    pub fn apply(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        payload: &[u8],
    ) -> anyhow::Result<()> {
        let b = decode_block(payload)?;
        if !self.versions.admit(b.id, b.version) {
            return Ok(());
        }
        if b.is_removal() {
            self.blocks.remove(&b.id);
            return Ok(());
        }
        let normals = smooth_normals(&b.positions, &b.indices);
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh block vertices"),
            contents: &block::pack_vertices(&b.positions, &normals, &b.colors),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh block indices"),
            contents: &block::pack_indices(&b.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        // Replacing the entry drops the old buffers; wgpu keeps them alive until in-flight work ends.
        self.blocks.insert(
            b.id,
            GpuBlock { vertices, indices, index_count: b.indices.len() as u32 },
        );
        Ok(())
    }

    /// Drops every block and forgets all versions (e.g. when the room or ENU anchor changes).
    pub fn clear(&mut self) {
        self.blocks.clear();
        self.versions.clear();
    }

    /// Number of blocks currently held on the GPU.
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Records every block into `pass`. `view_proj` maps absolute ENU metres to clip space
    /// (column-major); see [`absolute_view_proj`]. The pass's colour target and depth attachment
    /// must match the formats given to [`MeshPass::new`].
    ///
    /// The uniform is written with `queue.write_buffer`, which lands before the next submit, so
    /// call this once per submitted frame (the last call's matrix wins otherwise).
    pub fn draw(&self, queue: &wgpu::Queue, pass: &mut wgpu::RenderPass<'_>, view_proj: [[f32; 4]; 4]) {
        if self.blocks.is_empty() {
            return;
        }
        let mut bytes = Vec::with_capacity(UNIFORM_SIZE as usize);
        for v in view_proj.iter().flatten().chain(LIGHT.iter()) {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        queue.write_buffer(&self.uniform_buf, 0, &bytes);

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        for b in self.blocks.values() {
            pass.set_vertex_buffer(0, b.vertices.slice(..));
            pass.set_index_buffer(b.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..b.index_count, 0, 0..1);
        }
    }
}
