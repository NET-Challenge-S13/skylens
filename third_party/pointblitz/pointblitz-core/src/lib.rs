//! PointBlitz render core: scene, chunk store, GPU buffers, point pipeline, capture.
//!
//! No window, network or browser API here (decision 0003). Targets (`native`, `web`, `server`)
//! own the device, the surface and the frame loop, and call into this crate.

pub mod camera;
pub mod headless;
pub mod layers;
pub mod renderer;
pub mod scene;
pub mod timer;

pub use camera::Camera;
pub use headless::Headless;
pub use layers::{Blend, Fog, FrameStyle, LineRenderer, LineSet, Material, MeshData, MeshRenderer, Shading, TextureImage, Topology, begin_pass};
pub use renderer::{PointSize, Renderer};
pub use scene::{Inserted, Scene};
pub use timer::GpuTimer;
