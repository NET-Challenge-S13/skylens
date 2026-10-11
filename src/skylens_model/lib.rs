//! Compute server library. Each `.rs` module sits next to the `.py` file it ports,
//! and `mod.rs` plays the role of `__init__.py`.

pub mod app; // app.py: REST (same endpoints as app.py) + gRPC ModelService
pub mod datasets;
pub mod detect; // contract between the UNet and the reconstruction frame hook
pub mod models;
pub mod utils;
