//! Contracts shared by every Rust component.
//!
//! - [`proto`]: messages generated from `proto/` (the same files the Go core
//!   generates from). The `grpc` feature adds the `ModelService` client and server.
//! - [`geo`]: GPS <-> local ENU conversion, numerically identical to
//!   `src/skylens_model/utils/geo.py`.

pub mod geo;

pub mod proto {
    pub mod model {
        pub mod v1 {
            include!(concat!(env!("OUT_DIR"), "/skylens.model.v1.rs"));
        }
    }
    pub mod viewer {
        pub mod v1 {
            include!(concat!(env!("OUT_DIR"), "/skylens.viewer.v1.rs"));
        }
    }
    pub mod payload {
        pub mod v1 {
            include!(concat!(env!("OUT_DIR"), "/skylens.payload.v1.rs"));
        }
    }
}
