//! Rust twin of `utils/geo.py`.
//!
//! Rust has ONE implementation of the GPS <-> ENU conversion, in
//! `skylens_shared::geo`; this module only re-exports it so the model crate keeps
//! the same module layout as the Python package.

pub use skylens_shared::geo::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reexport_round_trips() {
        let anchor = Gps {
            lat: 36.3685,
            lon: 127.3475,
            alt: 30.0,
        };
        let enu = Enu {
            e: 12.0,
            n: -4.0,
            u: 1.5,
        };
        let back = gps_to_enu(enu_to_gps(enu, anchor), anchor);
        assert!((back.e - enu.e).abs() < 1e-6);
        assert!((back.n - enu.n).abs() < 1e-6);
        assert!((back.u - enu.u).abs() < 1e-9);
    }
}
