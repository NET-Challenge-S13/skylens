//! GPS <-> local ENU (East/North/Up) conversion.
//!
//! Rust twin of `src/skylens_model/utils/geo.py`. Both use the same
//! equirectangular small-area approximation, the same Earth radius and the same
//! scene axes (x = East, y = Up, z = -North). Change one, change the other.

/// Earth radius in metres. Must match `_R` in geo.py.
pub const EARTH_RADIUS_M: f64 = 6_378_137.0;
const DEG: f64 = std::f64::consts::PI / 180.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gps {
    pub lat: f64,
    pub lon: f64,
    /// Metres, ellipsoidal or relative; used as-is for Up.
    pub alt: f64,
}

/// Reference point that defines the local ENU origin.
pub type GeoAnchor = Gps;

/// Local East/North/Up offset in metres from a [`GeoAnchor`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Enu {
    pub e: f64,
    pub n: f64,
    pub u: f64,
}

/// Scene axes: x = East, y = Up, z = -North.
pub type SceneVec3 = [f64; 3];

pub fn gps_to_enu(gps: Gps, anchor: GeoAnchor) -> Enu {
    let d_lat = (gps.lat - anchor.lat) * DEG;
    let d_lon = (gps.lon - anchor.lon) * DEG;
    Enu {
        e: d_lon * EARTH_RADIUS_M * (anchor.lat * DEG).cos(),
        n: d_lat * EARTH_RADIUS_M,
        u: gps.alt - anchor.alt,
    }
}

pub fn enu_to_gps(enu: Enu, anchor: GeoAnchor) -> Gps {
    Gps {
        lat: anchor.lat + (enu.n / EARTH_RADIUS_M) / DEG,
        lon: anchor.lon + (enu.e / (EARTH_RADIUS_M * (anchor.lat * DEG).cos())) / DEG,
        alt: anchor.alt + enu.u,
    }
}

pub fn enu_to_scene(enu: Enu) -> SceneVec3 {
    [enu.e, enu.u, -enu.n]
}

pub fn scene_to_enu(v: SceneVec3) -> Enu {
    Enu { e: v[0], u: v[1], n: -v[2] }
}

pub fn gps_to_scene(gps: Gps, anchor: GeoAnchor) -> SceneVec3 {
    enu_to_scene(gps_to_enu(gps, anchor))
}

pub fn scene_to_gps(v: SceneVec3, anchor: GeoAnchor) -> Gps {
    enu_to_gps(scene_to_enu(v), anchor)
}

impl From<crate::proto::model::v1::Gps> for Gps {
    fn from(g: crate::proto::model::v1::Gps) -> Self {
        Gps { lat: g.lat, lon: g.lon, alt: g.alt }
    }
}

impl From<Gps> for crate::proto::model::v1::Gps {
    fn from(g: Gps) -> Self {
        crate::proto::model::v1::Gps { lat: g.lat, lon: g.lon, alt: g.alt }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Default anchor of the model server (SKYLENS_ANCHOR in API.md).
    const ANCHOR: GeoAnchor = Gps { lat: 36.3685, lon: 127.3475, alt: 30.0 };

    #[test]
    fn anchor_maps_to_origin() {
        assert_eq!(gps_to_enu(ANCHOR, ANCHOR), Enu { e: 0.0, n: 0.0, u: 0.0 });
    }

    #[test]
    fn matches_geo_py_values() {
        // Expected values computed with the same formulas as geo.py:
        // e = dlon*R*cos(lat0), n = dlat*R, u = alt - alt0.
        let g = Gps { lat: 36.3695, lon: 127.3490, alt: 42.5 };
        let enu = gps_to_enu(g, ANCHOR);
        let e = 0.0015 * DEG * EARTH_RADIUS_M * (36.3685 * DEG).cos();
        let n = 0.0010 * DEG * EARTH_RADIUS_M;
        assert!((enu.e - e).abs() < 1e-6, "e {} vs {}", enu.e, e);
        assert!((enu.n - n).abs() < 1e-6, "n {} vs {}", enu.n, n);
        assert!((enu.u - 12.5).abs() < 1e-12);
    }

    #[test]
    fn round_trips() {
        let g = Gps { lat: 36.3712, lon: 127.3441, alt: 55.0 };
        let back = enu_to_gps(gps_to_enu(g, ANCHOR), ANCHOR);
        assert!((back.lat - g.lat).abs() < 1e-12);
        assert!((back.lon - g.lon).abs() < 1e-12);
        assert!((back.alt - g.alt).abs() < 1e-12);
    }

    #[test]
    fn scene_axes_are_east_up_minus_north() {
        let v = enu_to_scene(Enu { e: 1.0, n: 2.0, u: 3.0 });
        assert_eq!(v, [1.0, 3.0, -2.0]);
        assert_eq!(scene_to_enu(v), Enu { e: 1.0, n: 2.0, u: 3.0 });
    }
}
