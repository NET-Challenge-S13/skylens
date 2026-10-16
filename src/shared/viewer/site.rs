//! The operating area the core announces (`viewer.v1.Site`, sent first on every connection).
//!
//! `Site.anchor` is the ENU origin of every chunk, terrain tile and detection. Screens never pick
//! their own: until a `Site` arrives there is no frame, and GPS cannot be placed in the scene.
//!
//! Positions here are pointblitz ENU metres, `[east, north, up]` (z up). The legacy three.js scene
//! used `[east, up, -north]`; [`legacy_scene_to_enu`] converts when porting legacy offsets.

use skylens_shared::geo::{Enu, GeoAnchor, Gps, enu_to_gps, gps_to_enu};
use skylens_shared::proto::model::v1 as model;
use skylens_shared::proto::viewer::v1 as viewer;

/// The operating area and its ENU frame.
#[derive(Debug, Clone, PartialEq)]
pub struct SiteFrame {
    pub anchor: GeoAnchor,
    pub name: String,
    /// Bounding box (degrees).
    pub min_lat: f64,
    pub min_lon: f64,
    pub max_lat: f64,
    pub max_lon: f64,
    /// Satellite imagery available (VWorld key present on the core).
    pub imagery: bool,
    /// Building footprints available.
    pub buildings: bool,
    pub building_count: u32,
}

impl SiteFrame {
    /// Adopts a `Site` message. None when it carries no anchor (the frame would be undefined).
    pub fn from_site(site: &viewer::Site) -> Option<SiteFrame> {
        let a = site.anchor.as_ref()?;
        let anchor = Gps { lat: a.lat, lon: a.lon, alt: a.alt };
        if !(anchor.lat.is_finite() && anchor.lon.is_finite() && anchor.alt.is_finite()) {
            return None;
        }
        Some(SiteFrame {
            anchor,
            name: site.name.clone(),
            min_lat: site.min_lat,
            min_lon: site.min_lon,
            max_lat: site.max_lat,
            max_lon: site.max_lon,
            imagery: site.imagery,
            buildings: site.buildings,
            building_count: site.building_count,
        })
    }

    /// GPS to ENU metres `[e, n, u]` about the anchor.
    pub fn enu(&self, gps: Gps) -> [f64; 3] {
        let e = gps_to_enu(gps, self.anchor);
        [e.e, e.n, e.u]
    }

    /// Protobuf GPS to ENU metres.
    pub fn enu_of(&self, gps: &model::Gps) -> [f64; 3] {
        self.enu(Gps { lat: gps.lat, lon: gps.lon, alt: gps.alt })
    }

    /// ENU metres back to GPS.
    pub fn gps(&self, enu: [f64; 3]) -> Gps {
        enu_to_gps(Enu { e: enu[0], n: enu[1], u: enu[2] }, self.anchor)
    }

    /// ENU metres back to a protobuf GPS (route waypoints).
    pub fn gps_msg(&self, enu: [f64; 3]) -> model::Gps {
        self.gps(enu).into()
    }

    /// Whether a GPS position lies inside the bounding box. An empty box contains nothing.
    pub fn contains(&self, lat: f64, lon: f64) -> bool {
        self.has_bounds()
            && (self.min_lat..=self.max_lat).contains(&lat)
            && (self.min_lon..=self.max_lon).contains(&lon)
    }

    pub fn has_bounds(&self) -> bool {
        self.max_lat > self.min_lat && self.max_lon > self.min_lon
    }

    /// The bounding box in ENU metres at anchor altitude: `(min [e, n], max [e, n])`.
    pub fn enu_bounds(&self) -> Option<([f64; 2], [f64; 2])> {
        if !self.has_bounds() {
            return None;
        }
        let alt = self.anchor.alt;
        let lo = self.enu(Gps { lat: self.min_lat, lon: self.min_lon, alt });
        let hi = self.enu(Gps { lat: self.max_lat, lon: self.max_lon, alt });
        Some(([lo[0], lo[1]], [hi[0], hi[1]]))
    }

    /// Centre of the bounding box in ENU (anchor altitude), or the origin without bounds.
    pub fn enu_center(&self) -> [f64; 3] {
        match self.enu_bounds() {
            Some((lo, hi)) => [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, 0.0],
            None => [0.0; 3],
        }
    }
}

/// Legacy three.js scene coordinates `[east, up, -north]` to ENU `[east, north, up]`.
pub fn legacy_scene_to_enu(v: [f64; 3]) -> [f64; 3] {
    [v[0], -v[2], v[1]]
}

/// ENU `[east, north, up]` to legacy three.js scene coordinates `[east, up, -north]`.
pub fn enu_to_legacy_scene(v: [f64; 3]) -> [f64; 3] {
    [v[0], v[2], -v[1]]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn site() -> viewer::Site {
        viewer::Site {
            anchor: Some(model::Gps { lat: 36.3685, lon: 127.3475, alt: 30.0 }),
            min_lat: 36.36,
            min_lon: 127.34,
            max_lat: 36.38,
            max_lon: 127.36,
            name: "daejeon".into(),
            imagery: true,
            buildings: true,
            building_count: 6191,
            ..Default::default()
        }
    }

    #[test]
    fn adopts_the_core_anchor_as_origin() {
        let f = SiteFrame::from_site(&site()).unwrap();
        assert_eq!(f.enu(f.anchor), [0.0, 0.0, 0.0]);
        assert_eq!(f.name, "daejeon");
        assert_eq!(f.building_count, 6191);
    }

    #[test]
    fn site_without_anchor_has_no_frame() {
        let mut s = site();
        s.anchor = None;
        assert!(SiteFrame::from_site(&s).is_none());
        let mut s = site();
        s.anchor = Some(model::Gps { lat: f64::NAN, lon: 0.0, alt: 0.0 });
        assert!(SiteFrame::from_site(&s).is_none());
    }

    #[test]
    fn north_east_up_axes_and_round_trip() {
        let f = SiteFrame::from_site(&site()).unwrap();
        let p = f.enu(Gps { lat: 36.3695, lon: 127.3490, alt: 42.5 });
        assert!(p[0] > 0.0 && p[1] > 0.0);
        assert!((p[2] - 12.5).abs() < 1e-12);
        let back = f.gps(p);
        assert!((back.lat - 36.3695).abs() < 1e-12 && (back.lon - 127.3490).abs() < 1e-12);
        let m = f.gps_msg(p);
        assert!((m.alt - 42.5).abs() < 1e-9);
        assert_eq!(f.enu_of(&m), f.enu(back));
    }

    #[test]
    fn bounds_and_containment() {
        let f = SiteFrame::from_site(&site()).unwrap();
        assert!(f.contains(36.37, 127.35));
        assert!(!f.contains(36.40, 127.35));
        let (lo, hi) = f.enu_bounds().unwrap();
        assert!(lo[0] < 0.0 && lo[1] < 0.0 && hi[0] > 0.0 && hi[1] > 0.0);
        let mut s = site();
        s.max_lat = s.min_lat;
        let f = SiteFrame::from_site(&s).unwrap();
        assert!(f.enu_bounds().is_none());
        assert!(!f.contains(36.36, 127.35));
        assert_eq!(f.enu_center(), [0.0; 3]);
    }

    #[test]
    fn legacy_scene_axes_convert_both_ways() {
        let enu = [1.0, 2.0, 3.0];
        assert_eq!(enu_to_legacy_scene(enu), [1.0, 3.0, -2.0]);
        assert_eq!(legacy_scene_to_enu(enu_to_legacy_scene(enu)), enu);
    }
}
