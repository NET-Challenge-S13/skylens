//! Adapters from cumulus3d results to the engine's own types: zone clouds to
//! [`ZonePoint`]s and registered cameras to fusion [`View`]s.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use cumulus3d_cli::events::ZoneRange;
use cumulus3d_core::io::ply::PointCloud;
use cumulus3d_core::{ImageId, Reconstruction, Vec3};

use super::chunks::ZonePoint;
use super::fusion::{Projector, View};

/// Image ids of the zone: registered images whose position lies in the zone.
/// Images with an unknown position are kept rather than lost.
pub fn zone_images(
    model: &Reconstruction,
    range: ZoneRange,
    positions: &HashMap<String, usize>,
) -> Vec<ImageId> {
    model
        .registered_images()
        .into_iter()
        .filter(|id| {
            let Some(im) = model.image(*id) else {
                return false;
            };
            positions
                .get(&im.name)
                .is_none_or(|p| *p >= range.lo && *p < range.hi)
        })
        .collect()
}

/// The zone's points. The dense cloud when there is one; otherwise (no CUDA
/// densification) the sparse points observed by the zone's images.
pub fn zone_points(
    cloud: &PointCloud,
    model: &Reconstruction,
    range: ZoneRange,
    positions: &HashMap<String, usize>,
) -> Vec<ZonePoint> {
    if !cloud.is_empty() {
        return (0..cloud.len())
            .map(|i| ZonePoint {
                pos: cloud.positions[i],
                rgb: cloud.colors.get(i).copied().unwrap_or([255, 255, 255]),
                normal: cloud.normals.get(i).copied().unwrap_or([0.0; 3]),
            })
            .collect();
    }
    let ids: HashSet<ImageId> = zone_images(model, range, positions).into_iter().collect();
    model
        .points3d()
        .filter(|(_, p)| p.track.iter().any(|t| ids.contains(&t.image_id)))
        .map(|(_, p)| ZonePoint {
            pos: [p.xyz.x as f32, p.xyz.y as f32, p.xyz.z as f32],
            rgb: p.color,
            normal: [0.0; 3],
        })
        .collect()
}

/// Fusion views of the zone's registered cameras, keyed by stored image name.
pub fn zone_views(
    model: &Reconstruction,
    range: ZoneRange,
    positions: &HashMap<String, usize>,
) -> HashMap<String, View> {
    let mut out = HashMap::new();
    for id in zone_images(model, range, positions) {
        let (Some(im), Some(pose)) = (model.image(id), model.world_to_cam(id)) else {
            continue;
        };
        let Some(cam) = model.camera(im.camera_id).cloned() else {
            continue;
        };
        let r = pose.rotation_matrix();
        let rotation: [[f64; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| r[(i, j)]));
        let t = pose.translation;
        let (width, height) = (cam.width as u32, cam.height as u32);
        let project: Projector = Arc::new(move |p: [f64; 3]| {
            cam.cam_to_img(&Vec3::new(p[0], p[1], p[2]))
                .map(|v| [v.x, v.y])
        });
        out.insert(
            im.name.clone(),
            View::new(width, height, rotation, [t.x, t.y, t.z], project),
        );
    }
    out
}

/// ENU positions of a point list (fusion input).
pub fn positions_f64(points: &[ZonePoint]) -> Vec<[f64; 3]> {
    points.iter().map(|p| p.pos.map(f64::from)).collect()
}
