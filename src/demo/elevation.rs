//! Ground elevation at the takeoff point, from the same DEM the core builds its terrain from
//! (AWS Open Data terrarium tiles, SRTM ~30 m, zoom 12; `src/skylens_core/terrain/dem.go`).
//!
//! Using the core's source matters more than absolute accuracy: the tower draws the terrain as
//! DEM minus the anchor altitude and a drone at `alt - anchor.alt`, so a drone sent at
//! `DEM ground + rel_alt` hovers exactly `rel_alt` above the drawn ground.
//!
//! The tile is read from the core's terrain cache when it is there (so a second run works
//! offline) and written to it after a download (so the core reuses it).

use std::path::{Path, PathBuf};
use std::time::Duration;

/// Native zoom of the DEM (`demZoom` in dem.go).
pub const DEM_ZOOM: u32 = 12;
const TILE: usize = 256;

/// Takeoff area of the recorded flight (the lead's analysis of the DJI records). `rel_alt_m`
/// in `track.json` is measured from here.
pub const TAKEOFF: (f64, f64) = (36.3656, 127.3424);

/// Used when neither `--ground-alt` nor the DEM is available: an estimate of the ground around
/// the takeoff point (Yuseong, Daejeon). Off by a few metres at most; the tower then draws the
/// drones that much too high or too low over the terrain.
pub const FALLBACK_GROUND_ALT_M: f64 = 50.0;

pub fn terrarium_url(z: u32, x: u32, y: u32) -> String {
    format!("https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png")
}

/// Terrarium encoding: `R * 256 + G + B / 256 - 32768` metres.
pub fn terrarium(r: u8, g: u8, b: u8) -> f64 {
    f64::from(r) * 256.0 + f64::from(g) + f64::from(b) / 256.0 - 32768.0
}

/// Fractional Web Mercator tile coordinates (the same formulas as the core's tilemath.go).
pub fn tile_xy(lat: f64, lon: f64, z: u32) -> (f64, f64) {
    let n = 2f64.powi(z as i32);
    let r = lat.to_radians();
    ((lon + 180.0) / 360.0 * n, (1.0 - (r.tan() + 1.0 / r.cos()).ln() / std::f64::consts::PI) / 2.0 * n)
}

/// The core's terrain cache folder: `SKYLENS_CORE_TERRAIN_CACHE`, else Go's
/// `os.UserCacheDir()/skylens/terrain`.
pub fn cache_dir(var: impl Fn(&str) -> Option<String>) -> Option<PathBuf> {
    if let Some(dir) = var("SKYLENS_CORE_TERRAIN_CACHE").filter(|s| !s.trim().is_empty()) {
        return Some(PathBuf::from(dir.trim()));
    }
    let user_cache = if cfg!(windows) {
        var("LocalAppData").or_else(|| var("LOCALAPPDATA")).map(PathBuf::from)
    } else if cfg!(target_os = "macos") {
        var("HOME").map(|h| Path::new(&h).join("Library").join("Caches"))
    } else {
        var("XDG_CACHE_HOME")
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .or_else(|| var("HOME").map(|h| Path::new(&h).join(".cache")))
    }?;
    Some(user_cache.join("skylens").join("terrain"))
}

/// `<cache>/terrarium/z/x/y.png`, the name the core caches the tile under.
pub fn tile_cache_path(cache: &Path, z: u32, x: u32, y: u32) -> PathBuf {
    cache.join("terrarium").join(z.to_string()).join(x.to_string()).join(format!("{y}.png"))
}

/// Elevations of one decoded terrarium PNG, row-major.
pub fn decode_tile(png: &[u8]) -> Result<Vec<f64>, String> {
    let img = image::load_from_memory_with_format(png, image::ImageFormat::Png)
        .map_err(|e| format!("not a PNG: {e}"))?
        .to_rgb8();
    if img.width() as usize != TILE || img.height() as usize != TILE {
        return Err(format!("tile is {}x{}, want 256x256", img.width(), img.height()));
    }
    Ok(img.pixels().map(|p| terrarium(p[0], p[1], p[2])).collect())
}

/// Bilinear elevation at lat/lon, pixel centres at `i + 0.5` (as `demSampler.elevation`).
/// `tile(x, y)` gives a tile's elevations; `None` when any of the four samples is missing.
pub fn sample(lat: f64, lon: f64, z: u32, mut tile: impl FnMut(u32, u32) -> Option<Vec<f64>>) -> Option<f64> {
    let (tx, ty) = tile_xy(lat, lon, z);
    let gx = tx * TILE as f64 - 0.5;
    let gy = ty * TILE as f64 - 0.5;
    let (x0, y0) = (gx.floor() as i64, gy.floor() as i64);
    let (fx, fy) = (gx - x0 as f64, gy - y0 as f64);
    let mut loaded: Vec<((i64, i64), Vec<f64>)> = Vec::new();
    let mut px = |x: i64, y: i64| -> Option<f64> {
        if x < 0 || y < 0 {
            return None;
        }
        let key = (x / TILE as i64, y / TILE as i64);
        if !loaded.iter().any(|(k, _)| *k == key) {
            let t = tile(key.0 as u32, key.1 as u32)?;
            if t.len() != TILE * TILE {
                return None;
            }
            loaded.push((key, t));
        }
        let t = &loaded.iter().find(|(k, _)| *k == key)?.1;
        let v = t[(y as usize % TILE) * TILE + x as usize % TILE];
        (v.is_finite() && (-1000.0..=9500.0).contains(&v)).then_some(v)
    };
    let a = px(x0, y0)?;
    let b = px(x0 + 1, y0)?;
    let c = px(x0, y0 + 1)?;
    let d = px(x0 + 1, y0 + 1)?;
    Some(a * (1.0 - fx) * (1.0 - fy) + b * fx * (1.0 - fy) + c * (1.0 - fx) * fy + d * fx * fy)
}

/// One tile: from the cache, else downloaded (and cached for the core).
fn load_tile(cache: Option<&Path>, z: u32, x: u32, y: u32, errors: &mut Vec<String>) -> Option<Vec<f64>> {
    let cached = cache.map(|c| tile_cache_path(c, z, x, y));
    if let Some(p) = &cached
        && let Ok(bytes) = std::fs::read(p)
        && let Ok(t) = decode_tile(&bytes)
    {
        return Some(t);
    }
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(5))
        .timeout(Duration::from_secs(15))
        .try_proxy_from_env(true)
        .build();
    let url = terrarium_url(z, x, y);
    let mut bytes = Vec::new();
    match agent.get(&url).call() {
        Ok(resp) => {
            use std::io::Read as _;
            if let Err(e) = resp.into_reader().take(8 << 20).read_to_end(&mut bytes) {
                errors.push(format!("{url}: {e}"));
                return None;
            }
        }
        Err(e) => {
            errors.push(format!("{url}: {e}"));
            return None;
        }
    }
    let t = match decode_tile(&bytes) {
        Ok(t) => t,
        Err(e) => {
            errors.push(format!("{url}: {e}"));
            return None;
        }
    };
    if let Some(p) = &cached
        && let Some(dir) = p.parent()
        && std::fs::create_dir_all(dir).is_ok()
    {
        let part = p.with_extension("png.part");
        if std::fs::write(&part, &bytes).is_ok() {
            let _ = std::fs::rename(&part, p);
        }
    }
    Some(t)
}

/// Ground elevation (metres above sea level) at a point from the DEM, or why not.
pub fn ground_alt(lat: f64, lon: f64) -> Result<f64, String> {
    let cache = cache_dir(|k| std::env::var(k).ok());
    let mut errors = Vec::new();
    match sample(lat, lon, DEM_ZOOM, |x, y| load_tile(cache.as_deref(), DEM_ZOOM, x, y, &mut errors)) {
        Some(v) => Ok(v),
        None if errors.is_empty() => Err("DEM 표본이 비어 있습니다".into()),
        None => Err(errors.join("; ")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terrarium_decoding() {
        assert_eq!(terrarium(128, 0, 0), 0.0);
        assert_eq!(terrarium(128, 50, 128), 50.5);
        assert_eq!(terrarium(127, 255, 0), -1.0);
    }

    #[test]
    fn tile_coordinates_match_the_core() {
        let (x, y) = tile_xy(TAKEOFF.0, TAKEOFF.1, DEM_ZOOM);
        // x = (127.3424 + 180) / 360 * 4096 = 3496.87; y from the Mercator formula = 1603.3
        assert_eq!((x.floor() as u32, y.floor() as u32), (3496, 1603));
        let (x, y) = tile_xy(0.0, 0.0, 1);
        assert!((x - 1.0).abs() < 1e-12 && (y - 1.0).abs() < 1e-12);
        assert_eq!(
            terrarium_url(12, 3496, 1603),
            "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/12/3496/1603.png"
        );
    }

    #[test]
    fn cache_follows_the_core() {
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |k: &str| pairs.iter().find(|(n, _)| *n == k).map(|(_, v)| v.to_string())
        };
        assert_eq!(
            cache_dir(env(&[("SKYLENS_CORE_TERRAIN_CACHE", "/tmp/c"), ("HOME", "/h"), ("LocalAppData", "C:/L")])),
            Some(PathBuf::from("/tmp/c"))
        );
        let default = cache_dir(env(&[("HOME", "/h"), ("LocalAppData", "C:/L")])).unwrap();
        assert!(default.ends_with(Path::new("skylens").join("terrain")), "{}", default.display());
        assert_eq!(cache_dir(env(&[])), None);
        assert_eq!(
            tile_cache_path(Path::new("/c"), 12, 3496, 1603),
            Path::new("/c").join("terrarium").join("12").join("3496").join("1603.png")
        );
    }

    #[test]
    fn bilinear_inside_one_tile() {
        // A ramp: elevation = column index.
        let ramp: Vec<f64> = (0..TILE * TILE).map(|i| (i % TILE) as f64).collect();
        let (tx, ty) = tile_xy(TAKEOFF.0, TAKEOFF.1, DEM_ZOOM);
        let want = tx.fract() * TILE as f64 - 0.5;
        let mut asked = Vec::new();
        let got = sample(TAKEOFF.0, TAKEOFF.1, DEM_ZOOM, |x, y| {
            asked.push((x, y));
            Some(ramp.clone())
        })
        .unwrap();
        assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        assert!(asked.iter().all(|&(x, y)| x == tx.floor() as u32 && y == ty.floor() as u32));
        assert_eq!(asked.len(), 1, "one tile, loaded once");
    }

    #[test]
    fn missing_or_void_samples_give_none() {
        assert_eq!(sample(TAKEOFF.0, TAKEOFF.1, DEM_ZOOM, |_, _| None), None);
        assert_eq!(sample(TAKEOFF.0, TAKEOFF.1, DEM_ZOOM, |_, _| Some(vec![-32768.0; TILE * TILE])), None);
        assert_eq!(sample(TAKEOFF.0, TAKEOFF.1, DEM_ZOOM, |_, _| Some(vec![1.0; 3])), None);
        let flat = sample(TAKEOFF.0, TAKEOFF.1, DEM_ZOOM, |_, _| Some(vec![48.25; TILE * TILE])).unwrap();
        assert!((flat - 48.25).abs() < 1e-9);
    }

    #[test]
    fn decodes_a_png_tile() {
        let mut img = image::RgbImage::new(256, 256);
        for p in img.pixels_mut() {
            *p = image::Rgb([128, 50, 128]);
        }
        let mut png = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).unwrap();
        let t = decode_tile(&png).unwrap();
        assert_eq!(t.len(), TILE * TILE);
        assert!(t.iter().all(|&v| v == 50.5));
        assert!(decode_tile(b"nope").is_err());
        let mut small = Vec::new();
        image::RgbImage::new(4, 4).write_to(&mut std::io::Cursor::new(&mut small), image::ImageFormat::Png).unwrap();
        assert!(decode_tile(&small).is_err());
    }
}
