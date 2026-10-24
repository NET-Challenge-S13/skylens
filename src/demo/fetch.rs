//! Downloads the recorded footage when it is not on disk yet: the release zip goes to
//! `data/skylens-demo-footage.zip.part`, is renamed once complete, checked to be a zip, extracted
//! into `data/` (giving `data/demo/...`) and deleted. Nothing in it may land outside `data/`.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use crate::paths;
use crate::procs::log;

const ZIP_NAME: &str = "skylens-demo-footage.zip";
/// Progress line every this many bytes when the size is unknown.
const STEP_UNKNOWN: u64 = 50 << 20;

/// The recorded-flight files that are missing, relative to the repo with `/`.
pub fn missing(repo: &Path) -> Vec<String> {
    paths::recorded_files(repo)
        .into_iter()
        .filter(|p| !p.is_file())
        .map(|p| p.strip_prefix(repo).unwrap_or(&p).display().to_string().replace('\\', "/"))
        .collect()
}

/// How to put the footage in place by hand (Korean, for the error).
pub fn manual_steps(repo: &Path, url: &str) -> String {
    let data = paths::data_dir(repo);
    format!(
        "    직접 받으려면:\n      1. 브라우저나 curl 로 {url} 을 받습니다.\n      2. 압축을 {} 안에 풉니다. data/demo/left.mp4 · center.mp4 · right.mp4 · track.json · camera.json 이 생겨야 합니다.\n         예: curl -L -o {ZIP_NAME} \"{url}\" && unzip {ZIP_NAME} -d \"{}\"\n    옛 합성 영상으로 돌리려면 --synthetic",
        data.display(),
        data.display()
    )
}

/// `12.3 / 456.7 MB (2%)`, or without the total when the server did not say.
pub fn progress_line(done: u64, total: Option<u64>) -> String {
    let mb = |b: u64| b as f64 / (1024.0 * 1024.0);
    match total {
        Some(t) if t > 0 => format!("{:.1} / {:.1} MB ({:.0}%)", mb(done), mb(t), done as f64 / t as f64 * 100.0),
        _ => format!("{:.1} MB", mb(done)),
    }
}

/// Whether a progress line is due: every 10% of a known size, else every 50 MB.
fn due(done: u64, last: u64, total: Option<u64>) -> bool {
    let step = match total {
        Some(t) if t > 0 => (t / 10).max(1),
        _ => STEP_UNKNOWN,
    };
    done / step > last / step
}

/// A zip member name as a path under the extraction folder, or `None` when it would escape it
/// (absolute, a drive prefix, `..`) or names nothing.
pub fn safe_relative(name: &str) -> Option<PathBuf> {
    let unified = name.replace('\\', "/");
    if unified.starts_with('/') {
        return None;
    }
    let mut out = PathBuf::new();
    for part in unified.split('/') {
        match part {
            "" | "." => {}
            ".." => return None,
            p => {
                let c = Path::new(p).components().next();
                if p.contains(':') || !matches!(c, Some(Component::Normal(_))) {
                    return None;
                }
                out.push(p);
            }
        }
    }
    (!out.as_os_str().is_empty()).then_some(out)
}

/// Whether the file starts like a zip (local file header `PK\x03\x04`).
pub fn looks_like_zip(path: &Path) -> bool {
    let mut magic = [0u8; 4];
    File::open(path).and_then(|mut f| f.read_exact(&mut magic)).is_ok() && magic == *b"PK\x03\x04"
}

/// Extracts every member into `dest`. Each file is written next to its target and renamed into
/// place, so an interrupted run never leaves a truncated clip behind. Returns the files written.
pub fn extract(zip: &Path, dest: &Path) -> Result<Vec<PathBuf>, String> {
    let file = File::open(zip).map_err(|e| format!("{}: {e}", zip.display()))?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e| format!("{}: zip 파일이 아닙니다 ({e})", zip.display()))?;
    // Check every name before writing anything.
    let mut names = Vec::with_capacity(archive.len());
    for i in 0..archive.len() {
        let entry = archive.by_index(i).map_err(|e| format!("zip 항목 {i}: {e}"))?;
        let name = entry.name().to_string();
        let rel = safe_relative(&name).ok_or_else(|| format!("zip 항목 {name:?} 가 압축 해제 폴더 밖을 가리킵니다"))?;
        names.push((rel, entry.is_dir()));
    }
    let mut written = Vec::new();
    for (i, (rel, is_dir)) in names.into_iter().enumerate() {
        let out = dest.join(&rel);
        if is_dir {
            std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
            continue;
        }
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        let mut entry = archive.by_index(i).map_err(|e| format!("zip 항목 {}: {e}", rel.display()))?;
        let mut part_name = out.file_name().unwrap_or_default().to_os_string();
        part_name.push(".part");
        let part = out.with_file_name(part_name);
        let mut f = File::create(&part).map_err(|e| format!("{}: {e}", part.display()))?;
        std::io::copy(&mut entry, &mut f).map_err(|e| format!("{}: {e}", rel.display()))?;
        f.flush().map_err(|e| format!("{}: {e}", part.display()))?;
        drop(f);
        std::fs::rename(&part, &out).map_err(|e| format!("{}: {e}", out.display()))?;
        written.push(out);
    }
    Ok(written)
}

/// Streams `url` into `dest`, logging progress. Returns the byte count.
fn download(url: &str, dest: &Path) -> Result<u64, String> {
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(15))
        .timeout_read(Duration::from_secs(60))
        .try_proxy_from_env(true)
        .build();
    let resp = agent.get(url).call().map_err(|e| match e {
        ureq::Error::Status(code, _) => format!("서버가 HTTP {code} 로 답했습니다"),
        ureq::Error::Transport(t) => format!("연결하지 못했습니다 ({t}). 프록시가 있으면 HTTPS_PROXY 를 설정하십시오"),
    })?;
    let total = resp.header("Content-Length").and_then(|v| v.trim().parse::<u64>().ok());
    log("fetch", &format!("받는 중: {url} ({})", total.map_or("크기 모름".to_string(), |t| progress_line(t, None))));
    let mut reader = resp.into_reader();
    let mut out = File::create(dest).map_err(|e| format!("{}: {e}", dest.display()))?;
    let mut buf = vec![0u8; 1 << 20];
    let (mut done, mut last) = (0u64, 0u64);
    loop {
        let n = reader.read(&mut buf).map_err(|e| format!("받는 도중 끊겼습니다 ({e})"))?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n]).map_err(|e| format!("{}: {e}", dest.display()))?;
        done += n as u64;
        if due(done, last, total) {
            log("fetch", &progress_line(done, total));
            last = done;
        }
    }
    out.flush().map_err(|e| format!("{}: {e}", dest.display()))?;
    if let Some(t) = total
        && done != t
    {
        return Err(format!("{t} 바이트 중 {done} 바이트만 받았습니다"));
    }
    log("fetch", &format!("다 받았습니다: {}", progress_line(done, None)));
    Ok(done)
}

/// Makes sure the recorded footage is on disk: downloads and extracts it when anything is
/// missing (unless `no_fetch`). The error is a Korean paragraph with the manual steps.
pub fn ensure(repo: &Path, url: &str, no_fetch: bool) -> Result<(), String> {
    let missing = missing(repo);
    if missing.is_empty() {
        return Ok(());
    }
    let manual = manual_steps(repo, url);
    if no_fetch {
        return Err(format!("촬영 영상이 없습니다: {}\n    --no-fetch 라 내려받지 않았습니다.\n{manual}", missing.join(", ")));
    }
    log("fetch", &format!("촬영 영상이 없어 내려받습니다 (없는 것: {})", missing.join(", ")));
    let data = paths::data_dir(repo);
    std::fs::create_dir_all(&data).map_err(|e| format!("{}: {e}\n{manual}", data.display()))?;
    let zip = data.join(ZIP_NAME);
    let part = data.join(format!("{ZIP_NAME}.part"));
    let fail = |what: String| format!("촬영 영상을 내려받지 못했습니다: {what}\n{manual}");
    download(url, &part).map_err(&fail)?;
    std::fs::rename(&part, &zip).map_err(|e| fail(format!("{}: {e}", zip.display())))?;
    if !looks_like_zip(&zip) {
        let _ = std::fs::remove_file(&zip);
        return Err(fail(format!("받은 파일이 zip 이 아닙니다 ({url}). 프록시나 로그인 페이지가 대신 왔을 수 있습니다")));
    }
    log("fetch", &format!("압축 해제: {}", data.display()));
    let written = extract(&zip, &data).map_err(&fail)?;
    let _ = std::fs::remove_file(&zip);
    log("fetch", &format!("파일 {}개를 풀었습니다", written.len()));
    let still = self::missing(repo);
    if !still.is_empty() {
        return Err(fail(format!("압축을 풀었지만 여전히 없습니다: {}", still.join(", "))));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn temp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("skylens-demo-fetch-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn zip_of(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut w = zip::ZipWriter::new(Cursor::new(Vec::new()));
        let opts = zip::write::FileOptions::default().compression_method(zip::CompressionMethod::Stored);
        for (name, body) in entries {
            if name.ends_with('/') {
                w.add_directory(*name, opts).unwrap();
            } else {
                w.start_file(*name, opts).unwrap();
                w.write_all(body).unwrap();
            }
        }
        w.finish().unwrap().into_inner()
    }

    #[test]
    fn names_that_escape_are_rejected() {
        assert_eq!(safe_relative("demo/left.mp4"), Some(PathBuf::from("demo").join("left.mp4")));
        assert_eq!(safe_relative("./demo//track.json"), Some(PathBuf::from("demo").join("track.json")));
        assert_eq!(safe_relative("demo\\camera.json"), Some(PathBuf::from("demo").join("camera.json")));
        assert_eq!(safe_relative("demo/"), Some(PathBuf::from("demo")));
        for bad in ["../evil", "demo/../../evil", "/etc/passwd", "\\\\server\\share", "C:/Windows/x", "C:evil", "", "./"] {
            assert_eq!(safe_relative(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn progress_in_megabytes() {
        assert_eq!(progress_line(5 << 20, Some(20 << 20)), "5.0 / 20.0 MB (25%)");
        assert_eq!(progress_line(3 << 19, None), "1.5 MB");
        assert_eq!(progress_line(1, Some(0)), "0.0 MB");
        assert!(due(11, 9, Some(100)) && !due(12, 11, Some(100)));
        assert!(due(STEP_UNKNOWN, 0, None) && !due(STEP_UNKNOWN - 1, 0, None));
    }

    #[test]
    fn extracts_into_the_data_folder() {
        let dir = temp("ok");
        let zip = dir.join("f.zip");
        let bytes = zip_of(&[
            ("demo/", b""),
            ("demo/left.mp4", b"L"),
            ("demo/center.mp4", b"C"),
            ("demo/right.mp4", b"R"),
            ("demo/track.json", b"{}"),
        ]);
        std::fs::write(&zip, bytes).unwrap();
        assert!(looks_like_zip(&zip));
        let data = dir.join("data");
        let written = extract(&zip, &data).unwrap();
        assert_eq!(written.len(), 4);
        assert_eq!(std::fs::read(data.join("demo").join("center.mp4")).unwrap(), b"C");
        assert!(!data.join("demo").join("center.mp4.part").exists());
        // The recorded-flight check sees the files (repo = dir, data = dir/data).
        assert!(missing(&dir).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn escaping_zip_writes_nothing() {
        let dir = temp("evil");
        let zip = dir.join("f.zip");
        std::fs::write(&zip, zip_of(&[("demo/left.mp4", b"L"), ("../evil.txt", b"x")])).unwrap();
        let data = dir.join("data");
        let err = extract(&zip, &data).unwrap_err();
        assert!(err.contains("evil.txt"), "{err}");
        assert!(!dir.join("evil.txt").exists());
        assert!(!data.join("demo").join("left.mp4").exists(), "checked before anything is written");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn not_a_zip() {
        let dir = temp("html");
        let f = dir.join("f.zip");
        std::fs::write(&f, b"<html>login</html>").unwrap();
        assert!(!looks_like_zip(&f));
        assert!(extract(&f, &dir.join("data")).is_err());
        assert!(!looks_like_zip(&dir.join("absent.zip")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn no_fetch_explains_the_manual_steps() {
        let dir = temp("nofetch");
        let err = ensure(&dir, "https://example.invalid/f.zip", true).unwrap_err();
        for needle in ["data/demo/left.mp4", "data/demo/track.json", "--no-fetch", "https://example.invalid/f.zip", "--synthetic"] {
            assert!(err.contains(needle), "{needle} not in:\n{err}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
