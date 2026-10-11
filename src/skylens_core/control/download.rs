//! Saving a KMZ the bridge built. The web build hands the browser a download like the legacy
//! code (a blob URL on a hidden `<a download>`); the native window writes it into the user's
//! Downloads folder (`SKYLENS_DOWNLOAD_DIR` overrides).

#[cfg(not(target_arch = "wasm32"))]
pub fn save(file_name: &str, bytes: &[u8]) -> Result<(), String> {
    use std::path::PathBuf;
    let dir = std::env::var_os("SKYLENS_DOWNLOAD_DIR")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")).map(|h| PathBuf::from(h).join("Downloads")))
        .filter(|d| d.is_dir())
        .unwrap_or_else(|| PathBuf::from("."));
    let name = safe_file_name(file_name);
    let path = dir.join(&name);
    std::fs::write(&path, bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    skylens_viewer::log(&format!("[mission] saved {}", path.display()));
    Ok(())
}

#[cfg(target_arch = "wasm32")]
pub fn save(file_name: &str, bytes: &[u8]) -> Result<(), String> {
    use wasm_bindgen::JsCast as _;
    let err = |e: wasm_bindgen::JsValue| format!("{e:?}");
    let window = web_sys::window().ok_or("no window")?;
    let document = window.document().ok_or("no document")?;
    let parts = js_sys::Array::of1(&js_sys::Uint8Array::from(bytes));
    let blob = web_sys::Blob::new_with_u8_array_sequence(&parts).map_err(err)?;
    let url = web_sys::Url::create_object_url_with_blob(&blob).map_err(err)?;
    let anchor: web_sys::HtmlAnchorElement = document.create_element("a").map_err(err)?.dyn_into().map_err(|_| "not an anchor".to_string())?;
    anchor.set_href(&url);
    anchor.set_download(&safe_file_name(file_name));
    let _ = anchor.style().set_property("display", "none");
    if let Some(body) = document.body() {
        let _ = body.append_child(&anchor);
    }
    anchor.click();
    anchor.remove();
    // Revoked a second later, as the legacy code did, so the download can start.
    let revoke = wasm_bindgen::closure::Closure::once_into_js(move || {
        let _ = web_sys::Url::revoke_object_url(&url);
    });
    let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(revoke.unchecked_ref(), 1000);
    Ok(())
}

/// A file name without path separators or characters Windows refuses.
pub fn safe_file_name(name: &str) -> String {
    let cleaned: String = name.chars().map(|c| if matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|') || c.is_control() { '_' } else { c }).collect();
    let trimmed = cleaned.trim_matches(|c: char| c == '.' || c.is_whitespace());
    if trimmed.is_empty() { "mission.kmz".to_string() } else { trimmed.to_string() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_names_stay_in_the_folder() {
        assert_eq!(safe_file_name("skylens-drone-1.kmz"), "skylens-drone-1.kmz");
        assert_eq!(safe_file_name("../../evil.kmz"), "_.._evil.kmz");
        assert_eq!(safe_file_name(".."), "mission.kmz");
    }
}
