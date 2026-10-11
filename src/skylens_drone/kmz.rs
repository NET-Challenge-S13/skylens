//! Minimal zip helpers shared by the mission builder and the cloud packager.
//!
//! A KMZ is a plain deflated zip. Entry order matters to the callers (Python
//! picked "the first name that ends with ..."), so names are returned in
//! central-directory order rather than through a hash map.

use std::io::{Cursor, Read, Write};

use zip::write::FileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

/// Entry names in archive order. `Err` means the bytes are not a readable zip.
pub fn names(bytes: &[u8]) -> Result<Vec<String>, zip::result::ZipError> {
    let mut archive = ZipArchive::new(Cursor::new(bytes))?;
    let mut out = Vec::with_capacity(archive.len());
    for index in 0..archive.len() {
        out.push(archive.by_index(index)?.name().to_owned());
    }
    Ok(out)
}

/// Read one entry by exact name.
pub fn read(bytes: &[u8], name: &str) -> Result<Vec<u8>, zip::result::ZipError> {
    let mut archive = ZipArchive::new(Cursor::new(bytes))?;
    let mut file = archive.by_name(name)?;
    let mut out = Vec::new();
    file.read_to_end(&mut out)?;
    Ok(out)
}

/// Write a deflated zip holding `entries` in the given order.
pub fn write(entries: &[(&str, &[u8])]) -> Result<Vec<u8>, zip::result::ZipError> {
    let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
    let options = FileOptions::default().compression_method(CompressionMethod::Deflated);
    for (name, data) in entries {
        writer.start_file(*name, options)?;
        writer.write_all(data)?;
    }
    Ok(writer.finish()?.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_entries_in_order() {
        let bytes = write(&[("b.txt", &b"two"[..]), ("a.txt", &b"one"[..])]).unwrap();
        assert_eq!(names(&bytes).unwrap(), vec!["b.txt", "a.txt"]);
        assert_eq!(read(&bytes, "a.txt").unwrap(), b"one");
    }

    #[test]
    fn rejects_non_zip_bytes() {
        assert!(names(b"not a zip").is_err());
    }
}
