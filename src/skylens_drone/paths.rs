//! Path helpers: the repository root the defaults resolve against, and the
//! `~` expansion Python's `Path.expanduser()` gave every configured path.

use std::path::{Path, PathBuf};

/// Repository root (two levels above this crate). The Python bridge used
/// `Path(__file__).parents[4]`, which is the same directory.
pub fn repo_root() -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().and_then(Path::parent).unwrap_or(manifest).to_path_buf()
}

/// Expand a leading `~` (alone or followed by a separator) to the home
/// directory. Anything else is returned unchanged.
pub fn expand_user(value: &str) -> PathBuf {
    expand_user_with(value, home_dir())
}

fn home_dir() -> Option<PathBuf> {
    let var = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
    std::env::var_os(var).filter(|v| !v.is_empty()).map(PathBuf::from)
}

fn expand_user_with(value: &str, home: Option<PathBuf>) -> PathBuf {
    let rest = match value.strip_prefix('~') {
        Some(rest) if rest.is_empty() || rest.starts_with('/') || rest.starts_with('\\') => rest,
        _ => return PathBuf::from(value),
    };
    match home {
        Some(home) => home.join(rest.trim_start_matches(['/', '\\'])),
        None => PathBuf::from(value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_only_a_leading_tilde() {
        let home = Some(PathBuf::from("/home/pilot"));
        assert_eq!(expand_user_with("~", home.clone()), PathBuf::from("/home/pilot"));
        assert_eq!(
            expand_user_with("~/missions/a.kmz", home.clone()),
            PathBuf::from("/home/pilot").join("missions/a.kmz")
        );
        assert_eq!(expand_user_with("~other/x", home.clone()), PathBuf::from("~other/x"));
        assert_eq!(expand_user_with("rel/~/x", home), PathBuf::from("rel/~/x"));
        assert_eq!(expand_user_with("~/x", None), PathBuf::from("~/x"));
    }

    #[test]
    fn root_holds_the_workspace_manifest() {
        assert!(repo_root().join("Cargo.toml").is_file());
        assert!(repo_root().join("src").join("skylens_drone").is_dir());
    }
}
