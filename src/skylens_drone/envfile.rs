//! The bridge's settings file: `.env.mission` at the repository root, the same way the core
//! reads `.env.vworld`. Copy `.env.mission.example` there and fill in what you need.
//!
//! Lookup order for every `SKYLENS_*` variable: the process environment first (a non-empty
//! value wins), then the file. `SKYLENS_MISSION_ENV` points at another file. A missing file
//! is not an error: the bridge then runs on the environment and its built-in defaults.
//! Values are never logged, only the file path.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Variable that overrides the file location.
pub const ENV_FILE_VAR: &str = "SKYLENS_MISSION_ENV";
/// File name looked up at the repository root.
pub const DEFAULT_FILE: &str = ".env.mission";

/// `KEY=VALUE` pairs read from the settings file.
#[derive(Debug, Default, Clone)]
pub struct EnvFile {
    pub path: Option<PathBuf>,
    vars: HashMap<String, String>,
}

impl EnvFile {
    /// Where the file is expected: `SKYLENS_MISSION_ENV` if set, else `<root>/.env.mission`.
    pub fn location(root: &Path, get: impl Fn(&str) -> Option<String>) -> PathBuf {
        match get(ENV_FILE_VAR).map(|v| v.trim().to_owned()).filter(|v| !v.is_empty()) {
            Some(p) => crate::paths::expand_user(&p),
            None => root.join(DEFAULT_FILE),
        }
    }

    /// Read the file at `path`. A missing file gives an empty set; an unreadable one is an error.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(text) => Ok(Self { path: Some(path.to_path_buf()), vars: parse(&text) }),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(anyhow::Error::new(e).context(format!("reading {}", path.display()))),
        }
    }

    pub fn get(&self, key: &str) -> Option<String> {
        self.vars.get(key).cloned()
    }

    pub fn len(&self) -> usize {
        self.vars.len()
    }

    /// The process environment first, then this file.
    pub fn lookup<'a>(&'a self, env: impl Fn(&str) -> Option<String> + 'a) -> impl Fn(&str) -> Option<String> + 'a {
        move |k: &str| env(k).filter(|v| !v.trim().is_empty()).or_else(|| self.get(k))
    }
}

/// `KEY=VALUE` lines. Blank lines and `#` comments are skipped, an `export ` prefix is
/// allowed, and one pair of matching quotes around the value is removed. Later lines win.
pub fn parse(text: &str) -> HashMap<String, String> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut vars = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        let Some((key, value)) = line.split_once('=') else { continue };
        let key = key.trim();
        if key.is_empty() || key.contains(char::is_whitespace) {
            continue;
        }
        vars.insert(key.to_owned(), unquote(value.trim()).to_owned());
    }
    vars
}

fn unquote(v: &str) -> &str {
    for q in ['"', '\''] {
        if v.len() >= 2 && v.starts_with(q) && v.ends_with(q) {
            return &v[1..v.len() - 1];
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env_of<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |k: &str| pairs.iter().find(|(key, _)| *key == k).map(|(_, v)| (*v).to_owned())
    }

    #[test]
    fn parses_pairs_comments_quotes_and_export() {
        let vars = parse(
            "\u{feff}# comment\n\nSKYLENS_MISSION_PORT=9000\n  export SKYLENS_MISSION_HOST = 0.0.0.0 \n\
             SKYLENS_DJI_APP_INFO=\"Android SkyLens\"\nSKYLENS_DJI_WK_KEY='a=b'\nnot a pair\n=empty\n\
             SKYLENS_MISSION_PORT=9001\n# SKYLENS_DJI_MC_TOKEN=commented\n",
        );
        assert_eq!(vars.get("SKYLENS_MISSION_PORT").map(String::as_str), Some("9001"));
        assert_eq!(vars.get("SKYLENS_MISSION_HOST").map(String::as_str), Some("0.0.0.0"));
        assert_eq!(vars.get("SKYLENS_DJI_APP_INFO").map(String::as_str), Some("Android SkyLens"));
        assert_eq!(vars.get("SKYLENS_DJI_WK_KEY").map(String::as_str), Some("a=b"));
        assert!(!vars.contains_key("SKYLENS_DJI_MC_TOKEN"));
        assert_eq!(vars.len(), 4);
    }

    #[test]
    fn environment_wins_over_the_file_unless_empty() {
        let file = EnvFile { path: None, vars: parse("A=file\nB=file\nC=file\n") };
        let pairs = [("A", "env"), ("B", "  ")];
        let get = file.lookup(env_of(&pairs));
        assert_eq!(get("A").as_deref(), Some("env"));
        assert_eq!(get("B").as_deref(), Some("file"));
        assert_eq!(get("C").as_deref(), Some("file"));
        assert_eq!(get("D"), None);
    }

    #[test]
    fn location_defaults_to_the_root_and_follows_the_override() {
        let root = Path::new("/repo");
        assert_eq!(EnvFile::location(root, env_of(&[])), root.join(".env.mission"));
        let pairs = [(ENV_FILE_VAR, "/etc/skylens/mission.env")];
        assert_eq!(EnvFile::location(root, env_of(&pairs)), PathBuf::from("/etc/skylens/mission.env"));
    }

    #[test]
    fn missing_file_is_empty_and_present_file_is_read() {
        let dir = std::env::temp_dir().join(format!("skylens-envfile-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let missing = EnvFile::load(&dir.join("absent.env")).unwrap();
        assert_eq!(missing.len(), 0);
        assert!(missing.path.is_none());
        let path = dir.join(".env.mission");
        std::fs::write(&path, "SKYLENS_MISSION_PORT=9100\n").unwrap();
        let file = EnvFile::load(&path).unwrap();
        assert_eq!(file.get("SKYLENS_MISSION_PORT").as_deref(), Some("9100"));
        assert_eq!(file.path.as_deref(), Some(path.as_path()));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_example_file_parses_to_only_the_documented_defaults() {
        let example = crate::paths::repo_root().join(".env.mission.example");
        let vars = parse(&std::fs::read_to_string(example).expect(".env.mission.example at the root"));
        // Every secret is commented out in the example; only plain settings are active.
        for secret in ["SKYLENS_DJI_MC_TOKEN", "SKYLENS_DJI_WK_KEY"] {
            assert!(!vars.contains_key(secret), "{secret} must stay commented out in the example");
        }
        assert!(vars.keys().all(|k| k.starts_with("SKYLENS_")));
    }
}
