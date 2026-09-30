//! The on-disk layout `purpose serve` owns under its root:
//!
//! ```text
//! <root>/registry.json              built models
//! <root>/themes/<name>/sources/     uploaded training material
//! <root>/themes/<name>/model/       the exported model
//! <root>/jobs/<id>.json, <id>.log   build jobs and their logs
//! <root>/places.toml                optional placement-catalogue overrides
//! <root>/serve-token                the bearer token, when generated
//! ```

use std::path::{Path, PathBuf};

use crate::error::Error;

#[derive(Debug, Clone)]
pub struct Workspace {
    root: PathBuf,
}

impl Workspace {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn sources_dir(&self, theme: &str) -> PathBuf {
        self.root.join("themes").join(theme).join("sources")
    }

    pub fn model_dir(&self, theme: &str) -> PathBuf {
        self.root.join("themes").join(theme).join("model")
    }

    pub fn registry_path(&self) -> PathBuf {
        self.root.join("registry.json")
    }

    pub fn jobs_dir(&self) -> PathBuf {
        self.root.join("jobs")
    }

    pub fn places_path(&self) -> PathBuf {
        self.root.join("places.toml")
    }

    pub fn token_path(&self) -> PathBuf {
        self.root.join("serve-token")
    }
}

/// Theme names become directory names, so only a conservative character set
/// is accepted: no path separators, no `..`, nothing a filesystem treats specially.
pub fn validate_theme_name(name: &str) -> Result<(), Error> {
    let ok = !name.is_empty()
        && name.len() <= 64
        && !name.starts_with('.')
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.');
    if ok {
        Ok(())
    } else {
        Err(Error::Config(format!(
            "invalid theme name '{name}': use letters, digits, '-', '_' or '.', not starting with '.'"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn theme_names_cannot_escape_the_root() {
        for bad in ["", "..", ".hidden", "a/b", "a\\b", "x y", &"a".repeat(65)] {
            assert!(validate_theme_name(bad).is_err(), "{bad:?}");
        }
        for good in ["absicht", "nfdi4cat-model", "v0.2_test"] {
            assert!(validate_theme_name(good).is_ok(), "{good:?}");
        }
    }
}
