use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid record: {0}")]
    InvalidRecord(String),
    #[error("crypto: {0}")]
    Crypto(String),
    #[error("corrupt log line {line}: {reason}")]
    CorruptLog { line: usize, reason: String },
    #[error("data directory is locked by another process ({0}); if no chigutiro process is running, delete that file")]
    Locked(PathBuf),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

impl Error {
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Error::Io {
            path: path.into(),
            source,
        }
    }
}
