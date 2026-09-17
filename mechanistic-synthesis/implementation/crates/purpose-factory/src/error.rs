use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("source error: {0}")]
    Source(String),

    #[error("corpus error: {0}")]
    Corpus(String),

    #[error("training error: {0}")]
    Train(String),

    #[error("export error: {0}")]
    Export(String),

    #[error("config error: {0}")]
    Config(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<Error> for purpose_core::Error {
    fn from(e: Error) -> Self {
        purpose_core::Error::Internal(e.to_string())
    }
}
