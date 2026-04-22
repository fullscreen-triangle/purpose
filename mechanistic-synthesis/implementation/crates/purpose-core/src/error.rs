use thiserror::Error;

/// Unified error type for the core and its direct consumers.
#[derive(Debug, Error)]
pub enum Error {
    #[error("compilation error: {0}")]
    Compile(String),

    #[error("type error: {0}")]
    Type(String),

    #[error("provider error: {0}")]
    Provider(String),

    #[error("parse error: {0}")]
    Parse(String),

    #[error("internal error: {0}")]
    Internal(String),
}
