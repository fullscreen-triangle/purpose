//! Core types for the Purpose Model Factory.
//!
//! This crate defines the durable interfaces on which every other crate
//! depends: the vaHera AST, the Domain Connector, the Resolver trait, the
//! Operation signature, the Type system, and a minimal Value type.
//!
//! Nothing in this crate performs I/O, ML inference, or long-running work.

pub mod domain;
pub mod error;
pub mod operation;
pub mod typecheck;
pub mod types;
pub mod vahera;

pub use domain::{Domain, Resolver};
pub use error::Error;
pub use operation::Operation;
pub use types::Type;
pub use vahera::{VaHera, Value};
