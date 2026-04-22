//! Operation registry, provider trait, and vaHera executor.
//!
//! A `Provider` knows how to invoke one or more named operations (e.g. an
//! HTTP endpoint, a local model, a database). An `OperationRegistry` maps
//! operation names to their declared signatures plus the provider that
//! serves them. An `Executor` walks a vaHera fragment and dispatches each
//! call to its registered provider.

pub mod executor;
pub mod provider;
pub mod providers;
pub mod registry;

pub use executor::Executor;
pub use provider::Provider;
pub use registry::OperationRegistry;
