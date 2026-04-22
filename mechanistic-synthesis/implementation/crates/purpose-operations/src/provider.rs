use async_trait::async_trait;
use purpose_core::{Error, Value};
use std::collections::BTreeMap;

/// A provider executes one or more operations.
///
/// Providers are the integration point for external services (UniProt,
/// PDB, HuggingFace Inference API), local models (Candle), databases, or
/// any other substrate that the runtime calls into. A single provider may
/// serve many operations; a single operation is served by exactly one
/// provider in the registry.
#[async_trait]
pub trait Provider: Send + Sync {
    /// Invoke `op` with the given named arguments. Returns the operation's
    /// output value.
    async fn invoke(
        &self,
        op: &str,
        args: &BTreeMap<String, Value>,
    ) -> Result<Value, Error>;
}
