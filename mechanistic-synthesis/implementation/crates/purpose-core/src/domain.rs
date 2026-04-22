use async_trait::async_trait;
use std::sync::Arc;

use crate::error::Error;
use crate::operation::Operation;
use crate::vahera::VaHera;

/// A Domain Connector: the rigid, minimal interface a domain presents to
/// the Purpose runtime.
///
/// The factory produces a resolver by consuming (in a future version) a
/// richer Connector including a pure-intent sampler, theoretical corpus,
/// and verifier. The MVP Connector carries only what is needed to serve
/// queries from a pretrained or hand-coded resolver.
#[derive(Clone)]
pub struct Domain {
    pub name: String,
    pub operations: Vec<Operation>,
    pub resolver: Arc<dyn Resolver>,
}

/// A Resolver compiles natural-language utterances (or vaHera fragments
/// with holes) into fully-resolved vaHera fragments.
///
/// For the MVP a Resolver is a hand-coded function; in future stages the
/// factory produces trained LoRA adapters implementing this same trait.
#[async_trait]
pub trait Resolver: Send + Sync {
    /// Compile a natural-language utterance to a vaHera fragment.
    async fn compile(&self, utterance: &str) -> Result<VaHera, Error>;
}
