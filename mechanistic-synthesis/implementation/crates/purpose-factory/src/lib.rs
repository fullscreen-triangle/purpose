//! The `purpose-factory` crate: the training loop reserved by
//! `integration.md` §1.1. Ingests sources (local files, email, web pages)
//! for a named theme, builds a verified training corpus, and trains either
//! a small from-scratch LoRA-adapted causal LM or a LoRA adapter over a
//! downloaded pretrained LLaMA-family checkpoint, merging and exporting the
//! result as a self-contained model directory — a full model artifact, not
//! an adapter-only checkpoint, per the acquisition pipeline in
//! `absicht/docs/research-domain-specific-models`.

pub mod batch;
pub mod contract;
pub mod corpus;
pub mod error;
pub mod factory;
pub mod hf;
pub mod lora;
pub mod model;
pub mod pretrained_model;
pub mod server;
pub mod source;
pub mod theme_config;
pub mod tokenizer;
pub mod train;

pub use contract::{
    BaseModelSpec, HeuristicVerifier, PretrainedConfig, ScratchConfig, ThemeContract, TrainingSpec,
    Verifier,
};
pub use error::Error;
pub use factory::{Factory, Registry, ThemeModel};
pub use source::{Document, ImapSource, LocalFileSource, SourceProvider, UrlSource};
pub use theme_config::ThemeConfig;
