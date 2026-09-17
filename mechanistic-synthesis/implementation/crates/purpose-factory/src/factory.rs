use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::contract::{BaseModelSpec, ThemeContract};
use crate::corpus;
use crate::error::Error;
use crate::source::fetch_all;
use crate::train;

/// Metadata about a built theme model, written to the local registry so
/// other tools (the CLI's `factory list`, the TypeScript client) can
/// discover what has been produced without re-running the factory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThemeModel {
    pub name: String,
    pub path: PathBuf,
    pub document_count: usize,
    pub example_count: usize,
    pub vocab_size: usize,
}

pub struct Factory;

impl Factory {
    /// Runs the full pipeline: fetch sources, build a verified corpus, train
    /// (from scratch, or LoRA-adapting a downloaded pretrained checkpoint,
    /// per the theme's `BaseModelSpec`), merge and export it.
    pub async fn build(contract: ThemeContract, out_dir: &Path) -> Result<ThemeModel, Error> {
        tracing::info!(theme = %contract.name, sources = contract.sources.len(), "fetching sources");
        let docs = fetch_all(&contract.sources).await?;
        if docs.is_empty() {
            return Err(Error::Source(format!(
                "theme '{}' produced no documents from its sources",
                contract.name
            )));
        }
        let document_count = docs.len();

        let (example_count, vocab_size) = match &contract.base_model {
            BaseModelSpec::Scratch(cfg) => {
                tracing::info!(document_count, "building corpus (from scratch)");
                let corpus = corpus::build(
                    &docs,
                    contract.verifier.as_ref(),
                    cfg.vocab_size,
                    cfg.block_size,
                );
                if corpus.examples.is_empty() {
                    return Err(Error::Corpus(format!(
                        "theme '{}' produced no admissible training examples; verifier rejected everything",
                        contract.name
                    )));
                }
                let example_count = corpus.examples.len();
                let vocab_size = corpus.tokenizer.vocab_size();

                tracing::info!(example_count, vocab_size, "training (from scratch)");
                let trained = train::run_scratch(cfg, &contract.training, corpus)?;

                tracing::info!(out_dir = %out_dir.display(), "exporting");
                train::export_scratch(&trained, out_dir)?;

                (example_count, vocab_size)
            }
            BaseModelSpec::Pretrained(pretrained) => {
                tracing::info!(document_count, repo = %pretrained.repo, "building corpus (pretrained)");
                // Pretrained models use a fixed context window as the corpus
                // window size too; a reasonable default when the checkpoint's
                // own max_position_embeddings isn't known before download.
                let admitted = corpus::admitted_texts(&docs, contract.verifier.as_ref(), 2048);
                if admitted.is_empty() {
                    return Err(Error::Corpus(format!(
                        "theme '{}' produced no admissible training examples; verifier rejected everything",
                        contract.name
                    )));
                }

                let trained =
                    train::run_pretrained(pretrained, &contract.training, 512, &admitted).await?;
                let example_count = admitted.len();
                let vocab_size = trained.vocab_size;

                tracing::info!(out_dir = %out_dir.display(), "exporting");
                train::export_pretrained(&trained, out_dir)?;

                (example_count, vocab_size)
            }
        };

        Ok(ThemeModel {
            name: contract.name,
            path: out_dir.to_path_buf(),
            document_count,
            example_count,
            vocab_size,
        })
    }
}

/// A local registry of built theme models, so other tools can discover them
/// without re-running the factory. Stored as one JSON file, `[]` if empty.
pub struct Registry {
    path: PathBuf,
}

impl Registry {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn load(&self) -> Result<Vec<ThemeModel>, Error> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }
        let raw = std::fs::read_to_string(&self.path)?;
        serde_json::from_str(&raw).map_err(|e| Error::Config(format!("registry parse: {e}")))
    }

    pub fn record(&self, model: &ThemeModel) -> Result<(), Error> {
        let mut models = self.load()?;
        models.retain(|m| m.name != model.name);
        models.push(model.clone());
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(&models)
            .map_err(|e| Error::Config(format!("registry serialize: {e}")))?;
        std::fs::write(&self.path, json)?;
        Ok(())
    }
}
