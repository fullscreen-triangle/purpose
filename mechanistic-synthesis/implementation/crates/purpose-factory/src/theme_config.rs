use std::path::Path;

use serde::Deserialize;

use crate::contract::{
    BaseModelSpec, HeuristicVerifier, PretrainedConfig, ScratchConfig, ThemeContract, TrainingSpec,
};
use crate::error::Error;
use crate::source::{ImapSource, LocalFileSource, SourceProvider, UrlSource};

/// On-disk `theme.toml` shape. Parses into a `ThemeContract` via
/// `ThemeConfig::into_contract`. Kept as a separate serde-friendly struct so
/// the contract types themselves stay free of file-format concerns.
#[derive(Debug, Deserialize)]
pub struct ThemeConfig {
    pub name: String,
    #[serde(default)]
    pub sources: SourcesConfig,
    #[serde(default)]
    pub model: ModelConfig,
    #[serde(default)]
    pub training: TrainingConfig,
}

#[derive(Debug, Default, Deserialize)]
pub struct SourcesConfig {
    #[serde(default)]
    pub local_files: Vec<LocalFileConfig>,
    #[serde(default)]
    pub imap: Vec<ImapConfig>,
    #[serde(default)]
    pub urls: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct LocalFileConfig {
    pub root: String,
    #[serde(default)]
    pub extensions: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct ImapConfig {
    pub host: String,
    #[serde(default = "default_imap_port")]
    pub port: u16,
    pub username: String,
    /// App-password or pre-obtained OAuth token; the factory does not
    /// perform an OAuth consent flow itself. Read from this field literally,
    /// or from an env var if `password_env` is set instead.
    #[serde(default)]
    pub password: Option<String>,
    #[serde(default)]
    pub password_env: Option<String>,
    #[serde(default = "default_mailbox")]
    pub mailbox: String,
    #[serde(default = "default_search")]
    pub search: String,
}

fn default_imap_port() -> u16 {
    993
}
fn default_mailbox() -> String {
    "INBOX".to_string()
}
fn default_search() -> String {
    "ALL".to_string()
}

#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_n_layer")]
    pub n_layer: usize,
    #[serde(default = "default_n_head")]
    pub n_head: usize,
    #[serde(default = "default_n_embd")]
    pub n_embd: usize,
    #[serde(default = "default_block_size")]
    pub block_size: usize,
    /// When set, trains by LoRA-adapting this pretrained HuggingFace
    /// checkpoint instead of training a model from scratch; the other
    /// `[model]` fields (vocab_size, n_layer, ...) are then ignored.
    #[serde(default)]
    pub pretrained: Option<PretrainedModelConfig>,
}

#[derive(Debug, Deserialize)]
pub struct PretrainedModelConfig {
    /// `"owner/name"`, e.g. `"TinyLlama/TinyLlama-1.1B-Chat-v1.0"`.
    pub repo: String,
    #[serde(default)]
    pub revision: Option<String>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        let d = ScratchConfig::default();
        Self {
            vocab_size: d.vocab_size,
            n_layer: d.n_layer,
            n_head: d.n_head,
            n_embd: d.n_embd,
            block_size: d.block_size,
            pretrained: None,
        }
    }
}

fn default_vocab_size() -> usize {
    ScratchConfig::default().vocab_size
}
fn default_n_layer() -> usize {
    ScratchConfig::default().n_layer
}
fn default_n_head() -> usize {
    ScratchConfig::default().n_head
}
fn default_n_embd() -> usize {
    ScratchConfig::default().n_embd
}
fn default_block_size() -> usize {
    ScratchConfig::default().block_size
}

#[derive(Debug, Deserialize)]
pub struct TrainingConfig {
    #[serde(default = "default_epochs")]
    pub epochs: usize,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_lr")]
    pub learning_rate: f64,
    #[serde(default = "default_lora_rank")]
    pub lora_rank: usize,
    #[serde(default = "default_lora_alpha")]
    pub lora_alpha: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        let d = TrainingSpec::default();
        Self {
            epochs: d.epochs,
            batch_size: d.batch_size,
            learning_rate: d.learning_rate,
            lora_rank: d.lora_rank,
            lora_alpha: d.lora_alpha,
        }
    }
}

fn default_epochs() -> usize {
    TrainingSpec::default().epochs
}
fn default_batch_size() -> usize {
    TrainingSpec::default().batch_size
}
fn default_lr() -> f64 {
    TrainingSpec::default().learning_rate
}
fn default_lora_rank() -> usize {
    TrainingSpec::default().lora_rank
}
fn default_lora_alpha() -> f64 {
    TrainingSpec::default().lora_alpha
}

impl ThemeConfig {
    pub fn from_toml_str(raw: &str) -> Result<Self, Error> {
        toml::from_str(raw).map_err(|e| Error::Config(format!("theme.toml parse: {e}")))
    }

    pub fn from_file(path: &Path) -> Result<Self, Error> {
        let raw = std::fs::read_to_string(path)?;
        Self::from_toml_str(&raw)
    }

    pub fn into_contract(self) -> Result<ThemeContract, Error> {
        let mut sources: Vec<Box<dyn SourceProvider>> = Vec::new();

        for lf in self.sources.local_files {
            let mut source = LocalFileSource::new(lf.root);
            if !lf.extensions.is_empty() {
                source.extensions = lf.extensions;
            }
            sources.push(Box::new(source));
        }

        for imap_cfg in self.sources.imap {
            let password = match (imap_cfg.password, imap_cfg.password_env) {
                (Some(p), _) => p,
                (None, Some(env)) => std::env::var(&env).map_err(|_| {
                    Error::Config(format!("imap source: env var '{env}' not set"))
                })?,
                (None, None) => {
                    return Err(Error::Config(
                        "imap source requires 'password' or 'password_env'".into(),
                    ))
                }
            };
            sources.push(Box::new(ImapSource {
                host: imap_cfg.host,
                port: imap_cfg.port,
                username: imap_cfg.username,
                password,
                mailbox: imap_cfg.mailbox,
                search: imap_cfg.search,
            }));
        }

        if !self.sources.urls.is_empty() {
            sources.push(Box::new(UrlSource::new(self.sources.urls)));
        }

        if sources.is_empty() {
            return Err(Error::Config(format!(
                "theme '{}' declares no sources",
                self.name
            )));
        }

        let base_model = match self.model.pretrained {
            Some(p) => BaseModelSpec::Pretrained(PretrainedConfig {
                repo: p.repo,
                revision: p.revision,
            }),
            None => BaseModelSpec::Scratch(ScratchConfig {
                vocab_size: self.model.vocab_size,
                n_layer: self.model.n_layer,
                n_head: self.model.n_head,
                n_embd: self.model.n_embd,
                block_size: self.model.block_size,
            }),
        };

        let training = TrainingSpec {
            epochs: self.training.epochs,
            batch_size: self.training.batch_size,
            learning_rate: self.training.learning_rate,
            lora_rank: self.training.lora_rank,
            lora_alpha: self.training.lora_alpha,
        };

        Ok(ThemeContract {
            name: self.name,
            sources,
            base_model,
            verifier: Box::new(HeuristicVerifier::default()),
            training,
        })
    }
}

/// Scaffolds a starter `theme.toml` for `purpose factory init <name>`.
pub fn scaffold(name: &str) -> String {
    format!(
        r#"name = "{name}"

[sources]
local_files = [
    # {{ root = "path/to/papers", extensions = ["tex", "pdf", "md"] }}
]
urls = [
    # "https://example.com/article"
]

# [[sources.imap]]
# host = "imap.gmail.com"
# username = "you@example.com"
# password_env = "THEME_IMAP_PASSWORD"
# mailbox = "INBOX"
# search = "ALL"

[model]
# From-scratch model shape (ignored if [model.pretrained] is set below).
vocab_size = 8192
n_layer = 4
n_head = 4
n_embd = 256
block_size = 256

# Uncomment to LoRA-adapt a real pretrained checkpoint instead of training
# from scratch (requires network access on first build; cached afterward).
# Only unsharded checkpoints (a single model.safetensors) are supported.
# [model.pretrained]
# repo = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# revision = "main"

[training]
epochs = 3
batch_size = 8
learning_rate = 0.0003
lora_rank = 8
lora_alpha = 16.0
"#
    )
}
