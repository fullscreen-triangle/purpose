//! Downloads a pretrained checkpoint's `config.json`, `tokenizer.json`, and
//! `model.safetensors` from the HuggingFace Hub, caching under the hub's own
//! cache directory (`HF_HOME`/`~/.cache/huggingface` by default) so a repeat
//! `purpose factory build` against the same repo does no network work.
//!
//! v1 targets single-shard checkpoints (`model.safetensors`, no sharded
//! index) — the small model families (TinyLlama-class and below) this
//! factory is meant to adapt fit in one shard. A caller with a larger,
//! sharded repo gets a clear error rather than a silent partial load.

use std::path::PathBuf;

use hf_hub::HFClient;

use crate::error::Error;

pub struct PretrainedFiles {
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    pub weights: PathBuf,
}

/// `repo` is `"owner/name"`, e.g. `"TinyLlama/TinyLlama-1.1B-Chat-v1.0"`.
pub async fn fetch(repo: &str, revision: Option<&str>) -> Result<PretrainedFiles, Error> {
    let (owner, name) = repo.split_once('/').ok_or_else(|| {
        Error::Config(format!(
            "pretrained repo '{repo}' must be in 'owner/name' form"
        ))
    })?;

    let client = HFClient::new().map_err(|e| Error::Source(format!("hf-hub client: {e}")))?;
    let handle = client.model(owner, name);

    let config = handle
        .download_file()
        .filename("config.json")
        .maybe_revision(revision.map(str::to_string))
        .send()
        .await
        .map_err(|e| Error::Source(format!("download {repo}/config.json: {e}")))?;
    let tokenizer = handle
        .download_file()
        .filename("tokenizer.json")
        .maybe_revision(revision.map(str::to_string))
        .send()
        .await
        .map_err(|e| Error::Source(format!("download {repo}/tokenizer.json: {e}")))?;
    let weights = handle
        .download_file()
        .filename("model.safetensors")
        .maybe_revision(revision.map(str::to_string))
        .send()
        .await
        .map_err(|e| {
            Error::Source(format!(
                "download {repo}/model.safetensors: {e} \
                 (sharded checkpoints — model-00001-of-*.safetensors — are not supported in v1)"
            ))
        })?;

    Ok(PretrainedFiles {
        config,
        tokenizer,
        weights,
    })
}
