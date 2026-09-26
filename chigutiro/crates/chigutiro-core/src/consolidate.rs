//! The slow timescale: turning the voice corpus into model weights.
//!
//! Receivers learn on every ingest; weights learn in rounds. Each round
//! retrains a LoRA adapter *from the base checkpoint* over the whole current
//! voice corpus, via `purpose factory build` — never by stacking a new
//! adapter on the last one. That costs compute, and buys two properties:
//! nothing is forgotten by drift between rounds, and an erasure reaches the
//! weights at the next round, after which every older model is deleted.
//!
//! The corpus is written in plaintext for the factory to read and deleted as
//! soon as the build ends, succeed or fail. The exported model is plaintext
//! by necessity; keep the data directory on an encrypted volume.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::Error;
use crate::store::State;
use crate::voice::VoiceDoc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationConfig {
    /// A `purpose` binary built with the `factory` subcommand. Without it,
    /// consolidation is reported as unavailable and nothing else changes.
    pub purpose_bin: Option<PathBuf>,
    /// Hugging Face repo with a single unsharded `model.safetensors`.
    pub base_model: String,
    pub block_size: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub lora_rank: usize,
    pub lora_alpha: f64,
    /// New voice documents that make a round due.
    pub min_new_docs: usize,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        ConsolidationConfig {
            purpose_bin: None,
            base_model: "Qwen/Qwen2.5-0.5B-Instruct".into(),
            block_size: 256,
            epochs: 2,
            batch_size: 4,
            learning_rate: 2e-4,
            lora_rank: 8,
            lora_alpha: 16.0,
            min_new_docs: 50,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Running,
    Succeeded,
    Failed,
    /// A newer round succeeded; this model's files were deleted.
    Superseded,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Consolidation {
    pub version: u64,
    pub started: DateTime<Utc>,
    pub finished: Option<DateTime<Utc>>,
    pub status: Status,
    pub reason: String,
    pub voice_docs: usize,
    /// Every voice document up to this sequence number is in this model.
    pub highest_seq: u64,
    pub voice_erasure_epoch: u64,
    pub base_model: String,
    pub model_dir: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

pub fn latest_succeeded(state: &State) -> Option<&Consolidation> {
    state.consolidations.iter().rev().find(|c| c.status == Status::Succeeded)
}

/// Whether the current model may still encode something since erased.
pub fn tainted(state: &State) -> bool {
    latest_succeeded(state).is_some_and(|c| c.voice_erasure_epoch < state.voice_erasure_epoch)
}

/// Why a round is due now, if it is.
pub fn due(state: &State, docs: &[VoiceDoc], cfg: &ConsolidationConfig) -> Option<String> {
    if docs.is_empty() || state.consolidations.iter().any(|c| c.status == Status::Running) {
        return None;
    }
    match latest_succeeded(state) {
        None => (docs.len() >= cfg.min_new_docs).then(|| format!("first round: {} voice documents", docs.len())),
        Some(_) if tainted(state) => Some("an erasure removed voice material the current model was trained on".into()),
        Some(last) => {
            let new = docs.iter().filter(|d| d.seq > last.highest_seq).count();
            (new >= cfg.min_new_docs).then(|| format!("{new} new voice documents since v{}", last.version))
        }
    }
}

pub struct Plan {
    pub version: u64,
    pub corpus_dir: PathBuf,
    pub theme_path: PathBuf,
    pub model_dir: PathBuf,
    pub registry_path: PathBuf,
}

/// Writes the corpus and `theme.toml` for the next round and returns the
/// `Running` record to append to state.
pub fn prepare(
    root: &Path,
    state: &State,
    docs: &[VoiceDoc],
    cfg: &ConsolidationConfig,
    reason: String,
    now: DateTime<Utc>,
) -> Result<(Plan, Consolidation), Error> {
    let version = state.consolidations.iter().map(|c| c.version).max().unwrap_or(0) + 1;
    let dir = absolute(&root.join("consolidations").join(format!("v{version}")))?;
    let corpus_dir = dir.join("corpus");
    fs::create_dir_all(&corpus_dir).map_err(|e| Error::io(&corpus_dir, e))?;
    for d in docs {
        let path = corpus_dir.join(format!("{:010}.txt", d.seq));
        fs::write(&path, &d.text).map_err(|e| Error::io(&path, e))?;
    }
    let model_dir = dir.join("model");
    let theme_path = dir.join("theme.toml");
    fs::write(&theme_path, theme_toml(version, &corpus_dir, cfg)?).map_err(|e| Error::io(&theme_path, e))?;
    let plan = Plan { version, corpus_dir, theme_path, model_dir: model_dir.clone(), registry_path: dir.join("registry.json") };
    let record = Consolidation {
        version,
        started: now,
        finished: None,
        status: Status::Running,
        reason,
        voice_docs: docs.len(),
        highest_seq: docs.iter().map(|d| d.seq).max().unwrap_or(0),
        voice_erasure_epoch: state.voice_erasure_epoch,
        base_model: cfg.base_model.clone(),
        model_dir,
        error: None,
    };
    Ok((plan, record))
}

fn theme_toml(version: u64, corpus_dir: &Path, cfg: &ConsolidationConfig) -> Result<String, Error> {
    // TOML literal strings take Windows paths verbatim, but cannot hold `'`.
    let root = corpus_dir.display().to_string();
    if root.contains('\'') {
        return Err(Error::InvalidRecord(format!("data directory path contains a quote: {root}")));
    }
    Ok(format!(
        r#"# Written by chigutiro for consolidation round v{version}. Regenerated each round.
name = "chigutiro-voice-v{version}"

[sources]
local_files = [ {{ root = '{root}', extensions = ["txt"] }} ]

[model.pretrained]
repo = "{repo}"
block_size = {block}

[training]
epochs = {epochs}
batch_size = {batch}
learning_rate = {lr}
lora_rank = {rank}
lora_alpha = {alpha:.1}
"#,
        repo = cfg.base_model,
        block = cfg.block_size,
        epochs = cfg.epochs,
        batch = cfg.batch_size,
        lr = cfg.learning_rate,
        rank = cfg.lora_rank,
        alpha = cfg.lora_alpha,
    ))
}

/// Runs the factory build. Blocking: call it off the request path. The
/// plaintext corpus is removed whatever the outcome.
pub fn run(plan: &Plan, purpose_bin: &Path) -> Result<(), String> {
    let output = Command::new(purpose_bin)
        .args(["factory", "build"])
        .arg(&plan.theme_path)
        .arg("--out")
        .arg(&plan.model_dir)
        .arg("--registry")
        .arg(&plan.registry_path)
        .output();
    let _ = fs::remove_dir_all(&plan.corpus_dir);
    let output = output.map_err(|e| format!("could not start {}: {e}", purpose_bin.display()))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let tail: String = stderr.chars().rev().take(1500).collect::<Vec<_>>().into_iter().rev().collect();
        return Err(format!("purpose factory build exited with {}: {tail}", output.status));
    }
    // A starting point for serving the result through Ollama. Whether Ollama
    // can import this architecture from safetensors depends on its version —
    // untested here, so the file says so.
    let modelfile = format!(
        "# Import with: ollama create chigutiro-voice-v{v} -f Modelfile\n\
         # Untested: requires an Ollama version that imports this architecture from safetensors.\n\
         FROM {dir}\n",
        v = plan.version,
        dir = plan.model_dir.display()
    );
    let _ = fs::write(plan.model_dir.join("Modelfile"), modelfile);
    Ok(())
}

/// Records the outcome. On success, every earlier model is deleted and marked
/// superseded — the step that makes erasure reach the weights. Returns the
/// directories removed.
pub fn finish(state: &mut State, version: u64, result: Result<(), String>, now: DateTime<Utc>) -> Vec<PathBuf> {
    let mut removed = Vec::new();
    let succeeded = result.is_ok();
    if let Some(c) = state.consolidations.iter_mut().find(|c| c.version == version) {
        c.finished = Some(now);
        match result {
            Ok(()) => c.status = Status::Succeeded,
            Err(e) => {
                c.status = Status::Failed;
                c.error = Some(e);
            }
        }
    }
    if succeeded {
        for c in state.consolidations.iter_mut().filter(|c| c.version < version && c.status == Status::Succeeded) {
            if let Some(dir) = c.model_dir.parent() {
                if fs::remove_dir_all(dir).is_ok() {
                    removed.push(dir.to_path_buf());
                }
            }
            c.status = Status::Superseded;
        }
    }
    removed
}

fn absolute(path: &Path) -> Result<PathBuf, Error> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    let cwd = std::env::current_dir().map_err(|e| Error::io(path, e))?;
    Ok(cwd.join(path))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(seq: u64) -> VoiceDoc {
        VoiceDoc { id: format!("d{seq}"), seq, text: "a sufficiently long sample of the owner's own writing".into() }
    }

    #[test]
    fn due_tracks_new_docs_and_erasure() {
        let cfg = ConsolidationConfig { min_new_docs: 2, ..Default::default() };
        let mut state = State::default();
        assert!(due(&state, &[doc(1)], &cfg).is_none());
        assert!(due(&state, &[doc(1), doc(2)], &cfg).is_some());

        let dir = tempfile::tempdir().unwrap();
        let (plan, rec) = prepare(dir.path(), &state, &[doc(1), doc(2)], &cfg, "test".into(), Utc::now()).unwrap();
        let theme = fs::read_to_string(&plan.theme_path).unwrap();
        assert!(theme.contains("extensions = [\"txt\"]") && theme.contains("lora_alpha = 16.0"), "{theme}");
        assert_eq!(fs::read_dir(&plan.corpus_dir).unwrap().count(), 2);
        state.consolidations.push(rec);
        assert!(due(&state, &[doc(1), doc(2)], &cfg).is_none(), "not while running");

        fs::create_dir_all(&plan.model_dir).unwrap();
        finish(&mut state, 1, Ok(()), Utc::now());
        assert!(due(&state, &[doc(1), doc(2), doc(3)], &cfg).is_none());
        assert!(due(&state, &[doc(1), doc(2), doc(3), doc(4)], &cfg).is_some());

        state.voice_erasure_epoch += 1;
        assert!(tainted(&state));
        assert!(due(&state, &[doc(1)], &cfg).unwrap().contains("erasure"));

        // A second success deletes the first model.
        let (plan2, rec2) = prepare(dir.path(), &state, &[doc(1)], &cfg, "erasure".into(), Utc::now()).unwrap();
        state.consolidations.push(rec2);
        fs::create_dir_all(&plan2.model_dir).unwrap();
        let removed = finish(&mut state, 2, Ok(()), Utc::now());
        assert_eq!(removed.len(), 1);
        assert!(!plan.model_dir.exists());
        assert_eq!(state.consolidations[0].status, Status::Superseded);
        assert!(!tainted(&state));
    }
}
