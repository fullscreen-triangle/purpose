use crate::source::SourceProvider;

/// The Domain Contract for a theme, per `absicht`'s `def:contract`: a name, a
/// set of sources standing in for the extremal-regime sampler, a base model
/// to adapt, and a verifier deciding which generated examples are admissible.
pub struct ThemeContract {
    pub name: String,
    pub sources: Vec<Box<dyn SourceProvider>>,
    pub base_model: BaseModelSpec,
    pub verifier: Box<dyn Verifier>,
    pub training: TrainingSpec,
}

/// Which base model to train.
#[derive(Debug, Clone)]
pub enum BaseModelSpec {
    /// Train a small GPT-2-style causal LM from scratch on the theme corpus
    /// only — no external model download. Appropriate for a narrow theme
    /// where the corpus itself is the only source of domain knowledge, or
    /// where network access to HuggingFace isn't available.
    Scratch(ScratchConfig),
    /// Download a pretrained LLaMA-family checkpoint from the HuggingFace
    /// Hub and LoRA-adapt it (`pretrained_model::LoraLlama`). Produces a
    /// genuinely capable model; requires network access on first build
    /// (cached afterward) and a repo with an unsharded `model.safetensors`.
    Pretrained(PretrainedConfig),
}

#[derive(Debug, Clone)]
pub struct PretrainedConfig {
    /// `"owner/name"`, e.g. `"TinyLlama/TinyLlama-1.1B-Chat-v1.0"`.
    pub repo: String,
    pub revision: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ScratchConfig {
    pub vocab_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
    pub block_size: usize,
}

impl Default for ScratchConfig {
    fn default() -> Self {
        // Small enough to train on CPU in minutes for a single-theme corpus.
        Self {
            vocab_size: 8192,
            n_layer: 4,
            n_head: 4,
            n_embd: 256,
            block_size: 256,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrainingSpec {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub lora_rank: usize,
    pub lora_alpha: f64,
}

impl Default for TrainingSpec {
    fn default() -> Self {
        Self {
            epochs: 3,
            batch_size: 8,
            learning_rate: 3e-4,
            lora_rank: 8,
            lora_alpha: 16.0,
        }
    }
}

/// Decides whether a (prompt, completion) pair is admissible for training,
/// per `absicht`'s `Verd: Cands x Cands' -> {0,1}`.
pub trait Verifier: Send + Sync {
    fn verify(&self, prompt: &str, completion: &str) -> bool;
}

/// Default verifier: rejects degenerate examples (too short, too repetitive,
/// non-UTF8-clean) without requiring the caller to write a domain verifier
/// by hand. Sufficient for v1; callers with a real domain oracle should
/// supply their own `Verifier` instead.
pub struct HeuristicVerifier {
    pub min_completion_chars: usize,
}

impl Default for HeuristicVerifier {
    fn default() -> Self {
        Self {
            min_completion_chars: 16,
        }
    }
}

impl Verifier for HeuristicVerifier {
    fn verify(&self, _prompt: &str, completion: &str) -> bool {
        let trimmed = completion.trim();
        if trimmed.chars().count() < self.min_completion_chars {
            return false;
        }
        let unique_words: std::collections::HashSet<&str> = trimmed.split_whitespace().collect();
        let word_count = trimmed.split_whitespace().count();
        // Reject near-total repetition (e.g. "the the the the...").
        word_count == 0 || unique_words.len() * 3 >= word_count
    }
}
