//! Where to train a model, and how: ranks every place Kundai can train on
//! (the laptop, Apphub's GPUs, free notebooks, rented and academic GPUs) for
//! one training run, by feasibility first and then cost and time.
//!
//! The estimates are deliberately simple and every one is shown with its
//! reason, so a wrong number can be traced to the assumption behind it:
//! - memory: model weights + LoRA/optimizer state + activations + logits
//! - time: `6 · N · T / (peak FLOP/s · efficiency)`: forward, backward and
//!   gradient-checkpoint recompute per token, for N parameters and T tokens
//! - cost: hours × hourly price
//!
//! The laptop row is calibrated against a measured run (Qwen2.5-0.5B, LoRA,
//! 256-token steps, about 18 s per step on an i7-8650U), not against a
//! datasheet. Prices and GPU figures change; `Catalog::load` lets a
//! `places.toml` override the compiled-in defaults.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::Error;

/// How much exposure the training data can tolerate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Privacy {
    /// Published or public material: any place is fine.
    Public,
    /// Work material not meant to leave the group or institution.
    Internal,
    /// Personal data (own email, health, finances, other people's messages).
    Private,
}

/// Who controls the machine the data lands on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Custody {
    /// Kundai's own hardware.
    Own,
    /// A university or national research cluster, under its terms of use.
    Institutional,
    /// A commercial provider's own data centre, in the EU, contractually bound.
    SecureCloud,
    /// Private hosts or free notebook services: someone else's machine.
    Shared,
}

impl Custody {
    fn allows(self, privacy: Privacy) -> Result<(), &'static str> {
        match (privacy, self) {
            (Privacy::Public, _) => Ok(()),
            (_, Custody::Own) => Ok(()),
            (Privacy::Internal, Custody::Institutional | Custody::SecureCloud) => Ok(()),
            (Privacy::Private, Custody::SecureCloud) => Ok(()),
            (Privacy::Private, Custody::Institutional) => {
                Err("personal data: university clusters' terms cover research data, not private data")
            }
            (_, Custody::Shared) => Err("someone else's machine: not for non-public data"),
        }
    }
}

/// What the model is meant to get better at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Goal {
    /// The field's language, style and way of reasoning.
    Vocabulary,
    /// A task with worked examples (Q&A, extraction, code).
    Tasks,
    /// Recalling specific facts, identifiers and numbers.
    Facts,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Method {
    Auto,
    Lora,
    Qlora,
    Full,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Hardware {
    Cpu,
    Gpu,
}

/// One place to train. `peak_tflops` is dense bf16/fp16 tensor throughput
/// (for the CPU row: the calibrated effective rate), `efficiency` the
/// fraction of it a small-batch LoRA run actually achieves.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Place {
    pub id: String,
    pub name: String,
    pub hardware: Hardware,
    pub memory_gb: f64,
    pub peak_tflops: f64,
    pub efficiency: f64,
    pub price_per_hour_eur: f64,
    /// Longest single run the place allows, in hours; `None` if unlimited.
    pub session_limit_h: Option<f64>,
    pub custody: Custody,
    /// Whether `purpose serve` can launch and monitor a run here itself.
    pub automatic: bool,
    pub how_to: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Catalog {
    pub places: Vec<Place>,
}

fn place(
    id: &str,
    name: &str,
    hardware: Hardware,
    memory_gb: f64,
    peak_tflops: f64,
    efficiency: f64,
    price: f64,
    session_limit_h: Option<f64>,
    custody: Custody,
    automatic: bool,
    how_to: &str,
) -> Place {
    Place {
        id: id.into(),
        name: name.into(),
        hardware,
        memory_gb,
        peak_tflops,
        efficiency,
        price_per_hour_eur: price,
        session_limit_h,
        custody,
        automatic,
        how_to: how_to.into(),
    }
}

/// 6 · 494M params · 256 tokens / 18 s ≈ 0.042 TFLOP/s effective on the
/// laptop (Qwen2.5-0.5B LoRA run, 2026-09-26), used with efficiency 1.
const LAPTOP_EFFECTIVE_TFLOPS: f64 = 6.0 * 494e6 * 256.0 / 18.0 / 1e12;

impl Default for Catalog {
    /// Prices and limits as known in mid-2026; override with `places.toml`.
    fn default() -> Self {
        use Custody::*;
        use Hardware::*;
        let gpu_eff = 0.25;
        Self {
            places: vec![
                place("laptop", "This laptop (CPU)", Cpu, 10.0, LAPTOP_EFFECTIVE_TFLOPS, 1.0, 0.0, Some(24.0), Own, true,
                    "Started and monitored by purpose serve. Only LoRA on Qwen2/LLaMA-family bases; keep the laptop awake."),
                place("apphub-4090", "Apphub RTX 4090 (Uni Greifswald)", Gpu, 24.0, 165.0, gpu_eff, 0.0, Some(48.0), Institutional, false,
                    "JupyterLab session, software Deep Learning, Advanced: RTX 4090. Upload a training bundle, run run.sh; needs the university VPN off campus."),
                place("apphub-3090", "Apphub RTX 3090 (Uni Greifswald)", Gpu, 24.0, 71.0, gpu_eff, 0.0, Some(48.0), Institutional, false,
                    "As Apphub 4090, choosing RTX 3090."),
                place("apphub-2070", "Apphub RTX 2070 (Uni Greifswald)", Gpu, 8.0, 30.0, gpu_eff, 0.0, Some(48.0), Institutional, false,
                    "As Apphub 4090, choosing RTX 2070. No bf16: trains in fp16."),
                place("kisski-a100", "KISSKI A100 80GB (GWDG, free for German research)", Gpu, 80.0, 312.0, gpu_eff, 0.0, Some(48.0), Institutional, false,
                    "Apply once at kisski.gwdg.de; then a batch job on the cluster."),
                place("kaggle-t4", "Kaggle notebook T4 (free, ~30 GPU-h/week)", Gpu, 15.0, 65.0, 0.15, 0.0, Some(12.0), Shared, false,
                    "New notebook, accelerator GPU T4, upload the bundle as a dataset, run run.sh in a cell."),
                place("colab-t4", "Google Colab T4 (free tier)", Gpu, 15.0, 65.0, 0.15, 0.0, Some(4.0), Shared, false,
                    "Runtime type T4; sessions are cut without warning, so short runs only."),
                place("vast-4090", "Vast.ai RTX 4090 (private hosts)", Gpu, 24.0, 165.0, gpu_eff, 0.40, None, Shared, false,
                    "Rent a 4090 with a PyTorch image, upload the bundle, run run.sh, destroy the instance when done."),
                place("runpod-4090", "RunPod RTX 4090, Secure Cloud EU", Gpu, 24.0, 165.0, gpu_eff, 0.65, None, SecureCloud, false,
                    "Secure Cloud, EU region, PyTorch template; upload the bundle, run run.sh, stop the pod when done."),
                place("runpod-a100", "RunPod A100 80GB, Secure Cloud EU", Gpu, 80.0, 312.0, gpu_eff, 1.70, None, SecureCloud, false,
                    "As RunPod 4090, choosing A100 80GB."),
            ],
        }
    }
}

impl Catalog {
    /// The defaults, with any place in `path` (a `places.toml` holding
    /// `[[places]]` tables) replacing the default of the same `id` or added.
    pub fn load(path: &Path) -> Result<Self, Error> {
        let mut catalog = Self::default();
        if !path.exists() {
            return Ok(catalog);
        }
        let raw = std::fs::read_to_string(path)?;
        let overrides: Catalog = toml::from_str(&raw)
            .map_err(|e| Error::Config(format!("{}: {e}", path.display())))?;
        for p in overrides.places {
            match catalog.places.iter_mut().find(|q| q.id == p.id) {
                Some(q) => *q = p,
                None => catalog.places.push(p),
            }
        }
        Ok(catalog)
    }
}

/// One training run to place.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanRequest {
    /// Model size in billions of parameters, e.g. 0.5, 3, 7.
    pub params_b: f64,
    /// Training tokens per epoch.
    pub tokens: f64,
    #[serde(default = "default_epochs")]
    pub epochs: f64,
    #[serde(default = "default_seq")]
    pub seq_len: f64,
    #[serde(default = "default_vocab")]
    pub vocab: f64,
    #[serde(default = "default_method")]
    pub method: Method,
    pub privacy: Privacy,
    pub goal: Goal,
}

fn default_epochs() -> f64 {
    3.0
}
fn default_seq() -> f64 {
    1024.0
}
fn default_vocab() -> f64 {
    151_936.0 // Qwen2.5
}
fn default_method() -> Method {
    Method::Auto
}

#[derive(Debug, Clone, Serialize)]
pub struct Advice {
    /// The approach to take overall: `lora`, `qlora`, `full`, or `retrieval`.
    pub approach: String,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PlaceOption {
    pub place: String,
    pub name: String,
    pub feasible: bool,
    pub why_not: Vec<String>,
    pub method: Method,
    pub memory_gb: f64,
    pub hours: f64,
    pub cost_eur: f64,
    pub automatic: bool,
    pub how_to: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub advice: Advice,
    pub options: Vec<PlaceOption>,
}

/// Layers and hidden size for a parameter count, from the Qwen2.5 family
/// (nearest size), since activation memory depends on shape, not only N.
fn shape(params_b: f64) -> (f64, f64) {
    const SHAPES: &[(f64, f64, f64)] = &[
        (0.5, 24.0, 896.0),
        (1.5, 28.0, 1536.0),
        (3.0, 36.0, 2048.0),
        (7.0, 28.0, 3584.0),
        (14.0, 48.0, 5120.0),
        (32.0, 64.0, 5120.0),
        (72.0, 80.0, 8192.0),
    ];
    let &(_, layers, hidden) = SHAPES
        .iter()
        .min_by(|a, b| (a.0 - params_b).abs().total_cmp(&(b.0 - params_b).abs()))
        .unwrap();
    (layers, hidden)
}

/// Trainable fraction of the weights for rank-16 LoRA on every projection
/// (~40M of 7.6B for Qwen2.5-7B); each costs 16 bytes with grads and Adam state.
const LORA_FRACTION: f64 = 0.006;

/// Peak training memory in GB for one sequence per step.
fn memory_gb(req: &PlanRequest, method: Method, hardware: Hardware) -> f64 {
    let n = req.params_b * 1e9;
    let (layers, hidden) = shape(req.params_b);
    let seq = req.seq_len;
    let (weights, state, act_bytes) = match (hardware, method) {
        // Candle on CPU: F32 weights, no gradient checkpointing, so every
        // layer's activations are kept (~34 values per token per hidden unit).
        (Hardware::Cpu, _) => (n * 4.0, n * LORA_FRACTION * 16.0, 34.0 * seq * hidden * layers * 4.0),
        // Checkpointed GPU runs keep one input per layer plus one layer in full.
        (Hardware::Gpu, Method::Full) => (n * 16.0, 0.0, (2.0 * layers + 34.0) * seq * hidden * 2.0),
        (Hardware::Gpu, Method::Qlora) => (n * 0.55, n * LORA_FRACTION * 16.0, (2.0 * layers + 34.0) * seq * hidden * 2.0),
        (Hardware::Gpu, _) => (n * 2.0, n * LORA_FRACTION * 16.0, (2.0 * layers + 34.0) * seq * hidden * 2.0),
    };
    // fp32 logits, their softmax and gradient: large for a 152k vocabulary.
    let logits = seq * req.vocab * 4.0 * 3.0;
    let overhead = if hardware == Hardware::Gpu { 1.0e9 } else { 0.5e9 };
    (weights + state + act_bytes + logits + overhead) / 1e9
}

fn hours(req: &PlanRequest, p: &Place) -> f64 {
    let flops = 6.0 * req.params_b * 1e9 * req.tokens * req.epochs;
    flops / (p.peak_tflops * 1e12 * p.efficiency) / 3600.0
}

pub fn advise(req: &PlanRequest) -> Advice {
    let mut reasons = Vec::new();
    let approach = if req.goal == Goal::Facts {
        reasons.push(
            "Facts and numbers are best served by retrieval (look them up at answer time), not \
             fine-tuning: a fine-tune learns the vocabulary, then invents plausible facts. \
             doerr-lab-model v0.1 did exactly that."
                .into(),
        );
        reasons.push("Train only for the field's language, and put the facts in a searchable index.".into());
        "retrieval"
    } else {
        match req.method {
            Method::Full => {
                reasons.push("Full fine-tuning requested.".into());
                "full"
            }
            Method::Qlora => {
                reasons.push("QLoRA requested: 4-bit base weights, LoRA adapters on top.".into());
                "qlora"
            }
            _ if req.tokens * req.epochs >= 1e8 && req.params_b <= 3.0 => {
                reasons.push("Over 10⁸ training tokens on a small model: enough data for full fine-tuning.".into());
                "full"
            }
            _ => {
                reasons.push(
                    "LoRA: trains ~1% of the weights, needs a fraction of the memory, and is \
                     right for up to ~10⁸ tokens."
                        .into(),
                );
                "lora"
            }
        }
    };
    if req.tokens < 50_000.0 {
        reasons.push(format!(
            "Only {:.0}k tokens: expect the model to pick up style and terms, not content. \
             Adding question-and-answer examples helps more than more epochs.",
            req.tokens / 1e3
        ));
    }
    if req.goal == Goal::Tasks {
        reasons.push("Train on worked examples in chat form, with the loss on the answers only.".into());
    }
    Advice {
        approach: approach.into(),
        reasons,
    }
}

/// Ranks every place for `req`: feasible ones first, then cheapest, then fastest.
pub fn plan(req: &PlanRequest, catalog: &Catalog) -> Plan {
    let advice = advise(req);
    let wanted = match advice.approach.as_str() {
        "full" => Method::Full,
        "qlora" => Method::Qlora,
        _ => Method::Lora,
    };

    let mut options: Vec<PlaceOption> = catalog
        .places
        .iter()
        .map(|p| {
            let mut why_not = Vec::new();
            let method = match (p.hardware, wanted) {
                (Hardware::Cpu, Method::Full | Method::Qlora) => {
                    why_not.push("the laptop trainer only does LoRA".into());
                    Method::Lora
                }
                // On a GPU, fall back from LoRA to QLoRA when bf16 weights don't fit.
                (Hardware::Gpu, Method::Lora) if req.method == Method::Auto
                    && memory_gb(req, Method::Lora, p.hardware) > p.memory_gb * 0.9 =>
                {
                    Method::Qlora
                }
                (_, m) => m,
            };
            let mem = memory_gb(req, method, p.hardware);
            if mem > p.memory_gb * 0.9 {
                why_not.push(format!("needs ~{mem:.1} GB, has {:.0} GB", p.memory_gb));
            }
            let h = hours(req, p);
            if let Some(limit) = p.session_limit_h {
                if h > limit {
                    why_not.push(format!("~{h:.1} h exceeds the {limit:.0} h session limit"));
                }
            }
            if let Err(reason) = p.custody.allows(req.privacy) {
                why_not.push(reason.into());
            }
            PlaceOption {
                place: p.id.clone(),
                name: p.name.clone(),
                feasible: why_not.is_empty(),
                why_not,
                method,
                memory_gb: (mem * 10.0).round() / 10.0,
                hours: (h * 100.0).round() / 100.0,
                cost_eur: (h * p.price_per_hour_eur * 100.0).round() / 100.0,
                automatic: p.automatic,
                how_to: p.how_to.clone(),
            }
        })
        .collect();

    options.sort_by(|a, b| {
        b.feasible
            .cmp(&a.feasible)
            .then(a.cost_eur.total_cmp(&b.cost_eur))
            .then(a.hours.total_cmp(&b.hours))
    });
    Plan { advice, options }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(params_b: f64, tokens: f64, privacy: Privacy) -> PlanRequest {
        PlanRequest {
            params_b,
            tokens,
            epochs: 1.0,
            seq_len: 1024.0,
            vocab: 151_936.0,
            method: Method::Auto,
            privacy,
            goal: Goal::Vocabulary,
        }
    }

    fn option<'a>(plan: &'a Plan, id: &str) -> &'a PlaceOption {
        plan.options.iter().find(|o| o.place == id).unwrap()
    }

    #[test]
    fn laptop_estimate_matches_the_measured_run() {
        // 150 steps of 256 tokens took ~45 min per epoch (18 s/step).
        let mut r = req(0.494, 150.0 * 256.0, Privacy::Private);
        r.seq_len = 256.0;
        let laptop = Catalog::default().places.into_iter().find(|p| p.id == "laptop").unwrap();
        let h = hours(&r, &laptop);
        let measured = 150.0 * 18.0 / 3600.0;
        assert!(h / measured > 1.0 / 1.5 && h / measured < 1.5, "{h} vs {measured}");
        // And its memory estimate fits in the ~5 GB that run had free, while
        // 512 tokens x batch 2 (which ran out of memory) must not.
        assert!(memory_gb(&r, Method::Lora, Hardware::Cpu) < 5.0);
        r.seq_len = 1024.0;
        assert!(memory_gb(&r, Method::Lora, Hardware::Cpu) > 5.0);
    }

    #[test]
    fn private_data_never_goes_to_shared_or_institutional_machines() {
        let p = plan(&req(0.5, 1e5, Privacy::Private), &Catalog::default());
        for id in ["vast-4090", "kaggle-t4", "colab-t4", "apphub-4090", "kisski-a100"] {
            assert!(!option(&p, id).feasible, "{id} should be excluded for private data");
        }
        assert!(option(&p, "runpod-4090").feasible);
        assert!(option(&p, "laptop").feasible);
    }

    #[test]
    fn facts_are_answered_with_retrieval() {
        let mut r = req(3.0, 1e6, Privacy::Internal);
        r.goal = Goal::Facts;
        assert_eq!(advise(&r).approach, "retrieval");
    }

    #[test]
    fn seven_b_on_8_gb_needs_qlora_and_24_gb_takes_lora() {
        let mut r = req(7.0, 1e6, Privacy::Internal);
        r.seq_len = 512.0;
        let p = plan(&r, &Catalog::default());
        let small = option(&p, "apphub-2070");
        assert_eq!(small.method, Method::Qlora);
        assert!(small.feasible, "{:?}", small.why_not);
        let p = plan(&req(7.0, 1e6, Privacy::Internal), &Catalog::default());
        let big = option(&p, "apphub-4090");
        assert_eq!(big.method, Method::Lora);
        assert!(big.feasible, "{:?}", big.why_not);
    }

    #[test]
    fn free_feasible_places_rank_first_and_slow_laptop_runs_are_refused() {
        // The NFDI4Cat run: 7B, ~1.1M tokens, 3 epochs, internal data.
        let mut r = req(7.0, 1.1e6, Privacy::Internal);
        r.epochs = 3.0;
        let p = plan(&r, &Catalog::default());
        assert!(p.options[0].feasible && p.options[0].cost_eur == 0.0);
        assert!(!option(&p, "laptop").feasible);
        let apphub = option(&p, "apphub-4090");
        assert!(apphub.hours > 0.3 && apphub.hours < 3.0, "{}", apphub.hours);
    }

    #[test]
    fn places_toml_overrides_and_extends_the_defaults() {
        let dir = std::env::temp_dir().join(format!("pf-places-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("places.toml");
        std::fs::write(
            &path,
            r#"
[[places]]
id = "vast-4090"
name = "Vast 4090"
hardware = "gpu"
memory_gb = 24.0
peak_tflops = 165.0
efficiency = 0.25
price_per_hour_eur = 0.30
custody = "shared"
automatic = false
how_to = "x"

[[places]]
id = "home-3060"
name = "Home 3060"
hardware = "gpu"
memory_gb = 12.0
peak_tflops = 50.0
efficiency = 0.25
price_per_hour_eur = 0.0
custody = "own"
automatic = false
how_to = "y"
"#,
        )
        .unwrap();
        let c = Catalog::load(&path).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(c.places.iter().find(|p| p.id == "vast-4090").unwrap().price_per_hour_eur, 0.30);
        assert!(c.places.iter().any(|p| p.id == "home-3060"));
        assert_eq!(c.places.len(), Catalog::default().places.len() + 1);
    }
}
