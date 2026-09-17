use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use rand::seq::SliceRandom;
use rand::thread_rng;

use crate::batch::build_batch;
use crate::contract::{PretrainedConfig, ScratchConfig, TrainingSpec};
use crate::corpus::Corpus;
use crate::error::Error;
use crate::hf;
use crate::model::GptModel;
use crate::pretrained_model::{self, LoraLlama};
use crate::tokenizer::Tokenizer;

pub struct TrainedScratchModel {
    pub model: GptModel,
    pub tokenizer: Tokenizer,
}

/// Trains a `GptModel` from scratch (base weights + LoRA adapters, all
/// trainable) on `corpus` using AdamW, per `absicht`'s acquisition pipeline:
/// train once, at this theme's own extremal-regime corpus, rather than per
/// restriction.
pub fn run_scratch(
    cfg: &ScratchConfig,
    training: &TrainingSpec,
    corpus: Corpus,
) -> Result<TrainedScratchModel, Error> {
    if corpus.examples.is_empty() {
        return Err(Error::Train(
            "corpus has no admitted examples; nothing to train on".into(),
        ));
    }

    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let model = GptModel::new(cfg, training.lora_rank, training.lora_alpha, vb)
        .map_err(|e| Error::Train(format!("model init: {e}")))?;

    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: training.learning_rate,
            ..ParamsAdamW::default()
        },
    )
    .map_err(|e| Error::Train(format!("optimizer init: {e}")))?;

    let eos_id = corpus.tokenizer.eos_id;
    let mut indices: Vec<usize> = (0..corpus.examples.len()).collect();
    let mut rng = thread_rng();

    for epoch in 0..training.epochs {
        indices.shuffle(&mut rng);
        let mut total_loss = 0f32;
        let mut n_batches = 0usize;

        for batch_idx in indices.chunks(training.batch_size.max(1)) {
            let batch: Vec<&Vec<u32>> = batch_idx
                .iter()
                .map(|&i| &corpus.examples[i].token_ids)
                .collect();
            let Some((input_ids, targets)) = build_batch(&batch, eos_id, &device) else {
                continue;
            };

            let logits = model
                .forward(&input_ids)
                .map_err(|e| Error::Train(format!("forward: {e}")))?;
            let loss = sequence_cross_entropy(&logits, &targets)?;

            opt.backward_step(&loss)
                .map_err(|e| Error::Train(format!("backward step: {e}")))?;

            total_loss += loss
                .to_scalar::<f32>()
                .map_err(|e| Error::Train(format!("read loss: {e}")))?;
            n_batches += 1;
        }

        let avg_loss = if n_batches > 0 {
            total_loss / n_batches as f32
        } else {
            f32::NAN
        };
        tracing::info!(epoch, avg_loss, "purpose-factory training epoch complete (scratch)");
    }

    Ok(TrainedScratchModel {
        model,
        tokenizer: corpus.tokenizer,
    })
}

fn sequence_cross_entropy(logits: &Tensor, targets: &Tensor) -> Result<Tensor, Error> {
    let (b, t, v) = logits
        .dims3()
        .map_err(|e| Error::Train(format!("logits shape: {e}")))?;
    let logits_flat = logits
        .reshape((b * t, v))
        .map_err(|e| Error::Train(format!("reshape logits: {e}")))?;
    let targets_flat = targets
        .reshape(b * t)
        .map_err(|e| Error::Train(format!("reshape targets: {e}")))?;
    candle_nn::loss::cross_entropy(&logits_flat, &targets_flat)
        .map_err(|e| Error::Train(format!("loss: {e}")))
}

/// Merges LoRA adapters into the base weights and writes a self-contained
/// `.safetensors` + `config.json` + `tokenizer.json` bundle — a full,
/// loadable model artifact rather than an adapter-only export.
pub fn export_scratch(trained: &TrainedScratchModel, out_dir: &Path) -> Result<(), Error> {
    std::fs::create_dir_all(out_dir)?;

    let merged = trained
        .model
        .merge_lora()
        .map_err(|e| Error::Export(format!("merge lora: {e}")))?;

    let tensors: HashMap<String, Tensor> = merged.named_tensors().into_iter().collect();
    candle_core::safetensors::save(&tensors, out_dir.join("model.safetensors"))
        .map_err(|e| Error::Export(format!("safetensors write: {e}")))?;

    let cfg = merged.config();
    let config_json = serde_json::json!({
        "architecture": "purpose-factory-gpt-mini",
        "vocab_size": cfg.vocab_size,
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "n_embd": cfg.n_embd,
        "block_size": cfg.block_size,
    });
    std::fs::write(
        out_dir.join("config.json"),
        serde_json::to_string_pretty(&config_json).map_err(|e| Error::Export(e.to_string()))?,
    )?;

    let tokenizer_json =
        serde_json::to_string_pretty(&trained.tokenizer).map_err(|e| Error::Export(e.to_string()))?;
    std::fs::write(out_dir.join("tokenizer.json"), tokenizer_json)?;

    Ok(())
}

// =====================================================================
// Pretrained (LoRA-adapted) path
// =====================================================================

pub struct TrainedPretrainedModel {
    pub model: LoraLlama,
    pub tokenizer_path: std::path::PathBuf,
    pub vocab_size: usize,
}

/// Downloads a pretrained checkpoint, LoRA-adapts it (only `q_proj`,
/// `v_proj`, `gate_proj` are trainable in every block; everything else stays
/// frozen), and trains on `admitted_texts` using the checkpoint's own
/// tokenizer. Unlike the from-scratch path, no new vocabulary is trained —
/// the pretrained tokenizer is authoritative.
pub async fn run_pretrained(
    pretrained: &PretrainedConfig,
    training: &TrainingSpec,
    block_size: usize,
    admitted_texts: &[String],
) -> Result<TrainedPretrainedModel, Error> {
    if admitted_texts.is_empty() {
        return Err(Error::Train(
            "corpus has no admitted examples; nothing to train on".into(),
        ));
    }

    tracing::info!(repo = %pretrained.repo, "downloading pretrained checkpoint");
    let files = hf::fetch(&pretrained.repo, pretrained.revision.as_deref()).await?;

    let cfg = pretrained_model::load_config(&files.config)
        .map_err(|e| Error::Train(format!("load config.json: {e}")))?;
    let tokenizer = tokenizers::Tokenizer::from_file(&files.tokenizer)
        .map_err(|e| Error::Train(format!("load tokenizer.json: {e}")))?;

    let device = Device::Cpu;
    // Safety: the checkpoint file is not mutated concurrently by this
    // process; `files.weights` is exclusively owned by this build.
    let frozen_vb = unsafe {
        pretrained_model::load_frozen_weights(&[files.weights.clone()], DType::F32, &device)
            .map_err(|e| Error::Train(format!("load checkpoint weights: {e}")))?
    };

    let varmap = VarMap::new();
    let lora_vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let model = LoraLlama::load(
        frozen_vb,
        lora_vb,
        &cfg,
        training.lora_rank,
        training.lora_alpha,
        &device,
    )
    .map_err(|e| Error::Train(format!("model init: {e}")))?;

    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: training.learning_rate,
            ..ParamsAdamW::default()
        },
    )
    .map_err(|e| Error::Train(format!("optimizer init: {e}")))?;

    let examples: Vec<Vec<u32>> = admitted_texts
        .iter()
        .filter_map(|text| {
            let enc = tokenizer.encode(text.as_str(), true).ok()?;
            let mut ids = enc.get_ids().to_vec();
            ids.truncate(block_size);
            (ids.len() >= 8).then_some(ids)
        })
        .collect();
    if examples.is_empty() {
        return Err(Error::Train(
            "no example tokenized to at least 8 tokens under the pretrained tokenizer".into(),
        ));
    }

    let pad_id = tokenizer
        .token_to_id("</s>")
        .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
        .unwrap_or(0);

    let mut indices: Vec<usize> = (0..examples.len()).collect();
    let mut rng = thread_rng();

    for epoch in 0..training.epochs {
        indices.shuffle(&mut rng);
        let mut total_loss = 0f32;
        let mut n_batches = 0usize;

        for batch_idx in indices.chunks(training.batch_size.max(1)) {
            let batch: Vec<&Vec<u32>> = batch_idx.iter().map(|&i| &examples[i]).collect();
            let Some((input_ids, targets)) = build_batch(&batch, pad_id, &device) else {
                continue;
            };

            let logits = model
                .forward_train(&input_ids)
                .map_err(|e| Error::Train(format!("forward: {e}")))?;
            let loss = sequence_cross_entropy(&logits, &targets)?;

            opt.backward_step(&loss)
                .map_err(|e| Error::Train(format!("backward step: {e}")))?;

            total_loss += loss
                .to_scalar::<f32>()
                .map_err(|e| Error::Train(format!("read loss: {e}")))?;
            n_batches += 1;
        }

        let avg_loss = if n_batches > 0 {
            total_loss / n_batches as f32
        } else {
            f32::NAN
        };
        tracing::info!(epoch, avg_loss, "purpose-factory training epoch complete (pretrained)");
    }

    Ok(TrainedPretrainedModel {
        model,
        tokenizer_path: files.tokenizer,
        vocab_size: cfg.vocab_size,
    })
}

/// Merges LoRA adapters into the base checkpoint's weights and writes a
/// self-contained `.safetensors` + `config.json` + `tokenizer.json` bundle —
/// tensor names match the original checkpoint's naming, so this is a
/// drop-in replacement for it in any HF-compatible loader.
pub fn export_pretrained(trained: &TrainedPretrainedModel, out_dir: &Path) -> Result<(), Error> {
    std::fs::create_dir_all(out_dir)?;

    let merged = trained
        .model
        .merge_lora()
        .map_err(|e| Error::Export(format!("merge lora: {e}")))?;

    let tensors: HashMap<String, Tensor> = merged.named_tensors().into_iter().collect();
    candle_core::safetensors::save(&tensors, out_dir.join("model.safetensors"))
        .map_err(|e| Error::Export(format!("safetensors write: {e}")))?;

    let cfg = merged.config();
    let config_json = serde_json::json!({
        "architecture": "purpose-factory-lora-llama",
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "vocab_size": cfg.vocab_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads(),
        "rms_norm_eps": cfg.rms_norm_eps,
        "rope_theta": cfg.rope_theta,
        "max_position_embeddings": cfg.max_position_embeddings,
        "tie_word_embeddings": cfg.tie_word_embeddings,
    });
    std::fs::write(
        out_dir.join("config.json"),
        serde_json::to_string_pretty(&config_json).map_err(|e| Error::Export(e.to_string()))?,
    )?;

    std::fs::copy(&trained.tokenizer_path, out_dir.join("tokenizer.json"))?;

    Ok(())
}
