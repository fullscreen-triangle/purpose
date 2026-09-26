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

/// 1.0 where the target at (row, t) is a real next token of that row's
/// sequence, 0.0 on right-padding — matching `build_batch`'s layout.
fn target_mask(batch: &[&Vec<u32>], device: &Device) -> Result<Tensor, Error> {
    let max_len = batch.iter().map(|ids| ids.len()).max().unwrap_or(0);
    let flat: Vec<f32> = batch
        .iter()
        .flat_map(|ids| (0..max_len).map(move |t| if t + 1 < ids.len() { 1.0 } else { 0.0 }))
        .collect();
    Tensor::from_vec(flat, (batch.len(), max_len), device)
        .map_err(|e| Error::Train(format!("target mask: {e}")))
}

/// Mean next-token cross-entropy over the positions where `mask` is 1.
fn masked_sequence_cross_entropy(logits: &Tensor, targets: &Tensor, mask: &Tensor) -> Result<Tensor, Error> {
    let tr = |e: candle_core::Error| Error::Train(format!("masked loss: {e}"));
    let (b, t, v) = logits.dims3().map_err(tr)?;
    let log_probs = candle_nn::ops::log_softmax(&logits.reshape((b * t, v)).map_err(tr)?, 1).map_err(tr)?;
    let picked = log_probs
        .gather(&targets.reshape((b * t, 1)).map_err(tr)?, 1)
        .map_err(tr)?
        .reshape(b * t)
        .map_err(tr)?;
    let mask = mask.reshape(b * t).map_err(tr)?;
    let n = mask.sum_all().map_err(tr)?;
    let nll = (picked * &mask).map_err(tr)?.sum_all().map_err(tr)?.neg().map_err(tr)?;
    nll.broadcast_div(&n).map_err(tr)
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
    /// The checkpoint's own `config.json`, copied verbatim on export.
    pub config_path: std::path::PathBuf,
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
    let steps_per_epoch = examples.len().div_ceil(training.batch_size.max(1));
    tracing::info!(examples = examples.len(), steps_per_epoch, "pretrained training start");
    let started = std::time::Instant::now();

    for epoch in 0..training.epochs {
        indices.shuffle(&mut rng);
        let mut total_loss = 0f32;
        let mut n_batches = 0usize;

        for batch_idx in indices.chunks(training.batch_size.max(1)) {
            let batch: Vec<&Vec<u32>> = batch_idx.iter().map(|&i| &examples[i]).collect();
            let Some((input_ids, targets)) = build_batch(&batch, pad_id, &device) else {
                continue;
            };
            // Right-padding must not be trained as next-token targets: the
            // pretrained tokenizer's pad id is a real token (`<|endoftext|>`).
            let mask = target_mask(&batch, &device)?;

            let logits = model
                .forward_train(&input_ids)
                .map_err(|e| Error::Train(format!("forward: {e}")))?;
            let loss = masked_sequence_cross_entropy(&logits, &targets, &mask)?;

            opt.backward_step(&loss)
                .map_err(|e| Error::Train(format!("backward step: {e}")))?;

            total_loss += loss
                .to_scalar::<f32>()
                .map_err(|e| Error::Train(format!("read loss: {e}")))?;
            n_batches += 1;
            tracing::info!(
                epoch,
                step = n_batches,
                steps_per_epoch,
                avg_loss = total_loss / n_batches as f32,
                elapsed_s = started.elapsed().as_secs(),
                "pretrained step"
            );
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
        config_path: files.config,
        tokenizer_path: files.tokenizer,
        vocab_size: cfg.vocab_size,
    })
}

/// Merges LoRA adapters into the base checkpoint's weights and writes a
/// self-contained `.safetensors` + `config.json` + `tokenizer.json` bundle —
/// tensor names match the original checkpoint's naming and `config.json` is
/// the checkpoint's own (LoRA merging changes values, never shapes), so this
/// is a drop-in replacement for it in any HF-compatible loader.
pub fn export_pretrained(trained: &TrainedPretrainedModel, out_dir: &Path) -> Result<(), Error> {
    std::fs::create_dir_all(out_dir)?;

    let merged = trained
        .model
        .merge_lora()
        .map_err(|e| Error::Export(format!("merge lora: {e}")))?;

    let tensors: HashMap<String, Tensor> = merged.named_tensors().into_iter().collect();
    candle_core::safetensors::save(&tensors, out_dir.join("model.safetensors"))
        .map_err(|e| Error::Export(format!("safetensors write: {e}")))?;

    std::fs::copy(&trained.config_path, out_dir.join("config.json"))?;
    std::fs::copy(&trained.tokenizer_path, out_dir.join("tokenizer.json"))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// A 2-layer Qwen2-shaped checkpoint (q/k/v biases, tied embeddings),
    /// random values, written the way HF writes one.
    fn tiny_qwen2_checkpoint(dir: &Path) -> (std::path::PathBuf, std::path::PathBuf, BTreeSet<String>) {
        let (h, inter, vocab, layers, kv) = (16usize, 32usize, 40usize, 2usize, 8usize);
        let dev = Device::Cpu;
        let r = |shape: &[usize]| Tensor::randn(0f32, 0.02, shape, &dev).unwrap();
        let mut t: HashMap<String, Tensor> = HashMap::new();
        t.insert("model.embed_tokens.weight".into(), r(&[vocab, h]));
        t.insert("model.norm.weight".into(), r(&[h]));
        for i in 0..layers {
            let p = format!("model.layers.{i}");
            t.insert(format!("{p}.input_layernorm.weight"), r(&[h]));
            t.insert(format!("{p}.post_attention_layernorm.weight"), r(&[h]));
            t.insert(format!("{p}.self_attn.q_proj.weight"), r(&[h, h]));
            t.insert(format!("{p}.self_attn.q_proj.bias"), r(&[h]));
            t.insert(format!("{p}.self_attn.k_proj.weight"), r(&[kv, h]));
            t.insert(format!("{p}.self_attn.k_proj.bias"), r(&[kv]));
            t.insert(format!("{p}.self_attn.v_proj.weight"), r(&[kv, h]));
            t.insert(format!("{p}.self_attn.v_proj.bias"), r(&[kv]));
            t.insert(format!("{p}.self_attn.o_proj.weight"), r(&[h, h]));
            t.insert(format!("{p}.mlp.gate_proj.weight"), r(&[inter, h]));
            t.insert(format!("{p}.mlp.up_proj.weight"), r(&[inter, h]));
            t.insert(format!("{p}.mlp.down_proj.weight"), r(&[h, inter]));
        }
        let names = t.keys().cloned().collect();
        let weights = dir.join("in.safetensors");
        candle_core::safetensors::save(&t, &weights).unwrap();
        let config = dir.join("in-config.json");
        std::fs::write(
            &config,
            r#"{"architectures":["Qwen2ForCausalLM"],"model_type":"qwen2","hidden_size":16,"intermediate_size":32,"vocab_size":40,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"rms_norm_eps":1e-6,"rope_theta":1000000.0,"max_position_embeddings":64,"tie_word_embeddings":true}"#,
        )
        .unwrap();
        (weights, config, names)
    }

    #[test]
    fn qwen2_export_round_trips_tensor_names_and_config() {
        let dir = std::env::temp_dir().join(format!("pf-qwen2-export-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (weights, config_path, in_names) = tiny_qwen2_checkpoint(&dir);
        let tokenizer_path = dir.join("tok.json");
        std::fs::write(&tokenizer_path, "{}").unwrap();

        let cfg = pretrained_model::load_config(&config_path).unwrap();
        assert!(cfg.qkv_bias(), "qwen2 must default to q/k/v biases");
        let device = Device::Cpu;
        let frozen = unsafe { pretrained_model::load_frozen_weights(&[weights.clone()], DType::F32, &device).unwrap() };
        let varmap = VarMap::new();
        let lora_vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = LoraLlama::load(frozen, lora_vb, &cfg, 4, 8.0, &device).unwrap();
        let logits = model.forward_train(&Tensor::new(&[[1u32, 2, 3, 4]], &device).unwrap()).unwrap();
        assert_eq!(logits.dims(), &[1, 4, 40]);

        let trained = TrainedPretrainedModel { model, config_path: config_path.clone(), tokenizer_path, vocab_size: 40 };
        let out = dir.join("out");
        export_pretrained(&trained, &out).unwrap();

        let exported = candle_core::safetensors::load(out.join("model.safetensors"), &device).unwrap();
        let out_names: BTreeSet<String> = exported.keys().cloned().collect();
        assert_eq!(out_names, in_names, "exported tensor set must equal the checkpoint's");

        // B starts at zero, so an untrained merge must reproduce the biases exactly.
        let original = candle_core::safetensors::load(&weights, &device).unwrap();
        let key = "model.layers.1.self_attn.q_proj.bias";
        let diff = (&exported[key] - &original[key]).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert_eq!(diff, 0.0);

        assert_eq!(std::fs::read(out.join("config.json")).unwrap(), std::fs::read(&config_path).unwrap());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn padding_does_not_change_masked_loss() {
        let device = Device::Cpu;
        let short = vec![1u32, 2, 3];
        let long = vec![4u32, 5, 6, 7, 8];
        let logits_for = |batch: &[&Vec<u32>]| {
            let len = batch.iter().map(|b| b.len()).max().unwrap();
            Tensor::randn(0f32, 1.0, (batch.len(), len, 10), &device).unwrap()
        };
        // Same logits for the real positions; padded tail gets arbitrary values.
        let batch = [&short, &long];
        let (_, targets) = build_batch(&batch, 0, &device).unwrap();
        let logits = logits_for(&batch);
        let mask = target_mask(&batch, &device).unwrap();
        let a = masked_sequence_cross_entropy(&logits, &targets, &mask).unwrap().to_scalar::<f32>().unwrap();

        let noise = Tensor::randn(0f32, 50.0, (1, 2, 10), &device).unwrap();
        let row0 = Tensor::cat(&[logits.narrow(0, 0, 1).unwrap().narrow(1, 0, 3).unwrap(), noise], 1).unwrap();
        let perturbed = Tensor::cat(&[row0, logits.narrow(0, 1, 1).unwrap()], 0).unwrap();
        let b = masked_sequence_cross_entropy(&perturbed, &targets, &mask).unwrap().to_scalar::<f32>().unwrap();
        assert!((a - b).abs() < 1e-5, "{a} vs {b}");

        // A row's last real token has no real next token, so it is masked too.
        let m = mask.to_vec2::<f32>().unwrap();
        assert_eq!(m[0], vec![1.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(m[1], vec![1.0, 1.0, 1.0, 1.0, 0.0]);
    }
}
