//! A LoRA-adapted LLaMA-family causal LM, loaded from a real pretrained
//! HuggingFace checkpoint. Architecturally mirrors
//! `candle_transformers::models::llama` (verified against that crate's
//! source, v0.11), but is written directly against public `candle_core`/
//! `candle_nn` primitives rather than depending on that crate: its `Llama`
//! type keeps `q_proj`/`k_proj`/`v_proj`/`o_proj` private, so there is no way
//! to splice `LoraLinear` into its attention from outside. Vendoring the
//! architecture here, with `q`/`v` as `LoraLinear`, is what makes LoRA
//! adaptation of a real checkpoint possible at all.

use std::path::Path;

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{embedding, linear, linear_no_bias, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};
use serde::Deserialize;

use crate::lora::LoraLinear;

/// HuggingFace `config.json` shape for LLaMA-family models (LLaMA, TinyLlama,
/// Mistral-architecture, and compatible others). Field names and optionality
/// match the config.json emitted by these model families; deserializes
/// directly with no manual mapping.
#[derive(Debug, Clone, Deserialize)]
pub struct HfLlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// `"llama"`, `"qwen2"`, `"mistral"`, ... — decides architecture
    /// defaults the config does not state explicitly (see `qkv_bias`).
    #[serde(default)]
    pub model_type: Option<String>,
    /// Explicit q/k/v bias flag, when the config carries one.
    #[serde(default)]
    pub attention_bias: Option<bool>,
}

fn default_rms_eps() -> f64 {
    1e-5
}
fn default_rope_theta() -> f32 {
    10_000.0
}
fn default_max_pos() -> usize {
    4096
}

impl HfLlamaConfig {
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Whether q/k/v projections carry biases. Qwen2 checkpoints have them
    /// but their config.json does not say so, hence the `model_type` default.
    pub fn qkv_bias(&self) -> bool {
        self.attention_bias
            .unwrap_or(self.model_type.as_deref() == Some("qwen2"))
    }
}

/// q/k/v projection: biased for architectures that have q/k/v biases.
fn qkv_linear(cfg: &HfLlamaConfig, in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    if cfg.qkv_bias() {
        linear(in_dim, out_dim, vb)
    } else {
        linear_no_bias(in_dim, out_dim, vb)
    }
}

struct RotaryCache {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryCache {
    fn new(cfg: &HfLlamaConfig, device: &Device) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let theta: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
            .collect();
        let theta = Tensor::new(theta, device)?;
        let idx = Tensor::arange(0u32, cfg.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_position_embeddings, 1))?;
        let freqs = idx.matmul(&theta.reshape((1, theta.elem_count()))?)?;
        Ok(Self {
            cos: freqs.cos()?,
            sin: freqs.sin()?,
        })
    }

    fn slice(&self, seq_len: usize) -> Result<(Tensor, Tensor)> {
        Ok((
            self.cos.narrow(0, 0, seq_len)?,
            self.sin.narrow(0, 0, seq_len)?,
        ))
    }
}

struct Attention {
    q: LoraLinear,
    k: Linear,
    v: LoraLinear,
    o: Linear,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
}

impl Attention {
    /// `frozen_vb` reads base weights from the pretrained checkpoint;
    /// `lora_vb` (VarMap-backed) creates the trainable A/B parameters. The
    /// two must be `pp`-ed to the same relative path so tensor names line up
    /// with the checkpoint's `model.layers.{i}.self_attn.*` naming.
    fn load(
        frozen_vb: VarBuilder,
        lora_vb: VarBuilder,
        cfg: &HfLlamaConfig,
        lora_rank: usize,
        lora_alpha: f64,
    ) -> Result<Self> {
        let size_q = cfg.hidden_size;
        let size_kv = cfg.head_dim() * cfg.num_key_value_heads();

        let q_base = qkv_linear(cfg, cfg.hidden_size, size_q, frozen_vb.pp("q_proj"))?;
        let v_base = qkv_linear(cfg, cfg.hidden_size, size_kv, frozen_vb.pp("v_proj"))?;

        Ok(Self {
            q: LoraLinear::from_frozen_base(q_base, lora_rank, lora_alpha, lora_vb.pp("q_proj"))?,
            k: qkv_linear(cfg, cfg.hidden_size, size_kv, frozen_vb.pp("k_proj"))?,
            v: LoraLinear::from_frozen_base(v_base, lora_rank, lora_alpha, lora_vb.pp("v_proj"))?,
            o: linear_no_bias(size_q, cfg.hidden_size, frozen_vb.pp("o_proj"))?,
            n_head: cfg.num_attention_heads,
            n_kv_head: cfg.num_key_value_heads(),
            head_dim: cfg.head_dim(),
        })
    }

    fn forward(&self, x: &Tensor, rotary: &RotaryCache, causal_mask: &Tensor) -> Result<Tensor> {
        let (b, t, _c) = x.dims3()?;

        let q = self
            .q
            .forward(x)?
            .reshape((b, t, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k
            .forward(x)?
            .reshape((b, t, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v
            .forward(x)?
            .reshape((b, t, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let (cos, sin) = rotary.slice(t)?;
        let q = candle_nn::rotary_emb::rope(&q, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope(&k, &cos, &sin)?;

        let n_rep = self.n_head / self.n_kv_head;
        let k = repeat_kv(k, n_rep)?;
        let v = repeat_kv(v, n_rep)?;

        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let att = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let att = causal_mask
            .broadcast_as(att.shape())?
            .where_cond(&att, &neg_inf_like(&att)?)?;
        let att = candle_nn::ops::softmax(&att, D::Minus1)?;

        let y = att.matmul(&v)?;
        let y = y.transpose(1, 2)?.contiguous()?.reshape((b, t, ()))?;
        self.o.forward(&y)
    }

    fn merge_lora(&self) -> Result<MergedAttention> {
        Ok(MergedAttention {
            q: self.q.merge_into_base()?,
            k: self.k.clone(),
            v: self.v.merge_into_base()?,
            o: self.o.clone(),
            n_head: self.n_head,
            n_kv_head: self.n_kv_head,
            head_dim: self.head_dim,
        })
    }
}

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (b, n_kv_head, t, head_dim) = x.dims4()?;
    Tensor::cat(&vec![&x; n_rep], 2)?.reshape((b, n_kv_head * n_rep, t, head_dim))
}

fn neg_inf_like(t: &Tensor) -> Result<Tensor> {
    Tensor::full(f32::NEG_INFINITY, t.shape(), t.device())?.to_dtype(t.dtype())
}

fn causal_mask(t: usize, device: &Device) -> Result<Tensor> {
    let mask = Tensor::tril2(t, DType::U8, device)?;
    mask.reshape((1, 1, t, t))?.ge(1u8)
}

struct Mlp {
    gate: LoraLinear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    fn load(
        frozen_vb: VarBuilder,
        lora_vb: VarBuilder,
        cfg: &HfLlamaConfig,
        lora_rank: usize,
        lora_alpha: f64,
    ) -> Result<Self> {
        let gate_base = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, frozen_vb.pp("gate_proj"))?;
        Ok(Self {
            gate: LoraLinear::from_frozen_base(gate_base, lora_rank, lora_alpha, lora_vb.pp("gate_proj"))?,
            up: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, frozen_vb.pp("up_proj"))?,
            down: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, frozen_vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gated = (candle_nn::ops::silu(&self.gate.forward(x)?)? * self.up.forward(x)?)?;
        self.down.forward(&gated)
    }

    fn merge_lora(&self) -> Result<MergedMlp> {
        Ok(MergedMlp {
            gate: self.gate.merge_into_base()?,
            up: self.up.clone(),
            down: self.down.clone(),
        })
    }
}

struct Block {
    input_ln: RmsNorm,
    attn: Attention,
    post_attn_ln: RmsNorm,
    mlp: Mlp,
}

impl Block {
    fn load(
        frozen_vb: VarBuilder,
        lora_vb: VarBuilder,
        cfg: &HfLlamaConfig,
        lora_rank: usize,
        lora_alpha: f64,
    ) -> Result<Self> {
        Ok(Self {
            input_ln: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, frozen_vb.pp("input_layernorm"))?,
            attn: Attention::load(
                frozen_vb.pp("self_attn"),
                lora_vb.pp("self_attn"),
                cfg,
                lora_rank,
                lora_alpha,
            )?,
            post_attn_ln: rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                frozen_vb.pp("post_attention_layernorm"),
            )?,
            mlp: Mlp::load(frozen_vb.pp("mlp"), lora_vb.pp("mlp"), cfg, lora_rank, lora_alpha)?,
        })
    }

    fn forward(&self, x: &Tensor, rotary: &RotaryCache, mask: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x = self.attn.forward(&self.input_ln.forward(x)?, rotary, mask)?;
        let x = (x + residual)?;
        let residual = &x;
        let mlp_out = self.mlp.forward(&self.post_attn_ln.forward(&x)?)?;
        mlp_out + residual
    }
}

/// A LoRA-adapted LLaMA-family model loaded from a pretrained checkpoint.
/// `q_proj`, `v_proj` in every attention block and `gate_proj` in every MLP
/// are `LoraLinear` (trainable low-rank adapters over frozen base weights);
/// everything else loads as ordinary frozen `Linear`/`RmsNorm`/`Embedding`.
pub struct LoraLlama {
    embed: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Linear,
    rotary: RotaryCache,
    cfg: HfLlamaConfig,
}

impl LoraLlama {
    /// `frozen_vb` must be built over the checkpoint's safetensors (e.g. via
    /// `load_frozen_weights`) — every non-LoRA tensor, and every LoRA
    /// layer's base weight, is read from it and never trained. `lora_vb`
    /// must be `VarBuilder::from_varmap` over a fresh `VarMap` — only the
    /// tensors it creates (the LoRA A/B matrices) are trainable.
    pub fn load(
        frozen_vb: VarBuilder,
        lora_vb: VarBuilder,
        cfg: &HfLlamaConfig,
        lora_rank: usize,
        lora_alpha: f64,
        device: &Device,
    ) -> Result<Self> {
        let embed = embedding(cfg.vocab_size, cfg.hidden_size, frozen_vb.pp("model.embed_tokens"))?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, frozen_vb.pp("lm_head"))?
        };
        let ln_f = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, frozen_vb.pp("model.norm"))?;

        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let path = format!("model.layers.{i}");
            blocks.push(Block::load(
                frozen_vb.pp(&path),
                lora_vb.pp(&path),
                cfg,
                lora_rank,
                lora_alpha,
            )?);
        }

        let rotary = RotaryCache::new(cfg, device)?;

        Ok(Self {
            embed,
            blocks,
            ln_f,
            lm_head,
            rotary,
            cfg: cfg.clone(),
        })
    }

    pub fn config(&self) -> &HfLlamaConfig {
        &self.cfg
    }

    /// Full-sequence logits (batch, seq_len, vocab_size) for teacher-forcing
    /// loss — unlike the reference implementation's `forward`, which returns
    /// only the last position's logits (sufficient for generation, not for
    /// computing a next-token loss at every position during training).
    pub fn forward_train(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_b, t) = input_ids.dims2()?;
        let device = input_ids.device();

        let mut x = self.embed.forward(input_ids)?;
        let mask = causal_mask(t, device)?;
        for block in &self.blocks {
            x = block.forward(&x, &self.rotary, &mask)?;
        }
        let x = self.ln_f.forward(&x)?;
        self.lm_head.forward(&x)
    }

    pub fn merge_lora(&self) -> Result<MergedLoraLlama> {
        let mut merged_blocks = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            merged_blocks.push(MergedBlock {
                input_ln: block.input_ln.clone(),
                attn: block.attn.merge_lora()?,
                post_attn_ln: block.post_attn_ln.clone(),
                mlp: block.mlp.merge_lora()?,
            });
        }
        Ok(MergedLoraLlama {
            embed: self.embed.clone(),
            blocks: merged_blocks,
            ln_f: self.ln_f.clone(),
            lm_head: self.lm_head.clone(),
            rotary_device: self.rotary.cos.device().clone(),
            cfg: self.cfg.clone(),
        })
    }
}

struct MergedAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
}

impl MergedAttention {
    fn forward(&self, x: &Tensor, rotary: &RotaryCache, mask: &Tensor) -> Result<Tensor> {
        let (b, t, _c) = x.dims3()?;
        let q = self
            .q
            .forward(x)?
            .reshape((b, t, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k
            .forward(x)?
            .reshape((b, t, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v
            .forward(x)?
            .reshape((b, t, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let (cos, sin) = rotary.slice(t)?;
        let q = candle_nn::rotary_emb::rope(&q, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope(&k, &cos, &sin)?;

        let n_rep = self.n_head / self.n_kv_head;
        let k = repeat_kv(k, n_rep)?;
        let v = repeat_kv(v, n_rep)?;

        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let att = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let att = mask.broadcast_as(att.shape())?.where_cond(&att, &neg_inf_like(&att)?)?;
        let att = candle_nn::ops::softmax(&att, D::Minus1)?;

        let y = att.matmul(&v)?;
        let y = y.transpose(1, 2)?.contiguous()?.reshape((b, t, ()))?;
        self.o.forward(&y)
    }
}

struct MergedMlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl MergedMlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gated = (candle_nn::ops::silu(&self.gate.forward(x)?)? * self.up.forward(x)?)?;
        self.down.forward(&gated)
    }
}

struct MergedBlock {
    input_ln: RmsNorm,
    attn: MergedAttention,
    post_attn_ln: RmsNorm,
    mlp: MergedMlp,
}

/// A LoRA-free mirror of `LoraLlama`, produced by `merge_lora`, for export —
/// every projection is a plain `Linear`.
pub struct MergedLoraLlama {
    embed: Embedding,
    blocks: Vec<MergedBlock>,
    ln_f: RmsNorm,
    lm_head: Linear,
    rotary_device: Device,
    cfg: HfLlamaConfig,
}

impl MergedLoraLlama {
    pub fn config(&self) -> &HfLlamaConfig {
        &self.cfg
    }

    pub fn forward_train(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_b, t) = input_ids.dims2()?;
        let rotary = RotaryCache::new(&self.cfg, &self.rotary_device)?;
        let mask = causal_mask(t, &self.rotary_device)?;

        let mut x = self.embed.forward(input_ids)?;
        for block in &self.blocks {
            let residual = &x;
            let attn_out = block.attn.forward(&block.input_ln.forward(&x)?, &rotary, &mask)?;
            x = (&attn_out + residual)?;
            let residual = &x;
            let mlp_out = block.mlp.forward(&block.post_attn_ln.forward(&x)?)?;
            x = (&mlp_out + residual)?;
        }
        let x = self.ln_f.forward(&x)?;
        self.lm_head.forward(&x)
    }

    /// Every named tensor, in the same `model.layers.{i}.*` naming HF
    /// checkpoints use, so the exported file is a drop-in replacement for
    /// the base checkpoint (only these tensors' values differ). A tied
    /// `lm_head` is not written, and q/k/v biases are, when the checkpoint
    /// has them — so the tensor set matches the checkpoint's exactly.
    pub fn named_tensors(&self) -> Vec<(String, Tensor)> {
        let mut out = vec![
            (
                "model.embed_tokens.weight".to_string(),
                self.embed.embeddings().clone(),
            ),
            ("model.norm.weight".to_string(), self.ln_f.weight().clone()),
        ];
        if !self.cfg.tie_word_embeddings {
            out.push(("lm_head.weight".to_string(), self.lm_head.weight().clone()));
        }

        for (i, block) in self.blocks.iter().enumerate() {
            let p = format!("model.layers.{i}");
            out.push((
                format!("{p}.input_layernorm.weight"),
                block.input_ln.weight().clone(),
            ));
            out.push((
                format!("{p}.post_attention_layernorm.weight"),
                block.post_attn_ln.weight().clone(),
            ));
            out.push((format!("{p}.self_attn.q_proj.weight"), block.attn.q.weight().clone()));
            out.push((format!("{p}.self_attn.k_proj.weight"), block.attn.k.weight().clone()));
            out.push((format!("{p}.self_attn.v_proj.weight"), block.attn.v.weight().clone()));
            out.push((format!("{p}.self_attn.o_proj.weight"), block.attn.o.weight().clone()));
            for (name, lin) in [("q_proj", &block.attn.q), ("k_proj", &block.attn.k), ("v_proj", &block.attn.v)] {
                if let Some(bias) = lin.bias() {
                    out.push((format!("{p}.self_attn.{name}.bias"), bias.clone()));
                }
            }
            out.push((format!("{p}.mlp.gate_proj.weight"), block.mlp.gate.weight().clone()));
            out.push((format!("{p}.mlp.up_proj.weight"), block.mlp.up.weight().clone()));
            out.push((format!("{p}.mlp.down_proj.weight"), block.mlp.down.weight().clone()));
        }
        out
    }
}

/// Loads a `HfLlamaConfig` from a downloaded `config.json`.
pub fn load_config(path: &Path) -> Result<HfLlamaConfig> {
    let raw = std::fs::read_to_string(path)?;
    serde_json::from_str(&raw).map_err(|e| candle_core::Error::Msg(format!("config.json parse: {e}")))
}

/// Loads a frozen `VarBuilder` over one or more mmapped safetensors files —
/// the base checkpoint weights, read by `LoraLlama::load`'s `frozen_vb` and
/// never mutated during training.
///
/// # Safety
/// Mmaps the given files; the caller must not mutate them concurrently.
pub unsafe fn load_frozen_weights(
    paths: &[std::path::PathBuf],
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    unsafe { VarBuilder::from_mmaped_safetensors(paths, dtype, device) }
}
