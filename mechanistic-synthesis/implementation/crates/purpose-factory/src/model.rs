use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, layer_norm, linear, Embedding, LayerNorm, Linear, VarBuilder};

use crate::contract::ScratchConfig;
use crate::lora::LoraLinear;

/// A small GPT-2-style decoder-only transformer, trained from scratch on a
/// single theme's corpus. Attention's query/value projections and the MLP's
/// up-projection are LoRA-adapted (`LoraLinear`); everything else (key
/// projection, embeddings, layer norms) trains as ordinary dense parameters,
/// following the common LoRA convention of adapting only a subset of
/// projections rather than the whole network.
pub struct GptModel {
    token_emb: Embedding,
    pos_emb: Embedding,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    lm_head: Linear,
    cfg: ScratchConfig,
}

struct Block {
    ln1: LayerNorm,
    attn: CausalSelfAttention,
    ln2: LayerNorm,
    mlp_up: LoraLinear,
    mlp_down: Linear,
}

struct CausalSelfAttention {
    q: LoraLinear,
    k: Linear,
    v: LoraLinear,
    out: Linear,
    n_head: usize,
    head_dim: usize,
}

impl GptModel {
    pub fn new(cfg: &ScratchConfig, lora_rank: usize, lora_alpha: f64, vb: VarBuilder) -> Result<Self> {
        let token_emb = embedding(cfg.vocab_size, cfg.n_embd, vb.pp("wte"))?;
        let pos_emb = embedding(cfg.block_size, cfg.n_embd, vb.pp("wpe"))?;

        let mut blocks = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let vb_b = vb.pp(format!("h.{i}"));
            let head_dim = cfg.n_embd / cfg.n_head;
            blocks.push(Block {
                ln1: layer_norm(cfg.n_embd, 1e-5, vb_b.pp("ln1"))?,
                attn: CausalSelfAttention {
                    q: LoraLinear::new(cfg.n_embd, cfg.n_embd, lora_rank, lora_alpha, vb_b.pp("attn.q"))?,
                    k: linear(cfg.n_embd, cfg.n_embd, vb_b.pp("attn.k"))?,
                    v: LoraLinear::new(cfg.n_embd, cfg.n_embd, lora_rank, lora_alpha, vb_b.pp("attn.v"))?,
                    out: linear(cfg.n_embd, cfg.n_embd, vb_b.pp("attn.out"))?,
                    n_head: cfg.n_head,
                    head_dim,
                },
                ln2: layer_norm(cfg.n_embd, 1e-5, vb_b.pp("ln2"))?,
                mlp_up: LoraLinear::new(
                    cfg.n_embd,
                    cfg.n_embd * 4,
                    lora_rank,
                    lora_alpha,
                    vb_b.pp("mlp.up"),
                )?,
                mlp_down: linear(cfg.n_embd * 4, cfg.n_embd, vb_b.pp("mlp.down"))?,
            });
        }

        let ln_f = layer_norm(cfg.n_embd, 1e-5, vb.pp("ln_f"))?;
        let lm_head = linear(cfg.n_embd, cfg.vocab_size, vb.pp("lm_head"))?;

        Ok(Self {
            token_emb,
            pos_emb,
            blocks,
            ln_f,
            lm_head,
            cfg: cfg.clone(),
        })
    }

    /// `input_ids`: (batch, seq_len) of token ids. Returns logits of shape
    /// (batch, seq_len, vocab_size).
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_b, t) = input_ids.dims2()?;
        let device = input_ids.device();

        let tok = self.token_emb.forward(input_ids)?;
        let positions = Tensor::arange(0u32, t as u32, device)?;
        let pos = self.pos_emb.forward(&positions)?; // (t, n_embd)
        let mut x = tok.broadcast_add(&pos)?;

        let mask = causal_mask(t, device)?;
        for block in &self.blocks {
            x = block.forward(&x, &mask)?;
        }

        let x = self.ln_f.forward(&x)?;
        self.lm_head.forward(&x)
    }

    pub fn config(&self) -> &ScratchConfig {
        &self.cfg
    }

    /// Merges every LoRA adapter into its base projection and rebuilds a
    /// plain (adapter-free) transformer of identical shape — the "full
    /// exported model" artifact.
    pub fn merge_lora(&self) -> Result<MergedGptModel> {
        let mut merged_blocks = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            merged_blocks.push(MergedBlock {
                ln1: block.ln1.clone(),
                attn: MergedAttention {
                    q: block.attn.q.merge_into_base()?,
                    k: block.attn.k.clone(),
                    v: block.attn.v.merge_into_base()?,
                    out: block.attn.out.clone(),
                    n_head: block.attn.n_head,
                    head_dim: block.attn.head_dim,
                },
                ln2: block.ln2.clone(),
                mlp_up: block.mlp_up.merge_into_base()?,
                mlp_down: block.mlp_down.clone(),
            });
        }
        Ok(MergedGptModel {
            token_emb: self.token_emb.clone(),
            pos_emb: self.pos_emb.clone(),
            blocks: merged_blocks,
            ln_f: self.ln_f.clone(),
            lm_head: self.lm_head.clone(),
            cfg: self.cfg.clone(),
        })
    }
}

impl Block {
    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let attn_out = self.attn.forward(&self.ln1.forward(x)?, mask)?;
        let x = (x + attn_out)?;
        let mlp_hidden = self.mlp_up.forward(&self.ln2.forward(&x)?)?.gelu_erf()?;
        let mlp_out = self.mlp_down.forward(&mlp_hidden)?;
        x + mlp_out
    }
}

impl CausalSelfAttention {
    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        attention_forward(
            x,
            mask,
            self.n_head,
            self.head_dim,
            |x| self.q.forward(x),
            |x| self.k.forward(x),
            |x| self.v.forward(x),
            &self.out,
        )
    }
}

/// A merged (LoRA-free) mirror of `GptModel`, produced by `merge_lora`, used
/// only for export — every projection is a plain `Linear` so it serializes
/// as an ordinary dense checkpoint that any Candle-based loader (or a
/// converter to another runtime) can read without knowing about LoRA.
pub struct MergedGptModel {
    token_emb: Embedding,
    pos_emb: Embedding,
    blocks: Vec<MergedBlock>,
    ln_f: LayerNorm,
    lm_head: Linear,
    cfg: ScratchConfig,
}

struct MergedBlock {
    ln1: LayerNorm,
    attn: MergedAttention,
    ln2: LayerNorm,
    mlp_up: Linear,
    mlp_down: Linear,
}

struct MergedAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    n_head: usize,
    head_dim: usize,
}

impl MergedGptModel {
    pub fn config(&self) -> &ScratchConfig {
        &self.cfg
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_b, t) = input_ids.dims2()?;
        let device = input_ids.device();

        let tok = self.token_emb.forward(input_ids)?;
        let positions = Tensor::arange(0u32, t as u32, device)?;
        let pos = self.pos_emb.forward(&positions)?;
        let mut x = tok.broadcast_add(&pos)?;

        let mask = causal_mask(t, device)?;
        for block in &self.blocks {
            let attn_out = block.attn.forward(&block.ln1.forward(&x)?, &mask)?;
            x = (&x + attn_out)?;
            let mlp_hidden = block.mlp_up.forward(&block.ln2.forward(&x)?)?.gelu_erf()?;
            let mlp_out = block.mlp_down.forward(&mlp_hidden)?;
            x = (&x + mlp_out)?;
        }

        let x = self.ln_f.forward(&x)?;
        self.lm_head.forward(&x)
    }

    /// Every named tensor in this model, for safetensors export.
    pub fn named_tensors(&self) -> Vec<(String, Tensor)> {
        let mut out = vec![
            ("wte.weight".to_string(), self.token_emb.embeddings().clone()),
            ("wpe.weight".to_string(), self.pos_emb.embeddings().clone()),
            ("ln_f.weight".to_string(), self.ln_f.weight().clone()),
        ];
        if let Some(b) = self.ln_f.bias() {
            out.push(("ln_f.bias".to_string(), b.clone()));
        }
        out.push(("lm_head.weight".to_string(), self.lm_head.weight().clone()));
        if let Some(b) = self.lm_head.bias() {
            out.push(("lm_head.bias".to_string(), b.clone()));
        }

        for (i, block) in self.blocks.iter().enumerate() {
            let p = format!("h.{i}");
            out.push((format!("{p}.ln1.weight"), block.ln1.weight().clone()));
            if let Some(b) = block.ln1.bias() {
                out.push((format!("{p}.ln1.bias"), b.clone()));
            }
            out.push((format!("{p}.attn.q.weight"), block.attn.q.weight().clone()));
            out.push((format!("{p}.attn.k.weight"), block.attn.k.weight().clone()));
            out.push((format!("{p}.attn.v.weight"), block.attn.v.weight().clone()));
            out.push((format!("{p}.attn.out.weight"), block.attn.out.weight().clone()));
            if let Some(b) = block.attn.out.bias() {
                out.push((format!("{p}.attn.out.bias"), b.clone()));
            }
            out.push((format!("{p}.ln2.weight"), block.ln2.weight().clone()));
            if let Some(b) = block.ln2.bias() {
                out.push((format!("{p}.ln2.bias"), b.clone()));
            }
            out.push((format!("{p}.mlp.up.weight"), block.mlp_up.weight().clone()));
            out.push((format!("{p}.mlp.down.weight"), block.mlp_down.weight().clone()));
        }
        out
    }
}

impl MergedAttention {
    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        attention_forward(
            x,
            mask,
            self.n_head,
            self.head_dim,
            |x| self.q.forward(x),
            |x| self.k.forward(x),
            |x| self.v.forward(x),
            &self.out,
        )
    }
}

fn attention_forward(
    x: &Tensor,
    mask: &Tensor,
    n_head: usize,
    head_dim: usize,
    q_proj: impl Fn(&Tensor) -> Result<Tensor>,
    k_proj: impl Fn(&Tensor) -> Result<Tensor>,
    v_proj: impl Fn(&Tensor) -> Result<Tensor>,
    out_proj: &Linear,
) -> Result<Tensor> {
    let (b, t, c) = x.dims3()?;

    let q = q_proj(x)?
        .reshape((b, t, n_head, head_dim))?
        .transpose(1, 2)?
        .contiguous()?;
    let k = k_proj(x)?
        .reshape((b, t, n_head, head_dim))?
        .transpose(1, 2)?
        .contiguous()?;
    let v = v_proj(x)?
        .reshape((b, t, n_head, head_dim))?
        .transpose(1, 2)?
        .contiguous()?;

    let scale = 1f64 / (head_dim as f64).sqrt();
    let att = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
    let att = mask.broadcast_as(att.shape())?.where_cond(&att, &neg_inf_like(&att)?)?;
    let att = candle_nn::ops::softmax(&att, candle_core::D::Minus1)?;

    let y = att.matmul(&v)?; // (b, n_head, t, head_dim)
    let y = y.transpose(1, 2)?.contiguous()?.reshape((b, t, c))?;
    out_proj.forward(&y)
}

fn neg_inf_like(t: &Tensor) -> Result<Tensor> {
    Tensor::full(f32::NEG_INFINITY, t.shape(), t.device())?.to_dtype(t.dtype())
}

/// (1, 1, t, t) boolean mask, true where attention is allowed (j <= i).
fn causal_mask(t: usize, device: &Device) -> Result<Tensor> {
    let mask = Tensor::tril2(t, DType::U8, device)?;
    let mask = mask.reshape((1, 1, t, t))?;
    mask.ge(1u8)
}
