use candle_core::{Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

/// A linear layer with a frozen base weight and a trainable low-rank update,
/// `y = x W^T + b + (alpha/r) x A^T B^T`, per LoRA (Hu et al. 2022): the base
/// weight is loaded/initialized once and never receives gradients; only `a`
/// and `b` are trained. Merging (`merge_into_base`) folds the update into a
/// plain `Linear` for export, matching this crate's "full exported model"
/// requirement rather than shipping an adapter-only artifact.
#[derive(Clone)]
pub struct LoraLinear {
    base: Linear,
    a: Tensor, // (r, in_dim)
    b: Tensor, // (out_dim, r)
    scale: f64,
}

impl LoraLinear {
    /// Both the base weight and the LoRA `a`/`b` parameters come from the
    /// same `VarBuilder` (a `VarMap`-backed one, so both are trainable) —
    /// used for the from-scratch model, where there is no pretrained
    /// checkpoint to keep frozen.
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        rank: usize,
        alpha: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let base = candle_nn::linear_no_bias(in_dim, out_dim, vb.pp("base"))?;
        let (a, b) = Self::init_ab(in_dim, out_dim, rank, vb.pp("lora"))?;
        Ok(Self {
            base,
            a,
            b,
            scale: alpha / rank as f64,
        })
    }

    /// The base weight comes from a frozen, pretrained `Linear` (loaded by
    /// the caller from a mmapped checkpoint `VarBuilder` and never trained);
    /// the LoRA `a`/`b` parameters come from a separate, `VarMap`-backed
    /// `VarBuilder` so only they receive gradients. This is what makes LoRA
    /// adaptation of a real pretrained checkpoint possible: the base weight
    /// is never wrapped in a `Var`, so the optimizer never sees it.
    pub fn from_frozen_base(
        base: Linear,
        rank: usize,
        alpha: f64,
        lora_vb: VarBuilder,
    ) -> Result<Self> {
        let in_dim = base.weight().dim(1)?;
        let out_dim = base.weight().dim(0)?;
        let (a, b) = Self::init_ab(in_dim, out_dim, rank, lora_vb)?;
        Ok(Self {
            base,
            a,
            b,
            scale: alpha / rank as f64,
        })
    }

    fn init_ab(in_dim: usize, out_dim: usize, rank: usize, vb: VarBuilder) -> Result<(Tensor, Tensor)> {
        // LoRA convention: A initialized small-random, B initialized to zero,
        // so the adapter starts as a no-op and training moves it from there.
        let a = vb.get_with_hints(
            (rank, in_dim),
            "a",
            candle_nn::Init::Randn {
                mean: 0.,
                stdev: 0.02,
            },
        )?;
        let b = vb.get_with_hints((out_dim, rank), "b", candle_nn::Init::Const(0.))?;
        Ok((a, b))
    }

    /// Folds the LoRA update into the base weight and returns a plain
    /// `Linear`, for use when exporting a merged, standalone model.
    pub fn merge_into_base(&self) -> Result<Linear> {
        let delta = (self.b.matmul(&self.a)? * self.scale)?; // (out_dim, in_dim)
        let merged = (self.base.weight() + delta)?;
        Ok(Linear::new(merged, self.base.bias().cloned()))
    }

    pub fn lora_vars(&self) -> Vec<&Tensor> {
        vec![&self.a, &self.b]
    }
}

impl Module for LoraLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let base_out = self.base.forward(x)?;
        let lora_out = x.broadcast_matmul(&self.a.t()?)?.broadcast_matmul(&self.b.t()?)?;
        base_out + (lora_out * self.scale)?
    }
}
