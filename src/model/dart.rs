// DART: full model combining patch embedding, transformer stack, and output projection.
// Paper: https://arxiv.org/abs/2410.08159, §3, §B.1

use candle_core::{Result, Tensor};
use candle_nn::{self as nn, Embedding, Module, VarBuilder};

use super::config::DartConfig;
use super::transformer::{
    build_blockwise_causal_mask, build_rope_cache_from_inv_freq, RmsNorm, TransformerBlock,
};

/// §3.2, §B.1 — The DART model for class-conditional image generation.
///
/// Architecture:
///   1. Patch embedding: project VAE latent patches into hidden_size
///   2. Class embedding (via learned embedding table -> AdaLN conditioning)
///   3. Stack of N transformer blocks with block-wise causal attention
///   4. Final norm + linear projection back to patch_dim
///
/// Input:  noisy VAE latent patches concatenated across T noise levels
/// Output: predicted clean patches (v-prediction parameterization)
pub struct DartModel {
    config: DartConfig,
    patch_embed: nn::Linear,
    class_embed: Embedding,
    blocks: Vec<TransformerBlock>,
    final_norm: RmsNorm,
    output_proj: nn::Linear,
    /// Pre-built block-wise causal mask for full training sequence.
    mask: Tensor,
    /// RoPE inverse frequencies loaded from checkpoint.
    rope_inv_freq: Tensor,
}

impl DartModel {
    pub fn new(config: DartConfig, vb: VarBuilder) -> Result<Self> {
        let device = vb.device();
        let patch_dim = config.patch_dim();

        // §B.1 — Patch embedding: project each token from patch_dim to hidden_size
        let patch_embed = nn::linear(patch_dim, config.hidden_size, vb.pp("patch_embed"))?;

        // §B.1 — Class embedding for ImageNet (1000 classes + 1 for unconditional/CFG)
        let class_embed = nn::embedding(
            config.num_classes + 1,
            config.hidden_size,
            vb.pp("class_embed"),
        )?;

        // §B.1, Table 2 — Transformer blocks
        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let block = TransformerBlock::new(&config, vb.pp(format!("blocks.{i}")))?;
            blocks.push(block);
        }

        let final_norm = RmsNorm::new(config.hidden_size, vb.pp("final_norm"))?;
        let output_proj = nn::linear(config.hidden_size, patch_dim, vb.pp("output_proj"))?;

        // Load RoPE inv_freq from checkpoint if present, otherwise compute from config
        let total_half_dims: usize = config.rope_axes_dim.iter().map(|d| d / 2).sum();
        let rope_inv_freq = match vb.get(total_half_dims, "rope.inv_freq") {
            Ok(t) => t,
            Err(_) => {
                // Compute 3D decomposed inv_freq from config
                let mut all: Vec<f32> = Vec::new();
                for &dim in &config.rope_axes_dim {
                    for i in 0..(dim / 2) {
                        all.push(1.0 / 10000f32.powf(2.0 * i as f32 / dim as f32));
                    }
                }
                Tensor::from_vec(all, total_half_dims, device)?
            }
        };

        // Pre-build the block-wise causal mask for full training sequence
        let mask = build_blockwise_causal_mask(
            config.num_steps,
            config.num_tokens,
            device,
        )?;

        Ok(Self {
            config,
            patch_embed,
            class_embed,
            blocks,
            final_norm,
            output_proj,
            mask,
            rope_inv_freq,
        })
    }

    /// Forward pass for training or full-sequence inference.
    ///
    /// # Arguments
    /// * `x` — Noisy patches concatenated across T steps: (batch, T*K, patch_dim)
    ///          where K = num_tokens and patch_dim = patch_size^2 * vae_channels.
    /// * `class_ids` — Class labels: (batch,). Use num_classes for unconditional.
    ///
    /// # Returns
    /// Predicted clean patches: (batch, T*K, patch_dim) in v-prediction space.
    pub fn forward(&self, x: &Tensor, class_ids: &Tensor) -> Result<Tensor> {
        let (_batch, seq_len, _) = x.dims3()?;

        let mut h = self.patch_embed.forward(x)?;
        let cond = self.class_embed.forward(class_ids)?;

        let mask = self.mask.unsqueeze(0)?.unsqueeze(0)?;
        let num_blocks = seq_len / self.config.num_tokens;
        let rope = build_rope_cache_from_inv_freq(
            num_blocks, self.config.num_tokens,
            &self.config.rope_axes_dim, &self.rope_inv_freq, 0, h.device(),
        )?;

        for block in &self.blocks {
            h = block.forward(&h, Some(&cond), Some(&mask), &rope)?;
        }

        let h = self.final_norm.forward(&h)?;
        self.output_proj.forward(&h)
    }

    /// Forward pass for a single denoising step during inference.
    ///
    /// During sampling, we only have the current and noisier steps (x_t:T).
    /// This performs a forward pass on the partial sequence and returns
    /// the prediction for step t-1 only.
    ///
    /// # Arguments
    /// * `x_partial` — Noisy patches for steps t..T: (batch, (T-t+1)*K, patch_dim)
    /// * `class_ids` — Class labels: (batch,)
    /// * `current_step` — Which step t we're predicting from (0-indexed from T)
    ///
    /// # Returns
    /// Predicted clean patches for step t: (batch, K, patch_dim)
    pub fn forward_step(
        &self,
        x_partial: &Tensor,
        class_ids: &Tensor,
        step_offset: usize,
    ) -> Result<Tensor> {
        let (_batch, seq_len, _) = x_partial.dims3()?;
        let k = self.config.num_tokens;
        let num_blocks = seq_len / k;

        let mut h = self.patch_embed.forward(x_partial)?;
        let cond = self.class_embed.forward(class_ids)?;

        let mask = build_blockwise_causal_mask(num_blocks, k, h.device())?;
        let mask = mask.unsqueeze(0)?.unsqueeze(0)?;

        let rope = build_rope_cache_from_inv_freq(
            num_blocks, k, &self.config.rope_axes_dim, &self.rope_inv_freq,
            step_offset, h.device(),
        )?;

        for block in &self.blocks {
            h = block.forward(&h, Some(&cond), Some(&mask), &rope)?;
        }

        let h = self.final_norm.forward(&h)?;
        let out = self.output_proj.forward(&h)?;
        out.narrow(1, 0, k)
    }

    pub fn config(&self) -> &DartConfig {
        &self.config
    }

}
