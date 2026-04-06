// DART non-Markovian sampling loop.
// Paper: https://arxiv.org/abs/2410.08159, §3.2

use candle_core::{Device, Result, Tensor};

use crate::model::DartModel;
use super::schedule::CosineSchedule;

/// §3.2 — Non-Markovian sampling for DART.
///
/// Unlike standard diffusion which only conditions on x_t,
/// DART conditions on the entire trajectory x_{t:T}:
///
///   p_θ(x_{t-1} | x_{t:T}) = N(√γ_{t-1} · x̂_0(x_{t:T}), (1-γ_{t-1})I)
///
/// Sampling procedure:
///   1. Initialize x_T ~ N(0, I)
///   2. For t = T down to 1:
///      a. Concatenate x_{t:T} into one sequence
///      b. Run transformer to predict x̂_0
///      c. Sample x_{t-1} = √γ_{t-1} · x̂_0 + √(1-γ_{t-1}) · ε
///   3. Return x_0
pub struct DartSampler {
    schedule: CosineSchedule,
}

impl DartSampler {
    pub fn new(num_steps: usize) -> Self {
        Self {
            schedule: CosineSchedule::new(num_steps),
        }
    }

    /// §3.2, Algorithm 2 — Run the full DART sampling loop.
    ///
    /// Uses growing sequences: start with x_T, iteratively sample x_{T-1}..x_0.
    /// Each step conditions on all previously denoised blocks (non-Markovian).
    ///
    /// # Arguments
    /// * `model` — The DART transformer model.
    /// * `class_ids` — Class labels (batch,). Use model.config().num_classes for unconditional.
    /// * `cfg_scale` — Classifier-free guidance scale (§4.1: typically 1.5).
    /// * `device` — Computation device.
    ///
    /// # Returns
    /// Clean predicted patches: (batch, K, patch_dim)
    pub fn sample(
        &self,
        model: &DartModel,
        class_ids: &Tensor,
        cfg_scale: f64,
        device: &Device,
    ) -> Result<Tensor> {
        let config = model.config();
        let batch = class_ids.dims1()?;
        let k = config.num_tokens;
        let patch_dim = config.patch_dim();
        let t_steps = config.num_steps;

        // Initialize x_T ~ N(0, I)
        let x_t = Tensor::randn(0f32, 1f32, (batch, k, patch_dim), device)?;

        // Growing sequence: [x_t, x_{t+1}, ..., x_T]
        // Position 0 = current step (cleanest), last = x_T (noisiest)
        let mut blocks: Vec<Tensor> = vec![x_t];

        let mut x0_pred = blocks[0].clone(); // will be overwritten

        for step_idx in 0..t_steps {
            let t = t_steps - step_idx; // T, T-1, ..., 1

            // Concatenate growing sequence
            let block_refs: Vec<&Tensor> = blocks.iter().collect();
            let x_seq = Tensor::cat(&block_refs, 1)?;

            // step_offset aligns RoPE positions with training layout
            let step_offset = t - 1;
            let pred = if cfg_scale > 1.0 + 1e-6 {
                let cond_pred = model.forward_step(&x_seq, class_ids, step_offset)?;
                let uncond_ids = Tensor::full(
                    config.num_classes as u32,
                    class_ids.shape(),
                    device,
                )?;
                let uncond_pred = model.forward_step(&x_seq, &uncond_ids, step_offset)?;
                let diff = (&cond_pred - &uncond_pred)?;
                (uncond_pred + diff * cfg_scale)?
            } else {
                model.forward_step(&x_seq, class_ids, step_offset)?
            };

            // v-prediction → x̂_0: x̂_0 = α_t · x_t − σ_t · v̂_t
            let alpha = self.schedule.alpha_bar[t];
            let sigma = self.schedule.sqrt_one_minus_gamma[t];
            x0_pred = ((&blocks[0] * alpha)? - (pred * sigma)?)?;

            if t == 1 {
                break; // Final step: return x̂_0
            }

            // Sample x_{t-1} = √γ_{t-1} · x̂_0 + √(1-γ_{t-1}) · ε
            let target = t - 1;
            let (sqrt_g, sqrt_1mg) = self.schedule.add_noise_params(target);
            let noise = Tensor::randn(0f32, 1f32, (batch, k, patch_dim), device)?;
            let x_prev = ((x0_pred.clone() * sqrt_g)? + (noise * sqrt_1mg)?)?;

            // Prepend x_{t-1} to form [x_{t-1}, x_t, ..., x_T]
            blocks.insert(0, x_prev);
        }

        Ok(x0_pred)
    }

    pub fn schedule(&self) -> &CosineSchedule {
        &self.schedule
    }
}
