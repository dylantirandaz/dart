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

    /// §3.2 — Run the full DART sampling loop.
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

        // Step 1: Initialize x_T ~ N(0, I)
        // We store all T noise levels. Index 0 = step T (noisiest), index T-1 = step 1.
        let mut x_levels: Vec<Tensor> = Vec::with_capacity(t_steps);
        for _ in 0..t_steps {
            let noise = Tensor::randn(0f32, 1f32, (batch, k, patch_dim), device)?;
            x_levels.push(noise);
        }

        // Step 2: Denoise from t=T down to t=1
        // At iteration i, we denoise step (T - i) using steps (T-i)..T
        for i in 0..t_steps {
            let current_step = t_steps - i; // t = T, T-1, ..., 1
            let target_step = current_step - 1; // t-1 = T-1, T-2, ..., 0

            // Gather x_{t:T} — that's x_levels[i..] in our storage
            // (from current noisiest remaining to the initial T)
            let partial_levels: Vec<&Tensor> = x_levels[i..].iter().collect();
            let x_partial = Tensor::cat(&partial_levels, 1)?; // (B, (T-i)*K, patch_dim)

            // Run model to get prediction for the cleanest chunk
            let pred = if cfg_scale > 1.0 + 1e-6 {
                // Classifier-free guidance: run conditional and unconditional
                let cond_pred = model.forward_step(&x_partial, class_ids, i)?;

                let uncond_ids = Tensor::full(
                    config.num_classes as u32,
                    class_ids.shape(),
                    device,
                )?;
                let uncond_pred = model.forward_step(&x_partial, &uncond_ids, i)?;

                // §4.1 — CFG: pred = uncond + scale * (cond - uncond)
                let diff = (&cond_pred - &uncond_pred)?;
                (uncond_pred + diff * cfg_scale)?
            } else {
                model.forward_step(&x_partial, class_ids, i)?
            };
            // pred: (B, K, patch_dim) — the model's x̂_0 prediction

            // §B.3 — v-prediction: convert model output to x̂_0
            // x̂_0 = α_t · x_t - σ_t · v̂_t
            let alpha = self.schedule.alpha_bar[current_step] as f32;
            let sigma = self.schedule.sqrt_one_minus_gamma[current_step] as f32;
            let x_t_chunk = &x_levels[i].narrow(1, 0, k)?;
            let x0_pred = ((x_t_chunk * alpha as f64)? - (pred * sigma as f64)?)?;

            if target_step == 0 {
                // Final step: x_0 = x̂_0 (no noise added)
                x_levels[i] = x0_pred;
            } else {
                // §3.2 — Sample x_{t-1} = √γ_{t-1} · x̂_0 + √(1-γ_{t-1}) · ε
                let (sqrt_g, sqrt_1mg) = self.schedule.add_noise_params(target_step);
                let noise = Tensor::randn(0f32, 1f32, (batch, k, patch_dim), device)?;
                let x_prev =
                    ((x0_pred * sqrt_g)? + (noise * sqrt_1mg)?)?;

                // Store the denoised result. We insert it before the current position
                // so that x_levels[i] now holds x_{t-1}
                x_levels[i] = x_prev;
            }
        }

        // Return x_0 — stored in x_levels[0] after the final iteration
        Ok(x_levels[0].clone())
    }

    pub fn schedule(&self) -> &CosineSchedule {
        &self.schedule
    }
}
