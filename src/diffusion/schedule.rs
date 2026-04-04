// Cosine noise schedule for DART's non-Markovian diffusion.
// Paper: https://arxiv.org/abs/2410.08159, §3.1, §B.3

/// §B.3 — Cosine noise schedule.
///
/// DART uses a cosine schedule for the signal rate γ_t:
///   ᾱ_t = cos(π/2 · t/T)
///
/// This defines:
///   - γ_t = ᾱ_t²  (signal variance)
///   - σ_t² = 1 - γ_t (noise variance)
///   - SNR_t = γ_t / (1 - γ_t)
///
/// The schedule is non-Markovian: each x_t is independently noised from x_0:
///   x_t = √γ_t · x_0 + √(1 - γ_t) · ε,  ε ~ N(0, I)
pub struct CosineSchedule {
    /// Number of denoising steps T (§4.1: default 16).
    pub num_steps: usize,
    /// Signal rate γ_t for each step t = 0..T.
    /// gamma[0] = 1.0 (clean), gamma[T] ≈ 0.0 (pure noise).
    pub gamma: Vec<f64>,
    /// √γ_t — scale factor for signal component.
    pub sqrt_gamma: Vec<f64>,
    /// √(1 - γ_t) — scale factor for noise component.
    pub sqrt_one_minus_gamma: Vec<f64>,
    /// α_t values from cosine schedule.
    pub alpha_bar: Vec<f64>,
}

impl CosineSchedule {
    pub fn new(num_steps: usize) -> Self {
        let mut gamma = Vec::with_capacity(num_steps + 1);
        let mut sqrt_gamma = Vec::with_capacity(num_steps + 1);
        let mut sqrt_one_minus_gamma = Vec::with_capacity(num_steps + 1);
        let mut alpha_bar = Vec::with_capacity(num_steps + 1);

        for i in 0..=num_steps {
            // §B.3 — ᾱ_t = cos(π/2 · t/T), where t=0 is clean and t=T is noise
            let t_frac = i as f64 / num_steps as f64;
            let alpha = (std::f64::consts::FRAC_PI_2 * t_frac).cos();
            let g = alpha * alpha;

            alpha_bar.push(alpha);
            gamma.push(g);
            sqrt_gamma.push(g.sqrt());
            sqrt_one_minus_gamma.push((1.0 - g).sqrt());
        }

        Self {
            num_steps,
            gamma,
            sqrt_gamma,
            sqrt_one_minus_gamma,
            alpha_bar,
        }
    }

    /// §3.1 — Add noise to clean data x_0 at step t.
    /// x_t = √γ_t · x_0 + √(1 - γ_t) · ε
    pub fn add_noise_params(&self, step: usize) -> (f64, f64) {
        (self.sqrt_gamma[step], self.sqrt_one_minus_gamma[step])
    }

    /// §B.3 — v-prediction: convert between model output v and x_0 prediction.
    /// v_t = α_t · x_0 - σ_t · x_t
    /// x_0 = α_t · x_t + σ_t · v_t  (rearranged: actually x_0 = (x_t + σ_t · v_t) / α_t ... hmm)
    ///
    /// More precisely with the paper's convention:
    /// v_t = α_t · ε - σ_t · x_0  (Salimans & Ho 2022 convention varies)
    /// x̂_0 = α_t · x_t - σ_t · v̂_t
    pub fn v_to_x0(&self, v: &[f64], x_t: &[f64], step: usize) -> Vec<f64> {
        let alpha = self.alpha_bar[step];
        let sigma = self.sqrt_one_minus_gamma[step];
        v.iter()
            .zip(x_t.iter())
            .map(|(&vi, &xi)| alpha * xi - sigma * vi)
            .collect()
    }

    /// Loss weight ω_t from §3.2:
    /// ω_t = 1 - γ_t / γ_{t-1}  (for the simplified DART objective)
    pub fn loss_weight(&self, step: usize) -> f64 {
        if step == 0 {
            return 1.0;
        }
        1.0 - self.gamma[step] / self.gamma[step - 1]
    }

    /// SNR-based weight from §3.1:
    /// ω_t = √(Σ_{τ=t}^{T} (1 - γ_τ) / γ_τ)
    pub fn snr_weight(&self, step: usize) -> f64 {
        let sum: f64 = (step..=self.num_steps)
            .map(|tau| (1.0 - self.gamma[tau]) / self.gamma[tau].max(1e-10))
            .sum();
        sum.sqrt()
    }
}
