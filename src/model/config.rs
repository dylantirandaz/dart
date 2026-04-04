// DART model configurations from Table 2 (§B.1)
// Paper: https://arxiv.org/abs/2410.08159

use serde::{Deserialize, Serialize};

/// §B.1, Table 2 — DART model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DartConfig {
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Hidden dimension of the transformer.
    pub hidden_size: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Channels per attention head (§B.1: 64).
    pub head_dim: usize,
    /// Total model parameters (approximate, for reference).
    pub num_params_m: f64,
    /// VAE latent channels (SD v1.4 VAE: 4 latent channels).
    pub vae_channels: usize,
    /// Patch size for patchifying VAE latents (§B.1: 2).
    pub patch_size: usize,
    /// Number of denoising steps T (§4.1: default 16).
    pub num_steps: usize,
    /// Number of image tokens K per step (§B.1: 256 for 256x256).
    pub num_tokens: usize,
    /// Number of classes for class-conditional generation (ImageNet: 1000).
    pub num_classes: usize,
    /// Whether to use AdaLN for class conditioning (§B.1: true for C2I).
    pub use_adaln: bool,
    /// Whether to use SwiGLU FFN (§B.1: true).
    pub use_swiglu: bool,
    /// RoPE axis dimensions (§B.1: [16, 24, 24]).
    pub rope_axes_dim: [usize; 3],
    /// Classifier-free guidance scale for sampling.
    pub cfg_scale: f64,
}

impl DartConfig {
    /// §B.1, Table 2 — DART-S: 48M parameters.
    pub fn small() -> Self {
        Self {
            num_layers: 12,
            hidden_size: 384,
            num_heads: 6,
            head_dim: 64,
            num_params_m: 48.0,
            vae_channels: 4,
            patch_size: 2,
            num_steps: 16,
            num_tokens: 256,
            num_classes: 1000,
            use_adaln: true,
            use_swiglu: true,
            rope_axes_dim: [16, 24, 24],
            cfg_scale: 1.5,
        }
    }

    /// §B.1, Table 2 — DART-B: 141M parameters.
    pub fn base() -> Self {
        Self {
            num_layers: 12,
            hidden_size: 768,
            num_heads: 12,
            head_dim: 64,
            num_params_m: 141.0,
            ..Self::small()
        }
    }

    /// §B.1, Table 2 — DART-L: 464M parameters.
    pub fn large() -> Self {
        Self {
            num_layers: 24,
            hidden_size: 1024,
            num_heads: 16,
            head_dim: 64,
            num_params_m: 464.0,
            ..Self::small()
        }
    }

    /// §B.1, Table 2 — DART-XL: 812M parameters (class-conditional with AdaLN).
    pub fn xlarge() -> Self {
        Self {
            num_layers: 28,
            hidden_size: 1152,
            num_heads: 18,
            head_dim: 64,
            num_params_m: 812.0,
            ..Self::small()
        }
    }

    /// Dimension of each patch token = patch_size^2 * vae_channels.
    pub fn patch_dim(&self) -> usize {
        self.patch_size * self.patch_size * self.vae_channels
    }

    /// Total sequence length during training = num_steps * num_tokens.
    pub fn total_seq_len(&self) -> usize {
        self.num_steps * self.num_tokens
    }

    /// FFN intermediate size. SwiGLU uses 8/3 * hidden (rounded to nearest 256).
    pub fn ffn_hidden_size(&self) -> usize {
        let raw = (self.hidden_size * 8) / 3;
        // Round up to nearest multiple of 256 for efficiency
        ((raw + 255) / 256) * 256
    }
}
