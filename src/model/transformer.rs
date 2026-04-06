// DART transformer blocks: attention with RoPE, SwiGLU FFN, AdaLN.
// Paper: https://arxiv.org/abs/2410.08159, §B.1

use candle_core::{Device, Result, Tensor};
use candle_nn::{self as nn, Module, VarBuilder};

use super::config::DartConfig;

// ---------------------------------------------------------------------------
// RMSNorm (per-head variant used in §B.1)
// ---------------------------------------------------------------------------

pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps: 1e-6 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let variance = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let x_norm = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        x_norm.broadcast_mul(&self.weight)
    }
}

// ---------------------------------------------------------------------------
// Rotary Position Embeddings (§B.1: axes [16, 24, 24])
// ---------------------------------------------------------------------------

/// §B.1 — 3-axis decomposed RoPE: (denoising_step, spatial_h, spatial_w).
///
/// Each token's position is decomposed into 3 coordinates. Each axis gets
/// independent sinusoidal frequencies, then they're concatenated.
pub fn build_rope_cache_3d(
    num_steps: usize,
    num_tokens: usize,
    axes_dim: &[usize; 3],
    step_offset: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let grid_size = (num_tokens as f64).sqrt() as usize;
    let seq_len = num_steps * num_tokens;

    // Build per-axis inv_freq and positions
    let mut all_raw_freqs: Vec<Tensor> = Vec::new();

    for (axis_idx, &dim) in axes_dim.iter().enumerate() {
        let half = dim / 2;
        let theta: Vec<f32> = (0..half)
            .map(|i| 1.0 / 10000f32.powf(2.0 * i as f32 / dim as f32))
            .collect();
        let theta = Tensor::from_vec(theta, half, device)?;

        // Build position vector for this axis
        let positions: Vec<f32> = (0..seq_len)
            .map(|p| {
                let block = p / num_tokens;
                let token = p % num_tokens;
                match axis_idx {
                    0 => (block + step_offset) as f32,  // denoising step
                    1 => (token / grid_size) as f32,     // spatial row
                    _ => (token % grid_size) as f32,     // spatial col
                }
            })
            .collect();
        let positions = Tensor::from_vec(positions, seq_len, device)?;

        let freqs = positions.unsqueeze(1)?.broadcast_mul(&theta.unsqueeze(0)?)?;
        all_raw_freqs.push(freqs);
    }

    // Concatenate raw freqs across axes: (S, 32), then double: (S, 64)
    let raw_refs: Vec<&Tensor> = all_raw_freqs.iter().collect();
    let raw = Tensor::cat(&raw_refs, 1)?;

    let cos = raw.cos()?;
    let sin = raw.sin()?;
    let cos = Tensor::cat(&[&cos, &cos], 1)?;
    let sin = Tensor::cat(&[&sin, &sin], 1)?;

    Ok((cos, sin))
}

/// Apply RoPE to query/key tensors.
/// x shape: (batch, heads, seq_len, head_dim)
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_, _, seq_len, head_dim) = x.dims4()?;
    let half = head_dim / 2;

    let cos = cos.narrow(0, 0, seq_len)?.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.narrow(0, 0, seq_len)?.unsqueeze(0)?.unsqueeze(0)?;
    let cos = cos.narrow(3, 0, head_dim)?;
    let sin = sin.narrow(3, 0, head_dim)?;

    // Rotate: [x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin]
    let x1 = x.narrow(3, 0, half)?;
    let x2 = x.narrow(3, half, half)?;
    let cos1 = cos.narrow(3, 0, half)?;
    let sin1 = sin.narrow(3, 0, half)?;

    let rotated_x1 = (x1.broadcast_mul(&cos1)? - x2.broadcast_mul(&sin1)?)?;
    let rotated_x2 = (x2.broadcast_mul(&cos1)? + x1.broadcast_mul(&sin1)?)?;

    Tensor::cat(&[&rotated_x1, &rotated_x2], 3)
}

// ---------------------------------------------------------------------------
// Adaptive Layer Norm (§B.1: used for class conditioning in C2I)
// ---------------------------------------------------------------------------

pub struct AdaLN {
    linear: nn::Linear,
}

impl AdaLN {
    /// Projects conditioning vector to 6 * hidden_size parameters:
    /// (shift1, scale1, gate1, shift2, scale2, gate2) for pre/post attention.
    pub fn new(hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let linear = nn::linear(hidden_size, 6 * hidden_size, vb.pp("linear"))?;
        Ok(Self { linear })
    }

    /// Returns (shift1, scale1, gate1, shift2, scale2, gate2) each of shape (B, 1, D).
    pub fn forward(&self, cond: &Tensor) -> Result<[Tensor; 6]> {
        let out = self.linear.forward(cond)?; // (B, 6*D)
        let d = out.dim(1)? / 6;
        let chunks: Vec<Tensor> = (0..6)
            .map(|i| out.narrow(1, i * d, d).and_then(|t| t.unsqueeze(1)))
            .collect::<Result<_>>()?;
        Ok([
            chunks[0].clone(),
            chunks[1].clone(),
            chunks[2].clone(),
            chunks[3].clone(),
            chunks[4].clone(),
            chunks[5].clone(),
        ])
    }
}

fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    x.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift)
}

// ---------------------------------------------------------------------------
// Multi-Head Self-Attention with RoPE and block-wise causal mask
// ---------------------------------------------------------------------------

pub struct Attention {
    qkv: nn::Linear,
    out_proj: nn::Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    head_dim: usize,
}

impl Attention {
    pub fn new(config: &DartConfig, vb: VarBuilder) -> Result<Self> {
        let total_dim = config.num_heads * config.head_dim;
        let qkv = nn::linear_no_bias(config.hidden_size, 3 * total_dim, vb.pp("qkv"))?;
        let out_proj = nn::linear_no_bias(total_dim, config.hidden_size, vb.pp("out_proj"))?;

        // §B.1 — "per-head RMSNorm applied"
        let q_norm = RmsNorm::new(config.head_dim, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(config.head_dim, vb.pp("k_norm"))?;

        Ok(Self {
            qkv,
            out_proj,
            q_norm,
            k_norm,
            num_heads: config.num_heads,
            head_dim: config.head_dim,
        })
    }

    /// Forward with block-wise causal mask and RoPE.
    /// x: (batch, seq_len, hidden_size)
    /// mask: optional attention mask (seq_len, seq_len) — 0 = attend, -inf = block
    /// rope: (cos, sin) each (max_seq, head_dim)
    pub fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        rope: &(Tensor, Tensor),
    ) -> Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;
        let total_dim = self.num_heads * self.head_dim;

        // QKV projection
        let qkv = self.qkv.forward(x)?; // (B, S, 3*total_dim)
        let q = qkv.narrow(2, 0, total_dim)?;
        let k = qkv.narrow(2, total_dim, total_dim)?;
        let v = qkv.narrow(2, 2 * total_dim, total_dim)?;

        // Reshape to (B, heads, S, head_dim)
        let reshape = |t: Tensor| -> Result<Tensor> {
            t.reshape((b, seq_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)
        };
        let mut q = reshape(q)?;
        let mut k = reshape(k)?;
        let v = reshape(v)?;

        // §B.1 — per-head RMSNorm on Q, K
        // Apply to each head independently: reshape to (B*heads*S, head_dim)
        let flat_shape = (b * self.num_heads * seq_len, self.head_dim);
        q = self
            .q_norm
            .forward(&q.reshape(flat_shape)?)?
            .reshape((b, self.num_heads, seq_len, self.head_dim))?;
        k = self
            .k_norm
            .forward(&k.reshape(flat_shape)?)?
            .reshape((b, self.num_heads, seq_len, self.head_dim))?;

        // §B.1 — Apply RoPE
        let (ref cos, ref sin) = *rope;
        q = apply_rope(&q, cos, sin)?;
        k = apply_rope(&k, cos, sin)?;

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let attn_weights = q.matmul(&k.transpose(2, 3)?)? / scale; // (B, H, S, S)

        // Apply block-wise causal mask if provided
        let attn_weights = attn_weights?;
        let attn_weights = if let Some(m) = mask {
            attn_weights.broadcast_add(m)?
        } else {
            attn_weights
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_out = attn_weights.matmul(&v)?; // (B, H, S, head_dim)

        // Reshape back to (B, S, total_dim)
        let attn_out = attn_out
            .transpose(1, 2)?
            .reshape((b, seq_len, total_dim))?;

        self.out_proj.forward(&attn_out)
    }
}

// ---------------------------------------------------------------------------
// SwiGLU Feed-Forward Network (§B.1)
// ---------------------------------------------------------------------------

pub struct SwiGluFfn {
    w1: nn::Linear,
    w2: nn::Linear,
    w3: nn::Linear,
}

impl SwiGluFfn {
    pub fn new(config: &DartConfig, vb: VarBuilder) -> Result<Self> {
        let ffn_dim = config.ffn_hidden_size();
        let w1 = nn::linear_no_bias(config.hidden_size, ffn_dim, vb.pp("w1"))?;
        let w2 = nn::linear_no_bias(ffn_dim, config.hidden_size, vb.pp("w2"))?;
        let w3 = nn::linear_no_bias(config.hidden_size, ffn_dim, vb.pp("w3"))?;
        Ok(Self { w1, w2, w3 })
    }

    /// SwiGLU: out = W2(silu(W1(x)) * W3(x))
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.w1.forward(x)?.silu()?;
        let up = self.w3.forward(x)?;
        self.w2.forward(&(gate * up)?)
    }
}

// ---------------------------------------------------------------------------
// Transformer Block: pre-norm with AdaLN modulation
// ---------------------------------------------------------------------------

pub struct TransformerBlock {
    attn: Attention,
    ffn: SwiGluFfn,
    norm1: RmsNorm,
    norm2: RmsNorm,
    adaln: Option<AdaLN>,
}

impl TransformerBlock {
    pub fn new(config: &DartConfig, vb: VarBuilder) -> Result<Self> {
        let attn = Attention::new(config, vb.pp("attn"))?;
        let ffn = SwiGluFfn::new(config, vb.pp("ffn"))?;
        let norm1 = RmsNorm::new(config.hidden_size, vb.pp("norm1"))?;
        let norm2 = RmsNorm::new(config.hidden_size, vb.pp("norm2"))?;

        let adaln = if config.use_adaln {
            Some(AdaLN::new(config.hidden_size, vb.pp("adaln"))?)
        } else {
            None
        };

        Ok(Self {
            attn,
            ffn,
            norm1,
            norm2,
            adaln,
        })
    }

    /// Forward pass with optional AdaLN conditioning.
    /// x: (B, S, D), cond: (B, D) conditioning vector, mask: optional attention mask.
    pub fn forward(
        &self,
        x: &Tensor,
        cond: Option<&Tensor>,
        mask: Option<&Tensor>,
        rope: &(Tensor, Tensor),
    ) -> Result<Tensor> {
        if let (Some(adaln), Some(c)) = (&self.adaln, cond) {
            // §B.1 — AdaLN modulation for class-conditional generation
            let [shift1, scale1, gate1, shift2, scale2, gate2] = adaln.forward(c)?;

            // Pre-norm attention with modulation
            let h = modulate(&self.norm1.forward(x)?, &shift1, &scale1)?;
            let h = self.attn.forward(&h, mask, rope)?;
            let x = (x + h.broadcast_mul(&gate1)?)?;

            // Pre-norm FFN with modulation
            let h = modulate(&self.norm2.forward(&x)?, &shift2, &scale2)?;
            let h = self.ffn.forward(&h)?;
            x + h.broadcast_mul(&gate2)?
        } else {
            // Standard pre-norm without AdaLN
            let h = self.attn.forward(&self.norm1.forward(x)?, mask, rope)?;
            let x = (x + h)?;
            let h = self.ffn.forward(&self.norm2.forward(&x)?)?;
            x + h
        }
    }
}

// ---------------------------------------------------------------------------
// Block-wise causal mask for DART's non-Markovian attention
// ---------------------------------------------------------------------------

/// §3.2 — Build a block-wise causal attention mask.
///
/// DART processes T noise levels, each with K tokens, concatenated into one sequence.
/// The mask ensures:
///   - Tokens at step t can attend to all tokens at steps t..T (their own and noisier)
///   - Tokens at step t CANNOT attend to steps 1..t-1 (cleaner future predictions)
///
/// This is "block-wise causal": causal at the block level, full attention within blocks.
///
/// Returns: (total_seq, total_seq) tensor with 0.0 for allowed and -inf for blocked.
pub fn build_blockwise_causal_mask(
    num_steps: usize,
    num_tokens: usize,
    device: &Device,
) -> Result<Tensor> {
    let total = num_steps * num_tokens;
    let mut mask_data = vec![0.0f32; total * total];

    for q_step in 0..num_steps {
        for k_step in 0..num_steps {
            if k_step < q_step {
                // Block q_step from attending to earlier (cleaner) steps
                for qi in 0..num_tokens {
                    for ki in 0..num_tokens {
                        let q_idx = q_step * num_tokens + qi;
                        let k_idx = k_step * num_tokens + ki;
                        mask_data[q_idx * total + k_idx] = f32::NEG_INFINITY;
                    }
                }
            }
            // Steps >= q_step: full bidirectional attention within and across blocks
        }
    }

    Tensor::from_vec(mask_data, (total, total), device)
}
