"""
DART model definitions and non-Markovian sampling (PyTorch).

Paper: https://arxiv.org/abs/2410.08159
Architecture details: §3, §B.1

Used by train_cloud.py for Modal cloud training. Defines the model,
cosine schedule, block-wise causal mask, sampling loop, and patch utils.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image


# ---------------------------------------------------------------------------
# Model configs from Table 2
# ---------------------------------------------------------------------------

CONFIGS = {
    "small":  dict(num_layers=12, hidden_size=384,  num_heads=6,  head_dim=64),
    "base":   dict(num_layers=12, hidden_size=768,  num_heads=12, head_dim=64),
    "large":  dict(num_layers=24, hidden_size=1024, num_heads=16, head_dim=64),
    "xlarge": dict(num_layers=28, hidden_size=1152, num_heads=18, head_dim=64),
}

PATCH_SIZE = 2
VAE_CHANNELS = 4
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * VAE_CHANNELS  # 16
GRID_SIZE = 16  # spatial grid = 16x16 patches for 256x256 images
NUM_TOKENS = GRID_SIZE * GRID_SIZE  # 256 tokens per denoising step
ROPE_AXES_DIM = (16, 24, 24)  # §B.1: (denoising_step, spatial_h, spatial_w)


# ---------------------------------------------------------------------------
# Cosine noise schedule (§B.3)
# ---------------------------------------------------------------------------

class CosineSchedule:
    def __init__(self, num_steps=16):
        self.num_steps = num_steps
        t = torch.linspace(0, 1, num_steps + 1)
        self.alpha_bar = torch.cos(math.pi / 2 * t)
        self.gamma = self.alpha_bar ** 2
        self.sqrt_gamma = self.gamma.sqrt()
        self.sqrt_one_minus_gamma = (1 - self.gamma).sqrt()

        # §3.2 Eq.7 — SNR-based per-step loss weights: ω_t = ∑(τ=t→T) γ_τ/(1-γ_τ)
        snr = self.gamma / (1 - self.gamma).clamp(min=1e-8)
        self.omega = torch.zeros(num_steps + 1)
        for step in range(1, num_steps + 1):
            self.omega[step] = snr[step:num_steps + 1].sum()

    def add_noise(self, x0, t_idx, noise=None):
        """x_t = sqrt(gamma_t) * x_0 + sqrt(1 - gamma_t) * eps"""
        if noise is None:
            noise = torch.randn_like(x0)
        sg = self.sqrt_gamma[t_idx]
        sng = self.sqrt_one_minus_gamma[t_idx]
        if x0.dim() == 3:
            sg = sg.view(-1, 1, 1)
            sng = sng.view(-1, 1, 1)
        return sg.to(x0.device) * x0 + sng.to(x0.device) * noise


# ---------------------------------------------------------------------------
# Transformer components
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, ffn_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class AdaLN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, 6 * dim)

    def forward(self, c):
        return self.linear(c).chunk(6, dim=-1)


class RotaryEmbedding3D(nn.Module):
    """3-axis decomposed RoPE (§B.1): (denoising_step, spatial_h, spatial_w).

    Each token's position is decomposed into 3 coordinates, and each axis
    gets independent sinusoidal frequencies. This lets the model distinguish
    spatial vs temporal positions within the denoising sequence.
    """
    def __init__(self, axes_dim=ROPE_AXES_DIM):
        super().__init__()
        self.axes_dim = axes_dim
        self.axis_half_dims = [d // 2 for d in axes_dim]
        # Not a buffer — prevents checkpoints from overwriting with stale values
        inv_freqs = []
        for dim in axes_dim:
            inv_freqs.append(1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim)))
        self.inv_freq = torch.cat(inv_freqs)

    def forward(self, num_steps, num_tokens, device, step_offset=0):
        """Build 3D RoPE frequencies for a sequence of num_steps * num_tokens tokens.

        Args:
            num_steps: number of denoising step blocks in the sequence
            num_tokens: tokens per step (256)
            device: computation device
            step_offset: starting step index (for sampling partial sequences)
        """
        grid_size = int(num_tokens ** 0.5)

        # 3D positions for each token
        step_pos = (torch.arange(num_steps, device=device) + step_offset
                    ).repeat_interleave(num_tokens).float()
        spatial = torch.arange(num_tokens, device=device)
        row_pos = (spatial // grid_size).repeat(num_steps).float()
        col_pos = (spatial % grid_size).repeat(num_steps).float()

        # Per-axis frequencies, concatenated
        positions = [step_pos, row_pos, col_pos]
        freqs_parts = []
        offset = 0
        for pos, half_dim in zip(positions, self.axis_half_dims):
            axis_inv_freq = self.inv_freq[offset:offset + half_dim].to(device)
            freqs_parts.append(torch.outer(pos, axis_inv_freq))
            offset += half_dim

        raw_freqs = torch.cat(freqs_parts, dim=-1)  # (S, 32)
        return torch.cat([raw_freqs, raw_freqs], dim=-1)  # (S, 64)


def apply_rope(x, freqs):
    d = x.shape[-1] // 2
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos[..., :d] - x2 * sin[..., :d],
                      x2 * cos[..., :d] + x1 * sin[..., :d]], dim=-1)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, head_dim):
        super().__init__()
        total = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(dim, 3 * total, bias=False)
        self.out_proj = nn.Linear(total, dim, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, x, mask, rope_freqs):
        B, S, _ = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        q = apply_rope(q, rope_freqs)
        k = apply_rope(k, rope_freqs)

        # Use flash attention when available, fall back to manual
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=self.head_dim ** -0.5
        )
        out = out.transpose(1, 2).reshape(B, S, -1)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, head_dim, use_adaln=True):
        super().__init__()
        ffn_dim = ((dim * 8 // 3 + 255) // 256) * 256
        self.attn = Attention(dim, num_heads, head_dim)
        self.ffn = SwiGLU(dim, ffn_dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.adaln = AdaLN(dim) if use_adaln else None

    def forward(self, x, cond, mask, rope_freqs):
        if self.adaln is not None and cond is not None:
            s1, sc1, g1, s2, sc2, g2 = self.adaln(cond)
            s1, sc1, g1 = [t.unsqueeze(1) for t in [s1, sc1, g1]]
            s2, sc2, g2 = [t.unsqueeze(1) for t in [s2, sc2, g2]]
            h = self.norm1(x) * (1 + sc1) + s1
            h = self.attn(h, mask, rope_freqs)
            x = x + h * g1
            h = self.norm2(x) * (1 + sc2) + s2
            h = self.ffn(h)
            x = x + h * g2
        else:
            x = x + self.attn(self.norm1(x), mask, rope_freqs)
            x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# DART Model
# ---------------------------------------------------------------------------

class DartModel(nn.Module):
    def __init__(self, cfg, num_steps=16, num_classes=1000):
        super().__init__()
        dim = cfg["hidden_size"]
        self.patch_embed = nn.Linear(PATCH_DIM, dim)
        self.class_embed = nn.Embedding(num_classes + 1, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, cfg["num_heads"], cfg["head_dim"])
            for _ in range(cfg["num_layers"])
        ])
        self.final_norm = RMSNorm(dim)
        self.output_proj = nn.Linear(dim, PATCH_DIM)
        self.rope = RotaryEmbedding3D()
        self.num_steps = num_steps

    def forward(self, x, class_ids, mask=None, step_offset=0):
        B, S, _ = x.shape
        h = self.patch_embed(x)
        cond = self.class_embed(class_ids)
        num_blocks = S // NUM_TOKENS
        rope_freqs = self.rope(num_blocks, NUM_TOKENS, h.device, step_offset)
        for block in self.blocks:
            h = block(h, cond, mask, rope_freqs)
        return self.output_proj(self.final_norm(h))


def build_blockwise_causal_mask(num_steps, num_tokens, device):
    total = num_steps * num_tokens
    mask = torch.zeros(total, total, device=device)
    for q_step in range(num_steps):
        for k_step in range(q_step):
            qs, qe = q_step * num_tokens, (q_step + 1) * num_tokens
            ks, ke = k_step * num_tokens, (k_step + 1) * num_tokens
            mask[qs:qe, ks:ke] = float("-inf")
    return mask.unsqueeze(0).unsqueeze(0)


def patchify(latents, patch_size=2):
    B, C, H, W = latents.shape
    pH, pW = H // patch_size, W // patch_size
    x = latents.reshape(B, C, pH, patch_size, pW, patch_size)
    x = x.permute(0, 2, 4, 3, 5, 1)
    return x.reshape(B, pH * pW, patch_size * patch_size * C)


def unpatchify(patches, patch_size=2, channels=4):
    """(B, K, patch_dim) -> (B, C, H, W) — inverse of patchify."""
    B, K, D = patches.shape
    grid_size = int(K ** 0.5)
    x = patches.reshape(B, grid_size, grid_size, patch_size, patch_size, channels)
    x = x.permute(0, 5, 1, 3, 2, 4)
    return x.reshape(B, channels, grid_size * patch_size, grid_size * patch_size)


# ---------------------------------------------------------------------------
# Sampling (§3.2 — Non-Markovian)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample(model, schedule, num_classes, num_steps, num_tokens, class_ids,
           cfg_scale=1.0, device="cuda"):
    """DART non-Markovian sampling (Algorithm 2 from §3.2).

    Uses growing sequences: start with x_T, iteratively sample x_{T-1}..x_0.
    Each step conditions on all previously denoised blocks (non-Markovian).
    """
    model.eval()
    B = class_ids.shape[0]

    # Initialize x_T ~ N(0, I)
    x_T = torch.randn(B, num_tokens, PATCH_DIM, device=device)

    # Growing sequence: [x_t, x_{t+1}, ..., x_T]
    # Position 0 = current step (cleanest), last position = x_T (noisiest)
    # This matches training where position 0 = step 1 (cleanest)
    blocks = [x_T]

    for step_idx in range(num_steps):
        t = num_steps - step_idx   # T, T-1, ..., 1

        x_seq = torch.cat(blocks, dim=1)
        mask = build_blockwise_causal_mask(len(blocks), num_tokens, device)

        # step_offset aligns RoPE positions with training layout
        step_offset = t - 1

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            v_pred = model(x_seq, class_ids, mask, step_offset)[:, :num_tokens, :]

            if cfg_scale > 1.0:
                uncond_ids = torch.full_like(class_ids, num_classes)
                v_uncond = model(x_seq, uncond_ids, mask, step_offset)[:, :num_tokens, :]
                v_pred = v_uncond + cfg_scale * (v_pred - v_uncond)

        # v-prediction → x̂_0: x̂_0 = α_t · x_t − σ_t · v̂_t
        alpha = schedule.alpha_bar[t].to(device)
        sigma = schedule.sqrt_one_minus_gamma[t].to(device)
        x0_pred = alpha * blocks[0] - sigma * v_pred

        if t == 1:
            model.train()
            return x0_pred

        # Sample x_{t-1} and prepend to sequence
        target = t - 1
        sg = schedule.sqrt_gamma[target].to(device)
        sng = schedule.sqrt_one_minus_gamma[target].to(device)
        x_prev = sg * x0_pred + sng * torch.randn_like(x0_pred)
        blocks = [x_prev] + blocks  # Prepend: [x_{t-1}, x_t, ..., x_T]

    model.train()
    return x0_pred


@torch.no_grad()
def generate_samples(model, ema_state, vae, vae_scale, schedule, num_classes,
                     num_steps, num_tokens, cfg_scale, device, output_path):
    """Generate sample images using EMA weights and save to disk."""
    # Swap to EMA weights
    train_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict({k: v.to(device) for k, v in ema_state.items()})

    # Generate one sample per class (up to 16)
    n = min(num_classes, 16)
    class_ids = torch.arange(n, device=device)
    patches = sample(model, schedule, num_classes, num_steps, num_tokens,
                     class_ids, cfg_scale, device)

    # Unpatchify → VAE decode → pixel images
    latents = unpatchify(patches, PATCH_SIZE, VAE_CHANNELS) / vae_scale
    pixels = vae.decode(latents).sample
    pixels = ((pixels + 1) / 2).clamp(0, 1)

    nrow = int(math.ceil(n ** 0.5))
    save_image(pixels, output_path, nrow=nrow, padding=2)

    # Restore training weights
    model.load_state_dict(train_state)
