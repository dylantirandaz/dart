"""
DART training script (PyTorch).
Trains a DART model on ImageNet and exports to safetensors for Rust inference.

Paper: https://arxiv.org/abs/2410.08159
Training details: §B.2, §4.1

Usage:
    # Train DART-S on ImageNet (fast, for testing)
    python train/train.py --size small --data-dir /path/to/imagenet --epochs 10

    # Train DART-XL (full paper recipe)
    python train/train.py --size xlarge --data-dir /path/to/imagenet --steps 500000

    # Train on any image folder (not ImageNet)
    python train/train.py --size small --data-dir /path/to/images --epochs 50
"""

import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from safetensors.torch import save_file


# ---------------------------------------------------------------------------
# Model configs from Table 2
# ---------------------------------------------------------------------------

CONFIGS = {
    "small":  dict(num_layers=12, hidden_size=384,  num_heads=6,  head_dim=64),
    "base":   dict(num_layers=12, hidden_size=768,  num_heads=12, head_dim=64),
    "large":  dict(num_layers=24, hidden_size=1024, num_heads=16, head_dim=64),
    "xlarge": dict(num_layers=28, hidden_size=1152, num_heads=18, head_dim=64),
}

NUM_CLASSES = 1000
PATCH_SIZE = 2
VAE_CHANNELS = 4  # SD v1.4 VAE latent channels
NUM_TOKENS = 256  # 32/2 * 32/2 for 256x256 images
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * VAE_CHANNELS  # 16


# ---------------------------------------------------------------------------
# Cosine noise schedule (§B.3)
# ---------------------------------------------------------------------------

class CosineSchedule:
    def __init__(self, num_steps=16):
        self.num_steps = num_steps
        # gamma[t] = cos^2(pi/2 * t/T)
        t = torch.linspace(0, 1, num_steps + 1)
        self.alpha_bar = torch.cos(math.pi / 2 * t)
        self.gamma = self.alpha_bar ** 2
        self.sqrt_gamma = self.gamma.sqrt()
        self.sqrt_one_minus_gamma = (1 - self.gamma).sqrt()

    def add_noise(self, x0, t, noise=None):
        """x_t = sqrt(gamma_t) * x_0 + sqrt(1 - gamma_t) * eps"""
        if noise is None:
            noise = torch.randn_like(x0)
        sg = self.sqrt_gamma[t].view(-1, 1, 1) if x0.dim() == 3 else self.sqrt_gamma[t]
        sng = self.sqrt_one_minus_gamma[t].view(-1, 1, 1) if x0.dim() == 3 else self.sqrt_one_minus_gamma[t]
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


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq=8192):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq = max_seq

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        return torch.cat([freqs, freqs], dim=-1)


def apply_rope(x, freqs):
    """x: (B, H, S, D), freqs: (S, D)"""
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
        q, k, v = qkv.unbind(2)  # (B, S, H, D)
        q = self.q_norm(q).transpose(1, 2)  # (B, H, S, D)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        q = apply_rope(q, rope_freqs)
        k = apply_rope(k, rope_freqs)

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn + mask
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, S, -1)
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

        rope_dim = sum([16, 24, 24])  # total head_dim
        self.rope = RotaryEmbedding(rope_dim)
        self.num_steps = num_steps
        self.num_classes = num_classes

    def forward(self, x, class_ids, mask=None):
        """
        x: (B, T*K, patch_dim) — noisy patches across all T steps
        class_ids: (B,) — class labels
        Returns: (B, T*K, patch_dim) — predicted clean patches (v-prediction)
        """
        B, S, _ = x.shape
        h = self.patch_embed(x)
        cond = self.class_embed(class_ids)

        rope_freqs = self.rope(S, h.device)

        for block in self.blocks:
            h = block(h, cond, mask, rope_freqs)

        h = self.final_norm(h)
        return self.output_proj(h)


def build_blockwise_causal_mask(num_steps, num_tokens, device):
    """Block-wise causal: step t can attend to steps t..T, not 0..t-1."""
    total = num_steps * num_tokens
    mask = torch.zeros(total, total, device=device)
    for q_step in range(num_steps):
        for k_step in range(num_steps):
            if k_step < q_step:
                qi_start = q_step * num_tokens
                qi_end = qi_start + num_tokens
                ki_start = k_step * num_tokens
                ki_end = ki_start + num_tokens
                mask[qi_start:qi_end, ki_start:ki_end] = float("-inf")
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)


# ---------------------------------------------------------------------------
# Patchify / Unpatchify
# ---------------------------------------------------------------------------

def patchify(latents, patch_size=2):
    """(B, C, H, W) -> (B, K, patch_dim)"""
    B, C, H, W = latents.shape
    pH = H // patch_size
    pW = W // patch_size
    x = latents.reshape(B, C, pH, patch_size, pW, patch_size)
    x = x.permute(0, 2, 4, 3, 5, 1)  # (B, pH, pW, ps, ps, C)
    return x.reshape(B, pH * pW, patch_size * patch_size * C)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = CONFIGS[args.size]
    print(f"Model: DART-{args.size.upper()} ({cfg})")

    # Dataset
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)
    print(f"Dataset: {len(dataset)} images, {len(loader)} batches")

    # VAE for encoding images to latents
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False
    vae_scale = 0.18215

    # Model
    num_steps = args.num_steps
    model = DartModel(cfg, num_steps=num_steps).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # §B.2 — AdamW, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01, grad_clip=2.0
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95),
        weight_decay=0.01, eps=1e-8,
    )

    # §B.2 — Cosine schedule with 10k warmup
    total_steps = args.steps if args.steps else len(loader) * args.epochs
    warmup_steps = min(10000, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    schedule = CosineSchedule(num_steps)
    mask = build_blockwise_causal_mask(num_steps, NUM_TOKENS, device)

    # EMA (§B.2: decay 0.9999)
    ema_state = {k: v.clone() for k, v in model.state_dict().items()}
    ema_decay = 0.9999

    # Training loop
    model.train()
    global_step = 0
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        for batch_idx, (images, labels) in enumerate(loader):
            if args.steps and global_step >= args.steps:
                break

            images = images.to(device)  # (B, 3, 256, 256)
            labels = labels.to(device)  # (B,)

            # Encode to VAE latent space
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample() * vae_scale  # (B, 4, 32, 32)

            # Patchify: (B, 4, 32, 32) -> (B, 256, 16)
            x0 = patchify(latents, PATCH_SIZE)  # (B, K, patch_dim)
            B, K, D = x0.shape

            # Create noisy versions for all T steps and concatenate
            all_noisy = []
            all_targets = []
            for t in range(1, num_steps + 1):
                noise = torch.randn_like(x0)
                x_t = schedule.add_noise(x0, t, noise)
                all_noisy.append(x_t)

                # §B.3 — v-prediction target: v = alpha * eps - sigma * x0
                alpha = schedule.alpha_bar[t].to(device)
                sigma = schedule.sqrt_one_minus_gamma[t].to(device)
                v_target = alpha * noise - sigma * x0
                all_targets.append(v_target)

            # Concatenate all steps: (B, T*K, D)
            x_input = torch.cat(all_noisy, dim=1)
            v_targets = torch.cat(all_targets, dim=1)

            # Forward pass
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                v_pred = model(x_input, labels, mask)
                loss = F.mse_loss(v_pred, v_targets)

            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()

            scheduler.step()

            # EMA update
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    ema_state[k].mul_(ema_decay).add_(v, alpha=1 - ema_decay)

            global_step += 1

            if global_step % 100 == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"  step {global_step}/{total_steps} | loss={loss.item():.4f} | lr={lr:.2e}")

            if global_step % args.save_every == 0:
                save_checkpoint(model, ema_state, args.output_dir, global_step, args.size)

        if args.steps and global_step >= args.steps:
            break
        print(f"Epoch {epoch+1}/{args.epochs} done")

    # Final save
    save_checkpoint(model, ema_state, args.output_dir, global_step, args.size)
    print(f"\nTraining complete. {global_step} steps.")


def save_checkpoint(model, ema_state, output_dir, step, size):
    """Save both regular and EMA weights in safetensors format for Rust inference."""
    # Map PyTorch keys to Rust model keys
    rust_state = {}
    for k, v in ema_state.items():
        rust_key = k  # Keys already match the Rust VarBuilder paths
        rust_state[rust_key] = v.cpu()

    path = os.path.join(output_dir, f"dart_{size}_step{step}.safetensors")
    save_file(rust_state, path)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="DART training")
    parser.add_argument("--size", choices=CONFIGS.keys(), default="small")
    parser.add_argument("--data-dir", type=str, required=True, help="ImageNet or image folder path")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=None, help="Max training steps (overrides epochs)")
    parser.add_argument("--num-steps", type=int, default=16, help="Number of denoising steps T")
    parser.add_argument("--save-every", type=int, default=10000, help="Save checkpoint every N steps")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
