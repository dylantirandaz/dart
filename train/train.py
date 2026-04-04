"""
DART training script (PyTorch).
Trains a DART model and exports to safetensors for Rust inference.

Paper: https://arxiv.org/abs/2410.08159
Training details: §B.2, §4.1

Usage:
    # Train DART-S on CIFAR-10 (auto-downloads, good for testing)
    python train/train.py --size small --dataset cifar10 --num-steps 4 --epochs 50

    # Train DART-S on an image folder
    python train/train.py --size small --data-dir /path/to/images --num-steps 8 --epochs 100

    # Train DART-B on ImageNet
    python train/train.py --size base --data-dir /path/to/imagenet --num-steps 16 --steps 500000
"""

import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from safetensors.torch import save_file
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


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq=8192):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        return torch.cat([freqs, freqs], dim=-1)


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
        rope_dim = sum([16, 24, 24])
        self.rope = RotaryEmbedding(rope_dim)
        self.num_steps = num_steps

    def forward(self, x, class_ids, mask=None):
        B, S, _ = x.shape
        h = self.patch_embed(x)
        cond = self.class_embed(class_ids)
        rope_freqs = self.rope(S, h.device)
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
    """DART non-Markovian sampling. Returns (B, K, patch_dim) clean patches."""
    model.eval()
    B = class_ids.shape[0]

    # Initialize all T levels with pure noise
    x_levels = [torch.randn(B, num_tokens, PATCH_DIM, device=device)
                for _ in range(num_steps)]

    for i in range(num_steps):
        current_step = num_steps - i   # T, T-1, ..., 1
        target_step = current_step - 1  # T-1, ..., 0
        num_blocks = num_steps - i

        # Concatenate remaining noisy levels: x_{t:T}
        x_partial = torch.cat(x_levels[i:], dim=1)
        mask = build_blockwise_causal_mask(num_blocks, num_tokens, device)

        # Model prediction (with AMP)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            v_pred = model(x_partial, class_ids, mask)[:, :num_tokens, :]

            if cfg_scale > 1.0:
                uncond_ids = torch.full_like(class_ids, num_classes)
                v_uncond = model(x_partial, uncond_ids, mask)[:, :num_tokens, :]
                v_pred = v_uncond + cfg_scale * (v_pred - v_uncond)

        # v-prediction → x̂_0: x̂_0 = α_t · x_t − σ_t · v̂_t
        alpha = schedule.alpha_bar[current_step].to(device)
        sigma = schedule.sqrt_one_minus_gamma[current_step].to(device)
        x0_pred = alpha * x_levels[i] - sigma * v_pred

        if target_step == 0:
            x_levels[i] = x0_pred
        else:
            sg = schedule.sqrt_gamma[target_step].to(device)
            sng = schedule.sqrt_one_minus_gamma[target_step].to(device)
            x_levels[i] = sg * x0_pred + sng * torch.randn_like(x0_pred)

    model.train()
    return x_levels[0]


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


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def get_cifar10_dataset(data_root="./data"):
    """CIFAR-10: 50K 32x32 images, 10 classes. Auto-downloads."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)


def get_imagefolder_dataset(data_dir):
    """Any folder of images organized as data_dir/class_name/image.jpg"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return datasets.ImageFolder(data_dir, transform=transform)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  VRAM: {vram_gb:.1f} GB")

    cfg = CONFIGS[args.size]
    print(f"Model: DART-{args.size.upper()} ({cfg})")

    # Dataset
    if args.dataset == "cifar10":
        dataset = get_cifar10_dataset()
        num_classes = 10
    else:
        dataset = get_imagefolder_dataset(args.data_dir)
        num_classes = len(dataset.classes)
    print(f"Dataset: {len(dataset)} images, {num_classes} classes")

    # Compute num_tokens from image resolution
    # 256x256 -> VAE latent 32x32 -> patchified with ps=2 -> 16x16 = 256 tokens
    num_tokens = (256 // 8 // PATCH_SIZE) ** 2  # 256
    num_steps = args.num_steps
    seq_len = num_steps * num_tokens

    print(f"Denoising steps (T): {num_steps}")
    print(f"Tokens per step (K): {num_tokens}")
    print(f"Total sequence length: {seq_len}")

    # Figure out batch size for available VRAM
    # Rough heuristic: seq_len * hidden_size * num_layers * 12 bytes per sample
    if device.type == "cuda":
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        bytes_per_sample = seq_len * cfg["hidden_size"] * cfg["num_layers"] * 12
        max_batch = max(1, int(vram_bytes * 0.5 / bytes_per_sample))
        effective_batch = min(args.batch_size, max_batch)
        if effective_batch < args.batch_size:
            print(f"  Reducing batch size {args.batch_size} -> {effective_batch} for VRAM")
    else:
        effective_batch = args.batch_size

    # Gradient accumulation to reach target batch size
    accum_steps = max(1, args.batch_size // effective_batch)
    print(f"Batch size: {effective_batch} x {accum_steps} accum = {effective_batch * accum_steps} effective")

    loader = DataLoader(
        dataset, batch_size=effective_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )

    # VAE for encoding images to latents
    print("Loading VAE (stabilityai/sd-vae-ft-ema)...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False
    vae_scale = 0.18215
    print("  VAE loaded")

    # Model
    model = DartModel(cfg, num_steps=num_steps, num_classes=num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Enable gradient checkpointing for memory efficiency
    if args.grad_checkpoint:
        for block in model.blocks:
            block._orig_forward = block.forward
            def make_ckpt_fn(blk):
                def ckpt_forward(x, cond, mask, rope_freqs):
                    return torch.utils.checkpoint.checkpoint(
                        blk._orig_forward, x, cond, mask, rope_freqs, use_reentrant=False
                    )
                return ckpt_forward
            block.forward = make_ckpt_fn(block)
        print("  Gradient checkpointing: enabled")

    # Optimizer (§B.2)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95),
        weight_decay=0.01, eps=1e-8,
    )

    total_steps = args.steps if args.steps else len(loader) * args.epochs // accum_steps
    warmup_steps = min(10000, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    schedule = CosineSchedule(num_steps)

    # Pre-build the block-wise causal mask
    mask = build_blockwise_causal_mask(num_steps, num_tokens, device)

    # EMA
    ema_state = {k: v.clone() for k, v in model.state_dict().items()}
    ema_decay = 0.9999

    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    model.train()
    global_step = 0
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    start_time = time.time()

    print(f"\nTraining for {total_steps} steps...")
    print(f"  Warmup: {warmup_steps} steps")
    print(f"  Save every: {args.save_every} steps")
    print()

    for epoch in range(args.epochs):
        for batch_idx, (images, labels) in enumerate(loader):
            if args.steps and global_step >= args.steps:
                break

            images = images.to(device)
            labels = labels.to(device)

            # §B.2 — 10% unconditional dropout for classifier-free guidance
            drop_mask = torch.rand(labels.shape[0]) < 0.1
            labels = labels.clone()
            labels[drop_mask] = num_classes  # unconditional token

            # Encode to VAE latent space
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample() * vae_scale

            # Patchify
            x0 = patchify(latents, PATCH_SIZE)
            B = x0.shape[0]

            # Create noisy versions for all T steps
            all_noisy = []
            all_targets = []
            for t in range(1, num_steps + 1):
                noise = torch.randn_like(x0)
                x_t = schedule.add_noise(x0, t, noise)
                all_noisy.append(x_t)
                alpha = schedule.alpha_bar[t].to(device)
                sigma = schedule.sqrt_one_minus_gamma[t].to(device)
                all_targets.append(alpha * noise - sigma * x0)

            x_input = torch.cat(all_noisy, dim=1)
            v_targets = torch.cat(all_targets, dim=1)

            # Forward + loss
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                v_pred = model(x_input, labels, mask)
                loss = F.mse_loss(v_pred, v_targets) / accum_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Step optimizer every accum_steps
            if (batch_idx + 1) % accum_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                # EMA
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        ema_state[k].mul_(ema_decay).add_(v, alpha=1 - ema_decay)

                global_step += 1

                if global_step % 50 == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = global_step / elapsed
                    lr = scheduler.get_last_lr()[0]
                    real_loss = loss.item() * accum_steps
                    eta = (total_steps - global_step) / max(steps_per_sec, 0.01)
                    print(f"  step {global_step:>6}/{total_steps} | "
                          f"loss={real_loss:.4f} | lr={lr:.2e} | "
                          f"{steps_per_sec:.1f} it/s | ETA {eta/60:.0f}m")

                if global_step % args.save_every == 0:
                    save_checkpoint(ema_state, args.output_dir, global_step, args.size)

                if args.sample_every and global_step % args.sample_every == 0:
                    img_path = os.path.join(args.output_dir, f"samples_step{global_step}.png")
                    print(f"  Generating samples...")
                    generate_samples(
                        model, ema_state, vae, vae_scale, schedule,
                        num_classes, num_steps, num_tokens,
                        args.cfg_scale, device, img_path,
                    )
                    print(f"  Saved: {img_path}")

                if args.steps and global_step >= args.steps:
                    break

        if args.steps and global_step >= args.steps:
            break
        print(f"Epoch {epoch+1}/{args.epochs} done ({global_step} steps)")

    save_checkpoint(ema_state, args.output_dir, global_step, args.size)
    elapsed = time.time() - start_time
    print(f"\nTraining complete. {global_step} steps in {elapsed/60:.1f} minutes.")


def save_checkpoint(ema_state, output_dir, step, size):
    state = {k: v.cpu().contiguous() for k, v in ema_state.items()}
    path = os.path.join(output_dir, f"dart_{size}_step{step}.safetensors")
    save_file(state, path)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DART training")
    parser.add_argument("--size", choices=CONFIGS.keys(), default="small")
    parser.add_argument("--dataset", choices=["cifar10", "folder"], default="folder",
                        help="cifar10 auto-downloads; folder uses --data-dir")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Target batch size (auto-reduced if VRAM insufficient)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=4,
                        help="Denoising steps T (4 for 16GB GPU, 16 for 80GB)")
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--grad-checkpoint", action="store_true", default=True,
                        help="Enable gradient checkpointing (saves VRAM)")
    parser.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")
    parser.add_argument("--sample-every", type=int, default=1000,
                        help="Generate sample images every N steps (0 to disable)")
    parser.add_argument("--cfg-scale", type=float, default=1.5,
                        help="Classifier-free guidance scale for sampling")
    args = parser.parse_args()

    if args.dataset == "folder" and args.data_dir is None:
        parser.error("--data-dir is required when --dataset=folder")

    train(args)
