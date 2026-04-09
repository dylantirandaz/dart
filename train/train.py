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


def get_food101_dataset(data_root="./data"):
    """Food-101: 101K images, 101 classes. Auto-downloads. Native high-res."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return datasets.Food101(root=data_root, split="train", download=True, transform=transform)


def get_flowers102_dataset(data_root="./data"):
    """Flowers-102: 8K images, 102 classes. Auto-downloads. Quick test dataset."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return datasets.Flowers102(root=data_root, split="train", download=True, transform=transform)


def get_imagefolder_dataset(data_dir):
    """Any folder of images organized as data_dir/class_name/image.jpg"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return datasets.ImageFolder(data_dir, transform=transform)


class CachedLatentDataset(torch.utils.data.Dataset):
    """Pre-encoded VAE latents loaded from numpy files."""
    def __init__(self, cache_path):
        import numpy as np
        print(f"  Loading latents into RAM...")
        self.latents = torch.from_numpy(np.load(cache_path + ".latents.npy"))
        self.labels = torch.from_numpy(np.load(cache_path + ".labels.npy"))
        print(f"  Loaded {len(self.labels)} latents ({self.latents.nbytes / 1e9:.1f} GB)")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self.latents[idx].copy()), int(self.labels[idx])


def cache_vae_latents(dataset, vae, vae_scale, device, cache_path, batch_size=32):
    """Encode entire dataset through VAE and save as memory-mapped numpy files.

    Uses pre-allocated numpy mmap files so RAM usage stays constant (~100MB)
    regardless of dataset size. Critical for ImageNet-scale (1.2M images).
    """
    import numpy as np
    latent_path = cache_path + ".latents.npy"
    label_path = cache_path + ".labels.npy"

    print(f"  Caching VAE latents to {cache_path}...")
    n = len(dataset)
    num_tokens = (256 // 8 // PATCH_SIZE) ** 2
    patch_dim = PATCH_SIZE * PATCH_SIZE * VAE_CHANNELS

    # Pre-allocate memory-mapped files on disk
    latents_mmap = np.lib.format.open_memmap(
        latent_path, mode="w+", dtype=np.float32, shape=(n, num_tokens, patch_dim))
    labels_mmap = np.lib.format.open_memmap(
        label_path, mode="w+", dtype=np.int64, shape=(n,))

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
    idx = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            latents = vae.encode(images.to(device)).latent_dist.sample() * vae_scale
            patches = patchify(latents, PATCH_SIZE).cpu().numpy()
            bs = patches.shape[0]
            latents_mmap[idx:idx + bs] = patches
            labels_mmap[idx:idx + bs] = labels.numpy()
            idx += bs
            if (i + 1) % 100 == 0:
                print(f"    {idx}/{n} images encoded")

    latents_mmap.flush()
    labels_mmap.flush()
    size_mb = (os.path.getsize(latent_path) + os.path.getsize(label_path)) / 1e6
    print(f"  Cached {n} latents ({size_mb:.0f} MB)")


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
        raw_dataset = get_cifar10_dataset()
        num_classes = 10
    elif args.dataset == "food101":
        raw_dataset = get_food101_dataset()
        num_classes = 101
    elif args.dataset == "flowers102":
        raw_dataset = get_flowers102_dataset()
        num_classes = 102
    else:
        raw_dataset = get_imagefolder_dataset(args.data_dir)
        num_classes = len(raw_dataset.classes)
    print(f"Dataset: {len(raw_dataset)} images, {num_classes} classes")

    # Compute num_tokens from image resolution
    # 256x256 -> VAE latent 32x32 -> patchified with ps=2 -> 16x16 = 256 tokens
    num_tokens = (256 // 8 // PATCH_SIZE) ** 2  # 256
    num_steps = args.num_steps
    seq_len = num_steps * num_tokens

    print(f"Denoising steps (T): {num_steps}")
    print(f"Tokens per step (K): {num_tokens}")
    print(f"Total sequence length: {seq_len}")

    # VAE for encoding images to latents
    print("Loading VAE (stabilityai/sd-vae-ft-ema)...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False
    vae_scale = 0.18215
    print("  VAE loaded")

    # Pre-cache VAE latents (encode once, train fast)
    # Save cache to output dir (local disk) to avoid OneDrive I/O issues
    cache_path = os.path.join(args.output_dir, f"{args.dataset}_latents")
    if not os.path.exists(cache_path + ".latents.npy"):
        cache_vae_latents(raw_dataset, vae, vae_scale, device, cache_path)
    else:
        print(f"  Using cached latents: {cache_path}")
    dataset = CachedLatentDataset(cache_path)

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
    use_amp = device.type == "cuda"

    # Resume from checkpoint
    if args.resume:
        pt_path = args.resume
        print(f"Resuming from {pt_path}...")
        ckpt = torch.load(pt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        ema_state = {k: v.to(device) for k, v in ckpt["ema"].items()}
        global_step = ckpt["global_step"]
        print(f"  Resumed at step {global_step}")

    start_time = time.time()
    resume_step = global_step

    # Compute starting epoch when resuming
    batches_per_epoch = len(loader)
    steps_per_epoch = batches_per_epoch // accum_steps
    start_epoch = global_step // steps_per_epoch if steps_per_epoch > 0 else 0
    start_batch = (global_step % steps_per_epoch) * accum_steps if steps_per_epoch > 0 else 0

    print(f"\nTraining for {total_steps} steps...")
    print(f"  Warmup: {warmup_steps} steps")
    print(f"  Save every: {args.save_every} steps")
    if global_step > 0:
        print(f"  Resuming from step {global_step} (epoch {start_epoch})")
    print()

    for epoch in range(start_epoch, args.epochs):
        for batch_idx, (x0, labels) in enumerate(loader):
            # Skip batches already completed in the resume epoch
            if epoch == start_epoch and batch_idx < start_batch:
                continue
            if args.steps and global_step >= args.steps:
                break

            x0 = x0.to(device)
            labels = labels.to(device)

            # §B.2 — 10% unconditional dropout for classifier-free guidance
            drop_mask = torch.rand(labels.shape[0]) < 0.1
            labels = labels.clone()
            labels[drop_mask] = num_classes  # unconditional token

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

            # Forward + loss (bf16: same dynamic range as fp32, no scaler needed)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                v_pred = model(x_input, labels, mask)
                loss = F.mse_loss(v_pred, v_targets) / accum_steps

            # Skip NaN losses to prevent poisoning weights
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            loss.backward()

            # Step optimizer every accum_steps
            if (batch_idx + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                # EMA (only update if weights are valid)
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        if not torch.isnan(v).any():
                            ema_state[k].mul_(ema_decay).add_(v, alpha=1 - ema_decay)

                global_step += 1

                if global_step % 50 == 0:
                    elapsed = time.time() - start_time
                    steps_done = global_step - resume_step
                    steps_per_sec = steps_done / max(elapsed, 1e-6)
                    lr = scheduler.get_last_lr()[0]
                    real_loss = loss.item() * accum_steps
                    eta = (total_steps - global_step) / max(steps_per_sec, 0.01)
                    print(f"  step {global_step:>6}/{total_steps} | "
                          f"loss={real_loss:.4f} | lr={lr:.2e} | "
                          f"{steps_per_sec:.1f} it/s | ETA {eta/60:.0f}m")

                if global_step % args.save_every == 0:
                    save_checkpoint(ema_state, args.output_dir, global_step, args.size,
                                    model, optimizer, scheduler)

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

    save_checkpoint(ema_state, args.output_dir, global_step, args.size,
                    model, optimizer, scheduler)
    elapsed = time.time() - start_time
    print(f"\nTraining complete. {global_step} steps in {elapsed/60:.1f} minutes.")


def save_checkpoint(ema_state, output_dir, step, size, model=None,
                    optimizer=None, scheduler=None):
    """Save EMA weights (safetensors for inference) + full training state (.pt for resume)."""
    # Inference weights (safetensors)
    ema_cpu = {k: v.cpu().contiguous() for k, v in ema_state.items()}
    st_path = os.path.join(output_dir, f"dart_{size}_step{step}.safetensors")
    save_file(ema_cpu, st_path)
    print(f"  Saved: {st_path}")

    # Full training state for resume
    if model is not None and optimizer is not None:
        train_state = {
            "global_step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "ema": ema_state,
        }
        pt_path = os.path.join(output_dir, f"dart_{size}_latest.pt")
        torch.save(train_state, pt_path)
        print(f"  Saved: {pt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DART training")
    parser.add_argument("--size", choices=CONFIGS.keys(), default="small")
    parser.add_argument("--dataset", choices=["cifar10", "food101", "flowers102", "folder"],
                        default="folder",
                        help="cifar10/food101/flowers102 auto-download; folder uses --data-dir")
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
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader workers (0 recommended on Windows)")
    parser.add_argument("--grad-checkpoint", action="store_true", default=True,
                        help="Enable gradient checkpointing (saves VRAM)")
    parser.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")
    parser.add_argument("--sample-every", type=int, default=1000,
                        help="Generate sample images every N steps (0 to disable)")
    parser.add_argument("--cfg-scale", type=float, default=1.5,
                        help="Classifier-free guidance scale for sampling")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to .pt checkpoint to resume training from")
    args = parser.parse_args()

    if args.dataset == "folder" and args.data_dir is None:
        parser.error("--data-dir is required when --dataset=folder")

    train(args)
