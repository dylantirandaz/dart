"""
DART cloud training on Modal (A100 GPU).

Usage:
    # First time: authenticate
    modal setup

    # Run training
    modal run train_cloud.py

    # Download results when done
    modal volume get dart-data checkpoints/ ./cloud_checkpoints/
"""

import modal
import os

app = modal.App("dart-training")

# Persistent volume for ImageNet cache + checkpoints
volume = modal.Volume.from_name("dart-data", create_if_missing=True)

# GPU image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1",
        "torchvision",
        "safetensors",
        "diffusers",
        "transformers",
        "accelerate",
        "datasets",
        "Pillow",
        "numpy",
    )
)

VOLUME_PATH = "/data"


def _iter_images(path):
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(".jpg"):
                yield os.path.join(root, f)
IMAGENET_PATH = f"{VOLUME_PATH}/imagenet256/train"
CACHE_PATH = f"{VOLUME_PATH}/imagenet_latents"
CHECKPOINT_DIR = f"{VOLUME_PATH}/checkpoints"


@app.function(
    gpu="A10G",
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=86400,
    memory=8192,
)
def prepare_imagenet():
    """Stream ImageNet through VAE directly into latent cache. Resumable."""
    import numpy as np
    import torch
    from torchvision import transforms
    from diffusers import AutoencoderKL
    from datasets import load_dataset

    patch_size, vae_channels = 2, 4
    num_tokens = (256 // 8 // patch_size) ** 2
    patch_dim = patch_size * patch_size * vae_channels
    n = 1281167

    latent_path = CACHE_PATH + ".latents.npy"
    label_path = CACHE_PATH + ".labels.npy"

    # Check if already complete or partially cached
    if os.path.exists(latent_path):
        existing = np.load(latent_path, mmap_mode="r")
        # Find last non-zero row (mmap is pre-allocated to full size,
        # so len(existing) == n even if encoding was interrupted)
        resume_idx = 0
        for i in range(len(existing) - 1, -1, -1):
            if existing[i].any():
                resume_idx = i + 1
                break
        if resume_idx >= n - 100:
            print(f"Latent cache complete: {resume_idx}/{n} images ({n - resume_idx} skipped)")
            return
        print(f"Resuming from {resume_idx}/{n}")
        del existing
    else:
        resume_idx = 0

    device = torch.device("cuda")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()
    vae_scale = 0.18215

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    print("Streaming ImageNet and encoding VAE latents...")
    ds = load_dataset("evanarlian/imagenet_1k_resized_256", split="train", streaming=True)

    os.makedirs(os.path.dirname(CACHE_PATH) or ".", exist_ok=True)

    if resume_idx == 0:
        latents_mmap = np.lib.format.open_memmap(
            latent_path, mode="w+", dtype=np.float32, shape=(n, num_tokens, patch_dim))
        labels_mmap = np.lib.format.open_memmap(
            label_path, mode="w+", dtype=np.int64, shape=(n,))
    else:
        latents_mmap = np.lib.format.open_memmap(latent_path, mode="r+")
        labels_mmap = np.lib.format.open_memmap(label_path, mode="r+")

    batch_images = []
    batch_labels = []
    batch_size = 64
    idx = 0
    skipped = 0

    with torch.no_grad():
        for example in ds:
            if idx < resume_idx:
                idx += 1
                continue

            img = transform(example["image"].convert("RGB"))
            batch_images.append(img)
            batch_labels.append(example["label"])

            if len(batch_images) == batch_size:
                images = torch.stack(batch_images).to(device)
                latents = vae.encode(images).latent_dist.sample() * vae_scale
                B, C, H, W = latents.shape
                pH, pW = H // patch_size, W // patch_size
                x = latents.reshape(B, C, pH, patch_size, pW, patch_size)
                x = x.permute(0, 2, 4, 3, 5, 1)
                patches = x.reshape(B, pH * pW, patch_size * patch_size * C)

                bs = patches.shape[0]
                latents_mmap[idx:idx + bs] = patches.cpu().numpy()
                labels_mmap[idx:idx + bs] = np.array(batch_labels, dtype=np.int64)
                idx += bs

                batch_images = []
                batch_labels = []

                if idx % 10000 < batch_size:
                    print(f"  {idx}/{n} images encoded")
                if idx % 50000 < batch_size:
                    latents_mmap.flush()
                    labels_mmap.flush()
                    volume.commit()

            if idx >= n:
                break

    # Handle remaining
    if batch_images:
        with torch.no_grad():
            images = torch.stack(batch_images).to(device)
            latents = vae.encode(images).latent_dist.sample() * vae_scale
            B, C, H, W = latents.shape
            pH, pW = H // patch_size, W // patch_size
            x = latents.reshape(B, C, pH, patch_size, pW, patch_size)
            x = x.permute(0, 2, 4, 3, 5, 1)
            patches = x.reshape(B, pH * pW, patch_size * patch_size * C)
            bs = patches.shape[0]
            latents_mmap[idx:idx + bs] = patches.cpu().numpy()
            labels_mmap[idx:idx + bs] = np.array(batch_labels, dtype=np.int64)
            idx += bs

    latents_mmap.flush()
    labels_mmap.flush()
    print(f"Cached {idx} latents")
    volume.commit()


@app.function(
    gpu="A100",
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=86400,
)
def train(
    num_steps_t: int = 8,
    total_steps: int = 800000,
    batch_size: int = 32,
    lr: float = 3e-4,
    size: str = "small",
    resume: bool = True,
):
    """Run DART training on A100."""
    import sys
    sys.path.insert(0, "/data/code")

    import torch
    import numpy as np
    import math
    import time
    from safetensors.torch import save_file
    from torchvision.utils import save_image

    # Copy training code to volume if needed
    _sync_code()

    # Import from training script
    from train.train import (
        DartModel, CONFIGS, CosineSchedule, build_blockwise_causal_mask,
        patchify, unpatchify, sample, generate_samples,
        PATCH_SIZE, VAE_CHANNELS, NUM_TOKENS, PATCH_DIM,
    )

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    cfg = CONFIGS[size]
    num_tokens = NUM_TOKENS
    seq_len = num_steps_t * num_tokens

    # Load cached latents
    print("Loading latent cache...")
    latents = torch.from_numpy(np.load(CACHE_PATH + ".latents.npy"))
    labels = torch.from_numpy(np.load(CACHE_PATH + ".labels.npy"))
    num_classes = int(labels.max().item()) + 1
    print(f"  {len(labels)} images, {num_classes} classes")

    dataset = torch.utils.data.TensorDataset(latents, labels)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

    # Model
    model = DartModel(cfg, num_steps=num_steps_t, num_classes=num_classes).to(device)
    model = torch.compile(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: DART-{size.upper()}, {n_params:,} params, T={num_steps_t}, batch={batch_size}")

    # Gradient checkpointing
    for block in model._orig_mod.blocks:
        block._orig_forward = block.forward
        def make_ckpt_fn(blk):
            def ckpt_forward(x, cond, mask, rope_freqs):
                return torch.utils.checkpoint.checkpoint(
                    blk._orig_forward, x, cond, mask, rope_freqs, use_reentrant=False)
            return ckpt_forward
        block.forward = make_ckpt_fn(block)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                                   weight_decay=0.01, eps=1e-8)

    warmup_steps = min(10000, total_steps // 10)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    schedule = CosineSchedule(num_steps_t)
    mask = build_blockwise_causal_mask(num_steps_t, num_tokens, device)

    # EMA
    ema_state = {k: v.clone() for k, v in model.state_dict().items()}
    ema_decay = 0.9999

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Resume
    global_step = 0
    pt_path = os.path.join(CHECKPOINT_DIR, f"dart_{size}_latest.pt")
    if resume and os.path.exists(pt_path):
        print(f"Resuming from {pt_path}...")
        ckpt = torch.load(pt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        ema_state = {k: v.to(device) for k, v in ckpt["ema"].items()}
        global_step = ckpt["global_step"]
        print(f"  Resumed at step {global_step}")

    model.train()
    start_time = time.time()
    resume_step = global_step

    accum_steps = 1
    batches_per_epoch = len(loader)
    steps_per_epoch = batches_per_epoch // accum_steps
    start_epoch = global_step // steps_per_epoch if steps_per_epoch > 0 else 0
    start_batch = (global_step % steps_per_epoch) * accum_steps if steps_per_epoch > 0 else 0

    print(f"\nTraining for {total_steps} steps...")
    print(f"  Warmup: {warmup_steps}, Save every: 10000\n")

    for epoch in range(start_epoch, 100000):
        for batch_idx, (x0, batch_labels) in enumerate(loader):
            if epoch == start_epoch and batch_idx < start_batch:
                continue
            if global_step >= total_steps:
                break

            x0 = x0.to(device)
            batch_labels = batch_labels.to(device).long()

            # CFG dropout
            drop_mask = torch.rand(batch_labels.shape[0]) < 0.1
            batch_labels = batch_labels.clone()
            batch_labels[drop_mask] = num_classes

            B = x0.shape[0]
            all_noisy, all_targets = [], []
            for t in range(1, num_steps_t + 1):
                noise = torch.randn_like(x0)
                x_t = schedule.add_noise(x0, t, noise)
                all_noisy.append(x_t)
                alpha = schedule.alpha_bar[t].to(device)
                sigma = schedule.sqrt_one_minus_gamma[t].to(device)
                all_targets.append(alpha * noise - sigma * x0)

            x_input = torch.cat(all_noisy, dim=1)
            v_targets = torch.cat(all_targets, dim=1)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                v_pred = model(x_input, batch_labels, mask)
                loss = torch.nn.functional.mse_loss(v_pred, v_targets)

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if not torch.isnan(v).any():
                        ema_state[k].mul_(ema_decay).add_(v, alpha=1 - ema_decay)

            global_step += 1

            if global_step % 100 == 0:
                elapsed = time.time() - start_time
                steps_done = global_step - resume_step
                sps = steps_done / max(elapsed, 1e-6)
                eta = (total_steps - global_step) / max(sps, 0.01)
                lr_now = scheduler.get_last_lr()[0]
                print(f"  step {global_step:>7}/{total_steps} | "
                      f"loss={loss.item():.4e} | lr={lr_now:.2e} | "
                      f"{sps:.1f} it/s | ETA {eta/60:.0f}m")

            if global_step % 10000 == 0:
                # Save checkpoint
                ema_cpu = {k: v.cpu().contiguous() for k, v in ema_state.items()}
                st_path = os.path.join(CHECKPOINT_DIR, f"dart_{size}_step{global_step}.safetensors")
                save_file(ema_cpu, st_path)
                print(f"  Saved: {st_path}")

                train_state = {
                    "global_step": global_step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "ema": ema_state,
                }
                torch.save(train_state, pt_path)
                volume.commit()

            if global_step >= total_steps:
                break

        if global_step >= total_steps:
            break
        print(f"Epoch {epoch + 1} done ({global_step} steps)")

    # Final save
    ema_cpu = {k: v.cpu().contiguous() for k, v in ema_state.items()}
    save_file(ema_cpu, os.path.join(CHECKPOINT_DIR, f"dart_{size}_step{global_step}.safetensors"))
    torch.save({
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "ema": ema_state,
    }, pt_path)
    volume.commit()

    elapsed = time.time() - start_time
    print(f"\nDone. {global_step} steps in {elapsed/3600:.1f} hours.")


@app.function(
    gpu="A100",
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=3600,
)
def sample_grid(
    checkpoint: str = "dart_small_step800000.safetensors",
    classes: str = "0,9,107,207,281,291,340,388,817,963,985,1",
    cfg_scale: float = 1.5,
    num_steps_t: int = 8,
    size: str = "small",
    grid_cols: int = 4,
):
    """Generate a grid of samples for fixed classes from a specific checkpoint."""
    import sys
    sys.path.insert(0, "/data/code")

    import torch
    from torchvision.utils import make_grid, save_image
    from diffusers import AutoencoderKL
    from safetensors.torch import load_file

    from train.train import (
        DartModel, CONFIGS, CosineSchedule,
        NUM_TOKENS, PATCH_SIZE, VAE_CHANNELS,
        sample, unpatchify,
    )

    device = torch.device("cuda")
    cfg = CONFIGS[size]
    num_classes = 1000

    ckpt_path = os.path.join(CHECKPOINT_DIR, checkpoint)
    print(f"Loading {ckpt_path}")
    state_dict = load_file(ckpt_path, device=str(device))
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model = DartModel(cfg, num_steps=num_steps_t, num_classes=num_classes).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()
    vae_scale = 0.18215
    schedule = CosineSchedule(num_steps_t)

    class_list = [int(c) for c in classes.split(",")]
    class_ids = torch.tensor(class_list, dtype=torch.long, device=device)

    with torch.no_grad():
        patches = sample(model, schedule, num_classes, num_steps_t,
                         NUM_TOKENS, class_ids, cfg_scale, device)
        latents = unpatchify(patches, PATCH_SIZE, VAE_CHANNELS) / vae_scale
        pixels = vae.decode(latents).sample
    pixels = ((pixels + 1) / 2).clamp(0, 1)

    grid = make_grid(pixels.cpu(), nrow=grid_cols, padding=2)
    out_dir = os.path.join(VOLUME_PATH, "sample_grids")
    os.makedirs(out_dir, exist_ok=True)
    tag = checkpoint.replace(".safetensors", "")
    out_path = os.path.join(out_dir, f"{tag}_cfg{cfg_scale}.png")
    save_image(grid, out_path)
    volume.commit()
    print(f"Saved grid to {out_path}")


@app.function(
    gpu="A100",
    image=image.pip_install("clean-fid"),
    volumes={VOLUME_PATH: volume},
    timeout=86400,
)
def fid_eval(
    checkpoint: str = "dart_small_step800000.safetensors",
    num_samples: int = 50000,
    batch_size: int = 32,
    cfg_scale: float = 1.5,
    num_steps_t: int = 8,
    size: str = "small",
):
    """Generate samples and compute FID vs ImageNet 256 reference set."""
    import sys
    sys.path.insert(0, "/data/code")

    import math
    import torch
    from torchvision import transforms
    from diffusers import AutoencoderKL
    from safetensors.torch import load_file
    from cleanfid import fid

    from train.train import (
        DartModel, CONFIGS, CosineSchedule,
        NUM_TOKENS, PATCH_SIZE, VAE_CHANNELS,
        sample, unpatchify,
    )

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    cfg = CONFIGS[size]
    num_classes = 1000

    ckpt_path = os.path.join(CHECKPOINT_DIR, checkpoint)
    print(f"Loading {ckpt_path}")
    state_dict = load_file(ckpt_path, device=str(device))
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model = DartModel(cfg, num_steps=num_steps_t, num_classes=num_classes).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()
    vae_scale = 0.18215

    schedule = CosineSchedule(num_steps_t)

    # Persist to volume so failure doesn't lose generation work
    ckpt_tag = checkpoint.replace(".safetensors", "").replace("/", "_")
    out_dir = os.path.join(VOLUME_PATH, "fid_samples", f"{ckpt_tag}_cfg{cfg_scale}")
    ref_dir = os.path.join(VOLUME_PATH, "fid_reference_imagenet")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)

    # Resume: count already-generated images
    existing = sorted(f for f in os.listdir(out_dir) if f.endswith(".png"))
    generated = len(existing)
    print(f"Existing: {generated}/{num_samples} generated samples in {out_dir}")

    if generated < num_samples:
        samples_per_class = math.ceil(num_samples / num_classes)
        # Resume at the class boundary we left off at
        start_class = generated // samples_per_class
        per_class_done = generated % samples_per_class

        for class_id in range(start_class, num_classes):
            if generated >= num_samples:
                break
            remaining = samples_per_class - (per_class_done if class_id == start_class else 0)
            remaining = min(remaining, num_samples - generated)
            while remaining > 0:
                this_batch = min(batch_size, remaining)
                class_ids = torch.full((this_batch,), class_id, dtype=torch.long, device=device)
                with torch.no_grad():
                    patches = sample(model, schedule, num_classes, num_steps_t,
                                     NUM_TOKENS, class_ids, cfg_scale, device)
                    latents = unpatchify(patches, PATCH_SIZE, VAE_CHANNELS) / vae_scale
                    pixels = vae.decode(latents).sample
                pixels = ((pixels + 1) / 2).clamp(0, 1)
                for i in range(this_batch):
                    img = transforms.ToPILImage()(pixels[i].cpu())
                    img.save(os.path.join(out_dir, f"{generated:06d}.png"))
                    generated += 1
                remaining -= this_batch
            if class_id % 50 == 0:
                print(f"  Class {class_id}/{num_classes}: {generated}/{num_samples}")
                volume.commit()
        volume.commit()

    # Build ImageNet reference from HuggingFace streaming if not already cached
    ref_existing = sorted(f for f in os.listdir(ref_dir) if f.endswith(".png"))
    if len(ref_existing) < num_samples:
        from datasets import load_dataset
        print(f"Building ImageNet reference set ({num_samples} images)...")
        ds = load_dataset("evanarlian/imagenet_1k_resized_256",
                          split="train", streaming=True)
        tfm = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
        ])
        idx = len(ref_existing)
        for example in ds:
            if idx >= num_samples:
                break
            img = tfm(example["image"].convert("RGB"))
            img.save(os.path.join(ref_dir, f"{idx:06d}.png"))
            idx += 1
            if idx % 5000 == 0:
                print(f"  Reference: {idx}/{num_samples}")
                volume.commit()
        volume.commit()

    print(f"Generated {generated} / Reference {len(os.listdir(ref_dir))} images")
    print("Computing FID (folder vs folder)...")
    score = fid.compute_fid(out_dir, ref_dir)
    print(f"\n{'='*40}")
    print(f"FID: {score:.2f}")
    print(f"{'='*40}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Samples: {generated}")
    print(f"  CFG: {cfg_scale}, Steps: {num_steps_t}")


def _sync_code():
    """Copy training code to volume so it can be imported."""
    import shutil
    code_dir = "/data/code/train"
    os.makedirs(code_dir, exist_ok=True)
    # The training script is mounted via the volume
    # We need to write it there on first run
    init_path = os.path.join(code_dir, "__init__.py")
    if not os.path.exists(init_path):
        open(init_path, "w").close()


@app.function(image=image, volumes={VOLUME_PATH: volume})
def check_cache():
    """Inspect the latent cache to see how many images are actually encoded."""
    import numpy as np
    latent_path = CACHE_PATH + ".latents.npy"
    label_path = CACHE_PATH + ".labels.npy"

    if not os.path.exists(latent_path):
        print("No latent cache found.")
        return

    latents = np.load(latent_path, mmap_mode="r")
    labels = np.load(label_path, mmap_mode="r")
    n = len(latents)

    # Find last non-zero entry
    last_filled = 0
    for i in range(n - 1, -1, -1):
        if latents[i].any():
            last_filled = i + 1
            break

    # Check for zero gaps in the filled range
    zero_gaps = 0
    for i in range(0, min(last_filled, 10000)):
        if not latents[i].any():
            zero_gaps += 1

    num_classes = int(labels[:last_filled].max()) + 1 if last_filled > 0 else 0

    print(f"Cache shape: {latents.shape}")
    print(f"Filled entries: {last_filled}/{n} ({100*last_filled/n:.1f}%)")
    print(f"Zero gaps in first 10K: {zero_gaps}")
    print(f"Classes found: {num_classes}")

    # Check checkpoints
    if os.path.exists(CHECKPOINT_DIR):
        files = os.listdir(CHECKPOINT_DIR)
        print(f"\nCheckpoints ({len(files)} files):")
        for f in sorted(files):
            size_mb = os.path.getsize(os.path.join(CHECKPOINT_DIR, f)) / 1e6
            print(f"  {f} ({size_mb:.0f} MB)")
    else:
        print("\nNo checkpoints found.")


@app.function(image=image, volumes={VOLUME_PATH: volume})
def clean_volume():
    """Delete old raw images from previous failed downloads."""
    import shutil
    imagenet_dir = VOLUME_PATH + "/imagenet256"
    if os.path.exists(imagenet_dir):
        print(f"Deleting {imagenet_dir}...")
        shutil.rmtree(imagenet_dir)
        print("  Done")
    volume.commit()


@app.function(image=image, volumes={VOLUME_PATH: volume})
def reset_all():
    """Delete latent cache and checkpoints for a fresh start."""
    import shutil
    for path in [
        CACHE_PATH + ".latents.npy",
        CACHE_PATH + ".labels.npy",
    ]:
        if os.path.exists(path):
            print(f"Deleting {path}...")
            os.remove(path)
    if os.path.exists(CHECKPOINT_DIR):
        print(f"Deleting {CHECKPOINT_DIR}...")
        shutil.rmtree(CHECKPOINT_DIR)
    volume.commit()
    print("Volume cleaned — ready for fresh run.")


@app.local_entrypoint()
def main():
    """Upload training code, prepare data, then train."""
    import shutil

    # Upload training code to volume
    print("Uploading training code to Modal volume...")
    local_train = os.path.join(os.path.dirname(__file__), "train", "train.py")
    with open(local_train, "rb") as f:
        code = f.read()

    # Write code to volume via a helper function
    upload_code.remote(code)

    # Clean old data from failed runs
    print("\nCleaning old data from volume...")
    clean_volume.remote()

    print("\nStep 1: Preparing ImageNet data + VAE latent cache...")
    print("  (Skip if already cached on volume)")
    prepare_imagenet.remote()

    print("\nStep 2: Starting training...")
    train.remote(
        num_steps_t=8,
        total_steps=800000,
        batch_size=32,
        lr=3e-4,
        size="small",
    )

    print("\nTraining complete!")
    print("Download checkpoints with:")
    print("  modal volume get dart-data checkpoints/ ./cloud_checkpoints/")


@app.function(image=image, volumes={VOLUME_PATH: volume})
def upload_code(code_bytes: bytes):
    """Write training code to the volume."""
    code_dir = "/data/code/train"
    os.makedirs(code_dir, exist_ok=True)
    with open(os.path.join(code_dir, "train.py"), "wb") as f:
        f.write(code_bytes)
    with open(os.path.join(code_dir, "__init__.py"), "w") as f:
        pass
    volume.commit()
    print("  Code uploaded to volume")
