"""
Compute FID (Frechet Inception Distance) for a trained DART model against CIFAR-10.

Generates N samples from the model, decodes them through the VAE, resizes to
match CIFAR-10 resolution, and computes FID using the clean-fid library.

Requirements:
    pip install clean-fid

Usage:
    python eval/fid.py --weights checkpoints/dart_small_step100000.safetensors --num-samples 10000 --cfg-scale 1.5
"""

import argparse
import math
import os
import sys
import tempfile

import torch
from torchvision import datasets, transforms

# Ensure the project root is importable so we can reuse train.train
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from train.train import (
    CONFIGS,
    CosineSchedule,
    DartModel,
    NUM_TOKENS,
    PATCH_DIM,
    PATCH_SIZE,
    VAE_CHANNELS,
    sample,
    unpatchify,
)


def load_model(weights_path, cfg, num_steps, num_classes, device):
    """Load a DART model from a safetensors checkpoint."""
    from safetensors.torch import load_file

    model = DartModel(cfg, num_steps=num_steps, num_classes=num_classes).to(device)
    state_dict = load_file(weights_path, device=str(device))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def generate_all_samples(model, schedule, num_classes, num_steps, num_tokens,
                         cfg_scale, vae, vae_scale, num_samples, batch_size,
                         output_dir, device):
    """Generate num_samples images, decode via VAE, and save as PNGs."""
    os.makedirs(output_dir, exist_ok=True)

    samples_per_class = math.ceil(num_samples / num_classes)
    generated = 0

    for class_id in range(num_classes):
        remaining_for_class = samples_per_class
        if generated >= num_samples:
            break

        while remaining_for_class > 0 and generated < num_samples:
            this_batch = min(batch_size, remaining_for_class, num_samples - generated)
            class_ids = torch.full((this_batch,), class_id, dtype=torch.long, device=device)

            patches = sample(
                model, schedule, num_classes, num_steps, num_tokens,
                class_ids, cfg_scale, device,
            )

            latents = unpatchify(patches, PATCH_SIZE, VAE_CHANNELS) / vae_scale

            with torch.no_grad():
                pixels = vae.decode(latents).sample

            pixels = ((pixels + 1) / 2).clamp(0, 1)

            for i in range(this_batch):
                img = transforms.ToPILImage()(pixels[i].cpu())
                img.save(os.path.join(output_dir, f"{generated:06d}.png"))
                generated += 1

            remaining_for_class -= this_batch

        print(f"  Class {class_id}: done ({generated}/{num_samples} total)")

    print(f"Generated {generated} images in {output_dir}")
    return generated


def save_cifar10_images(output_dir, data_root="./data"):
    """Save CIFAR-10 training images as PNGs for clean-fid comparison."""
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) >= 50000:
        print(f"  CIFAR-10 reference images already cached at {output_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    dataset = datasets.CIFAR10(root=data_root, train=True, download=True)

    print(f"  Saving {len(dataset)} CIFAR-10 images to {output_dir}...")
    for i in range(len(dataset)):
        img, _ = dataset[i]
        img.save(os.path.join(output_dir, f"{i:06d}.png"))
        if (i + 1) % 10000 == 0:
            print(f"    {i + 1}/{len(dataset)}")

    print(f"  Saved {len(dataset)} reference images")


def compute_fid(generated_dir, reference_dir):
    """Compute FID between two directories of images using clean-fid."""
    from cleanfid import fid

    score = fid.compute_fid(generated_dir, reference_dir)
    return score


def compute_fid_builtin(generated_dir, dataset_name, dataset_res, dataset_split):
    """Compute FID using clean-fid's precomputed reference statistics."""
    from cleanfid import fid

    return fid.compute_fid(
        generated_dir,
        dataset_name=dataset_name,
        dataset_res=dataset_res,
        dataset_split=dataset_split,
    )


def main():
    parser = argparse.ArgumentParser(description="Compute FID for DART model on CIFAR-10")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to safetensors checkpoint")
    parser.add_argument("--num-samples", type=int, default=10000,
                        help="Number of images to generate (default: 10000)")
    parser.add_argument("--cfg-scale", type=float, default=1.5,
                        help="Classifier-free guidance scale (default: 1.5)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Generation batch size (default: 10)")
    parser.add_argument("--num-steps", type=int, default=4,
                        help="Number of denoising steps (default: 4)")
    parser.add_argument("--size", choices=CONFIGS.keys(), default="small",
                        help="Model size (default: small)")
    parser.add_argument("--num-classes", type=int, default=1000,
                        help="Number of classes (10 for CIFAR-10, 1000 for ImageNet)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save generated images (default: temp dir)")
    parser.add_argument("--reference-dir", type=str, default=None,
                        help="Directory of reference images for FID (default: CIFAR-10)")
    parser.add_argument("--builtin-ref", type=str, default=None,
                        choices=["cifar10", "imagenet"],
                        help="Use clean-fid's precomputed reference stats (e.g. 'imagenet')")
    parser.add_argument("--data-root", type=str, default="./data",
                        help="Root directory for CIFAR-10 download (default: ./data)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    num_classes = args.num_classes
    cfg = CONFIGS[args.size]
    print(f"Model: DART-{args.size.upper()}")
    print(f"Checkpoint: {args.weights}")
    print(f"CFG scale: {args.cfg_scale}")
    print(f"Denoising steps: {args.num_steps}")
    print(f"Samples to generate: {args.num_samples}")

    # Load model
    print("\nLoading model...")
    model = load_model(args.weights, cfg, args.num_steps, num_classes, device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    # Load VAE
    print("Loading VAE (stabilityai/sd-vae-ft-ema)...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False
    vae_scale = 0.18215
    print("  VAE loaded")

    # Noise schedule
    schedule = CosineSchedule(args.num_steps)

    # Output directories
    use_temp = args.output_dir is None
    if use_temp:
        temp_root = tempfile.mkdtemp(prefix="dart_fid_")
        generated_dir = os.path.join(temp_root, "generated")
    else:
        generated_dir = args.output_dir

    reference_dir = os.path.join(args.data_root, "cifar10_fid_ref")

    # Prepare reference images (skip if using builtin stats)
    if args.builtin_ref:
        print(f"\nUsing clean-fid builtin reference: {args.builtin_ref}")
    elif args.reference_dir:
        reference_dir = args.reference_dir
        print(f"\nUsing reference images: {reference_dir}")
    else:
        print("\nPreparing CIFAR-10 reference images...")
        save_cifar10_images(reference_dir, args.data_root)

    # Generate samples
    print(f"\nGenerating {args.num_samples} samples...")
    generate_all_samples(
        model, schedule, num_classes, args.num_steps, NUM_TOKENS,
        args.cfg_scale, vae, vae_scale, args.num_samples, args.batch_size,
        generated_dir, device,
    )

    # Compute FID
    print("\nComputing FID...")
    if args.builtin_ref == "imagenet":
        fid_score = compute_fid_builtin(generated_dir, "imagenet", 256, "trainval")
    elif args.builtin_ref == "cifar10":
        fid_score = compute_fid_builtin(generated_dir, "cifar10", 32, "train")
    else:
        fid_score = compute_fid(generated_dir, reference_dir)
    print(f"\n{'='*40}")
    print(f"FID Score: {fid_score:.2f}")
    print(f"{'='*40}")
    print(f"  Model: DART-{args.size.upper()}")
    print(f"  Checkpoint: {args.weights}")
    print(f"  Samples: {args.num_samples}")
    print(f"  CFG scale: {args.cfg_scale}")
    print(f"  Steps: {args.num_steps}")

    # Cleanup temp directory
    if use_temp:
        import shutil
        print(f"\nCleaning up temp directory: {temp_root}")
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
