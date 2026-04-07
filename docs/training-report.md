# DART Training Report: First 100K Steps

**Date:** April 2026
**Paper:** [DART: Denoising Autoregressive Transformer for Scalable Text-to-Image Generation](https://arxiv.org/abs/2410.08159) (ICLR 2025)

## Overview

This report documents the first complete training run of our DART implementation --- a from-scratch reimplementation of Apple's Denoising Autoregressive Transformer in both Python (training) and Rust (inference). DART combines autoregressive and diffusion modeling: a transformer processes a sequence of progressively denoised image patches, with block-wise causal attention so each denoising step can condition on all noisier steps.

The goal of this cycle was to validate the full pipeline end-to-end: training convergence, sample quality progression, and cross-language inference compatibility.

## Architecture

| Component | Detail |
|-----------|--------|
| Model | DART-S (Small) |
| Parameters | 31.9M |
| Layers | 12 |
| Hidden dim | 384 |
| Attention heads | 6 (head_dim=64) |
| FFN | SwiGLU |
| Conditioning | AdaLN (class-conditional) |
| Position encoding | 3-axis decomposed RoPE (step, row, col) |
| Patch size | 2x2 over VAE latents |
| VAE | Stable Diffusion v1 (stabilityai/sd-vae-ft-ema) |
| Loss | MSE on v-prediction targets |

### 3-Axis Decomposed RoPE

Per Section B.1 of the paper, we decompose rotary position embeddings into three independent axes:

- **Denoising step** (16 dims): which noise level in the T-step sequence
- **Spatial row** (24 dims): vertical position in the 16x16 patch grid
- **Spatial column** (24 dims): horizontal position in the grid

This gives the attention mechanism explicit awareness of both spatial layout and denoising progress, rather than treating all 1024 tokens as a flat sequence.

## Training Configuration

| Setting | Value |
|---------|-------|
| Dataset | CIFAR-10 (50K images, 10 classes) |
| Image resolution | 32x32 upscaled to 256x256 |
| Denoising steps (T) | 4 |
| Tokens per step (K) | 256 (16x16 grid) |
| Total sequence length | 1024 |
| Batch size | 8 |
| Optimizer | AdamW (lr=3e-4, betas=0.9/0.95, wd=0.01) |
| LR schedule | Linear warmup (10K steps) + cosine decay |
| EMA decay | 0.9999 |
| Gradient clipping | 2.0 |
| Mixed precision | bf16 autocast |
| Gradient checkpointing | Enabled |
| CFG dropout | 10% unconditional |
| CFG scale (sampling) | 1.5 |
| Total steps | 100,000 |
| Hardware | NVIDIA RTX 4080 (16GB VRAM) |

## Sample Progression

Each grid shows one generated image per CIFAR-10 class (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) using classifier-free guidance at scale 1.5.

### Step 5,000 --- Noise with Emerging Color Bias

![Step 5K](samples/samples_step5000.png)

The model has learned basic color distributions per class but produces only noisy textures. No spatial structure yet.

### Step 20,000 --- Scene Structure Emerges

![Step 20K](samples/samples_step20000.png)

Coherent backgrounds appear: sky gradients, water, terrain. Some class-specific coloring (blue for airplane/ship, green/brown for animals). Objects are not yet distinguishable.

### Step 30,000 --- Color Differentiation

![Step 30K](samples/samples_step30000.png)

Stronger per-class color palettes. Blocky patch artifacts visible as the model learns to coordinate predictions across the 16x16 spatial grid.

### Step 75,000 --- Recognizable Objects

![Step 75K](samples/samples_step75000.png)

Clear object silhouettes: cars with windshields, horses in profile, ships on water, birds in flight. Backgrounds are coherent. The model has learned class-specific structure.

### Step 100,000 --- Final

![Step 100K](samples/samples_step100000.png)

Best quality achieved. Objects are distinguishable with correct proportions and class-appropriate contexts (horses on grass, ships on water, cars on roads). Remaining blur is inherent to CIFAR-10's 32x32 source resolution upscaled to 256x256.

## Training Dynamics

### Loss Curve

| Step Range | Loss | Notes |
|-----------|------|-------|
| 0 - 5K | ~0.45 - 0.50 | Initial learning, high variance |
| 5K - 20K | ~0.40 - 0.45 | Steady descent |
| 20K - 40K | ~0.33 - 0.40 | Accelerating improvement |
| 40K - 70K | ~0.30 - 0.35 | Fine-grained learning |
| 70K - 100K | ~0.28 - 0.35 | Convergence plateau |

Loss decreased monotonically with no NaN or divergence events across the full 100K steps.

### Training Speed

| Phase | Speed | Bottleneck |
|-------|-------|-----------|
| Steps 0-70K (no cache) | ~1.9 it/s | VAE encoding every batch |
| Steps 70K-100K (cached) | ~8 it/s | Pure transformer forward/backward |

Pre-caching all 50K VAE latents to disk (~820MB) eliminated per-batch VAE encoding and provided a **~4x speedup**.

## Challenges and Solutions

### 1. fp16 NaN Collapse at ~30K Steps

**Problem:** Training with fp16 mixed precision (via `torch.amp.GradScaler`) consistently produced NaN loss at approximately step 30,000-35,000. The GradScaler's dynamic loss scaling would shrink to near-zero after repeated NaN detections, causing all subsequent gradients to underflow.

**Solution:** Switched to bf16, which has the same exponent range as fp32 (8 bits vs fp16's 5 bits) and requires no loss scaling. Training ran the full 100K steps without a single NaN.

### 2. Windows DataLoader Shared Memory Exhaustion

**Problem:** Using `num_workers > 0` in PyTorch's DataLoader on Windows causes shared memory exhaustion on long training runs, crashing the process.

**Solution:** Default to `--workers 0` on Windows. The latent caching optimization later eliminated this as a bottleneck entirely.

### 3. Training Resilience

**Problem:** Long training runs on consumer hardware are vulnerable to crashes, power events, and OS interruptions.

**Solution:** Built checkpoint resume support (saves full optimizer/scheduler/EMA state every 5K steps) and a watchdog script that monitors the process and auto-restarts from the latest checkpoint on crash.

## Implementation Stack

- **Training:** Python + PyTorch with custom transformer, RoPE, and sampling
- **Inference:** Rust + [candle](https://github.com/huggingface/candle) (Hugging Face's Rust ML framework)
- **Model exchange:** Safetensors format (Python trains, Rust loads for inference)
- **VAE:** Stable Diffusion v1 encoder/decoder shared across both runtimes

The full pipeline is verified end-to-end: Python training produces safetensors checkpoints that the Rust binary loads for inference, decodes through the VAE, and outputs PNG images.

## Limitations

- **CIFAR-10 resolution ceiling:** The source images are 32x32, upscaled to 256x256 for the VAE. This creates an inherent blur ceiling --- the model cannot generate detail that doesn't exist in the training data.
- **T=4 denoising steps:** Limited by 16GB VRAM. The paper uses T=16 for best quality. More steps allow finer-grained denoising but require proportionally more memory (sequence length = T x 256).
- **No FID evaluation:** Quantitative metrics were not computed in this cycle. Visual quality assessment only.

## Next Steps

1. **Higher-quality dataset:** Train on a native 256x256 dataset (CelebA-HQ, LSUN Bedrooms, or similar) to remove the CIFAR-10 blur ceiling.
2. **Rust inference validation:** Generate samples from the Rust binary using the 100K checkpoint and compare against Python outputs.
3. **Quantitative evaluation:** Compute FID/IS scores against the CIFAR-10 test set.
4. **Scale up T:** With latent caching freeing VRAM, experiment with T=8 denoising steps for better sample quality.
