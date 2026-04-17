# DART: Denoising Autoregressive Transformer

Rust implementation of [DART: Denoising Autoregressive Transformer for Scalable Text-to-Image Generation](https://arxiv.org/abs/2410.08159) (ICLR 2025).

DART unifies autoregressive and diffusion models by denoising image patches through a non-Markovian process using a standard transformer architecture. Unlike traditional diffusion which only conditions on the previous noisy step, DART conditions on the entire noisy trajectory — enabling more efficient learning with fewer inference steps.

## Architecture

- **Transformer**: GPT-style with RoPE positional encoding, SwiGLU FFN, per-head RMSNorm
- **Conditioning**: Adaptive LayerNorm (AdaLN) for class-conditional generation
- **Attention**: Block-wise causal mask — tokens at step t attend to all tokens at steps t..T
- **Diffusion**: Non-Markovian cosine schedule with v-prediction parameterization
- **Tokenization**: VAE latent patches (SD v1.4), patch size 2x2

## Model Sizes (Table 2)

| Model | Layers | Hidden | Heads | Params |
|-------|--------|--------|-------|--------|
| DART-S | 12 | 384 | 6 | 48M |
| DART-B | 12 | 768 | 12 | 141M |
| DART-L | 24 | 1024 | 16 | 464M |
| DART-XL | 28 | 1152 | 18 | 812M |

## Inference (Rust)

```bash
# Build
cargo build --release

# Print model config and noise schedule
cargo run -- info --model-size xlarge

# Dry-run generation (random weights, verifies architecture)
cargo run -- generate --model-size small --class 207 --steps 16

# Generate with trained weights — downloads SD v1.4 VAE automatically
cargo run -- generate --model-size small --weights checkpoints/dart_small.safetensors \
  --class 207 --steps 16 --cfg-scale 1.5 --output-dir output/
```

The generate command will:
1. Load DART weights from safetensors
2. Run T-step non-Markovian denoising with classifier-free guidance
3. Download and run the SD v1.4 VAE decoder to convert latents to pixels
4. Save PNG images to the output directory

## Training (Modal)

Training runs on [Modal](https://modal.com) with an A100, streaming ImageNet from HuggingFace. Weights export to safetensors for Rust inference.

```bash
pip install modal
modal setup

# Train DART-S on ImageNet (resumable across 24h function timeouts)
modal run --detach train_cloud.py

# Compute FID against 50K ImageNet reference images
modal run --detach train_cloud.py::fid_eval --cfg-scale 1.5

# Generate a sample grid from a specific checkpoint
modal run train_cloud.py::sample_grid --checkpoint dart_small_step800000.safetensors

# Download the final checkpoint to use with Rust inference
modal volume get --force dart-data checkpoints/dart_small_step800000.safetensors ./dart_small.safetensors
```

Training details (§B.2): AdamW (lr=3e-4, betas=0.9/0.95), cosine LR with 10K warmup, gradient clip 2.0, EMA decay 0.9999, bf16 mixed precision. Checkpoints persist on a Modal volume so timeouts and restarts don't lose work.

If your safetensors checkpoint was saved from a `torch.compile`d model, strip the `_orig_mod.` prefix before loading in Rust:

```bash
python scripts/strip_compile_prefix.py raw_ckpt.safetensors dart_small.safetensors
```

## Project Structure

```
src/                        # Rust inference engine
├── main.rs                 # CLI: generate, info
├── lib.rs
├── vae.rs                  # SD v1.4 VAE decoder, unpatchify, PNG output
├── model/
│   ├── config.rs           # Model configurations (S/B/L/XL)
│   ├── transformer.rs      # Attention (RoPE + block-wise causal), SwiGLU, AdaLN
│   └── dart.rs             # Full DART model
└── diffusion/
    ├── schedule.rs         # Cosine noise schedule, v-prediction
    └── sampling.rs         # Non-Markovian sampling loop with CFG

train_cloud.py              # Modal app: training, FID eval, sample grids
train/train.py              # DART model definitions (imported by train_cloud.py)
scripts/strip_compile_prefix.py  # Utility to clean torch.compile'd checkpoints
```

## Paper Reference

```
@inproceedings{gu2025dart,
  title={DART: Denoising Autoregressive Transformer for Scalable Text-to-Image Generation},
  author={Gu, Jiatao and Wang, Yuyang and Zhang, Yizhe and Zhang, Qihang and Zhang, Dinghuai and Lipman, Yaron and Susskind, Josh and Benyosef, Navdeep},
  booktitle={International Conference on Learning Representations},
  year={2025}
}
```
