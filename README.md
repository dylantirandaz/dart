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

## Training (Python)

Training uses PyTorch for its mature ecosystem. Weights export to safetensors for Rust inference.

```bash
pip install -r train/requirements.txt

# Train DART-S on an image folder (testing)
python train/train.py --size small --data-dir /path/to/images --epochs 50

# Train DART-XL on ImageNet (full paper recipe, needs GPU)
python train/train.py --size xlarge --data-dir /path/to/imagenet --steps 500000

# Weights are saved to checkpoints/dart_{size}_step{N}.safetensors
# Then use with Rust inference:
cargo run -- generate --weights checkpoints/dart_small_step50000.safetensors --class 207
```

Training details (§B.2): AdamW (lr=3e-4, betas=0.9/0.95), cosine LR with 10k warmup, gradient clip 2.0, EMA decay 0.9999, bf16 mixed precision.

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

train/                      # Python training
├── train.py                # Full training script with VAE encoding, EMA, safetensors export
└── requirements.txt
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
