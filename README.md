# DART: Denoising Autoregressive Transformer

Rust implementation of [DART: Denoising Autoregressive Transformer for Scalable Text-to-Image Generation](https://arxiv.org/abs/2410.08159) (ICLR 2025).

DART unifies autoregressive and diffusion models by denoising image patches through a non-Markovian process using a standard transformer architecture. Unlike traditional diffusion which only conditions on the previous noisy step, DART conditions on the entire noisy trajectory — enabling more efficient learning with fewer inference steps.

## Architecture

- **Transformer**: GPT-style with RoPE positional encoding, SwiGLU FFN, per-head RMSNorm
- **Conditioning**: Adaptive LayerNorm (AdaLN) for class-conditional generation
- **Attention**: Block-wise causal mask — tokens at step t attend to all tokens at steps t..T
- **Diffusion**: Non-Markovian cosine schedule with v-prediction parameterization
- **Tokenization**: VAE latent patches (SD v1.4), patch size 2x2, 16 channels → 64-dim tokens

## Model Sizes (Table 2)

| Model | Layers | Hidden | Heads | Params |
|-------|--------|--------|-------|--------|
| DART-S | 12 | 384 | 6 | 48M |
| DART-B | 12 | 768 | 12 | 141M |
| DART-L | 24 | 1024 | 16 | 464M |
| DART-XL | 28 | 1152 | 18 | 812M |

## Usage

```bash
# Build
cargo build --release

# Print model config and noise schedule
cargo run -- info --model-size xlarge

# Dry-run generation (random weights, verifies architecture)
cargo run -- generate --model-size small --class 207 --steps 16

# Generate with trained weights (safetensors format)
cargo run -- generate --model-size xlarge --weights model.safetensors --class 207 --steps 16 --cfg-scale 1.5
```

## Project Structure

```
src/
├── main.rs                 # CLI entry point
├── lib.rs                  # Library root
├── model/
│   ├── config.rs           # Model configurations (S/B/L/XL) from Table 2
│   ├── transformer.rs      # Attention (RoPE + block-wise causal mask), SwiGLU FFN, AdaLN
│   └── dart.rs             # Full DART model: patch embed → transformer → output projection
└── diffusion/
    ├── schedule.rs         # Cosine noise schedule, v-prediction, loss weights
    └── sampling.rs         # Non-Markovian sampling loop with CFG support
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
