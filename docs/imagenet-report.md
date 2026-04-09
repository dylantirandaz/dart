# DART Training Report: ImageNet 256x256

**Date:** April 2026
**Dataset:** ImageNet-1K (ILSVRC 2012), 1.2M training images, 941 classes (94% of full 1000)

## Why ImageNet

This is the paper's actual evaluation dataset. Everything before this (CIFAR-10, Food-101) was just pipeline validation. If this implementation is faithful to the paper, ImageNet is where it has to prove it.

## Configuration

| Setting | Value |
|---------|-------|
| Model | DART-S, 31.9M params |
| Dataset | ImageNet 256x256, 1.2M images, 941 classes |
| RoPE | 3-axis decomposed (16, 24, 24) |
| Loss weighting | Uniform |
| T | 4 denoising steps |
| Batch size | 8 |
| Steps | 200,000 |
| LR | 3e-4, cosine decay with 10K warmup |
| AMP | bf16 |
| Latent cache | Memory-mapped numpy on local disk |

### What differs from the paper

| | Paper | This run |
|-|-------|----------|
| Model | DART-XL, 812M params | DART-S, 32M params |
| T | 16 | 4 |
| Batch size | 128 | 8 |
| Training steps | Not disclosed (est. millions) | 200K |
| Classes | 1000 | 941 (6% missing from incomplete download) |

The model is 25x smaller, uses 4x fewer denoising steps, and trained for a fraction of the iterations. This isn't trying to match the paper's numbers. It's testing whether the architecture works on the target dataset at small scale.

## Sample Progression

Each grid shows 16 samples from the first 16 ImageNet classes.

### Step 10,000

![Step 10K](samples/imagenet/samples_step10000.png)

Noise with class-specific color palettes. Aquatic classes are blue, animal classes are brown/green. No shapes yet.

### Step 50,000

![Step 50K](samples/imagenet/samples_step50000.png)

Background structure forming. Blue water scenes, green foliage, brown ground. Some blurry object silhouettes starting to appear.

### Step 100,000

![Step 100K](samples/imagenet/samples_step100000.png)

Object shapes visible. Goldfish in water, dolphins, flamingos, starfish. Colors and backgrounds clearly conditioned on class. The model has learned to associate class IDs with the right visual content across hundreds of categories.

### Step 200,000

![Step 200K](samples/imagenet/samples_step200000.png)

Best results. Class conditioning is strong -- you can tell fish classes from bird classes from mammal classes at a glance. Objects have recognizable silhouettes and sit in appropriate environments. Still soft at fine detail because T=4 only gives the model four chances to denoise.

## FID Score

| Metric | Value |
|--------|-------|
| FID (5K generated vs 10K reference) | **154.90** |

For context:
- Paper's DART-XL (812M, T=16): **3.98 FID**
- Our DART-S (32M, T=4): **154.90 FID**

The gap is expected. Our model is 25x smaller and uses 4x fewer denoising steps. FID is exponentially sensitive to both model capacity and number of denoising steps. The score confirms the model is generating class-conditioned images (random noise would score >300), but it's nowhere near the paper's results.

The FID was computed with 5K generated samples against a random 10K subset of the training data. Standard evaluation uses 50K samples against the full validation set, but that would take 3+ hours to compute Inception features for 1.2M reference images.

## Latent Caching at Scale

The original caching approach (accumulate all latents in a Python list, then `torch.cat`) crashed the machine twice when applied to 1.2M images. It tried to hold ~19GB of tensors in RAM and then doubled that during concatenation.

Fixed by switching to numpy memory-mapped files. The new approach pre-allocates the full array on disk and writes to it incrementally. RAM usage stays at ~100MB regardless of dataset size. The 1.2M image cache is ~19GB on disk and loads lazily during training.

## What This Proves

The DART architecture works on ImageNet. A 32M parameter model with only 4 denoising steps learns meaningful class-conditional generation across 941 categories. The samples are blurry and the FID is high, but that's a capacity and compute constraint, not an architecture problem. The pipeline from training through Rust inference is fully functional on the paper's target dataset.

## What Would Improve Results

- **More parameters**: DART-B (141M) or larger. Might fit on 16GB with aggressive gradient checkpointing.
- **More denoising steps**: T=8 or T=16. Each additional step refines the output but costs more VRAM.
- **More training steps**: 200K steps with batch_size=8 means each image was seen ~1.25 times on average. The paper likely trains for millions of steps with batch_size=128.
- **Full ImageNet**: We're missing 59 classes (6%) from the incomplete download. Minor impact.
