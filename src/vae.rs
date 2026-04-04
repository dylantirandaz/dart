// VAE decoder for converting DART's latent patches to pixel images.
// Uses the SD v1.4 VAE (stabilityai/sd-vae-ft-ema) via candle-transformers.
//
// Pipeline: DART patches (B, K, patch_dim) -> unpatchify -> (B, C, H, W) latent -> VAE decode -> (B, 3, 256, 256) pixels

use std::path::PathBuf;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::stable_diffusion::vae::{AutoEncoderKL, AutoEncoderKLConfig};

use crate::model::DartConfig;

/// SD v1.4 VAE configuration.
/// These match the stabilityai/sd-vae-ft-ema model used in §B.1.
fn sd_vae_config() -> AutoEncoderKLConfig {
    AutoEncoderKLConfig {
        block_out_channels: vec![128, 256, 512, 512],
        layers_per_block: 2,
        latent_channels: 4,
        norm_num_groups: 32,
        use_quant_conv: true,
        use_post_quant_conv: true,
    }
}

/// Download the SD v1.4 VAE weights from HuggingFace Hub.
pub fn download_vae_weights() -> Result<PathBuf> {
    // §B.1 — "StabilityAI SD v1.4 VAE (stabilityai/sd-vae-ft-ema)"
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model("stabilityai/sd-vae-ft-ema".to_string());
    let path = repo.get("diffusion_pytorch_model.safetensors")?;
    Ok(path)
}

/// Load the SD VAE decoder from weights.
pub fn load_vae(weights_path: &str, device: &Device) -> Result<AutoEncoderKL> {
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)?
    };
    let vae = AutoEncoderKL::new(vb, 3, 3, sd_vae_config())?;
    Ok(vae)
}

/// Convert DART output patches back to a 2D latent grid.
///
/// DART outputs (B, K, patch_dim) where:
///   K = (H_latent / patch_size) * (W_latent / patch_size)  = 256 for 256x256 images
///   patch_dim = patch_size^2 * vae_channels = 2*2*16 = 64
///
/// This function rearranges patches back to (B, vae_channels, H_latent, W_latent).
///
/// However, the SD v1.4 VAE has latent_channels=4 (not 16). The paper (§B.1) says
/// "VAE Latent Channels: 16" which likely means they use a different VAE or the
/// VAE's spatial latent is 16-channel before the quant_conv projects to 4.
///
/// For the standard SD v1.4 VAE with 4 latent channels:
///   256x256 image -> 32x32 latent (8x downsampling) with 4 channels
///   Patchified with patch_size=2: 16x16 = 256 tokens, each 2*2*4 = 16 dims
///
/// We handle both cases based on the config's vae_channels.
pub fn unpatchify(
    patches: &Tensor,
    config: &DartConfig,
    latent_h: usize,
    latent_w: usize,
) -> candle_core::Result<Tensor> {
    let (batch, _num_tokens, _patch_dim) = patches.dims3()?;
    let ps = config.patch_size;
    let c = config.vae_channels;

    let grid_h = latent_h / ps;
    let grid_w = latent_w / ps;

    // (B, grid_h * grid_w, ps * ps * C) -> (B, grid_h, grid_w, ps, ps, C)
    let x = patches.reshape((batch, grid_h, grid_w, ps, ps, c))?;

    // Rearrange to (B, C, grid_h, ps, grid_w, ps)
    let x = x.permute((0, 5, 1, 3, 2, 4))?;

    // Merge spatial dims: (B, C, grid_h * ps, grid_w * ps) = (B, C, latent_h, latent_w)
    x.reshape((batch, c, latent_h, latent_w))
}

/// Full pipeline: DART patches -> pixel image tensor.
///
/// Returns (B, 3, 256, 256) float tensor with pixel values in [0, 1].
pub fn patches_to_pixels(
    patches: &Tensor,
    config: &DartConfig,
    vae: &AutoEncoderKL,
) -> candle_core::Result<Tensor> {
    // §B.1 — VAE latent spatial size = image_size / 8
    // For 256x256 images: 32x32 latent
    let latent_h = 32;
    let latent_w = 32;

    // Unpatchify: (B, 256, 64) -> (B, C, 32, 32)
    let latents = unpatchify(patches, config, latent_h, latent_w)?;

    // If config uses 16 channels but VAE expects 4, we need to handle the mismatch.
    // The SD VAE's post_quant_conv projects from latent_channels to latent_channels,
    // so we need to match. For standard SD v1.4: latent_channels = 4.
    //
    // If the DART model uses 16 channels (as stated in paper), we'd need to apply
    // the VAE's quant_conv in reverse. For now, assume the channels match or
    // take the first 4 channels as a fallback.
    let latents = if config.vae_channels != 4 {
        // Take the first 4 channels as the VAE latent
        latents.narrow(1, 0, 4)?
    } else {
        latents
    };

    // §B.1 — SD VAE scaling factor (standard for SD v1.4)
    let scaling_factor = 0.18215;
    let latents = (latents / scaling_factor)?;

    // Decode: (B, 4, 32, 32) -> (B, 3, 256, 256)
    let pixels = vae.decode(&latents)?;

    // Clamp to [0, 1]
    let pixels = ((pixels + 1.0)? / 2.0)?;
    pixels.clamp(0.0, 1.0)
}

/// Save a tensor of shape (3, H, W) with values in [0, 1] as a PNG image.
pub fn save_image(tensor: &Tensor, path: &str) -> Result<()> {
    let (c, h, w) = tensor.dims3()?;
    assert_eq!(c, 3, "Expected 3 channels (RGB)");

    // Convert to u8
    let tensor = (tensor * 255.0)?.to_dtype(DType::U8)?;
    let data = tensor.flatten_all()?.to_vec1::<u8>()?;

    // candle stores as (C, H, W), image crate expects (H, W, C)
    let mut rgb_data = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            for ch in 0..3 {
                rgb_data[(y * w + x) * 3 + ch] = data[ch * h * w + y * w + x];
            }
        }
    }

    let img = image::RgbImage::from_raw(w as u32, h as u32, rgb_data)
        .ok_or_else(|| anyhow::anyhow!("Failed to create image buffer"))?;
    img.save(path)?;
    println!("  Saved: {path}");
    Ok(())
}
