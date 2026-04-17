// DART: Denoising Autoregressive Transformer for Scalable Text-to-Image Generation
// Paper: https://arxiv.org/abs/2410.08159 (ICLR 2025)
//
// Usage:
//   dart generate --model-size xl --class 207 --steps 16 --cfg-scale 1.5
//   dart generate --model-size small --steps 4   (dry-run with random weights)
//   dart info --model-size xlarge

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use clap::{Parser, Subcommand};

use dart::diffusion::DartSampler;
use dart::model::{DartConfig, DartModel};
use dart::vae;

#[derive(Parser)]
#[command(name = "dart", about = "DART image generation")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate images using DART.
    Generate {
        /// Model size: small, base, large, xlarge
        #[arg(long, default_value = "base")]
        model_size: String,

        /// ImageNet class ID (0-999). Use 1000 for unconditional.
        #[arg(long, default_value_t = 207)]
        class: u32,

        /// Number of denoising steps T.
        #[arg(long, default_value_t = 16)]
        steps: usize,

        /// Classifier-free guidance scale.
        #[arg(long, default_value_t = 1.5)]
        cfg_scale: f64,

        /// Number of classes (10 for CIFAR-10, 1000 for ImageNet).
        #[arg(long, default_value_t = 1000)]
        num_classes: usize,

        /// Batch size (number of images to generate).
        #[arg(long, default_value_t = 1)]
        batch: usize,

        /// Path to DART model weights (safetensors format).
        #[arg(long)]
        weights: Option<String>,

        /// Path to VAE weights. If omitted, downloads SD v1.4 VAE from HuggingFace.
        #[arg(long)]
        vae_weights: Option<String>,

        /// Output directory for generated images.
        #[arg(long, default_value = "output")]
        output_dir: String,

        /// Skip VAE decoding (output raw patches only).
        #[arg(long, default_value_t = false)]
        no_decode: bool,
    },

    /// Print model configuration and architecture info.
    Info {
        /// Model size: small, base, large, xlarge
        #[arg(long, default_value = "xlarge")]
        model_size: String,
    },
}

fn get_config(size: &str, steps: usize) -> Result<DartConfig> {
    let mut config = match size {
        "small" | "s" => DartConfig::small(),
        "base" | "b" => DartConfig::base(),
        "large" | "l" => DartConfig::large(),
        "xlarge" | "xl" => DartConfig::xlarge(),
        _ => anyhow::bail!("Unknown model size: {size}. Use small/base/large/xlarge."),
    };
    config.num_steps = steps;
    Ok(config)
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Info { model_size } => {
            let config = get_config(&model_size, 16)?;
            println!("DART Configuration ({model_size}):");
            println!("  Layers:       {}", config.num_layers);
            println!("  Hidden size:  {}", config.hidden_size);
            println!("  Heads:        {}", config.num_heads);
            println!("  Head dim:     {}", config.head_dim);
            println!("  FFN dim:      {}", config.ffn_hidden_size());
            println!("  Patch dim:    {}", config.patch_dim());
            println!("  Params:       ~{}M", config.num_params_m);
            println!("  Steps (T):    {}", config.num_steps);
            println!("  Tokens (K):   {}", config.num_tokens);
            println!("  Total seq:    {}", config.total_seq_len());
            println!("  RoPE axes:    {:?}", config.rope_axes_dim);
            println!("  AdaLN:        {}", config.use_adaln);
            println!("  SwiGLU:       {}", config.use_swiglu);

            let schedule = dart::diffusion::CosineSchedule::new(config.num_steps);
            println!("\nCosine noise schedule (gamma_t):");
            for (i, g) in schedule.gamma.iter().enumerate() {
                let bar = "#".repeat((g * 40.0) as usize);
                println!("  t={i:>2}: gamma={g:.4} {bar}");
            }
        }

        Commands::Generate {
            model_size,
            class,
            steps,
            cfg_scale,
            num_classes,
            batch,
            weights,
            vae_weights,
            output_dir,
            no_decode,
        } => {
            let mut config = get_config(&model_size, steps)?;
            config.num_classes = num_classes;
            let device = Device::cuda_if_available(0)?;

            println!("DART Generation");
            println!("  Device:    {}", if device.is_cuda() { "CUDA (GPU)" } else { "CPU" });
            println!("  Model:     {model_size} (~{}M params)", config.num_params_m);
            println!("  Class:     {class}");
            println!("  Steps:     {steps}");
            println!("  CFG scale: {cfg_scale}");
            println!("  Batch:     {batch}");

            // Load or initialize DART model
            let model = if let Some(ref path) = weights {
                println!("  Weights:   {path}");
                let vb = unsafe {
                    VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &device)?
                };
                DartModel::new(config.clone(), vb)?
            } else {
                println!("  Weights:   (random init -- dry run)");
                let var_map = candle_nn::VarMap::new();
                let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
                DartModel::new(config.clone(), vb)?
            };

            // Run sampling
            let sampler = DartSampler::new(config.num_steps);
            let class_ids = Tensor::full(class, batch, &device)?;

            println!("\nSampling ({steps} denoising steps)...");
            let patches = sampler.sample(&model, &class_ids, cfg_scale, &device)?;
            let (b, k, pd) = patches.dims3()?;
            println!("  Patches: ({b}, {k}, {pd})");


            if no_decode || weights.is_none() {
                if weights.is_none() {
                    println!("\n  Architecture verified (random weights).");
                    println!("  Provide --weights <path> for real generation.");
                } else {
                    println!("  Skipping VAE decode (--no-decode).");
                }
                return Ok(());
            }

            // Load VAE
            let vae_path = if let Some(ref p) = vae_weights {
                println!("  VAE:       {p}");
                p.clone()
            } else {
                println!("  Downloading SD v1.4 VAE from HuggingFace...");
                let path = vae::download_vae_weights()?;
                let p = path.to_string_lossy().to_string();
                println!("  VAE:       {p}");
                p
            };

            let vae_model = vae::load_vae(&vae_path, &device)?;

            // Decode patches to images
            println!("  Decoding latents to pixels...");
            let pixels = vae::patches_to_pixels(&patches, &config, &vae_model)?;
            let (b, c, h, w) = pixels.dims4()?;
            println!("  Pixels: ({b}, {c}, {h}, {w})");

            // Save images
            std::fs::create_dir_all(&output_dir)?;
            for i in 0..b {
                let img_tensor = pixels.get(i)?; // (3, H, W)
                let path = format!("{output_dir}/dart_class{class}_{i:03}.png");
                vae::save_image(&img_tensor, &path)?;
            }

            println!("\nDone. Generated {b} image(s) in {output_dir}/");
        }
    }

    Ok(())
}
