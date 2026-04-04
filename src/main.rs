// DART: Denoising Autoregressive Transformer for Scalable Text-to-Image Generation
// Paper: https://arxiv.org/abs/2410.08159 (ICLR 2025)
//
// Usage:
//   dart generate --model-size xl --class 207 --steps 16 --cfg-scale 1.5

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use clap::{Parser, Subcommand};

use dart::model::{DartConfig, DartModel};
use dart::diffusion::DartSampler;

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

        /// Batch size (number of images to generate).
        #[arg(long, default_value_t = 1)]
        batch: usize,

        /// Path to model weights (safetensors format).
        #[arg(long)]
        weights: Option<String>,
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

            // Show noise schedule
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
            batch,
            weights,
        } => {
            let config = get_config(&model_size, steps)?;
            let device = Device::Cpu;

            println!("DART Generation");
            println!("  Model:     {model_size} (~{}M params)", config.num_params_m);
            println!("  Class:     {class}");
            println!("  Steps:     {steps}");
            println!("  CFG scale: {cfg_scale}");
            println!("  Batch:     {batch}");

            if let Some(ref path) = weights {
                println!("  Weights:   {path}");
                let vb = unsafe {
                    VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &device)?
                };
                let model = DartModel::new(config.clone(), vb)?;
                let sampler = DartSampler::new(config.num_steps);

                let class_ids = Tensor::full(class, batch, &device)?;

                println!("\nSampling...");
                let output = sampler.sample(&model, &class_ids, cfg_scale, &device)?;
                let (b, k, pd) = output.dims3()?;
                println!("  Output: ({b}, {k}, {pd})");
                println!("  Done. Output contains VAE latent patches.");
                println!("  Decode with the SD v1.4 VAE to get pixel images.");
            } else {
                println!("  Weights:   (none -- dry run with random init)");
                println!("\nInitializing model with random weights...");

                let var_map = candle_nn::VarMap::new();
                let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
                let model = DartModel::new(config.clone(), vb)?;

                let sampler = DartSampler::new(config.num_steps);
                let class_ids = Tensor::full(class, batch, &device)?;

                println!("Running {steps}-step sampling (random weights -- output is noise)...");
                let output = sampler.sample(&model, &class_ids, cfg_scale, &device)?;
                let (b, k, pd) = output.dims3()?;
                println!("  Output shape: ({b}, {k}, {pd})");
                println!("  Patch dim: {pd} = {}x{}x{}", config.patch_size, config.patch_size, config.vae_channels);
                println!("\n  Architecture verified. Provide --weights <path> for real generation.");
            }
        }
    }

    Ok(())
}
