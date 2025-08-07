use clap::{Parser, Subcommand};
use ohms_adaptq::{
    VerificationConfig, VerificationEngine, Quantizer, QuantizationConfig, QuantizationMethod,
    UniversalLoader, ModelFetcher, parse_model_source,
};

#[derive(Parser)]
#[command(name = "apq", version, about = "OHMS Adaptive Quantization - Production CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Quantize {
        #[arg(long, help = "Model source: hf:<repo>[:file] | url:<http(s)://...> | ollama:<name> | file:/abs/path | /abs/path")] model: String,
        #[arg(long, default_value = "SpinQuant")] method: String,
        #[arg(long, default_value_t = 1.58)] weight_bits: f32,
        #[arg(long, default_value_t = 4)] activation_bits: u8,
        #[arg(long, default_value_t = 128)] group_size: usize,
        #[arg(long, default_value_t = true)] per_channel: bool,
    },
    Verify {
        #[arg(long)] original: String,
        #[arg(long)] quantized: String,
    },
    Info {
        #[arg(long, help = "Model source: hf:<repo>[:file] | url:<http(s)://...> | ollama:<name> | file:/abs/path | /abs/path")] 
        model: Option<String>,
    },
}

fn parse_method(name: &str) -> QuantizationMethod {
    match name.to_lowercase().as_str() {
        "spinquant" => QuantizationMethod::SpinQuant,
        "ternaryllmdlt" | "ternary" => QuantizationMethod::TernaryLLMDLT,
        "vptq" => QuantizationMethod::VPTQ,
        "duquant" => QuantizationMethod::DuQuant,
        "hyperternary" => QuantizationMethod::HyperTernary,
        "adaptivebits" => QuantizationMethod::AdaptiveBits,
        "zeroshot" => QuantizationMethod::ZeroShot,
        "neuralquant" => QuantizationMethod::NeuralQuant,
        "gptq" => QuantizationMethod::GPTQ,
        "awq" => QuantizationMethod::AWQ,
        "smoothquant" => QuantizationMethod::SmoothQuant,
        "bitnet" => QuantizationMethod::BitNet,
        "int4" => QuantizationMethod::INT4,
        "int8" => QuantizationMethod::INT8,
        _ => QuantizationMethod::SpinQuant,
    }
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Quantize { model, method, weight_bits, activation_bits, group_size, per_channel } => {
            let method = parse_method(&method);
            let config = QuantizationConfig {
                method,
                weight_bits,
                activation_bits,
                group_size,
                use_symmetric: true,
                per_channel,
                calibration_samples: 512,
            };

            // Remote-aware fetch → local path or special handle
            let source = parse_model_source(&model);
            let fetched = ModelFetcher::fetch(&source).map_err(|e| anyhow::anyhow!(e.to_string()))?;

            let mut loader = UniversalLoader::new();
            let model = loader
                .load_model(&fetched.local_path)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            let mut quantizer = Quantizer::new(config);
            let result = quantizer
                .quantize_model(&model)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            println!("Quantized {}: {:.1}x compression", result.metadata.original_model, result.metadata.compression_ratio);
        }
        Commands::Verify { original, quantized } => {
            let engine = VerificationEngine::new(VerificationConfig::default());
            let report = futures::executor::block_on(engine.verify_model(&original, &quantized))
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            println!("{}", engine.generate_report_string(&report));
        }
        Commands::Info { model } => {
            if let Some(model) = model {
                let source = parse_model_source(&model);
                let fetched = ModelFetcher::fetch(&source).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                let mut loader = UniversalLoader::new();
                let m = loader
                    .load_model(&fetched.local_path)
                    .map_err(|e| anyhow::anyhow!(e.to_string()))?;
                let total_params: u64 = m.layers.iter().map(|l| l.parameters).sum();
                println!(
                    "Model: {}\nFormat: {:?}\nArch: {}\nLayers: {}\nParams (sum): {}\nHidden size: {}\nHeads: {}\nContext length: {}\nPrecision: {:?}",
                    m.metadata.name,
                    m.format,
                    m.metadata.architecture,
                    m.layers.len(),
                    total_params,
                    m.metadata.hidden_size,
                    m.metadata.num_heads,
                    m.metadata.context_length,
                    m.metadata.precision
                );
            } else {
                println!("APQ CLI - available methods: SpinQuant, TernaryLLMDLT, VPTQ, DuQuant, HyperTernary, AdaptiveBits, ZeroShot, NeuralQuant, GPTQ, AWQ, SmoothQuant, BitNet, INT4, INT8");
                println!("\nUsage: apq info --model <hf:repo[:file]|url:...|ollama:name|file:/abs/path|/abs/path>");
            }
        }
    }

    Ok(())
}

