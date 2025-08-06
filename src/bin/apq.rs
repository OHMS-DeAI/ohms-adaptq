use clap::{Parser, Subcommand};
use ohms_adaptq::*;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "apq")]
#[command(about = "Adaptive Progressive Quantization CLI for OHMS")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Quantize a HuggingFace model
    Quantize {
        /// Model identifier or path
        model: String,
        /// Output directory for artifacts
        #[arg(short, long, default_value = "./artifacts")]
        output: PathBuf,
        /// Target bits for quantization (3-4)
        #[arg(short, long, default_value = "4")]
        bits: u8,
        /// Calibration dataset size
        #[arg(short, long, default_value = "512")]
        calibration_size: usize,
        /// Random seed for determinism
        #[arg(short, long)]
        seed: Option<u64>,
    },
    /// Verify existing artifacts
    Verify {
        /// Path to artifacts directory
        path: PathBuf,
    },
    /// Generate verification report
    Report {
        /// Path to artifacts directory
        path: PathBuf,
        /// Output format (json, yaml, text)
        #[arg(short, long, default_value = "json")]
        format: String,
    },
    /// Test determinism by running quantization multiple times
    TestDeterminism {
        /// Model identifier or path
        model: String,
        /// Number of runs to compare
        #[arg(short, long, default_value = "3")]
        runs: usize,
        /// Random seed
        #[arg(short, long, default_value = "42")]
        seed: u64,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::init();
    
    let cli = Cli::parse();

    match cli.command {
        Commands::Quantize { model, output, bits, calibration_size, seed } => {
            println!("🔧 Quantizing model: {}", model);
            
            let config = ApqConfig {
                target_bits: bits,
                calibration_size,
                chunk_size: 2 * 1024 * 1024, // 2 MiB
                fidelity_threshold: 0.1,
                seed,
            };

            let quantizer = ApqQuantizer::new(config);
            let result = quantizer.quantize_model(&model)?;
            
            // Create output directory
            std::fs::create_dir_all(&output)?;
            
            // Generate manifest
            let manifest = ManifestBuilder::from_quantization_result(&result, "1.0.0")?;
            
            // Write artifacts
            let manifest_json = serde_json::to_string_pretty(&manifest)?;
            std::fs::write(output.join("manifest.json"), manifest_json)?;
            
            // Generate model.meta (example)
            let model_meta = ManifestBuilder::build_model_meta(
                "phi", 
                &model, 
                &model, 
                32064, 
                4096, 
                "mit"
            );
            let meta_json = serde_json::to_string_pretty(&model_meta)?;
            std::fs::write(output.join("model.meta"), meta_json)?;
            
            // Generate verification report
            let calibration_set = vec!["Test prompt".to_string()];
            let verification = VerificationEngine::generate_report(&model, &[], &calibration_set)?;
            let verification_json = serde_json::to_string_pretty(&verification)?;
            std::fs::write(output.join("verification.json"), verification_json)?;
            
            // Create shards directory
            let shards_dir = output.join("shards");
            std::fs::create_dir_all(&shards_dir)?;
            
            // Write chunks
            for chunk in &result.chunks {
                std::fs::write(shards_dir.join(&chunk.id), &chunk.data)?;
            }
            
            println!("✅ Quantization complete. Artifacts written to: {}", output.display());
        }
        
        Commands::Verify { path } => {
            println!("🔍 Verifying artifacts at: {}", path.display());
            
            // Load manifest
            let manifest_path = path.join("manifest.json");
            let manifest_json = std::fs::read_to_string(manifest_path)?;
            let manifest: Manifest = serde_json::from_str(&manifest_json)?;
            
            // Load verification report
            let verification_path = path.join("verification.json");
            let verification_json = std::fs::read_to_string(verification_path)?;
            let verification: VerificationReport = serde_json::from_str(&verification_json)?;
            
            // Validate
            let warnings = VerificationEngine::validate_thresholds(&verification);
            
            if warnings.is_empty() {
                println!("✅ All verifications passed");
            } else {
                println!("⚠️  Verification warnings:");
                for warning in warnings {
                    println!("  - {}", warning);
                }
            }
            
            println!("📊 Overall status: {:?}", verification.overall_status);
        }
        
        Commands::Report { path, format } => {
            println!("📄 Generating report for: {}", path.display());
            
            let verification_path = path.join("verification.json");
            let verification_json = std::fs::read_to_string(verification_path)?;
            let verification: VerificationReport = serde_json::from_str(&verification_json)?;
            
            match format.as_str() {
                "json" => println!("{}", serde_json::to_string_pretty(&verification)?),
                "text" => {
                    println!("Verification Report");
                    println!("==================");
                    println!("Status: {:?}", verification.overall_status);
                    println!("Perplexity Delta: {:.4}", verification.perplexity_delta);
                    println!("Layer Errors: {:?}", verification.layer_errors);
                    println!("Fidelity Checks: {}/{} passed", 
                             verification.fidelity_checks.iter().filter(|c| c.passed).count(),
                             verification.fidelity_checks.len());
                }
                _ => {
                    eprintln!("❌ Unsupported format: {}", format);
                    std::process::exit(1);
                }
            }
        }
        
        Commands::TestDeterminism { model, runs, seed } => {
            println!("🎯 Testing determinism for model: {} ({} runs)", model, runs);
            
            let config = ApqConfig {
                target_bits: 4,
                calibration_size: 512,
                chunk_size: 2 * 1024 * 1024,
                fidelity_threshold: 0.1,
                seed: Some(seed),
            };
            
            let quantizer = ApqQuantizer::new(config);
            let is_deterministic = quantizer.verify_determinism(&model, runs)?;
            
            if is_deterministic {
                println!("✅ Model quantization is deterministic");
            } else {
                println!("❌ Model quantization is NOT deterministic");
                std::process::exit(1);
            }
        }
    }

    Ok(())
}