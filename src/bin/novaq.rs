use clap::{Parser, Subcommand};
use ohms_adaptq::{PublicNOVAQ, NOVAQConfig, CompressionStats};
use std::path::PathBuf;
use indicatif::{ProgressBar, ProgressStyle};
use nu_ansi_term::Color::{Green, Blue, Yellow, Red};

#[derive(Parser)]
#[command(name = "novaq")]
#[command(about = "NOVAQ - Normalized Outlier-Vector Additive Quantization")]
#[command(version = "2.0")]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compress model from Hugging Face repository
    #[command(name = "hf")]
    HuggingFace {
        /// Hugging Face repository (e.g., "meta-llama/Llama-3-8B")
        repo: String,
        /// Specific file to download (optional)
        #[arg(short, long)]
        file: Option<String>,
        /// Output path for compressed model
        #[arg(short, long, default_value = "novaq_compressed.bin")]
        output: PathBuf,
        /// Target bits per weight (default: 1.5)
        #[arg(short, long, default_value = "1.5")]
        bits: f32,
        /// Number of subspaces (default: 4)
        #[arg(short, long, default_value = "4")]
        subspaces: usize,
    },
    
    /// Compress model from Ollama
    #[command(name = "ollama")]
    Ollama {
        /// Ollama model name (e.g., "llama3:8b")
        model: String,
        /// Output path for compressed model
        #[arg(short, long, default_value = "novaq_compressed.bin")]
        output: PathBuf,
        /// Target bits per weight (default: 1.5)
        #[arg(short, long, default_value = "1.5")]
        bits: f32,
        /// Number of subspaces (default: 4)
        #[arg(short, long, default_value = "4")]
        subspaces: usize,
    },
    
    /// Compress model from URL
    #[command(name = "url")]
    Url {
        /// URL to download model from
        url: String,
        /// Output path for compressed model
        #[arg(short, long, default_value = "novaq_compressed.bin")]
        output: PathBuf,
        /// Target bits per weight (default: 1.5)
        #[arg(short, long, default_value = "1.5")]
        bits: f32,
        /// Number of subspaces (default: 4)
        #[arg(short, long, default_value = "4")]
        subspaces: usize,
    },
    
    /// Compress local model file
    #[command(name = "local")]
    Local {
        /// Path to local model file
        path: PathBuf,
        /// Output path for compressed model
        #[arg(short, long, default_value = "novaq_compressed.bin")]
        output: PathBuf,
        /// Target bits per weight (default: 1.5)
        #[arg(short, long, default_value = "1.5")]
        bits: f32,
        /// Number of subspaces (default: 4)
        #[arg(short, long, default_value = "4")]
        subspaces: usize,
    },
    
    /// Validate NOVAQ compressed model
    #[command(name = "validate")]
    Validate {
        /// Path to NOVAQ compressed model
        path: PathBuf,
    },
    
    /// Show NOVAQ compression statistics
    #[command(name = "stats")]
    Stats {
        /// Path to NOVAQ compressed model
        path: PathBuf,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let cli = Cli::parse();
    
    match &cli.command {
        Commands::HuggingFace { repo, file, output, bits, subspaces } => {
            compress_hf_model(repo, file.as_deref(), output, *bits, *subspaces)?;
        },
        Commands::Ollama { model, output, bits, subspaces } => {
            compress_ollama_model(model, output, *bits, *subspaces)?;
        },
        Commands::Url { url, output, bits, subspaces } => {
            compress_url_model(url, output, *bits, *subspaces)?;
        },
        Commands::Local { path, output, bits, subspaces } => {
            compress_local_model(path, output, *bits, *subspaces)?;
        },
        Commands::Validate { path } => {
            validate_model(path)?;
        },
        Commands::Stats { path } => {
            show_stats(path)?;
        },
    }
    
    Ok(())
}

fn compress_hf_model(repo: &str, file: Option<&str>, output: &PathBuf, bits: f32, subspaces: usize) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("{}", Green.bold().paint("🚀 NOVAQ Democratic Compression"));
    println!("{}", Blue.paint("Source: Hugging Face"));
    println!("{}", Blue.paint(&format!("Repository: {}", repo)));
    if let Some(f) = file {
        println!("{}", Blue.paint(&format!("File: {}", f)));
    }
    println!("{}", Blue.paint(&format!("Output: {}", output.display())));
    println!();
    
    let config = NOVAQConfig {
        target_bits: bits,
        num_subspaces: subspaces,
        ..Default::default()
    };
    
    let mut novaq = PublicNOVAQ::new(config);
    
    let progress = ProgressBar::new_spinner();
    progress.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {wide_msg}")
            .unwrap()
    );
    
    progress.set_message("Fetching model from Hugging Face...");
    let model = novaq.compress_hf_model(repo, file).map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
    progress.finish_and_clear();
    
    // Save compressed model
    progress.set_message("Saving compressed model...");
    let model_data = bincode::serialize(&model).map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
    std::fs::write(output, model_data)?;
    progress.finish_and_clear();
    
    // Show results
    let stats = novaq.get_compression_stats(&model);
    show_compression_results(&stats, &model)?;
    
    Ok(())
}

fn compress_ollama_model(model: &str, output: &PathBuf, bits: f32, subspaces: usize) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("{}", Green.bold().paint("🚀 NOVAQ Democratic Compression"));
    println!("{}", Blue.paint("Source: Ollama"));
    println!("{}", Blue.paint(&format!("Model: {}", model)));
    println!("{}", Blue.paint(&format!("Output: {}", output.display())));
    println!();
    
    let config = NOVAQConfig {
        target_bits: bits,
        num_subspaces: subspaces,
        ..Default::default()
    };
    
    let mut novaq = PublicNOVAQ::new(config);
    
    let progress = ProgressBar::new_spinner();
    progress.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {wide_msg}")
            .unwrap()
    );
    
    progress.set_message("Fetching model from Ollama...");
    let model = novaq.compress_ollama_model(model).map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
    progress.finish_and_clear();
    
    // Save compressed model
    progress.set_message("Saving compressed model...");
    let model_data = bincode::serialize(&model).map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
    std::fs::write(output, model_data)?;
    progress.finish_and_clear();
    
    // Show results
    let stats = novaq.get_compression_stats(&model);
    show_compression_results(&stats, &model)?;
    
    Ok(())
}

fn compress_url_model(url: &str, output: &PathBuf, bits: f32, subspaces: usize) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("{}", Green.bold().paint("🚀 NOVAQ Democratic Compression"));
    println!("{}", Blue.paint("Source: URL"));
    println!("{}", Blue.paint(&format!("URL: {}", url)));
    println!("{}", Blue.paint(&format!("Output: {}", output.display())));
    println!();
    
    let config = NOVAQConfig {
        target_bits: bits,
        num_subspaces: subspaces,
        ..Default::default()
    };
    
    let mut novaq = PublicNOVAQ::new(config);
    
    let progress = ProgressBar::new_spinner();
    progress.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {wide_msg}")
            .unwrap()
    );
    
    progress.set_message("Fetching model from URL...");
    let model = novaq.compress_url_model(url, None).map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
    progress.finish_and_clear();
    
    // Save compressed model
    progress.set_message("Saving compressed model...");
    let model_data = bincode::serialize(&model).map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
    std::fs::write(output, model_data)?;
    progress.finish_and_clear();
    
    // Show results
    let stats = novaq.get_compression_stats(&model);
    show_compression_results(&stats, &model)?;
    
    Ok(())
}

fn compress_local_model(path: &PathBuf, output: &PathBuf, bits: f32, subspaces: usize) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("{}", Green.bold().paint("🚀 NOVAQ Democratic Compression"));
    println!("{}", Blue.paint("Source: Local File"));
    println!("{}", Blue.paint(&format!("Path: {}", path.display())));
    println!("{}", Blue.paint(&format!("Output: {}", output.display())));
    println!();
    
    let config = NOVAQConfig {
        target_bits: bits,
        num_subspaces: subspaces,
        ..Default::default()
    };
    
    let mut novaq = PublicNOVAQ::new(config);
    
    let progress = ProgressBar::new_spinner();
    progress.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {wide_msg}")
            .unwrap()
    );
    
    progress.set_message("Loading local model...");
    let model = novaq.compress_local_model(path.to_str().unwrap()).map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
    progress.finish_and_clear();
    
    // Save compressed model
    progress.set_message("Saving compressed model...");
    let model_data = bincode::serialize(&model).map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
    std::fs::write(output, model_data)?;
    progress.finish_and_clear();
    
    // Show results
    let stats = novaq.get_compression_stats(&model);
    show_compression_results(&stats, &model)?;
    
    Ok(())
}

fn validate_model(path: &PathBuf) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("{}", Green.bold().paint("🔍 NOVAQ Model Validation"));
    println!("{}", Blue.paint(&format!("Model: {}", path.display())));
    println!();
    
    let model_data = std::fs::read(path)?;
    let model: ohms_adaptq::NOVAQModel = bincode::deserialize(&model_data).map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
    
    let novaq = PublicNOVAQ::new(model.config.clone());
    let validation = novaq.validate_model(&model).map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
    
    println!("{}", Yellow.bold().paint("Validation Results:"));
    println!("  Compression Ratio: {:.1}x", validation.compression_ratio);
    println!("  Bit Accuracy: {:.3}%", validation.bit_accuracy * 100.0);
    println!("  Quality Score: {:.3}", validation.quality_score);
    println!("  Status: {}", if validation.passed_validation { 
        Green.paint("✅ PASSED") 
    } else { 
        Red.paint("❌ FAILED") 
    });
    
    if !validation.issues.is_empty() {
        println!("  Issues:");
        for issue in &validation.issues {
            println!("    - {}", Red.paint(issue));
        }
    }
    
    Ok(())
}

fn show_stats(path: &PathBuf) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("{}", Green.bold().paint("📊 NOVAQ Compression Statistics"));
    println!("{}", Blue.paint(&format!("Model: {}", path.display())));
    println!();
    
    let model_data = std::fs::read(path)?;
    let model: ohms_adaptq::NOVAQModel = bincode::deserialize(&model_data).map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
    
    let novaq = PublicNOVAQ::new(model.config.clone());
    let stats = novaq.get_compression_stats(&model);
    
    println!("{}", Yellow.bold().paint("Compression Statistics:"));
    println!("  Compression Ratio: {:.1}x", stats.compression_ratio);
    println!("  Bit Accuracy: {:.3}%", stats.bit_accuracy * 100.0);
    println!("  Quality Score: {:.3}", stats.quality_score);
    println!("  Target Bits: {:.1}", stats.target_bits);
    println!("  Subspaces: {}", stats.num_subspaces);
    println!("  L1 Codebook Size: {}", stats.codebook_size_l1);
    println!("  L2 Codebook Size: {}", stats.codebook_size_l2);
    
    Ok(())
}

fn show_compression_results(stats: &CompressionStats, model: &ohms_adaptq::NOVAQModel) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("{}", Green.bold().paint("✅ NOVAQ Compression Complete!"));
    println!();
    println!("{}", Yellow.bold().paint("Compression Results:"));
    println!("  Compression Ratio: {:.1}x", stats.compression_ratio);
    println!("  Bit Accuracy: {:.3}%", stats.bit_accuracy * 100.0);
    println!("  Quality Score: {:.3}", stats.quality_score);
    println!("  Target Bits: {:.1}", stats.target_bits);
    println!("  Subspaces: {}", stats.num_subspaces);
    println!("  Weight Shapes: {}", model.weight_shapes.len());
    println!();
    println!("{}", Blue.paint("🎉 Model successfully compressed with NOVAQ!"));
    println!("{}", Blue.paint("   No restrictions, no gatekeeping - pure democratic access."));
    
    Ok(())
}