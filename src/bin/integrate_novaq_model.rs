use clap::{Parser, Subcommand};
use ohms_adaptq::{PublicNOVAQ, NOVAQModel};
use std::fs;
use std::path::Path;

#[derive(Parser)]
#[command(name = "integrate-novaq")]
#[command(about = "Integrate NOVAQ compressed models into OHMS catalog")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Integrate a NOVAQ compressed model into OHMS catalog
    Integrate {
        /// Path to NOVAQ compressed model file
        #[arg(short, long)]
        model_path: String,
        
        /// Model ID for the catalog
        #[arg(short, long)]
        model_id: String,
        
        /// Source model identifier
        #[arg(short, long, default_value = "unknown")]
        source_model: String,
        
        /// Output directory for catalog files
        #[arg(short, long, default_value = "catalog")]
        output_dir: String,
    },
    
    /// List available NOVAQ models in catalog
    List {
        /// Catalog directory
        #[arg(short, long, default_value = "catalog")]
        catalog_dir: String,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    
    match &cli.command {
        Commands::Integrate { model_path, model_id, source_model, output_dir } => {
            integrate_model(model_path, model_id, source_model, output_dir)?;
        }
        Commands::List { catalog_dir } => {
            list_models(catalog_dir)?;
        }
    }
    
    Ok(())
}

fn integrate_model(model_path: &str, model_id: &str, source_model: &str, output_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔗 Integrating NOVAQ model into OHMS catalog...");
    println!("Model: {}", model_path);
    println!("ID: {}", model_id);
    println!("Source: {}", source_model);
    
    // Load the NOVAQ model
    let model_data = fs::read(model_path)?;
    let model: NOVAQModel = bincode::deserialize(&model_data)?;
    
    // Create NOVAQ engine for validation
    let novaq = PublicNOVAQ::new(model.config.clone());
    let validation = novaq.validate_model(&model).map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
    let stats = novaq.get_compression_stats(&model);
    
    println!("✅ Model loaded successfully");
    println!("Compression Ratio: {:.1}x", stats.compression_ratio);
    println!("Bit Accuracy: {:.3}%", stats.bit_accuracy * 100.0);
    println!("Quality Score: {:.3}", stats.quality_score);
    println!("Validation Status: {}", if validation.passed_validation { "✅ PASSED" } else { "❌ FAILED" });
    
    // Create output directory
    let output_path = Path::new(output_dir);
    fs::create_dir_all(output_path)?;
    
    // Create catalog metadata
    let catalog_entry = create_catalog_entry(&model, model_id, source_model, &stats, &validation)?;
    
    // Save catalog entry
    let catalog_file = output_path.join(format!("{}.json", model_id));
    let catalog_json = serde_json::to_string_pretty(&catalog_entry)?;
    fs::write(&catalog_file, catalog_json)?;
    
    // Copy model file to catalog
    let model_file = output_path.join(format!("{}.bin", model_id));
    fs::copy(model_path, &model_file)?;
    
    println!("✅ Model integrated successfully!");
    println!("Catalog entry: {}", catalog_file.display());
    println!("Model file: {}", model_file.display());
    
    Ok(())
}

fn list_models(catalog_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    let catalog_path = Path::new(catalog_dir);
    if !catalog_path.exists() {
        println!("❌ Catalog directory does not exist: {}", catalog_dir);
        return Ok(());
    }
    
    println!("📋 Available NOVAQ models in catalog:");
    println!("Directory: {}", catalog_dir);
    println!();
    
    for entry in fs::read_dir(catalog_path)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            let catalog_data = fs::read_to_string(&path)?;
            let catalog_entry: serde_json::Value = serde_json::from_str(&catalog_data)?;
            
            if let Some(model_id) = catalog_entry["model_id"].as_str() {
                if let Some(compression_ratio) = catalog_entry["compression_stats"]["compression_ratio"].as_f64() {
                    if let Some(bit_accuracy) = catalog_entry["compression_stats"]["bit_accuracy"].as_f64() {
                        println!("📦 {}: {:.1}x compression, {:.1}% accuracy", 
                                model_id, compression_ratio, bit_accuracy * 100.0);
                    }
                }
            }
        }
    }
    
    Ok(())
}

#[derive(serde::Serialize)]
struct CatalogEntry {
    model_id: String,
    source_model: String,
    compression_stats: CompressionStats,
    validation: ValidationInfo,
    integration_date: String,
    model_file: String,
}

#[derive(serde::Serialize)]
struct CompressionStats {
    compression_ratio: f64,
    bit_accuracy: f64,
    quality_score: f64,
    target_bits: f64,
    subspaces: u32,
    l1_codebook_size: u32,
    l2_codebook_size: u32,
}

#[derive(serde::Serialize)]
struct ValidationInfo {
    passed_validation: bool,
    quality_score: f64,
    issues: Vec<String>,
}

fn create_catalog_entry(
    model: &NOVAQModel,
    model_id: &str,
    source_model: &str,
    stats: &ohms_adaptq::CompressionStats,
    validation: &ohms_adaptq::ValidationReport,
) -> Result<CatalogEntry, Box<dyn std::error::Error>> {
    Ok(CatalogEntry {
        model_id: model_id.to_string(),
        source_model: source_model.to_string(),
        compression_stats: CompressionStats {
            compression_ratio: stats.compression_ratio as f64,
            bit_accuracy: stats.bit_accuracy as f64,
            quality_score: stats.quality_score as f64,
            target_bits: model.config.target_bits as f64,
            subspaces: model.config.num_subspaces as u32,
            l1_codebook_size: model.config.codebook_size_l1 as u32,
            l2_codebook_size: model.config.codebook_size_l2 as u32,
        },
        validation: ValidationInfo {
            passed_validation: validation.passed_validation,
            quality_score: validation.quality_score as f64,
            issues: validation.issues.clone(),
        },
        integration_date: chrono::Utc::now().to_rfc3339(),
        model_file: format!("{}.bin", model_id),
    })
}
