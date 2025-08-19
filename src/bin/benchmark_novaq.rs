use clap::{Parser, Subcommand};
use ohms_adaptq::{PublicNOVAQ, NOVAQConfig, NOVAQModel, VerbosityLevel};
use std::fs;
use std::path::Path;
use std::time::Instant;
use serde::{Serialize, Deserialize};

#[derive(Parser)]
#[command(name = "benchmark-novaq")]
#[command(about = "Comprehensive NOVAQ compression benchmarking suite")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run comprehensive NOVAQ benchmarks
    Run {
        /// Output directory for benchmark results
        #[arg(short, long, default_value = "benchmarks")]
        output_dir: String,
        
        /// Number of benchmark iterations
        #[arg(short, long, default_value = "3")]
        iterations: u32,
        
        /// Verbosity level
        #[arg(short, long, default_value = "silent")]
        verbosity: String,
    },
    
    /// Benchmark specific model
    Model {
        /// HuggingFace model repository
        #[arg(short, long)]
        model: String,
        
        /// Target bits for quantization
        #[arg(short, long, default_value = "1.5")]
        bits: f32,
        
        /// Number of subspaces
        #[arg(short, long, default_value = "2")]
        subspaces: u32,
        
        /// Output file for results
        #[arg(short, long, default_value = "benchmark_results.json")]
        output: String,
    },
    
    /// Compare different quantization settings
    Compare {
        /// Model to benchmark
        #[arg(short, long)]
        model: String,
        
        /// Output file for comparison results
        #[arg(short, long, default_value = "comparison_results.json")]
        output: String,
    },
    
    /// Generate benchmark report
    Report {
        /// Benchmark results directory
        #[arg(short, long, default_value = "benchmarks")]
        results_dir: String,
        
        /// Output report file
        #[arg(short, long, default_value = "novaq_benchmark_report.md")]
        output: String,
    },
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct BenchmarkResult {
    model_name: String,
    config: BenchmarkConfig,
    performance: PerformanceMetrics,
    quality: QualityMetrics,
    timestamp: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct BenchmarkConfig {
    target_bits: f32,
    num_subspaces: u32,
    l1_codebook_size: u32,
    l2_codebook_size: u32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct PerformanceMetrics {
    compression_time_seconds: f64,
    compression_ratio: f64,
    memory_usage_mb: f64,
    throughput_mb_per_second: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct QualityMetrics {
    bit_accuracy: f64,
    quality_score: f64,
    validation_passed: bool,
    issues: Vec<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    
    match &cli.command {
        Commands::Run { output_dir, iterations, verbosity } => {
            run_comprehensive_benchmarks(output_dir, *iterations, verbosity)?;
        }
        Commands::Model { model, bits, subspaces, output } => {
            benchmark_single_model(model, *bits, *subspaces, output)?;
        }
        Commands::Compare { model, output } => {
            compare_quantization_settings(model, output)?;
        }
        Commands::Report { results_dir, output } => {
            generate_benchmark_report(results_dir, output)?;
        }
    }
    
    Ok(())
}

fn run_comprehensive_benchmarks(output_dir: &str, iterations: u32, verbosity: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Running comprehensive NOVAQ benchmarks...");
    println!("Output directory: {}", output_dir);
    println!("Iterations: {}", iterations);
    
    let verbosity_level = match verbosity.to_lowercase().as_str() {
        "silent" => VerbosityLevel::Silent,
        "standard" => VerbosityLevel::Standard,
        "detailed" => VerbosityLevel::Detailed,
        _ => VerbosityLevel::Silent,
    };
    
    // Create output directory
    let output_path = Path::new(output_dir);
    fs::create_dir_all(output_path)?;
    
    // Define benchmark models
    let models = vec![
        "microsoft/DialoGPT-small",
        "microsoft/DialoGPT-medium", 
        "microsoft/DialoGPT-large",
    ];
    
    // Define quantization configurations
    let configs = vec![
        (1.0, 2), // 1-bit, 2 subspaces
        (1.5, 2), // 1.5-bit, 2 subspaces
        (2.0, 2), // 2-bit, 2 subspaces
        (1.5, 4), // 1.5-bit, 4 subspaces
        (2.0, 4), // 2-bit, 4 subspaces
    ];
    
    let mut all_results = Vec::new();
    
    for model in &models {
        println!("\n📊 Benchmarking model: {}", model);
        
        for (bits, subspaces) in &configs {
            println!("  Configuration: {} bits, {} subspaces", bits, subspaces);
            
            let mut config_results = Vec::new();
            
            for i in 0..iterations {
                println!("    Iteration {}/{}", i + 1, iterations);
                
                match benchmark_model_config(model, *bits, *subspaces, verbosity_level) {
                    Ok(result) => {
                        config_results.push(result.clone());
                        println!("    ✅ Completed - {:.1}x compression, {:.1}% accuracy", 
                                result.performance.compression_ratio, 
                                result.quality.bit_accuracy * 100.0);
                    }
                    Err(e) => {
                        println!("    ❌ Failed: {}", e);
                    }
                }
            }
            
            // Calculate average results
            if !config_results.is_empty() {
                let avg_result = calculate_average_results(&config_results, model, *bits, *subspaces);
                all_results.push(avg_result);
                
                // Save individual results
                let config_file = output_path.join(format!("{}_{}bit_{}sub_{}.json", 
                    model.replace("/", "_"), bits, subspaces, chrono::Utc::now().timestamp()));
                let config_json = serde_json::to_string_pretty(&config_results)?;
                fs::write(config_file, config_json)?;
            }
        }
    }
    
    // Save comprehensive results
    let results_file = output_path.join("comprehensive_benchmark_results.json");
    let results_json = serde_json::to_string_pretty(&all_results)?;
    fs::write(&results_file, results_json)?;
    
    println!("\n✅ Comprehensive benchmarks completed!");
    println!("Results saved to: {}", results_file.display());
    
    Ok(())
}

fn benchmark_single_model(model: &str, bits: f32, subspaces: u32, output: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Benchmarking single model: {}", model);
    println!("Configuration: {} bits, {} subspaces", bits, subspaces);
    
    let result = benchmark_model_config(model, bits, subspaces, VerbosityLevel::Standard)?;
    
    let results_json = serde_json::to_string_pretty(&vec![result])?;
    fs::write(output, results_json)?;
    
    println!("✅ Benchmark completed!");
    println!("Results saved to: {}", output);
    
    Ok(())
}

fn compare_quantization_settings(model: &str, output: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Comparing quantization settings for: {}", model);
    
    let configs = vec![
        (1.0, 2), (1.5, 2), (2.0, 2), (4.0, 2),
        (1.5, 4), (2.0, 4), (4.0, 4),
    ];
    
    let mut comparison_results = Vec::new();
    
    for (bits, subspaces) in &configs {
        println!("Testing: {} bits, {} subspaces", bits, subspaces);
        
        match benchmark_model_config(model, *bits, *subspaces, VerbosityLevel::Silent) {
            Ok(result) => {
                comparison_results.push(result.clone());
                println!("  ✅ {:.1}x compression, {:.1}% accuracy", 
                        result.performance.compression_ratio, 
                        result.quality.bit_accuracy * 100.0);
            }
            Err(e) => {
                println!("  ❌ Failed: {}", e);
            }
        }
    }
    
    let results_json = serde_json::to_string_pretty(&comparison_results)?;
    fs::write(output, results_json)?;
    
    println!("✅ Comparison completed!");
    println!("Results saved to: {}", output);
    
    Ok(())
}

fn generate_benchmark_report(results_dir: &str, output: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("📋 Generating benchmark report...");
    
    let results_path = Path::new(results_dir);
    if !results_path.exists() {
        return Err("Results directory does not exist".into());
    }
    
    let mut report = String::new();
    report.push_str("# NOVAQ Compression Benchmark Report\n\n");
    report.push_str(&format!("Generated: {}\n\n", chrono::Utc::now().to_rfc3339()));
    
    // Find and load results
    let mut all_results = Vec::new();
    
    for entry in fs::read_dir(results_path)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            let data = fs::read_to_string(&path)?;
            let results: Vec<BenchmarkResult> = serde_json::from_str(&data)?;
            all_results.extend(results);
        }
    }
    
    if all_results.is_empty() {
        report.push_str("No benchmark results found.\n");
    } else {
        // Generate summary statistics
        report.push_str("## Summary Statistics\n\n");
        
        let avg_compression_ratio: f64 = all_results.iter()
            .map(|r| r.performance.compression_ratio)
            .sum::<f64>() / all_results.len() as f64;
        
        let avg_accuracy: f64 = all_results.iter()
            .map(|r| r.quality.bit_accuracy)
            .sum::<f64>() / all_results.len() as f64;
        
        let avg_quality_score: f64 = all_results.iter()
            .map(|r| r.quality.quality_score)
            .sum::<f64>() / all_results.len() as f64;
        
        report.push_str(&format!("- **Average Compression Ratio**: {:.1}x\n", avg_compression_ratio));
        report.push_str(&format!("- **Average Bit Accuracy**: {:.1}%\n", avg_accuracy * 100.0));
        report.push_str(&format!("- **Average Quality Score**: {:.3}\n", avg_quality_score));
        report.push_str(&format!("- **Total Benchmarks**: {}\n\n", all_results.len()));
        
        // Best performing configurations
        report.push_str("## Best Performing Configurations\n\n");
        
        let mut sorted_by_compression = all_results.clone();
        sorted_by_compression.sort_by(|a, b| b.performance.compression_ratio.partial_cmp(&a.performance.compression_ratio).unwrap());
        
        report.push_str("### Highest Compression Ratios\n\n");
        for (i, result) in sorted_by_compression.iter().take(5).enumerate() {
            report.push_str(&format!("{}. **{}** ({:.1} bits, {} subspaces)\n", 
                i + 1, result.model_name, result.config.target_bits, result.config.num_subspaces));
            report.push_str(&format!("   - Compression: {:.1}x\n", result.performance.compression_ratio));
            report.push_str(&format!("   - Accuracy: {:.1}%\n", result.quality.bit_accuracy * 100.0));
            report.push_str(&format!("   - Quality Score: {:.3}\n\n", result.quality.quality_score));
        }
        
        let mut sorted_by_accuracy = all_results.clone();
        sorted_by_accuracy.sort_by(|a, b| b.quality.bit_accuracy.partial_cmp(&a.quality.bit_accuracy).unwrap());
        
        report.push_str("### Highest Accuracy\n\n");
        for (i, result) in sorted_by_accuracy.iter().take(5).enumerate() {
            report.push_str(&format!("{}. **{}** ({:.1} bits, {} subspaces)\n", 
                i + 1, result.model_name, result.config.target_bits, result.config.num_subspaces));
            report.push_str(&format!("   - Accuracy: {:.1}%\n", result.quality.bit_accuracy * 100.0));
            report.push_str(&format!("   - Compression: {:.1}x\n", result.performance.compression_ratio));
            report.push_str(&format!("   - Quality Score: {:.3}\n\n", result.quality.quality_score));
        }
    }
    
    fs::write(output, report)?;
    println!("✅ Benchmark report generated: {}", output);
    
    Ok(())
}

fn benchmark_model_config(model: &str, bits: f32, subspaces: u32, verbosity: VerbosityLevel) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    
    // Create NOVAQ configuration
    let config = NOVAQConfig {
        target_bits: bits,
        num_subspaces: subspaces as usize,
        codebook_size_l1: 16,
        codebook_size_l2: 4,
        outlier_threshold: 0.01,
        teacher_model_path: None,
        refinement_iterations: 50,
        kl_weight: 1.0,
        cosine_weight: 0.5,
        learning_rate: 0.001,
        seed: 42,
    };
    
    let mut novaq = PublicNOVAQ::new_with_verbosity(config.clone(), verbosity);
    
    // Compress model
    let compressed_model = novaq.compress_hf_model(model, None).map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
    
    let compression_time = start_time.elapsed().as_secs_f64();
    
    // Get performance metrics
    let stats = novaq.get_compression_stats(&compressed_model);
    let validation = novaq.validate_model(&compressed_model).map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
    
    // Estimate memory usage (rough calculation)
    let memory_usage = estimate_memory_usage(&compressed_model);
    let throughput = memory_usage / compression_time;
    
    Ok(BenchmarkResult {
        model_name: model.to_string(),
        config: BenchmarkConfig {
            target_bits: config.target_bits,
            num_subspaces: config.num_subspaces as u32,
            l1_codebook_size: config.codebook_size_l1 as u32,
            l2_codebook_size: config.codebook_size_l2 as u32,
        },
        performance: PerformanceMetrics {
            compression_time_seconds: compression_time,
            compression_ratio: stats.compression_ratio as f64,
            memory_usage_mb: memory_usage,
            throughput_mb_per_second: throughput,
        },
        quality: QualityMetrics {
            bit_accuracy: stats.bit_accuracy as f64,
            quality_score: stats.quality_score as f64,
            validation_passed: validation.passed_validation,
            issues: validation.issues,
        },
        timestamp: chrono::Utc::now().to_rfc3339(),
    })
}

fn calculate_average_results(results: &[BenchmarkResult], model: &str, bits: f32, subspaces: u32) -> BenchmarkResult {
    let avg_compression_time = results.iter().map(|r| r.performance.compression_time_seconds).sum::<f64>() / results.len() as f64;
    let avg_compression_ratio = results.iter().map(|r| r.performance.compression_ratio).sum::<f64>() / results.len() as f64;
    let avg_memory_usage = results.iter().map(|r| r.performance.memory_usage_mb).sum::<f64>() / results.len() as f64;
    let avg_throughput = results.iter().map(|r| r.performance.throughput_mb_per_second).sum::<f64>() / results.len() as f64;
    
    let avg_bit_accuracy = results.iter().map(|r| r.quality.bit_accuracy).sum::<f64>() / results.len() as f64;
    let avg_quality_score = results.iter().map(|r| r.quality.quality_score).sum::<f64>() / results.len() as f64;
    let all_passed = results.iter().all(|r| r.quality.validation_passed);
    
    let mut all_issues = Vec::new();
    for result in results {
        all_issues.extend(result.quality.issues.clone());
    }
    
    BenchmarkResult {
        model_name: model.to_string(),
        config: BenchmarkConfig {
            target_bits: bits,
            num_subspaces: subspaces,
            l1_codebook_size: 16,
            l2_codebook_size: 4,
        },
        performance: PerformanceMetrics {
            compression_time_seconds: avg_compression_time,
            compression_ratio: avg_compression_ratio,
            memory_usage_mb: avg_memory_usage,
            throughput_mb_per_second: avg_throughput,
        },
        quality: QualityMetrics {
            bit_accuracy: avg_bit_accuracy,
            quality_score: avg_quality_score,
            validation_passed: all_passed,
            issues: all_issues,
        },
        timestamp: chrono::Utc::now().to_rfc3339(),
    }
}

fn estimate_memory_usage(model: &NOVAQModel) -> f64 {
    // Rough estimation based on model size
    let total_codebooks = model.vector_codebooks.level1_codebooks.iter().map(|c| c.len()).sum::<usize>();
    let total_indices = model.quantization_indices.level1_indices.iter().map(|i| i.len()).sum::<usize>();
    
    let total_bytes = (total_codebooks + total_indices) * 4; // Assuming f32
    total_bytes as f64 / (1024.0 * 1024.0) // Convert to MB
}
