pub mod manifest;
pub mod verification;
pub mod universal_loader;
pub mod model_fetcher;
pub mod novaq;
pub mod real_model_loader;
pub mod streaming_loader;

pub use manifest::*;
pub use verification::*;
pub use universal_loader::{UniversalModel, UniversalLoader, load_any_model, find_model};
pub use model_fetcher::{ModelFetcher, ModelSource, parse_model_source, FetchResult, ModelMetadata, ModelFormat};
pub use novaq::{NOVAQEngine, NOVAQConfig, NOVAQModel, WeightMatrix, QuantizationRecoveryManager, RecoveryStats, QuantizationProgressTracker, VerbosityLevel};
pub use real_model_loader::{RealModelLoader, ModelStats};
pub use streaming_loader::{StreamingModelLoader};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

// NOVAQ Democratic Access - No Restrictions
// Core compression technology available to everyone
// No admin controls - completely open access

/// Public NOVAQ compression interface - no restrictions
pub struct PublicNOVAQ {
    engine: NOVAQEngine,
    recovery_manager: QuantizationRecoveryManager,
    auto_recovery_enabled: bool,
    verbosity_level: VerbosityLevel,
}

impl PublicNOVAQ {
    pub fn new(config: NOVAQConfig) -> Self {
        let verbosity = match std::env::var("NOVAQ_VERBOSITY").as_deref() {
            Ok("silent") => VerbosityLevel::Silent,
            Ok("minimal") => VerbosityLevel::Minimal,
            Ok("detailed") => VerbosityLevel::Detailed,
            _ => VerbosityLevel::Standard,
        };
        
        Self {
            engine: NOVAQEngine::new(config.clone()),
            recovery_manager: QuantizationRecoveryManager::new(config),
            auto_recovery_enabled: true,
            verbosity_level: verbosity,
        }
    }
    
    /// Create with specific verbosity level
    pub fn new_with_verbosity(config: NOVAQConfig, verbosity: VerbosityLevel) -> Self {
        Self {
            engine: NOVAQEngine::new(config.clone()),
            recovery_manager: QuantizationRecoveryManager::new(config),
            auto_recovery_enabled: true,
            verbosity_level: verbosity,
        }
    }
    
    /// Compress any model with NOVAQ - no restrictions
    /// Uses automatic recovery if enabled (default: enabled)
    pub fn compress_model(&mut self, weights: Vec<WeightMatrix>) -> Result<NOVAQModel> {
        if self.auto_recovery_enabled {
            self.recovery_manager.quantize_with_recovery_and_progress(weights, self.verbosity_level)
        } else {
            let mut progress = QuantizationProgressTracker::new(self.verbosity_level);
            self.engine.quantize_model_with_progress(weights, &mut progress)
        }
    }
    
    /// Compress model without automatic recovery (original behavior)
    pub fn compress_model_basic(&mut self, weights: Vec<WeightMatrix>) -> Result<NOVAQModel> {
        self.engine.quantize_model(weights)
    }
    
    /// Enable or disable automatic recovery
    pub fn set_auto_recovery(&mut self, enabled: bool) {
        self.auto_recovery_enabled = enabled;
        if enabled {
            println!("🛡️  Automatic recovery enabled - quantization will attempt to recover from failures");
        } else {
            println!("⚠️  Automatic recovery disabled - quantization will fail immediately on errors");
        }
    }
    
    /// Get recovery statistics
    pub fn get_recovery_stats(&self) -> &RecoveryStats {
        self.recovery_manager.get_stats()
    }
    
    /// Print recovery statistics summary
    pub fn print_recovery_summary(&self) {
        self.recovery_manager.print_recovery_summary();
    }
    
    /// Reset recovery statistics
    pub fn reset_recovery_stats(&mut self) {
        self.recovery_manager.reset_stats();
    }
    
    /// Validate NOVAQ model quality
    pub fn validate_model(&self, model: &NOVAQModel) -> Result<ValidationReport> {
        let mut issues = Vec::new();
        
        // Realistic validation thresholds based on bit depth
        let min_compression_ratio = 2.0; // At least 2x compression
        let min_bit_accuracy = match model.config.target_bits {
            b if b <= 1.0 => 0.85,  // 1-bit: 85% accuracy is excellent
            b if b <= 2.0 => 0.90,  // 2-bit: 90% accuracy is excellent  
            b if b <= 4.0 => 0.95,  // 4-bit: 95% accuracy is excellent
            _ => 0.98,              // Higher bits: 98% accuracy expected
        };
        
        // Check compression ratio
        if model.compression_ratio < min_compression_ratio {
            issues.push(format!("Compression ratio {:.1}x below minimum {:.1}x", 
                               model.compression_ratio, min_compression_ratio));
        }
        
        // Check bit accuracy
        if model.bit_accuracy < min_bit_accuracy {
            issues.push(format!("Bit accuracy {:.1}% below minimum {:.1}% for {:.1}-bit quantization", 
                               model.bit_accuracy * 100.0, min_bit_accuracy * 100.0, model.config.target_bits));
        }
        
        // Quality score calculation
        let quality_score = (model.compression_ratio / 100.0 + model.bit_accuracy) / 2.0;
        
        let passed_validation = issues.is_empty();
        
        Ok(ValidationReport {
            compression_ratio: model.compression_ratio,
            bit_accuracy: model.bit_accuracy,
            quality_score,
            passed_validation,
            issues,
        })
    }

    /// Fetch and compress model from Hugging Face
    pub fn compress_hf_model(&mut self, repo: &str, file: Option<&str>) -> Result<NOVAQModel> {
        let source = ModelSource::HuggingFace { 
            repo: repo.to_string(), 
            file: file.map(|f| f.to_string()) 
        };
        
        let fetch_result = ModelFetcher::fetch(&source)?;
        let weights = RealModelLoader::load_model(&fetch_result)?;
        
        self.compress_model(weights)
    }

    /// Fetch and compress model from Ollama
    pub fn compress_ollama_model(&mut self, model: &str) -> Result<NOVAQModel> {
        let source = ModelSource::Ollama { 
            model: model.to_string() 
        };
        
        let fetch_result = ModelFetcher::fetch(&source)?;
        let weights = RealModelLoader::load_model(&fetch_result)?;
        
        self.compress_model(weights)
    }

    /// Fetch and compress model from URL
    pub fn compress_url_model(&mut self, url: &str, filename: Option<&str>) -> Result<NOVAQModel> {
        let source = ModelSource::Url { 
            url: url.to_string(), 
            filename: filename.map(|f| f.to_string()) 
        };
        
        let fetch_result = ModelFetcher::fetch(&source)?;
        let weights = RealModelLoader::load_model(&fetch_result)?;
        
        self.compress_model(weights)
    }

    /// Compress local model file
    pub fn compress_local_model(&mut self, path: &str) -> Result<NOVAQModel> {
        let source = ModelSource::LocalPath { 
            path: std::path::PathBuf::from(path) 
        };
        
        let fetch_result = ModelFetcher::fetch(&source)?;
        let weights = RealModelLoader::load_model(&fetch_result)?;
        
        self.compress_model(weights)
    }

    /// Get compression statistics
    pub fn get_compression_stats(&self, model: &NOVAQModel) -> CompressionStats {
        CompressionStats {
            compression_ratio: model.compression_ratio,
            bit_accuracy: model.bit_accuracy,
            quality_score: (model.compression_ratio / 100.0 + model.bit_accuracy) / 2.0,
            target_bits: model.config.target_bits,
            num_subspaces: model.config.num_subspaces,
            codebook_size_l1: model.config.codebook_size_l1,
            codebook_size_l2: model.config.codebook_size_l2,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ValidationReport {
    pub compression_ratio: f32,
    pub bit_accuracy: f32,
    pub quality_score: f32,
    pub passed_validation: bool,
    pub issues: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CompressionStats {
    pub compression_ratio: f32,
    pub bit_accuracy: f32,
    pub quality_score: f32,
    pub target_bits: f32,
    pub num_subspaces: usize,
    pub codebook_size_l1: usize,
    pub codebook_size_l2: usize,
}