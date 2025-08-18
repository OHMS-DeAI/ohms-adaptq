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
pub use novaq::{NOVAQEngine, NOVAQConfig, NOVAQModel, WeightMatrix};
pub use real_model_loader::{RealModelLoader, ModelStats};
pub use streaming_loader::{StreamingModelLoader};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

// NOVAQ Democratic Access - No Restrictions
// Core compression technology available to everyone
// No admin controls - completely open access

/// Public NOVAQ compression interface - no restrictions
pub struct PublicNOVAQ {
    engine: NOVAQEngine,
}

impl PublicNOVAQ {
    pub fn new(config: NOVAQConfig) -> Self {
        Self {
            engine: NOVAQEngine::new(config),
        }
    }
    
    /// Compress any model with NOVAQ - no restrictions
    pub fn compress_model(&mut self, weights: Vec<WeightMatrix>) -> Result<NOVAQModel> {
        self.engine.quantize_model(weights)
    }
    
    /// Validate NOVAQ model quality
    pub fn validate_model(&self, model: &NOVAQModel) -> Result<ValidationReport> {
        Ok(ValidationReport {
            compression_ratio: model.compression_ratio,
            bit_accuracy: model.bit_accuracy,
            quality_score: (model.compression_ratio / 100.0 + model.bit_accuracy) / 2.0,
            passed_validation: model.compression_ratio >= 93.0 && model.bit_accuracy >= 0.99,
            issues: Vec::new(),
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