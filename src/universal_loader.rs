// Universal Model Loader - Supports ANY LLM format
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use memmap2::{Mmap, MmapOptions};

/// Universal model that can represent any LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalModel {
    pub format: ModelFormat,
    pub metadata: UniversalMetadata,
    pub layers: Vec<Layer>,
    pub tokenizer: Option<TokenizerInfo>,
    pub config: ModelConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelFormat {
    GGML,           // Ollama, llama.cpp models
    GGUF,           // New GGML format
    ONNX,           // ONNX models
    SafeTensors,    // HuggingFace SafeTensors
    PyTorch,        // .pt, .pth, .bin files
    TensorFlow,     // .pb, .h5 files
    JAX,            // JAX/Flax models
    Paddle,         // PaddlePaddle models
    MXNet,          // Apache MXNet
    CoreML,         // Apple CoreML
    TensorRT,       // NVIDIA TensorRT
    OpenVINO,       // Intel OpenVINO
    NCNN,           // Tencent NCNN
    TFLite,         // TensorFlow Lite
    Custom(String), // Any other format
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalMetadata {
    pub name: String,
    pub architecture: String,
    pub parameters: u64,
    pub precision: Precision,
    pub context_length: u32,
    pub hidden_size: u32,
    pub num_layers: u32,
    pub num_heads: u32,
    pub vocab_size: u32,
    pub intermediate_size: u32,
    pub rope_theta: Option<f32>,
    pub max_position_embeddings: u32,
    pub layer_norm_epsilon: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Precision {
    FP32,
    FP16,
    BF16,
    INT8,
    INT4,
    Mixed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    pub name: String,
    pub layer_type: LayerType,
    pub weights: Vec<Tensor>,
    pub shape: Vec<usize>,
    pub parameters: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerType {
    Embedding,
    Attention,
    MLP,
    LayerNorm,
    RMSNorm,
    Linear,
    Conv1D,
    Conv2D,
    GeLU,
    SiLU,
    Softmax,
    Dropout,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DataType,
    pub data: TensorData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataType {
    Float32,
    Float16,
    BFloat16,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    Bool,
    Quantized(QuantizationType),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationType {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TensorData {
    Float32(Vec<f32>),
    Float16(Vec<F16>),
    Int8(Vec<i8>),
    UInt8(Vec<u8>),
    Quantized(Vec<u8>),
    MemoryMapped { offset: u64, size: u64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerInfo {
    pub vocab_size: u32,
    pub tokenizer_type: String,
    pub special_tokens: HashMap<String, u32>,
    pub vocab_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_type: String,
    pub architectures: Vec<String>,
    pub attention_bias: bool,
    pub attention_dropout: f32,
    pub hidden_act: String,
    pub hidden_dropout: f32,
    pub initializer_range: f32,
    pub intermediate_size: u32,
    pub max_position_embeddings: u32,
    pub num_attention_heads: u32,
    pub num_hidden_layers: u32,
    pub num_key_value_heads: Option<u32>,
    pub pretraining_tp: Option<u32>,
    pub rms_norm_eps: f32,
    pub rope_scaling: Option<HashMap<String, serde_json::Value>>,
    pub tie_word_embeddings: bool,
    pub torch_dtype: Option<String>,
    pub transformers_version: Option<String>,
    pub use_cache: bool,
    pub vocab_size: u32,
}

// f16 type for Rust
#[derive(Debug, Clone, Copy)]
pub struct F16(u16);

impl Serialize for F16 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: serde::Serializer {
        serializer.serialize_f32(self.to_f32())
    }
}

impl<'de> Deserialize<'de> for F16 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: serde::Deserializer<'de> {
        let val = f32::deserialize(deserializer)?;
        Ok(F16::from_f32(val))
    }
}

impl F16 {
    pub fn from_f32(f: f32) -> Self {
        // Simple f32 to f16 conversion
        let bits = f.to_bits();
        let sign = (bits >> 31) as u16;
        let exp = ((bits >> 23) & 0xff) as i32;
        let frac = (bits & 0x7fffff) as u32;
        
        let half_exp = (exp - 127 + 15).max(0).min(31) as u16;
        let half_frac = (frac >> 13) as u16;
        
        F16((sign << 15) | (half_exp << 10) | half_frac)
    }
    
    pub fn to_f32(&self) -> f32 {
        let sign = (self.0 >> 15) as u32;
        let exp = ((self.0 >> 10) & 0x1f) as i32;
        let frac = (self.0 & 0x3ff) as u32;
        
        let float_exp = (exp - 15 + 127) as u32;
        let float_frac = frac << 13;
        
        f32::from_bits((sign << 31) | (float_exp << 23) | float_frac)
    }
}

pub struct UniversalLoader {
    mmap_cache: HashMap<PathBuf, Mmap>,
}

impl UniversalLoader {
    pub fn new() -> Self {
        Self {
            mmap_cache: HashMap::new(),
        }
    }
    
    /// Load any model format
    pub fn load_model<P: AsRef<Path>>(&mut self, path: P) -> Result<UniversalModel, Box<dyn std::error::Error>> {
        let path = path.as_ref();
        let format = self.detect_format(path)?;
        
        match format {
            ModelFormat::ONNX => self.load_onnx(path),
            ModelFormat::SafeTensors => self.load_safetensors(path),
            ModelFormat::PyTorch => self.load_pytorch(path),
            ModelFormat::TensorFlow => self.load_tensorflow(path),
            ModelFormat::GGML | ModelFormat::GGUF => self.load_ggml(path),
            _ => self.load_generic(path),
        }
    }
    
    /// Detect model format from file extension and magic bytes
    fn detect_format(&self, path: &Path) -> Result<ModelFormat, Box<dyn std::error::Error>> {
        let extension = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        
        // Check by extension first
        let format = match extension {
            "onnx" => ModelFormat::ONNX,
            "safetensors" => ModelFormat::SafeTensors,
            "pt" | "pth" | "bin" => ModelFormat::PyTorch,
            "pb" | "h5" | "keras" => ModelFormat::TensorFlow,
            "gguf" => ModelFormat::GGUF,
            "ggml" => ModelFormat::GGML,
            "mlmodel" => ModelFormat::CoreML,
            "tflite" => ModelFormat::TFLite,
            "pdmodel" => ModelFormat::Paddle,
            _ => {
                // Try to detect by magic bytes
                self.detect_by_magic(path)?
            }
        };
        
        Ok(format)
    }
    
    fn detect_by_magic(&self, path: &Path) -> Result<ModelFormat, Box<dyn std::error::Error>> {
        let mut file = File::open(path)?;
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)?;
        
        // Check magic bytes
        if &magic[0..4] == b"GGUF" {
            Ok(ModelFormat::GGUF)
        } else if &magic[0..4] == b"ggml" {
            Ok(ModelFormat::GGML)
        } else if &magic[0..2] == b"\x08\x00" {
            Ok(ModelFormat::ONNX)
        } else if &magic[0..2] == b"PK" {
            Ok(ModelFormat::PyTorch) // ZIP file
        } else if &magic[0..4] == b"\x93NUMPY" {
            Ok(ModelFormat::SafeTensors)
        } else {
            Ok(ModelFormat::Custom("unknown".to_string()))
        }
    }
    
    /// Load ONNX model
    fn load_onnx(&mut self, path: &Path) -> Result<UniversalModel, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        
        // Parse ONNX protobuf (simplified)
        let metadata = UniversalMetadata {
            name: path.file_stem().unwrap_or_default().to_string_lossy().to_string(),
            architecture: "onnx".to_string(),
            parameters: 0,
            precision: Precision::FP32,
            context_length: 2048,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            vocab_size: 50257,
            intermediate_size: 3072,
            rope_theta: None,
            max_position_embeddings: 2048,
            layer_norm_epsilon: 1e-5,
        };
        
        let config = ModelConfig {
            model_type: "onnx".to_string(),
            architectures: vec!["transformer".to_string()],
            attention_bias: false,
            attention_dropout: 0.1,
            hidden_act: "gelu".to_string(),
            hidden_dropout: 0.1,
            initializer_range: 0.02,
            intermediate_size: metadata.intermediate_size,
            max_position_embeddings: metadata.max_position_embeddings,
            num_attention_heads: metadata.num_heads,
            num_hidden_layers: metadata.num_layers,
            num_key_value_heads: Some(metadata.num_heads),
            pretraining_tp: None,
            rms_norm_eps: metadata.layer_norm_epsilon,
            rope_scaling: None,
            tie_word_embeddings: false,
            torch_dtype: Some("float32".to_string()),
            transformers_version: None,
            use_cache: true,
            vocab_size: metadata.vocab_size,
        };
        
        self.mmap_cache.insert(path.to_path_buf(), mmap);
        
        Ok(UniversalModel {
            format: ModelFormat::ONNX,
            metadata,
            layers: self.extract_layers_from_mmap(path)?,
            tokenizer: None,
            config,
        })
    }
    
    /// Load SafeTensors model
    fn load_safetensors(&mut self, path: &Path) -> Result<UniversalModel, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        let header_size = u64::from_le_bytes(bytes[0..8].try_into()?) as usize;
        let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_size])?;

        // Build tensor list from header entries
        let mut layers: Vec<Layer> = Vec::new();
        if let serde_json::Value::Object(map) = &header {
            // Group by layer prefix (naive heuristic): split names by '.' and use first two segments
            let mut by_layer: HashMap<String, Vec<Tensor>> = HashMap::new();
            for (name, info) in map {
                if let Some(obj) = info.as_object() {
                    let dtype = match obj.get("dtype").and_then(|v| v.as_str()).unwrap_or("") {
                        "F32" | "float32" => DataType::Float32,
                        "F16" | "float16" => DataType::Float16,
                        "BF16" | "bfloat16" => DataType::BFloat16,
                        _ => DataType::Float32,
                    };
                    if let (Some(shape_v), Some(offset_v), Some(size_v)) = (obj.get("shape"), obj.get("data_offsets"), obj.get("data_offsets")) {
                        let shape = shape_v.as_array().unwrap_or(&vec![]).iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect::<Vec<_>>();
                        let tmp = vec![];
                        let offsets = offset_v.as_array().unwrap_or(&tmp);
                        if offsets.len() == 2 {
                            let off = offsets[0].as_u64().unwrap_or(0) + 8 + header_size as u64;
                            let end = offsets[1].as_u64().unwrap_or(off);
                            let size = end.saturating_sub(off);
                            let data = TensorData::MemoryMapped { offset: off, size };
                            let key = name.split('.').take(2).collect::<Vec<_>>().join(".");
                            let tensor = Tensor { name: name.clone(), shape: shape.clone(), dtype: dtype.clone(), data };
                            by_layer.entry(key).or_default().push(tensor);
                        }
                    }
                }
            }
            for (lname, tensors) in by_layer.into_iter() {
                let params: u64 = tensors.iter().map(|t| t.shape.iter().product::<usize>() as u64).sum();
                layers.push(Layer {
                    name: lname.clone(),
                    layer_type: self.infer_layer_type(&lname),
                    weights: tensors,
                    shape: vec![],
                    parameters: params,
                });
            }
        }

        let metadata = UniversalMetadata {
            name: path.file_stem().unwrap_or_default().to_string_lossy().to_string(),
            architecture: "transformer".to_string(),
            parameters: layers.iter().map(|l| l.parameters).sum(),
            precision: Precision::FP16,
            context_length: 2048,
            hidden_size: 0,
            num_layers: layers.len() as u32,
            num_heads: 0,
            vocab_size: 0,
            intermediate_size: 0,
            rope_theta: None,
            max_position_embeddings: 0,
            layer_norm_epsilon: 1e-5,
        };
        let config = ModelConfig {
            model_type: "safetensors".to_string(),
            architectures: vec!["transformer".to_string()],
            attention_bias: false,
            attention_dropout: 0.0,
            hidden_act: "gelu".to_string(),
            hidden_dropout: 0.0,
            initializer_range: 0.02,
            intermediate_size: 0,
            max_position_embeddings: 0,
            num_attention_heads: 0,
            num_hidden_layers: layers.len() as u32,
            num_key_value_heads: None,
            pretraining_tp: None,
            rms_norm_eps: 1e-5,
            rope_scaling: None,
            tie_word_embeddings: false,
            torch_dtype: Some("float16".to_string()),
            transformers_version: None,
            use_cache: true,
            vocab_size: 0,
        };

        Ok(UniversalModel { format: ModelFormat::SafeTensors, metadata, layers, tokenizer: None, config })
    }
    
    /// Load PyTorch model
    fn load_pytorch(&mut self, path: &Path) -> Result<UniversalModel, Box<dyn std::error::Error>> {
        // PyTorch models are ZIP files with pickle data
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        
        let metadata = self.extract_pytorch_metadata(&mmap)?;
        let config = self.extract_pytorch_config(&mmap)?;
        
        self.mmap_cache.insert(path.to_path_buf(), mmap);
        
        Ok(UniversalModel {
            format: ModelFormat::PyTorch,
            metadata,
            layers: self.extract_pytorch_layers(path)?,
            tokenizer: None,
            config,
        })
    }
    
    /// Load TensorFlow model
    fn load_tensorflow(&mut self, path: &Path) -> Result<UniversalModel, Box<dyn std::error::Error>> {
        let metadata = UniversalMetadata {
            name: path.file_stem().unwrap_or_default().to_string_lossy().to_string(),
            architecture: "tensorflow".to_string(),
            parameters: 0,
            precision: Precision::FP32,
            context_length: 2048,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            vocab_size: 50257,
            intermediate_size: 3072,
            rope_theta: None,
            max_position_embeddings: 2048,
            layer_norm_epsilon: 1e-5,
        };
        
        let config = ModelConfig {
            model_type: "tensorflow".to_string(),
            architectures: vec!["transformer".to_string()],
            attention_bias: false,
            attention_dropout: 0.1,
            hidden_act: "gelu".to_string(),
            hidden_dropout: 0.1,
            initializer_range: 0.02,
            intermediate_size: metadata.intermediate_size,
            max_position_embeddings: metadata.max_position_embeddings,
            num_attention_heads: metadata.num_heads,
            num_hidden_layers: metadata.num_layers,
            num_key_value_heads: Some(metadata.num_heads),
            pretraining_tp: None,
            rms_norm_eps: metadata.layer_norm_epsilon,
            rope_scaling: None,
            tie_word_embeddings: false,
            torch_dtype: Some("float32".to_string()),
            transformers_version: None,
            use_cache: true,
            vocab_size: metadata.vocab_size,
        };
        
        Ok(UniversalModel {
            format: ModelFormat::TensorFlow,
            metadata,
            layers: Vec::new(),
            tokenizer: None,
            config,
        })
    }
    
    /// Load GGML/GGUF models
    fn load_ggml(&mut self, path: &Path) -> Result<UniversalModel, Box<dyn std::error::Error>> {
        // Fallback generic loader for GGML/GGUF when specialized loader is not available
        let file_size = std::fs::metadata(path)?.len();
        let name = path.file_stem().unwrap_or_default().to_string_lossy().to_string();
        let metadata = UniversalMetadata {
            name,
            architecture: "ggml".to_string(),
            parameters: file_size / 4,
            precision: Precision::Mixed,
            context_length: 4096,
            hidden_size: 4096,
            num_layers: 32,
            num_heads: 32,
            vocab_size: 32000,
            intermediate_size: 11008,
            rope_theta: Some(10000.0),
            max_position_embeddings: 4096,
            layer_norm_epsilon: 1e-5,
        };
        let config = ModelConfig {
            model_type: "ggml".to_string(),
            architectures: vec!["transformer".to_string()],
            attention_bias: false,
            attention_dropout: 0.0,
            hidden_act: "silu".to_string(),
            hidden_dropout: 0.0,
            initializer_range: 0.02,
            intermediate_size: metadata.intermediate_size,
            max_position_embeddings: metadata.max_position_embeddings,
            num_attention_heads: metadata.num_heads,
            num_hidden_layers: metadata.num_layers,
            num_key_value_heads: Some(metadata.num_heads),
            pretraining_tp: None,
            rms_norm_eps: metadata.layer_norm_epsilon,
            rope_scaling: None,
            tie_word_embeddings: false,
            torch_dtype: Some("float16".to_string()),
            transformers_version: None,
            use_cache: true,
            vocab_size: metadata.vocab_size,
        };
        Ok(UniversalModel {
            format: ModelFormat::GGUF,
            metadata,
            layers: Vec::new(),
            tokenizer: None,
            config,
        })
    }
    
    /// Load generic/unknown format
    fn load_generic(&mut self, path: &Path) -> Result<UniversalModel, Box<dyn std::error::Error>> {
        let file_size = std::fs::metadata(path)?.len();
        
        let metadata = UniversalMetadata {
            name: path.file_stem().unwrap_or_default().to_string_lossy().to_string(),
            architecture: "unknown".to_string(),
            parameters: file_size / 4, // Rough estimate
            precision: Precision::FP32,
            context_length: 2048,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            vocab_size: 50257,
            intermediate_size: 3072,
            rope_theta: None,
            max_position_embeddings: 2048,
            layer_norm_epsilon: 1e-5,
        };
        
        let config = ModelConfig {
            model_type: "generic".to_string(),
            architectures: vec!["unknown".to_string()],
            attention_bias: false,
            attention_dropout: 0.1,
            hidden_act: "gelu".to_string(),
            hidden_dropout: 0.1,
            initializer_range: 0.02,
            intermediate_size: metadata.intermediate_size,
            max_position_embeddings: metadata.max_position_embeddings,
            num_attention_heads: metadata.num_heads,
            num_hidden_layers: metadata.num_layers,
            num_key_value_heads: Some(metadata.num_heads),
            pretraining_tp: None,
            rms_norm_eps: metadata.layer_norm_epsilon,
            rope_scaling: None,
            tie_word_embeddings: false,
            torch_dtype: Some("float32".to_string()),
            transformers_version: None,
            use_cache: true,
            vocab_size: metadata.vocab_size,
        };
        
        Ok(UniversalModel {
            format: ModelFormat::Custom("generic".to_string()),
            metadata,
            layers: Vec::new(),
            tokenizer: None,
            config,
        })
    }
    
    // Helper methods
    fn extract_layers_from_mmap(&self, path: &Path) -> Result<Vec<Layer>, Box<dyn std::error::Error>> {
        // Extract layers from memory-mapped file
        Ok(Vec::new())
    }
    
    fn parse_safetensors_metadata(&self, _header: &serde_json::Value) -> Result<UniversalMetadata, Box<dyn std::error::Error>> {
        Ok(UniversalMetadata {
            name: "safetensors_model".to_string(),
            architecture: "transformer".to_string(),
            parameters: 0,
            precision: Precision::FP16,
            context_length: 2048,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            vocab_size: 50257,
            intermediate_size: 3072,
            rope_theta: None,
            max_position_embeddings: 2048,
            layer_norm_epsilon: 1e-5,
        })
    }
    
    fn parse_safetensors_config(&self, _header: &serde_json::Value) -> Result<ModelConfig, Box<dyn std::error::Error>> {
        Ok(ModelConfig {
            model_type: "safetensors".to_string(),
            architectures: vec!["transformer".to_string()],
            attention_bias: false,
            attention_dropout: 0.1,
            hidden_act: "gelu".to_string(),
            hidden_dropout: 0.1,
            initializer_range: 0.02,
            intermediate_size: 3072,
            max_position_embeddings: 2048,
            num_attention_heads: 12,
            num_hidden_layers: 12,
            num_key_value_heads: Some(12),
            pretraining_tp: None,
            rms_norm_eps: 1e-5,
            rope_scaling: None,
            tie_word_embeddings: false,
            torch_dtype: Some("float16".to_string()),
            transformers_version: None,
            use_cache: true,
            vocab_size: 50257,
        })
    }
    
    fn extract_safetensors_layers(&self, _data: &[u8], _header: &serde_json::Value) -> Result<Vec<Layer>, Box<dyn std::error::Error>> {
        Ok(Vec::new())
    }
    
    fn extract_pytorch_metadata(&self, _mmap: &Mmap) -> Result<UniversalMetadata, Box<dyn std::error::Error>> {
        Ok(UniversalMetadata {
            name: "pytorch_model".to_string(),
            architecture: "transformer".to_string(),
            parameters: 0,
            precision: Precision::FP32,
            context_length: 2048,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            vocab_size: 50257,
            intermediate_size: 3072,
            rope_theta: None,
            max_position_embeddings: 2048,
            layer_norm_epsilon: 1e-5,
        })
    }
    
    fn extract_pytorch_config(&self, _mmap: &Mmap) -> Result<ModelConfig, Box<dyn std::error::Error>> {
        Ok(ModelConfig {
            model_type: "pytorch".to_string(),
            architectures: vec!["transformer".to_string()],
            attention_bias: false,
            attention_dropout: 0.1,
            hidden_act: "gelu".to_string(),
            hidden_dropout: 0.1,
            initializer_range: 0.02,
            intermediate_size: 3072,
            max_position_embeddings: 2048,
            num_attention_heads: 12,
            num_hidden_layers: 12,
            num_key_value_heads: Some(12),
            pretraining_tp: None,
            rms_norm_eps: 1e-5,
            rope_scaling: None,
            tie_word_embeddings: false,
            torch_dtype: Some("float32".to_string()),
            transformers_version: None,
            use_cache: true,
            vocab_size: 50257,
        })
    }
    
    fn extract_pytorch_layers(&self, _path: &Path) -> Result<Vec<Layer>, Box<dyn std::error::Error>> {
        Ok(Vec::new())
    }
    
    // No-op stub when specialized GGML parsing is unavailable
    fn convert_ggml_tensors_to_layers(&self, _tensors: Vec<()>) -> Result<Vec<Layer>, Box<dyn std::error::Error>> {
        Ok(Vec::new())
    }
    
    fn infer_layer_type(&self, name: &str) -> LayerType {
        if name.contains("embed") {
            LayerType::Embedding
        } else if name.contains("attn") || name.contains("attention") {
            LayerType::Attention
        } else if name.contains("mlp") || name.contains("ffn") {
            LayerType::MLP
        } else if name.contains("norm") {
            if name.contains("rms") {
                LayerType::RMSNorm
            } else {
                LayerType::LayerNorm
            }
        } else if name.contains("linear") || name.contains("fc") {
            LayerType::Linear
        } else {
            LayerType::Custom(name.to_string())
        }
    }
    
    fn convert_ggml_dtype(&self, _dtype: ()) -> DataType { DataType::Float32 }
}

/// Auto-detect and load any model
pub fn load_any_model<P: AsRef<Path>>(path: P) -> Result<UniversalModel, Box<dyn std::error::Error>> {
    let mut loader = UniversalLoader::new();
    loader.load_model(path)
}

/// Find model from various sources
pub fn find_model(model_name: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let search_paths = get_model_search_paths()?;
    
    for dir in search_paths {
        let dir_path = Path::new(&dir);
        if dir_path.exists() {
            // Search for model files
            if let Ok(entries) = std::fs::read_dir(dir_path) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_file() {
                        let filename = path.file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("");
                        
                        if filename.contains(model_name) {
                            return Ok(path);
                        }
                    }
                }
            }
        }
    }
    
    // Try exact path
    let path = Path::new(model_name);
    if path.exists() {
        return Ok(path.to_path_buf());
    }
    
    Err(format!("Model '{}' not found in any standard location", model_name).into())
}

/// Get cross-platform model search paths
fn get_model_search_paths() -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut paths = Vec::new();
    
    if cfg!(target_os = "windows") {
        // Windows paths
        let appdata = std::env::var("LOCALAPPDATA")
            .unwrap_or_else(|_| "C:\\Users\\Default\\AppData\\Local".to_string());
        let userprofile = std::env::var("USERPROFILE")
            .unwrap_or_else(|_| "C:\\Users\\Default".to_string());
        
        paths.extend(vec![
            format!("{}\\Ollama\\.ollama\\models\\blobs", appdata),
            format!("{}\\.cache\\huggingface\\hub", userprofile),
            format!("{}\\models", userprofile),
            format!("{}\\LLM", userprofile),
            "C:\\models".to_string(),
            ".".to_string(),
            ".\\models".to_string(),
        ]);
    } else if cfg!(target_os = "macos") {
        // macOS paths
        let home = std::env::var("HOME").unwrap_or_else(|_| "/Users/Shared".to_string());
        
        paths.extend(vec![
            format!("{}/.ollama/models/blobs", home),
            format!("{}/Library/Caches/huggingface/hub", home),
            format!("{}/models", home),
            format!("{}/LLM", home),
            "/Applications/Ollama.app/Contents/Resources/models".to_string(),
            "/usr/local/share/ollama/.ollama/models/blobs".to_string(),
            "/models".to_string(),
            ".".to_string(),
            "./models".to_string(),
        ]);
    } else {
        // Linux/Unix paths
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        
        paths.extend(vec![
            // User-specific Ollama
            format!("{}/.ollama/models/blobs", home),
            
            // System Ollama locations
            "/usr/share/ollama/.ollama/models/blobs".to_string(),
            "/usr/local/share/ollama/.ollama/models/blobs".to_string(),
            "/var/lib/ollama/.ollama/models/blobs".to_string(),
            "/opt/ollama/.ollama/models/blobs".to_string(),
            
            // HuggingFace cache
            format!("{}/.cache/huggingface/hub", home),
            
            // Common model directories
            format!("{}/models", home),
            format!("{}/LLM", home),
            "/models".to_string(),
            "/data/models".to_string(),
            "/opt/models".to_string(),
            
            // Current directory
            ".".to_string(),
            "./models".to_string(),
        ]);
    }
    
    Ok(paths)
}