use crate::{Result, WeightMatrix, NOVAQEngine, NOVAQModel};
use hf_hub::api::tokio::{Api, ApiBuilder, Repo, RepoType};
use reqwest::Client;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncSeekExt, SeekFrom};

/// Native HuggingFace integration using hf-hub crate
pub struct HuggingFaceStreamer {
    api: Api,
    client: Client,
}

impl HuggingFaceStreamer {
    pub fn new() -> Result<Self> {
        // Try to get HF token from environment or config
        let mut builder = ApiBuilder::new();
        
        if let Ok(token) = std::env::var("HF_TOKEN") {
            builder = builder.with_token(Some(token));
        } else if let Ok(token) = std::env::var("HUGGINGFACE_HUB_TOKEN") {
            builder = builder.with_token(Some(token));
        }
        
        let api = builder.build().map_err(|e| format!("Failed to create HF API: {}", e))?;
        let client = Client::new();
        
        Ok(Self { api, client })
    }
    
    /// Stream compress a HuggingFace model directly
    pub async fn stream_compress_hf_model(
        &self,
        repo_id: &str,
        filename: Option<&str>,
        novaq_engine: &mut NOVAQEngine,
        progress_callback: impl Fn(f32, &str) + Send + Sync,
    ) -> Result<NOVAQModel> {
        progress_callback(0.0, &format!("Connecting to HuggingFace: {}", repo_id));
        
        let repo = self.api.repo(Repo::new(repo_id.to_string(), RepoType::Model));
        
        // Get model info and file list
        let repo_info = repo.info().await
            .map_err(|e| format!("Failed to get repo info for {}: {}", repo_id, e))?;
        
        progress_callback(0.1, "Analyzing model files...");
        
        // Find the best model file to stream
        let target_file = self.find_best_model_file(&repo, filename).await?;
        progress_callback(0.2, &format!("Found model file: {}", target_file));
        
        // Stream and compress based on file type
        if target_file.ends_with(".safetensors") {
            self.stream_safetensors_from_hf(&repo, &target_file, novaq_engine, progress_callback).await
        } else if target_file.ends_with(".bin") {
            self.stream_pytorch_from_hf(&repo, &target_file, novaq_engine, progress_callback).await  
        } else {
            Err(format!("Unsupported model file format: {}", target_file).into())
        }
    }
    
    /// Find the best model file in the repo
    async fn find_best_model_file(&self, repo: &hf_hub::api::tokio::ApiRepo, filename: Option<&str>) -> Result<String> {
        if let Some(file) = filename {
            return Ok(file.to_string());
        }
        
        // Get file list from repo
        let files = repo.get("").await
            .map_err(|e| format!("Failed to list repo files: {}", e))?;
        
        // Try to read the file list (this is a simplified approach)
        // In practice, we'd use the HF API to list files properly
        
        // Priority order: model.safetensors > pytorch_model.bin > model.bin
        let preferred_files = [
            "model.safetensors",
            "pytorch_model.bin", 
            "model.bin",
            "consolidated.00.pth",
        ];
        
        for preferred in &preferred_files {
            // Check if file exists (simplified check)
            if let Ok(_) = repo.get(preferred).await {
                return Ok(preferred.to_string());
            }
        }
        
        Err("No compatible model file found in repository".into())
    }
    
    /// Stream SafeTensors from HuggingFace with progressive compression
    async fn stream_safetensors_from_hf(
        &self,
        repo: &hf_hub::api::tokio::ApiRepo,
        filename: &str,
        novaq_engine: &mut NOVAQEngine,
        progress_callback: impl Fn(f32, &str) + Send + Sync,
    ) -> Result<NOVAQModel> {
        progress_callback(0.3, "Downloading SafeTensors header...");
        
        // Get the file path for streaming
        let file_path = repo.get(filename).await
            .map_err(|e| format!("Failed to access {}: {}", filename, e))?;
        
        // Read SafeTensors header
        let mut file = fs::File::open(&file_path).await
            .map_err(|e| format!("Failed to open model file: {}", e))?;
        
        // Read header size (first 8 bytes)
        let mut size_bytes = [0u8; 8];
        file.read_exact(&mut size_bytes).await
            .map_err(|e| format!("Failed to read header size: {}", e))?;
        
        let header_size = u64::from_le_bytes(size_bytes) as usize;
        
        // Read header
        let mut header_bytes = vec![0u8; header_size];
        file.read_exact(&mut header_bytes).await
            .map_err(|e| format!("Failed to read header: {}", e))?;
        
        let header: HashMap<String, Value> = serde_json::from_slice(&header_bytes)
            .map_err(|e| format!("Failed to parse SafeTensors header: {}", e))?;
        
        progress_callback(0.4, "Processing tensors with NOVAQ...");
        
        // Process each tensor
        let mut weight_matrices = Vec::new();
        let tensor_names: Vec<_> = header.keys().filter(|k| *k != "__metadata__").collect();
        let total_tensors = tensor_names.len();
        
        for (i, tensor_name) in tensor_names.iter().enumerate() {
            let progress = 0.4 + (i as f32 / total_tensors as f32) * 0.5;
            progress_callback(progress, &format!("Processing tensor: {}", tensor_name));
            
            let tensor_info = &header[*tensor_name];
            
            // Extract tensor data
            let tensor_data = self.extract_tensor_from_file(&mut file, tensor_info).await?;
            
            // Get tensor shape
            let shape: Vec<usize> = tensor_info["shape"].as_array()
                .ok_or("Invalid tensor shape")?
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            
            let weight_matrix = WeightMatrix::new(tensor_data, shape, tensor_name.to_string());
            weight_matrices.push(weight_matrix);
        }
        
        progress_callback(0.9, "Finalizing NOVAQ compression...");
        
        let novaq_model = novaq_engine.quantize_model(weight_matrices)
            .map_err(|e| format!("NOVAQ compression failed: {}", e))?;
        
        progress_callback(1.0, "HuggingFace streaming compression complete!");
        
        Ok(novaq_model)
    }
    
    /// Extract tensor data from SafeTensors file
    async fn extract_tensor_from_file(
        &self,
        file: &mut fs::File,
        tensor_info: &Value,
    ) -> Result<Vec<f32>> {
        let data_offsets = tensor_info["data_offsets"].as_array()
            .ok_or("Invalid tensor data offsets")?;
        let start_offset = data_offsets[0].as_u64().unwrap();
        let end_offset = data_offsets[1].as_u64().unwrap();
        let tensor_size = (end_offset - start_offset) as usize;
        
        // Seek to tensor data
        let header_size = 8u64; // Size of header length field
        let actual_offset = header_size + start_offset;
        file.seek(SeekFrom::Start(actual_offset)).await
            .map_err(|e| format!("Failed to seek to tensor data: {}", e))?;
        
        // Read tensor data
        let mut tensor_bytes = vec![0u8; tensor_size];
        file.read_exact(&mut tensor_bytes).await
            .map_err(|e| format!("Failed to read tensor data: {}", e))?;
        
        // Convert bytes to f32 values (assuming fp32)
        let float_count = tensor_size / 4;
        let mut tensor_data = Vec::with_capacity(float_count);
        
        for chunk in tensor_bytes.chunks_exact(4) {
            let float_val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            tensor_data.push(float_val);
        }
        
        Ok(tensor_data)
    }
    
    /// Stream PyTorch model from HuggingFace
    async fn stream_pytorch_from_hf(
        &self,
        repo: &hf_hub::api::tokio::ApiRepo,
        filename: &str,
        novaq_engine: &mut NOVAQEngine,
        progress_callback: impl Fn(f32, &str) + Send + Sync,
    ) -> Result<NOVAQModel> {
        // TODO: Implement PyTorch streaming
        progress_callback(0.0, "PyTorch streaming not yet implemented for HF");
        Err("PyTorch streaming from HuggingFace not implemented yet".into())
    }
}

/// Native Ollama integration using Ollama REST API
pub struct OllamaStreamer {
    client: Client,
    base_url: String,
}

impl OllamaStreamer {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: "http://localhost:11434".to_string(),
        }
    }
    
    pub fn with_base_url(mut self, url: String) -> Self {
        self.base_url = url;
        self
    }
    
    /// Stream compress an Ollama model directly
    pub async fn stream_compress_ollama_model(
        &self,
        model_name: &str,
        novaq_engine: &mut NOVAQEngine,
        progress_callback: impl Fn(f32, &str) + Send + Sync,
    ) -> Result<NOVAQModel> {
        progress_callback(0.0, &format!("Connecting to Ollama: {}", model_name));
        
        // Check if model exists
        let models = self.list_models().await?;
        let model_info = models.iter()
            .find(|m| m["name"].as_str() == Some(model_name))
            .ok_or_else(|| format!("Model {} not found in Ollama. Run: ollama pull {}", model_name, model_name))?;
        
        progress_callback(0.1, "Found model in Ollama");
        
        // Get model details
        let model_details = self.get_model_info(model_name).await?;
        progress_callback(0.2, "Retrieved model information");
        
        // Access model files through Ollama's storage
        let model_path = self.find_ollama_model_path(model_name).await?;
        progress_callback(0.3, &format!("Located model files: {}", model_path));
        
        // Stream and compress the GGUF file
        self.stream_gguf_file(&model_path, novaq_engine, progress_callback).await
    }
    
    /// List available models in Ollama
    async fn list_models(&self) -> Result<Vec<Value>> {
        let url = format!("{}/api/tags", self.base_url);
        let response = self.client.get(&url).send().await
            .map_err(|e| format!("Failed to connect to Ollama API: {}. Is Ollama running?", e))?;
        
        if !response.status().is_success() {
            return Err(format!("Ollama API returned error: {}", response.status()).into());
        }
        
        let data: Value = response.json().await
            .map_err(|e| format!("Failed to parse Ollama response: {}", e))?;
        
        Ok(data["models"].as_array().unwrap_or(&vec![]).clone())
    }
    
    /// Get detailed model information
    async fn get_model_info(&self, model_name: &str) -> Result<Value> {
        let url = format!("{}/api/show", self.base_url);
        let payload = serde_json::json!({ "name": model_name });
        
        let response = self.client.post(&url)
            .json(&payload)
            .send().await
            .map_err(|e| format!("Failed to get model info: {}", e))?;
        
        if !response.status().is_success() {
            return Err(format!("Failed to get model info: {}", response.status()).into());
        }
        
        let data: Value = response.json().await
            .map_err(|e| format!("Failed to parse model info: {}", e))?;
        
        Ok(data)
    }
    
    /// Find the local path where Ollama stores the model
    async fn find_ollama_model_path(&self, model_name: &str) -> Result<String> {
        // Ollama typically stores models in ~/.ollama/models
        let home_dir = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .map_err(|_| "Could not determine home directory")?;
        
        let ollama_dir = format!("{}/.ollama/models", home_dir);
        
        // Search for the model files
        // This is a simplified approach - in practice we'd parse the manifest
        let manifests_dir = format!("{}/manifests", ollama_dir);
        let blobs_dir = format!("{}/blobs", ollama_dir);
        
        // For now, return a placeholder - this would need proper manifest parsing
        Ok(format!("{}/blobs/[model-hash]", ollama_dir))
    }
    
    /// Stream compress GGUF file from Ollama storage
    async fn stream_gguf_file(
        &self,
        model_path: &str,
        novaq_engine: &mut NOVAQEngine,
        progress_callback: impl Fn(f32, &str) + Send + Sync,
    ) -> Result<NOVAQModel> {
        progress_callback(0.4, "Parsing GGUF model format...");
        
        // TODO: Implement GGUF parsing and streaming compression
        // This would involve:
        // 1. Parse GGUF header
        // 2. Extract tensors progressively  
        // 3. Convert to WeightMatrix format
        // 4. Apply NOVAQ compression
        
        progress_callback(0.5, "GGUF streaming compression not fully implemented yet");
        
        Err("GGUF streaming compression from Ollama not implemented yet".into())
    }
}

impl Default for HuggingFaceStreamer {
    fn default() -> Self {
        Self::new().expect("Failed to create HuggingFace streamer")
    }
}

impl Default for OllamaStreamer {
    fn default() -> Self {
        Self::new()
    }
}