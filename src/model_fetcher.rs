use std::{fs, path::PathBuf};
use std::process::Command;
use anyhow::Context;
use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, USER_AGENT, ACCEPT};
use std::time::Duration;
use std::io::{Write, Read};
use indicatif::{ProgressBar, ProgressStyle};
use nu_ansi_term::Color::{White, Purple};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum ModelSource {
    HuggingFace { repo: String, file: Option<String> },
    Url { url: String, filename: Option<String> },
    Ollama { model: String },
    LocalPath { path: PathBuf },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub model_id: String,
    pub architecture: String,
    pub parameters: u64,
    pub model_type: String,
    pub tokenizer_config: Option<HashMap<String, serde_json::Value>>,
    pub config: Option<HashMap<String, serde_json::Value>>,
    pub files: Vec<String>,
    pub license: Option<String>,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct FetchResult {
    pub local_path: PathBuf,
    pub metadata: Option<ModelMetadata>,
    pub model_format: ModelFormat,
}

#[derive(Debug, Clone)]
pub enum ModelFormat {
    SafeTensors,
    PyTorch,
    GGUF,
    ONNX,
    Unknown,
}

impl ModelSource {
    /// Check if this source requires remote fetching
    pub fn is_remote(&self) -> bool {
        match self {
            ModelSource::LocalPath { .. } => false,
            _ => true,
        }
    }
    
    /// Convert source to direct URL for streaming
    pub fn to_url(&self) -> anyhow::Result<String> {
        match self {
            ModelSource::HuggingFace { repo, file } => {
                let file_name = file.as_deref().unwrap_or("model.safetensors");
                Ok(format!("https://huggingface.co/{}/resolve/main/{}", repo, file_name))
            }
            ModelSource::Url { url, .. } => Ok(url.clone()),
            ModelSource::Ollama { model } => {
                Err(anyhow::anyhow!("Ollama models require local pull first: ollama pull {}", model))
            }
            ModelSource::LocalPath { .. } => {
                Err(anyhow::anyhow!("Local paths don't have URLs"))
            }
        }
    }
}

pub struct ModelFetcher;

impl ModelFetcher {
    pub fn fetch(source: &ModelSource) -> anyhow::Result<FetchResult> {
        match source {
            ModelSource::LocalPath { path } => {
                let format = detect_model_format(path)?;
                Ok(FetchResult { 
                    local_path: path.clone(),
                    metadata: None,
                    model_format: format,
                })
            },
            ModelSource::Url { url, filename } => fetch_via_http(url, filename.as_deref()),
            ModelSource::HuggingFace { repo, file } => fetch_hf_with_metadata(repo, file.as_deref()),
            ModelSource::Ollama { model } => fetch_ollama_with_metadata(model),
        }
    }

    /// Fetch model metadata from Hugging Face
    pub fn fetch_metadata(repo: &str) -> anyhow::Result<ModelMetadata> {
        let token = std::env::var("HF_TOKEN").ok().or_else(|| std::env::var("HUGGINGFACE_HUB_TOKEN").ok());
        
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()?;
        
        let mut headers = HeaderMap::new();
        headers.insert(USER_AGENT, HeaderValue::from_static("ohms-adaptq/2.0"));
        headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
        
        if let Some(t) = token {
            let hv = HeaderValue::from_str(&format!("Bearer {}", t))
                .map_err(|_| anyhow::anyhow!("Invalid token format"))?;
            headers.insert(AUTHORIZATION, hv);
        }

        let url = format!("https://huggingface.co/api/models/{}", repo);
        let resp = client.get(&url).headers(headers).send()?;
        
        if !resp.status().is_success() {
            return Err(anyhow::anyhow!("Failed to fetch metadata: {}", resp.status()));
        }

        let metadata: serde_json::Value = resp.json()?;
        
        Ok(ModelMetadata {
            model_id: repo.to_string(),
            architecture: metadata["model_type"].as_str().unwrap_or("unknown").to_string(),
            parameters: metadata["safetensors"]["total"].as_u64().unwrap_or(0),
            model_type: metadata["model_type"].as_str().unwrap_or("unknown").to_string(),
            tokenizer_config: metadata["tokenizer_config"].as_object().map(|o| o.iter().map(|(k, v)| (k.clone(), v.clone())).collect()),
            config: metadata["config"].as_object().map(|o| o.iter().map(|(k, v)| (k.clone(), v.clone())).collect()),
            files: metadata["siblings"]
                .as_array()
                .map(|arr| arr.iter()
                    .filter_map(|v| v["rfilename"].as_str().map(|s| s.to_string()))
                    .collect())
                .unwrap_or_default(),
            license: metadata["license"].as_str().map(|s| s.to_string()),
            tags: metadata["tags"]
                .as_array()
                .map(|arr| arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect())
                .unwrap_or_default(),
        })
    }
}

fn fetch_via_http(url: &str, filename: Option<&str>) -> anyhow::Result<FetchResult> {
    let client = Client::builder()
        .timeout(Duration::from_secs(900))
        .build()?;
    let mut resp = client
        .get(url)
        .header(USER_AGENT, HeaderValue::from_static("ohms-adaptq/2.0"))
        .header(ACCEPT, HeaderValue::from_static("application/octet-stream"))
        .send()
        .with_context(|| format!("GET {} failed", url))?;
    anyhow::ensure!(resp.status().is_success(), "download failed: {}", resp.status());
    let name = filename.map(|s| s.to_string()).unwrap_or_else(|| infer_filename_from_url(url));
    let path = std::env::temp_dir().join(name);
    let mut file = std::fs::File::create(&path)?;

    let total = resp.content_length().unwrap_or(0);
    let bar = ProgressBar::new(total);
    bar.set_style(
        ProgressStyle::with_template("{prefix:.bold} {bar:40.cyan/blue} {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")
            .unwrap()
            .progress_chars("##-"),
    );
    bar.set_prefix(format!("{} {}",
        White.bold().paint("Downloading"),
        Purple.bold().paint("OHMS"),
    ));

    let mut buf = [0u8; 1 << 16];
    loop {
        let n = resp.read(&mut buf)?;
        if n == 0 { break; }
        file.write_all(&buf[..n])?;
        bar.inc(n as u64);
    }
    bar.finish_and_clear();
    file.flush()?;
    
    let format = detect_model_format(&path)?;
    Ok(FetchResult { 
        local_path: path,
        metadata: None,
        model_format: format,
    })
}

fn fetch_via_http_with_auth(url: &str, filename: &str, token: Option<String>) -> anyhow::Result<FetchResult> {
    let client = Client::builder()
        .timeout(Duration::from_secs(900))
        .build()?;
    let mut headers = HeaderMap::new();
    headers.insert(USER_AGENT, HeaderValue::from_static("ohms-adaptq/2.0"));
    headers.insert(ACCEPT, HeaderValue::from_static("application/octet-stream"));
    if let Some(t) = token {
        let hv = HeaderValue::from_str(&format!("Bearer {}", t))
            .map_err(|_| anyhow::anyhow!("Invalid token format"))?;
        headers.insert(AUTHORIZATION, hv);
    }
    let mut resp = client
        .get(url)
        .headers(headers)
        .send()
        .with_context(|| format!("GET {} failed", url))?;
    anyhow::ensure!(resp.status().is_success(), "download failed: {}", resp.status());
    let path = std::env::temp_dir().join(filename);
    let mut file = std::fs::File::create(&path)?;

    let total = resp.content_length().unwrap_or(0);
    let bar = ProgressBar::new(total);
    bar.set_style(
        ProgressStyle::with_template("{prefix:.bold} {bar:40.cyan/blue} {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")
            .unwrap()
            .progress_chars("##-"),
    );
    bar.set_prefix(format!("{} {}",
        White.bold().paint("Downloading"),
        Purple.bold().paint("OHMS"),
    ));

    let mut buf = [0u8; 1 << 16];
    loop {
        let n = resp.read(&mut buf)?;
        if n == 0 { break; }
        file.write_all(&buf[..n])?;
        bar.inc(n as u64);
    }
    bar.finish_and_clear();
    file.flush()?;
    
    let format = detect_model_format(&path)?;
    Ok(FetchResult { 
        local_path: path,
        metadata: None,
        model_format: format,
    })
}

fn fetch_hf_with_metadata(repo: &str, file: Option<&str>) -> anyhow::Result<FetchResult> {
    // First, fetch metadata
    let metadata = ModelFetcher::fetch_metadata(repo)?;
    
    // Prefer huggingface-cli with hf_transfer if present; fall back to HTTP
    if let Some(res) = try_hf_cli_download(repo, file)? {
        let format = detect_model_format(&res)?;
        return Ok(FetchResult { 
            local_path: res,
            metadata: Some(metadata),
            model_format: format,
        });
    }

    // Use lightweight HTTP with auth headers and try common filenames if none provided
    let token = std::env::var("HF_TOKEN").ok().or_else(|| std::env::var("HUGGINGFACE_HUB_TOKEN").ok());
    let try_files: Vec<String> = if let Some(f) = file { 
        vec![f.to_string()] 
    } else { 
        // Use metadata to find the best file
        let mut files = vec![
            "model.safetensors".into(),
            "consolidated.safetensors".into(),
            "pytorch_model.bin".into(),
        ];
        
        // Add files from metadata if available
        for file_name in &metadata.files {
            if file_name.ends_with(".safetensors") || file_name.ends_with(".bin") {
                files.push(file_name.clone());
            }
        }
        files
    };

    let base = format!("https://huggingface.co/{}/resolve/main/", repo);
    let mut last_err: Option<anyhow::Error> = None;
    for fname in try_files {
        let url = format!("{}{}", &base, &fname);
        match fetch_via_http_with_auth(&url, &fname, token.clone()) {
            Ok(mut result) => {
                result.metadata = Some(metadata);
                return Ok(result);
            },
            Err(e) => { last_err = Some(e); }
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("no compatible weight file found in repo '{}'; specify :<file>", repo)))
}

fn try_hf_cli_download(repo: &str, file: Option<&str>) -> anyhow::Result<Option<PathBuf>> {
    // Check if huggingface-cli exists
    let which = Command::new("bash").arg("-lc").arg("command -v huggingface-cli").output()?;
    if which.status.code().unwrap_or(1) != 0 { return Ok(None); }

    let tmp = std::env::temp_dir().join(format!("ohms-hf-{}", repo.replace('/', "_")));
    let _ = std::fs::create_dir_all(&tmp);

    // Build args
    let mut args: Vec<String> = vec![
        "download".into(),
        repo.into(),
        "--local-dir".into(), tmp.to_string_lossy().to_string(),
        "--resume-download".into(),
    ];
    if let Some(f) = file { 
        args.push("--include".into()); 
        args.push(f.into()); 
    } else {
        args.push("--include".into()); args.push("model.safetensors".into());
        args.push("--include".into()); args.push("consolidated.safetensors".into());
        args.push("--include".into()); args.push("pytorch_model.bin".into());
    }

    let mut cmd = Command::new("huggingface-cli");
    cmd.args(&args);
    // Enable accelerated transfer if available
    cmd.env("HF_HUB_ENABLE_HF_TRANSFER", "1");
    if let Ok(tok) = std::env::var("HF_TOKEN") { cmd.env("HF_TOKEN", tok); }
    let status = cmd.status()?;
    if !status.success() { return Ok(None); }

    // Resolve target file path
    if let Some(f) = file { 
        let p = tmp.join(f);
        if p.exists() { return Ok(Some(p)); }
    }
    for cand in ["model.safetensors", "consolidated.safetensors", "pytorch_model.bin"] {
        let p = tmp.join(cand);
        if p.exists() { return Ok(Some(p)); }
    }
    Ok(None)
}

fn fetch_ollama_with_metadata(model: &str) -> anyhow::Result<FetchResult> {
    // First, ensure model is pulled
    let status = std::process::Command::new("ollama").arg("pull").arg(model).status();
    anyhow::ensure!(status.map(|s| s.success()).unwrap_or(false), "ollama pull {} failed", model);

    // Try to get model info
    let output = std::process::Command::new("ollama")
        .arg("show")
        .arg(model)
        .output()?;
    
    let model_info = if output.status.success() {
        serde_json::from_slice::<serde_json::Value>(&output.stdout).ok()
    } else {
        None
    };

    // For Ollama, we need to export the model to get the actual file
    let export_path = std::env::temp_dir().join(format!("{}.gguf", model.replace(':', "_")));
    
    let export_status = std::process::Command::new("ollama")
        .arg("export")
        .arg(model)
        .arg(&export_path)
        .status()?;
    
    anyhow::ensure!(export_status.success(), "ollama export {} failed", model);

    // Create metadata from Ollama info
    let metadata = ModelMetadata {
        model_id: model.to_string(),
        architecture: model_info.as_ref()
            .and_then(|info| info["model_type"].as_str())
            .unwrap_or("gguf")
            .to_string(),
        parameters: model_info.as_ref()
            .and_then(|info| info["parameter_size"].as_u64())
            .unwrap_or(0),
        model_type: "gguf".to_string(),
        tokenizer_config: None,
        config: model_info.as_ref()
            .and_then(|info| info["config"].as_object())
            .map(|o| o.iter().map(|(k, v)| (k.clone(), v.clone())).collect()),
        files: vec![export_path.file_name().unwrap().to_string_lossy().to_string()],
        license: None,
        tags: vec!["ollama".to_string(), "gguf".to_string()],
    };

    Ok(FetchResult { 
        local_path: export_path,
        metadata: Some(metadata),
        model_format: ModelFormat::GGUF,
    })
}

fn detect_model_format(path: &PathBuf) -> anyhow::Result<ModelFormat> {
    let file_name = path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    
    if file_name.ends_with(".safetensors") {
        Ok(ModelFormat::SafeTensors)
    } else if file_name.ends_with(".bin") || file_name.ends_with(".pt") || file_name.ends_with(".pth") {
        Ok(ModelFormat::PyTorch)
    } else if file_name.ends_with(".gguf") {
        Ok(ModelFormat::GGUF)
    } else if file_name.ends_with(".onnx") {
        Ok(ModelFormat::ONNX)
    } else {
        // Try to detect by file header
        if let Ok(mut file) = std::fs::File::open(path) {
            let mut header = [0u8; 16];
            if let Ok(_) = file.read_exact(&mut header) {
                // Check for SafeTensors magic
                if &header[0..8] == b"__safetensors__" {
                    return Ok(ModelFormat::SafeTensors);
                }
                // Check for GGUF magic
                if &header[0..4] == b"GGUF" {
                    return Ok(ModelFormat::GGUF);
                }
                // Check for PyTorch magic
                if &header[0..8] == b"PK\x03\x04" {
                    return Ok(ModelFormat::PyTorch);
                }
            }
        }
        Ok(ModelFormat::Unknown)
    }
}

fn infer_filename_from_url(url: &str) -> String {
    url.split('/')
        .last()
        .filter(|s| !s.is_empty())
        .unwrap_or("model.bin")
        .to_string()
}

pub fn parse_model_source(s: &str) -> ModelSource {
    // Examples:
    // hf:meta-llama/Llama-3-8B:consolidated.safetensors
    // url:https://host/path/file.bin
    // ollama:llama3:8b
    // file:/abs/path/model.onnx
    if let Some(rest) = s.strip_prefix("hf:") {
        let mut parts = rest.splitn(2, ':');
        let repo = parts.next().unwrap_or("").to_string();
        let file = parts.next().map(|v| v.to_string());
        return ModelSource::HuggingFace { repo, file };
    }
    if let Some(rest) = s.strip_prefix("url:") {
        return ModelSource::Url { url: rest.to_string(), filename: None };
    }
    if let Some(rest) = s.strip_prefix("ollama:") {
        return ModelSource::Ollama { model: rest.to_string() };
    }
    if let Some(rest) = s.strip_prefix("file:") {
        return ModelSource::LocalPath { path: PathBuf::from(rest) };
    }
    // Default: treat as local path
    ModelSource::LocalPath { path: PathBuf::from(s) }
}

