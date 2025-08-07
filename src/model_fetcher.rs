use std::{fs, path::PathBuf};
use anyhow::Context;

#[derive(Debug, Clone)]
pub enum ModelSource {
    HuggingFace { repo: String, file: Option<String> },
    Url { url: String, filename: Option<String> },
    Ollama { model: String },
    LocalPath { path: PathBuf },
}

#[derive(Debug, Clone)]
pub struct FetchResult {
    pub local_path: PathBuf,
}

pub struct ModelFetcher;

impl ModelFetcher {
    pub fn fetch(source: &ModelSource) -> anyhow::Result<FetchResult> {
        match source {
            ModelSource::LocalPath { path } => Ok(FetchResult { local_path: path.clone() }),
            ModelSource::Url { url, filename } => fetch_via_http(url, filename.as_deref()),
            ModelSource::HuggingFace { repo, file } => fetch_hf(repo, file.as_deref()),
            ModelSource::Ollama { model } => fetch_ollama(model),
        }
    }
}

fn fetch_via_http(url: &str, filename: Option<&str>) -> anyhow::Result<FetchResult> {
    let resp = reqwest::blocking::get(url).with_context(|| format!("GET {} failed", url))?;
    anyhow::ensure!(resp.status().is_success(), "download failed: {}", resp.status());
    let bytes = resp.bytes()?;
    let name = filename.map(|s| s.to_string()).unwrap_or_else(|| infer_filename_from_url(url));
    let path = std::env::temp_dir().join(name);
    fs::write(&path, &bytes)?;
    Ok(FetchResult { local_path: path })
}

fn fetch_hf(repo: &str, file: Option<&str>) -> anyhow::Result<FetchResult> {
    // Use HuggingFace hub raw URLs for direct file fetch to avoid adding heavy deps.
    // e.g., https://huggingface.co/<repo>/resolve/main/<file>
    let target_file = file.unwrap_or("model.safetensors");
    let url = format!("https://huggingface.co/{}/resolve/main/{}", repo, target_file);
    fetch_via_http(&url, Some(target_file))
}

fn fetch_ollama(model: &str) -> anyhow::Result<FetchResult> {
    // Ollama stores blobs under ~/.ollama/models/blobs, but mapping model->blob is non-trivial.
    // Strategy: call `ollama pull` to ensure local cache, then advise user to provide a specific file if needed.
    // Here we try `ollama show --modelfile` to locate weights path if available.
    let status = std::process::Command::new("ollama").arg("pull").arg(model).status();
    anyhow::ensure!(status.map(|s| s.success()).unwrap_or(false), "ollama pull {} failed", model);

    // Fallback: export to a temporary file via `ollama show --model` (JSON) and return the model name as a handle.
    // Since Ollama models are not a single file universally, we return a pseudo path instructing loader to handle GGUF by name.
    let pseudo = PathBuf::from(format!("ollama://{}", model));
    Ok(FetchResult { local_path: pseudo })
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

