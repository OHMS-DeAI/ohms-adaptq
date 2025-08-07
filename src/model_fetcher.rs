use std::{fs, path::PathBuf};
use anyhow::Context;
use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, USER_AGENT, ACCEPT};
use std::time::Duration;
use std::io::{Write, Read};
use indicatif::{ProgressBar, ProgressStyle};
use nu_ansi_term::Color::{White, Purple};

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
    let client = Client::builder()
        .timeout(Duration::from_secs(900))
        .build()?;
    let mut resp = client
        .get(url)
        .header(USER_AGENT, HeaderValue::from_static("ohms-adaptq/0.9"))
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
    Ok(FetchResult { local_path: path })
}

fn fetch_via_http_with_auth(url: &str, filename: &str, token: Option<String>) -> anyhow::Result<FetchResult> {
    let client = Client::builder()
        .timeout(Duration::from_secs(900))
        .build()?;
    let mut headers = HeaderMap::new();
    headers.insert(USER_AGENT, HeaderValue::from_static("ohms-adaptq/0.9"));
    headers.insert(ACCEPT, HeaderValue::from_static("application/octet-stream"));
    if let Some(t) = token {
        let hv = HeaderValue::from_str(&format!("Bearer {}", t)).unwrap_or(HeaderValue::from_static(""));
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
    Ok(FetchResult { local_path: path })
}

fn fetch_hf(repo: &str, file: Option<&str>) -> anyhow::Result<FetchResult> {
    // Use lightweight HTTP with auth headers and try common filenames if none provided
    let token = std::env::var("HF_TOKEN").ok().or_else(|| std::env::var("HUGGINGFACE_HUB_TOKEN").ok());
    let try_files: Vec<String> = if let Some(f) = file { vec![f.to_string()] } else { vec![
        "model.safetensors".into(),
        "consolidated.safetensors".into(),
        "pytorch_model.bin".into(),
        // common first shards for sharded repos
        "model-00001-of-00001.safetensors".into(),
        "pytorch_model-00001-of-00001.bin".into(),
        "model-00001-of-000xx.safetensors".into(),
        "pytorch_model-00001-of-000xx.bin".into(),
    ]};

    let base = format!("https://huggingface.co/{}/resolve/main/", repo);
    let mut last_err: Option<anyhow::Error> = None;
    for fname in try_files {
        let url = format!("{}{}", &base, &fname);
        match fetch_via_http_with_auth(&url, &fname, token.clone()) {
            Ok(r) => return Ok(r),
            Err(e) => { last_err = Some(e); }
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("no compatible weight file found in repo '{}'; specify :<file>", repo)))
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

