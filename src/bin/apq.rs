use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use nu_ansi_term::Color::{Purple, White};
use ohms_adaptq::{
    VerificationConfig, VerificationEngine, Quantizer, QuantizationConfig, QuantizationMethod,
    UniversalLoader, ModelFetcher, parse_model_source,
};
use candid::Encode;
use std::str::FromStr;

#[derive(Parser)]
#[command(name = "apq", version, about = "OHMS Adaptive Quantization - APQ CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Quantize {
        #[arg(long, help = "Model source: hf:<repo>[:file] | url:<http(s)://...> | ollama:<name> | file:/abs/path | /abs/path")] model: String,
        #[arg(long, default_value = "SpinQuant")] method: String,
        #[arg(long, default_value_t = 1.58)] weight_bits: f32,
        #[arg(long, default_value_t = 4)] activation_bits: u8,
        #[arg(long, default_value_t = 128)] group_size: usize,
        #[arg(long, default_value_t = true)] per_channel: bool,
        #[arg(long, value_name = "N", help = "Override CPU threads (default: all cores)")] threads: Option<usize>,
        #[arg(long, value_name = "speed|balanced|quality", help = "Preset that overrides method/bits/calibration for speed or quality")] preset: Option<String>,
        #[arg(long, default_value_t = false, help = "Reduce RAM usage (bigger groups, fewer samples)")] low_mem: bool,
        #[arg(long, help = "Optional path to save SAPQ artifact (binary)")] out: Option<String>,
    },
    Verify {
        #[arg(long)] original: String,
        #[arg(long)] quantized: String,
    },
    Info {
        #[arg(long, help = "Model source: hf:<repo>[:file] | url:<http(s)://...> | ollama:<name> | file:/abs/path | /abs/path")] 
        model: Option<String>,
    },
    Net {
        #[command(subcommand)]
        command: NetCmd,
    },
    Pack {
        #[arg(long, help = "Input SAPQ artifact path (from --out in quantize)")] input: String,
        #[arg(long, help = "Output JSON artifact path for ohms-model publish")] out: String,
        #[arg(long, help = "Override architecture family (default: original_model)")] family: Option<String>,
    },
    Publish {
        #[arg(long, help = "Canister ID (principal text)")] canister: String,
        #[arg(long, help = "DFX identity to use (optional)")] identity: Option<String>,
        #[arg(long, help = "Network URL, e.g., https://ic0.app (default: local agent)")] network: Option<String>,
        #[arg(long, help = "Model ID to store under")] model_id: String,
        #[arg(long, help = "Source model descriptor (e.g., hf:repo:file or note)")] source_model: String,
        #[arg(long, help = "Path to serialized Super-APQ JSON artifact (with fields architecture, compressed_model, verification)")] artifact: String,
    },
}

#[derive(Subcommand)]
enum NetCmd {
    /// Boost bandwidth for APQ downloads (Linux: requires sudo). Best-effort.
    Boost { #[arg(long)] device: Option<String> },
    /// Reset traffic shaping rules applied by Boost
    Reset { #[arg(long)] device: Option<String> },
}

fn parse_method(name: &str) -> QuantizationMethod {
    match name.to_lowercase().as_str() {
        "spinquant" => QuantizationMethod::SpinQuant,
        "ternaryllmdlt" | "ternary" => QuantizationMethod::TernaryLLMDLT,
        "vptq" => QuantizationMethod::VPTQ,
        "duquant" => QuantizationMethod::DuQuant,
        "hyperternary" => QuantizationMethod::HyperTernary,
        "adaptivebits" => QuantizationMethod::AdaptiveBits,
        "zeroshot" => QuantizationMethod::ZeroShot,
        "neuralquant" => QuantizationMethod::NeuralQuant,
        "gptq" => QuantizationMethod::GPTQ,
        "awq" => QuantizationMethod::AWQ,
        "smoothquant" => QuantizationMethod::SmoothQuant,
        "bitnet" => QuantizationMethod::BitNet,
        "int4" => QuantizationMethod::INT4,
        "int8" => QuantizationMethod::INT8,
        _ => QuantizationMethod::SpinQuant,
    }
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Quantize { model, method, weight_bits, activation_bits, group_size, per_channel, threads, preset, low_mem, out } => {
            let banner = format!("{} {}", Purple.bold().paint("Ω"), White.bold().paint("APQ Quantize"));
            eprintln!("{}", banner);
            let stage = ProgressBar::new_spinner();
            stage.set_style(ProgressStyle::with_template("{spinner} {msg}").unwrap());
            stage.enable_steady_tick(std::time::Duration::from_millis(80));
            stage.set_message("Fetching model...");
            if let Some(n) = threads {
                let _ = rayon::ThreadPoolBuilder::new().num_threads(n).build_global();
            }
            let mut method = parse_method(&method);
            let config = QuantizationConfig {
                method,
                weight_bits,
                activation_bits,
                group_size,
                use_symmetric: true,
                per_channel,
                calibration_samples: 512,
            };
            let mut config = config;
            if let Some(p) = preset.as_deref() {
                match p {
                    "speed" => {
                        method = QuantizationMethod::INT8;
                        config.method = method;
                        config.weight_bits = 8.0;
                        config.activation_bits = 8;
                        config.group_size = 256;
                        config.per_channel = false;
                        config.calibration_samples = 0;
                    }
                    "balanced" => {
                        config.group_size = 256;
                        config.calibration_samples = 128;
                    }
                    "quality" => {
                        // keep provided settings, but more samples
                        config.calibration_samples = 1024;
                    }
                    _ => {}
                }
            }
            if low_mem {
                config.group_size = config.group_size.max(256);
                config.per_channel = false;
                if config.calibration_samples > 128 { config.calibration_samples = 128; }
            }

            // Remote-aware fetch → local path or special handle
            let source = parse_model_source(&model);
            let fetched = ModelFetcher::fetch(&source).map_err(|e| anyhow::anyhow!(e.to_string()))?;
            stage.set_message("Loading model...");

            let mut loader = UniversalLoader::new();
            let model = loader
                .load_model(&fetched.local_path)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            stage.set_message("Quantizing...");
            let mut quantizer = Quantizer::new(config);
            let result = quantizer
                .quantize_model(&model)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            stage.finish_and_clear();
            println!(
                "{} {}: {:.1}x compression",
                Purple.bold().paint("Ω"),
                result.metadata.original_model,
                result.metadata.compression_ratio
            );
            if let Some(path) = out {
                use std::path::Path;
                quantizer
                    .save_quantized_model(&result, Path::new(&path))
                    .map_err(|e| anyhow::anyhow!(e.to_string()))?;
                println!("{} Saved SAPQ artifact to {}", White.bold().paint("OK"), path);
            }
        }
        Commands::Pack { input, out, family } => {
            use std::path::Path;
            let banner = format!("{} {}", Purple.bold().paint("Ω"), White.bold().paint("APQ Pack"));
            eprintln!("{}", banner);
            let data = std::fs::read(&input)?;
            // Load quantized model for metadata
            let qm = ohms_adaptq::quantizer::Quantizer::load_quantized_model(Path::new(&input))
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            // Derive architecture fields
            let arch_family = family.unwrap_or_else(|| qm.metadata.original_model.clone());
            let layers = qm.layers.len() as u32;
            // Try estimate hidden size from first weight tensor shape second dim
            let hidden_size: u32 = qm.layers.iter()
                .flat_map(|l| l.weights.iter())
                .filter_map(|t| t.shape.get(1).copied())
                .next()
                .unwrap_or(0) as u32;

            #[derive(serde::Serialize)]
            struct ArchitectureSerde { family: String, layers: u32, hidden_size: u32 }
            #[derive(serde::Serialize)]
            struct CompressedMetaSerde { compression_ratio: f32, original_size: u64, compressed_size: u64 }
            #[derive(serde::Serialize)]
            struct CompressedModelSerde { data: String, metadata: CompressedMetaSerde }
            #[derive(serde::Serialize)]
            struct VerificationSerde { bit_accuracy: f32 }
            #[derive(serde::Serialize)]
            struct SuperQuantizedModelSerde {
                architecture: ArchitectureSerde,
                compressed_model: CompressedModelSerde,
                verification: VerificationSerde,
            }

            let artifact = SuperQuantizedModelSerde {
                architecture: ArchitectureSerde { family: arch_family, layers, hidden_size },
                compressed_model: CompressedModelSerde {
                    data: base64::encode(&data),
                    metadata: CompressedMetaSerde {
                        compression_ratio: qm.metadata.compression_ratio,
                        original_size: qm.metadata.original_size_bytes,
                        compressed_size: qm.metadata.quantized_size_bytes,
                    }
                },
                verification: VerificationSerde { bit_accuracy: qm.verification.accuracy_retention },
            };
            let json = serde_json::to_string_pretty(&artifact)?;
            std::fs::write(&out, json)?;
            println!("{} Wrote JSON artifact to {}", White.bold().paint("OK"), out);
        }
        Commands::Verify { original, quantized } => {
            let banner = format!("{} {}", Purple.bold().paint("Ω"), White.bold().paint("APQ Verify"));
            eprintln!("{}", banner);
            let stage = ProgressBar::new_spinner();
            stage.set_style(ProgressStyle::with_template("{spinner} {msg}").unwrap());
            stage.enable_steady_tick(std::time::Duration::from_millis(80));
            stage.set_message("Running verification...");
            let engine = VerificationEngine::new(VerificationConfig::default());
            let report = futures::executor::block_on(engine.verify_model(&original, &quantized))
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            stage.finish_and_clear();
            println!("{}", engine.generate_report_string(&report));
        }
        Commands::Info { model } => {
            if let Some(model) = model {
                let banner = format!("{} {}", Purple.bold().paint("Ω"), White.bold().paint("APQ Info"));
                eprintln!("{}", banner);
                let source = parse_model_source(&model);
                let fetched = ModelFetcher::fetch(&source).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                let mut loader = UniversalLoader::new();
                let m = loader
                    .load_model(&fetched.local_path)
                    .map_err(|e| anyhow::anyhow!(e.to_string()))?;
                let total_params: u64 = m.layers.iter().map(|l| l.parameters).sum();
                println!(
                    "Model: {}\nFormat: {:?}\nArch: {}\nLayers: {}\nParams (sum): {}\nHidden size: {}\nHeads: {}\nContext length: {}\nPrecision: {:?}",
                    m.metadata.name,
                    m.format,
                    m.metadata.architecture,
                    m.layers.len(),
                    total_params,
                    m.metadata.hidden_size,
                    m.metadata.num_heads,
                    m.metadata.context_length,
                    m.metadata.precision
                );
            } else {
                println!("APQ CLI - available methods: SpinQuant, TernaryLLMDLT, VPTQ, DuQuant, HyperTernary, AdaptiveBits, ZeroShot, NeuralQuant, GPTQ, AWQ, SmoothQuant, BitNet, INT4, INT8");
                println!("\nUsage: apq info --model <hf:repo[:file]|url:...|ollama:name|file:/abs/path|/abs/path>");
            }
        }
        Commands::Net { command } => {
            match command {
                NetCmd::Boost { device } => {
                    let dev = device.or_else(detect_default_iface).unwrap_or_else(|| "eth0".into());
                    eprintln!("{} Boosting download priority on {} (requires sudo)…", Purple.bold().paint("Ω"), dev);
                    // Best-effort; ignore failures
                    let _ = run("sudo", &["tc", "qdisc", "add", "dev", &dev, "root", "handle", "1:", "htb", "default", "20"]);
                    let _ = run("sudo", &["tc", "class", "add", "dev", &dev, "parent", "1:", "classid", "1:10", "htb", "rate", "1000mbit", "prio", "0"]);
                    let _ = run("sudo", &["tc", "class", "add", "dev", &dev, "parent", "1:", "classid", "1:20", "htb", "rate", "10mbit", "ceil", "1000mbit", "prio", "7"]);
                    let _ = run("sudo", &["iptables", "-t", "mangle", "-A", "OUTPUT", "-p", "tcp", "--dport", "443", "-j", "MARK", "--set-mark", "10"]);
                    let _ = run("sudo", &["tc", "filter", "add", "dev", &dev, "parent", "1:", "protocol", "ip", "handle", "10", "fw", "flowid", "1:10"]);
                    println!("{} Bandwidth boost active on {}. Run: apq net reset --device {} to undo.", White.bold().paint("OK"), dev, dev);
                }
                NetCmd::Reset { device } => {
                    let dev = device.or_else(detect_default_iface).unwrap_or_else(|| "eth0".into());
                    let _ = run("sudo", &["tc", "qdisc", "del", "dev", &dev, "root"]);
                    let _ = run("sudo", &["iptables", "-t", "mangle", "-F"]);
                    println!("{} Traffic shaping cleared on {}.", White.bold().paint("OK"), dev);
                }
            }
        },
        Commands::Publish { canister, identity: _identity, network, model_id, source_model, artifact } => {
            let banner = format!("{} {}", Purple.bold().paint("Ω"), White.bold().paint("APQ Publish"));
            eprintln!("{}", banner);
            let data = std::fs::read_to_string(&artifact)?;
            #[derive(serde::Deserialize)]
            struct SuperQuantizedModelSerde {
                architecture: ArchitectureSerde,
                compressed_model: CompressedModelSerde,
                verification: VerificationSerde,
            }
            #[derive(serde::Deserialize)]
            struct ArchitectureSerde { family: String, layers: u32, hidden_size: u32 }
            #[derive(serde::Deserialize)]
            struct CompressedModelSerde { data: String, metadata: CompressedMetaSerde }
            #[derive(serde::Deserialize)]
            struct CompressedMetaSerde { compression_ratio: f32, original_size: u64, compressed_size: u64 }
            #[derive(serde::Deserialize)]
            struct VerificationSerde { bit_accuracy: f32 }

            let parsed: SuperQuantizedModelSerde = serde_json::from_str(&data)?;
            // Expect base64 for data string; accept hex as fallback
            let bytes = if let Ok(b) = base64::decode(&parsed.compressed_model.data) { b } else { hex::decode(&parsed.compressed_model.data)? };

            // Recreate ohms-model's domain types inline for candid encoding
            #[derive(candid::CandidType, serde::Serialize)]
            struct QuantArch { family: String, layers: u32, hidden_size: u32 }
            #[derive(candid::CandidType, serde::Serialize)]
            struct CompressedMeta { compression_ratio: f32, original_size: u64, compressed_size: u64 }
            #[derive(candid::CandidType, serde::Serialize)]
            struct CompressedModel { data: Vec<u8>, metadata: CompressedMeta }
            #[derive(candid::CandidType, serde::Serialize, Clone)]
            struct Verification { bit_accuracy: f32 }
            #[derive(candid::CandidType, serde::Serialize)]
            struct SuperQuantizedModel { architecture: QuantArch, compressed_model: CompressedModel, verification: Verification }

            let sq = SuperQuantizedModel {
                architecture: QuantArch { family: parsed.architecture.family, layers: parsed.architecture.layers, hidden_size: parsed.architecture.hidden_size },
                compressed_model: CompressedModel { data: bytes, metadata: CompressedMeta { compression_ratio: parsed.compressed_model.metadata.compression_ratio, original_size: parsed.compressed_model.metadata.original_size, compressed_size: parsed.compressed_model.metadata.compressed_size }},
                verification: Verification { bit_accuracy: parsed.verification.bit_accuracy },
            };
            let verification = sq.verification.clone();

            // Encode candid args: (text, text, SuperQuantizedModel, Verification)
            let args = Encode!(&model_id, &source_model, &sq, &verification)?;

            // Build agent
            let url = network.unwrap_or_else(|| "http://127.0.0.1:4943".to_string());
            use std::env;
            use ic_agent::identity::BasicIdentity;
            let agent = if let Some(id_name) = _identity.clone() {
                let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
                let pem_path = format!("{}/.config/dfx/identity/{}/identity.pem", home, id_name);
                let pem = std::fs::read_to_string(&pem_path)
                    .map_err(|e| anyhow::anyhow!(format!("load identity {} failed: {}", pem_path, e)))?;
                let identity = BasicIdentity::from_pem(pem.as_bytes())
                    .map_err(|e| anyhow::anyhow!(e.to_string()))?;
                ic_agent::Agent::builder()
                    .with_identity(identity)
                    .with_url(url)
                    .build()?
            } else if let Ok(pem_path) = env::var("DFX_IDENTITY_PEM") {
                let pem = std::fs::read_to_string(&pem_path)
                    .map_err(|e| anyhow::anyhow!(format!("load identity {} failed: {}", pem_path, e)))?;
                let identity = BasicIdentity::from_pem(pem.as_bytes())
                    .map_err(|e| anyhow::anyhow!(e.to_string()))?;
                ic_agent::Agent::builder()
                    .with_identity(identity)
                    .with_url(url)
                    .build()?
            } else {
                ic_agent::Agent::builder()
                    .with_url(url)
                    .build()?
            };
            // On mainnet, fetch root key is not allowed; on local, it's required
            let _ = futures::executor::block_on(agent.fetch_root_key());

            let canister_id = ic_agent::export::Principal::from_str(&canister)?;
            let method = "submit_quantized_model";
            let fut = agent.update(&canister_id, method)
                .with_arg(args)
                .call_and_wait();
            let res = futures::executor::block_on(fut);
            match res {
                Ok(_) => println!("{} Published to {} as {}", White.bold().paint("OK"), canister, model_id),
                Err(e) => {
                    eprintln!("Publish failed: {}", e);
                    std::process::exit(1);
                }
            }
        }
    }

    Ok(())
}

fn run(cmd: &str, args: &[&str]) -> std::io::Result<std::process::ExitStatus> {
    std::process::Command::new(cmd).args(args).status()
}

fn detect_default_iface() -> Option<String> {
    let out = std::process::Command::new("bash")
        .arg("-lc")
        .arg("ip -br route get 8.8.8.8 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i==\"dev\") {print $(i+1); exit}}'")
        .output()
        .ok()?;
    if !out.status.success() { return None; }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}

