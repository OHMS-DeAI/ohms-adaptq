use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use nu_ansi_term::Color::{Purple, White};
use ohms_adaptq::{
    VerificationConfig, VerificationEngine, Quantizer, QuantizationConfig, QuantizationMethod,
    UniversalLoader, ModelFetcher, parse_model_source,
};

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
        Commands::Quantize { model, method, weight_bits, activation_bits, group_size, per_channel } => {
            let banner = format!("{} {}", Purple.bold().paint("Ω"), White.bold().paint("APQ Quantize"));
            eprintln!("{}", banner);
            let stage = ProgressBar::new_spinner();
            stage.set_style(ProgressStyle::with_template("{spinner} {msg}").unwrap());
            stage.enable_steady_tick(std::time::Duration::from_millis(80));
            stage.set_message("Fetching model...");
            let method = parse_method(&method);
            let config = QuantizationConfig {
                method,
                weight_bits,
                activation_bits,
                group_size,
                use_symmetric: true,
                per_channel,
                calibration_samples: 512,
            };

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

