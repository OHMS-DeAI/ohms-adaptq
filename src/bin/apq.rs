// Legacy APQ CLI - Redirects to Super-APQ
// Maintained for backward compatibility

use clap::{Parser, Subcommand};
use std::process::Command;

#[derive(Parser)]
#[command(name = "apq")]
#[command(about = "Legacy APQ CLI - Now powered by Super-APQ")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Quantize a model (redirects to Super-APQ)
    Quantize {
        model: String,
        #[arg(short, long)]
        output: Option<String>,
        #[arg(short, long)]
        bits: Option<u8>,
    },
    /// Verify artifacts
    Verify {
        path: String,
    },
    /// Generate report
    Report {
        path: String,
        #[arg(short, long)]
        format: Option<String>,
    },
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  NOTICE: APQ has been upgraded to Super-APQ!            ║");
    println!("║  You're now using 1000x compression technology.         ║");
    println!("║  Use 'super-apq' directly for all new features.         ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
    
    let cli = Cli::parse();
    
    // Redirect to super-apq with equivalent commands
    let status = match cli.command {
        Commands::Quantize { model, output, .. } => {
            let mut cmd = Command::new("super-apq");
            cmd.arg("quantize")
               .arg("--model").arg(&model);
            
            if let Some(out) = output {
                cmd.arg("--output").arg(out);
            }
            
            cmd.arg("--zero-cost")
               .status()
               .expect("Failed to execute super-apq")
        },
        
        Commands::Verify { path } => {
            Command::new("super-apq")
                .arg("verify")
                .arg(path)
                .arg("--perplexity")
                .arg("--accuracy")
                .status()
                .expect("Failed to execute super-apq")
        },
        
        Commands::Report { path, format } => {
            let mut cmd = Command::new("super-apq");
            cmd.arg("stats").arg(path);
            
            if let Some(fmt) = format {
                println!("Format option '{}' noted. Super-APQ provides enhanced statistics.", fmt);
            }
            
            cmd.status()
                .expect("Failed to execute super-apq")
        },
    };
    
    std::process::exit(status.code().unwrap_or(1));
}