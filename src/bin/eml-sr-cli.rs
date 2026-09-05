use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::Device;
use clap::{Parser, Subcommand};

use eml_sr::Trainer;

#[derive(Parser)]
#[command(name = "eml-sr-cli")]
#[command(about = "Gradient-based symbolic regression with EML trees")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Fit an EML tree to a CSV dataset.
    Train {
        /// Path to CSV with columns `x` and `y`.
        data: PathBuf,

        /// Tree depth (number of EML layers).
        #[arg(short, long, default_value_t = 3)]
        depth: usize,

        /// Number of training epochs.
        #[arg(short, long, default_value_t = 5000)]
        epochs: usize,

        /// Adam learning rate.
        #[arg(long, default_value_t = 1e-2)]
        lr: f64,

        /// Number of random seeds to try.
        #[arg(short, long, default_value_t = 5)]
        seeds: usize,

        /// Hardening regularization strength.
        #[arg(long, default_value_t = 1e-3)]
        hardening_lambda: f64,

        /// Optional output file for the recovered expression.
        #[arg(short, long)]
        out: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Train {
            data,
            depth,
            epochs,
            lr,
            seeds,
            hardening_lambda,
            out,
        } => {
            let (xs, ys) = load_csv(&data)?;
            println!("Loaded {} data points from {}", xs.len(), data.display());

            let device = Device::Cpu;
            let result = Trainer::new(depth)?
                .lr(lr)
                .epochs(epochs)
                .seeds(seeds)
                .hardening_lambda(hardening_lambda)
                .device(device)
                .fit(&xs, &ys)
                .context("training failed")?;

            println!("Trained for {} epochs", result.epochs);
            println!("Training MSE (best seed): {:.6e}", result.train_mse);
            println!("Snapped MSE:              {:.6e}", result.snapped_mse);
            println!("Expression:               {}", result.expr);

            if let Some(path) = out {
                std::fs::write(&path, result.expr.to_string())
                    .with_context(|| format!("writing {}", path.display()))?;
                println!("Wrote expression to {}", path.display());
            }
        }
    }

    Ok(())
}

fn load_csv(path: &std::path::Path) -> Result<(Vec<f64>, Vec<f64>)> {
    let mut xs = Vec::new();
    let mut ys = Vec::new();

    let mut reader =
        csv::Reader::from_path(path).with_context(|| format!("opening CSV {}", path.display()))?;

    for (idx, result) in reader.records().enumerate() {
        let record = result.with_context(|| format!("reading row {}", idx + 1))?;
        let x: f64 = record
            .get(0)
            .context("missing x column")?
            .parse()
            .with_context(|| format!("parsing x at row {}", idx + 1))?;
        let y: f64 = record
            .get(1)
            .context("missing y column")?
            .parse()
            .with_context(|| format!("parsing y at row {}", idx + 1))?;
        xs.push(x);
        ys.push(y);
    }

    Ok((xs, ys))
}
