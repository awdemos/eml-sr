use candle_core::{Device, Tensor};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::error::{Error, Result};
use crate::expr::Expr;
use crate::tree::EmlTree;

/// The outcome of a training run.
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Snapped symbolic expression.
    pub expr: Expr,
    /// Final training MSE before snapping.
    pub train_mse: f64,
    /// MSE of the snapped expression on the training data.
    pub snapped_mse: f64,
    /// Number of epochs actually trained.
    pub epochs: usize,
}

/// Builder/trainer for EML symbolic regression.
pub struct Trainer {
    depth: usize,
    lr: f64,
    epochs: usize,
    seeds: usize,
    hardening_lambda: f64,
    early_stop_tol: f64,
    patience: usize,
    device: Device,
}

impl Trainer {
    pub fn new(depth: usize) -> Result<Self> {
        if depth == 0 {
            return Err(Error::InvalidDepth(depth));
        }
        Ok(Self {
            depth,
            lr: 1e-2,
            epochs: 5_000,
            seeds: 5,
            hardening_lambda: 1e-3,
            early_stop_tol: 1e-12,
            patience: 500,
            device: Device::Cpu,
        })
    }

    pub fn lr(mut self, lr: f64) -> Self {
        self.lr = lr;
        self
    }

    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    pub fn seeds(mut self, seeds: usize) -> Self {
        self.seeds = seeds;
        self
    }

    pub fn hardening_lambda(mut self, lambda: f64) -> Self {
        self.hardening_lambda = lambda;
        self
    }

    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Fit an EML tree to `(xs, ys)` data.
    ///
    /// Tries `self.seeds` random initializations and returns the run with
    /// the lowest snapped MSE.
    pub fn fit(&self, xs: &[f64], ys: &[f64]) -> Result<TrainingResult> {
        if xs.is_empty() || ys.is_empty() {
            return Err(Error::EmptyDataset);
        }
        if xs.len() != ys.len() {
            Err(candle_core::Error::Msg(format!(
                "xs and ys must have the same length ({} vs {})",
                xs.len(),
                ys.len()
            )))?;
        }

        let xs_t = Tensor::new(xs.to_vec(), &self.device)?;
        let ys_t = Tensor::new(ys.to_vec(), &self.device)?;

        let mut best: Option<TrainingResult> = None;

        for seed in 0..self.seeds {
            let mut rng = StdRng::seed_from_u64(seed as u64);
            let tree = EmlTree::with_uniform_init(self.depth, &mut rng, self.device.clone())?;
            let result = self.fit_one(&tree, &xs_t, &ys_t)?;

            let keep = best
                .as_ref()
                .map(|b| result.snapped_mse < b.snapped_mse)
                .unwrap_or(true);
            if keep {
                best = Some(result);
            }
        }

        best.ok_or_else(|| Error::SnapFailed { mse: f64::INFINITY })
    }

    fn fit_one(&self, tree: &EmlTree, xs: &Tensor, ys: &Tensor) -> Result<TrainingResult> {
        let opt_params = ParamsAdamW {
            lr: self.lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 1e-4,
        };
        let mut opt = AdamW::new(vec![tree.params().clone()], opt_params)?;

        let mut best_loss = f64::INFINITY;
        let mut patience_counter = 0usize;
        let mut trained_epochs = 0usize;

        for epoch in 0..self.epochs {
            let pred = tree.forward(xs)?;
            let mse = mse_loss(&pred, ys)?;

            // Anneal hardening strength from 0 up to the configured lambda.
            let progress = (epoch as f64) / (self.epochs as f64).max(1.0);
            let lambda = self.hardening_lambda * progress * progress;
            let hard = tree.hardening_loss()?;
            let lambda_t = Tensor::new(lambda, &self.device)?;
            let loss = (mse.clone() + hard.broadcast_mul(&lambda_t)?)?;

            opt.backward_step(&loss)?;

            let loss_f: f64 = mse.to_vec0()?;
            trained_epochs = epoch + 1;

            if loss_f < best_loss - self.early_stop_tol {
                best_loss = loss_f;
                patience_counter = 0;
            } else {
                patience_counter += 1;
                if patience_counter >= self.patience {
                    break;
                }
            }
        }

        // Snap and evaluate the discrete expression.
        let snapped = tree.snap()?;
        let snapped_mse = eval_expr_mse(&snapped, xs, ys)?;
        let simplified = snapped.simplify();

        Ok(TrainingResult {
            expr: simplified,
            train_mse: best_loss,
            snapped_mse,
            epochs: trained_epochs,
        })
    }

    pub fn depth(&self) -> usize {
        self.depth
    }
}

fn mse_loss(pred: &Tensor, target: &Tensor) -> Result<Tensor> {
    let diff = (pred - target)?;
    let sq = diff.sqr()?;
    Ok(sq.mean_all()?)
}

fn eval_expr_mse(expr: &Expr, xs: &Tensor, ys: &Tensor) -> Result<f64> {
    let xs_vec: Vec<f64> = xs.to_vec1()?;
    let ys_vec: Vec<f64> = ys.to_vec1()?;
    let n = xs_vec.len();
    if n == 0 {
        return Ok(f64::INFINITY);
    }
    let mse: f64 = xs_vec
        .iter()
        .zip(&ys_vec)
        .map(|(x, y)| {
            let pred = expr.eval(*x);
            (pred - y).powi(2)
        })
        .sum::<f64>()
        / (n as f64);
    Ok(mse)
}
