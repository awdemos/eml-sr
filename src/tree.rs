use candle_core::{DType, Device, IndexOp, Tensor, Var};
use rand::Rng;

use crate::error::{Error, Result};
use crate::expr::Expr;
use crate::params::{input_blocks, n_params};

/// A trainable full binary EML tree.
///
/// Every internal node computes `eml(left, right) = exp(left) - ln(right)`.
/// Each node input is a softmax-weighted mixture of `1`, the input variable
/// `x`, and (for non-leaf layers) the output of a child sub-tree.
pub struct EmlTree {
    depth: usize,
    params: Var,
    blocks: Vec<(usize, usize, bool, usize, usize)>,
    device: Device,
}

impl EmlTree {
    /// Create a new tree of the given depth with initialized logits.
    ///
    /// `init` must have length `n_params(depth)`.
    pub fn new(depth: usize, init: &[f64], device: Device) -> Result<Self> {
        if depth == 0 {
            return Err(Error::InvalidDepth(depth));
        }
        let expected = n_params(depth);
        if init.len() != expected {
            Err(candle_core::Error::Msg(format!(
                "expected {expected} params, got {}",
                init.len()
            )))?;
        }

        let tensor = Tensor::new(init.to_vec(), &device)?.to_dtype(DType::F64)?;
        let params = Var::from_tensor(&tensor)?;
        let blocks = input_blocks(depth);

        Ok(Self {
            depth,
            params,
            blocks,
            device,
        })
    }

    /// Convenience constructor using biased initialization.
    pub fn with_biased_init<R: Rng>(depth: usize, rng: &mut R, device: Device) -> Result<Self> {
        let init = crate::params::init_biased(depth, rng);
        Self::new(depth, &init, device)
    }

    /// Convenience constructor using uniform initialization.
    pub fn with_uniform_init<R: Rng>(depth: usize, rng: &mut R, device: Device) -> Result<Self> {
        let init = crate::params::init_uniform(depth, rng);
        Self::new(depth, &init, device)
    }

    pub fn depth(&self) -> usize {
        self.depth
    }

    pub fn n_params(&self) -> usize {
        self.blocks.len() // each block is one input
    }

    pub fn params(&self) -> &Var {
        &self.params
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Hardening regularizer: sum of entropies of the input selectors.
    ///
    /// Minimizing entropy pushes each selector toward a one-hot choice of
    /// `1`, `x`, or a child output, which makes snapping more reliable.
    pub fn hardening_loss(&self) -> Result<Tensor> {
        let mut entropies = Vec::new();
        for (_, _, _, start, n_logits) in &self.blocks {
            let mut logits = Vec::with_capacity(*n_logits);
            for i in 0..*n_logits {
                logits.push(self.params.as_tensor().i(*start + i)?);
            }
            let logits = Tensor::stack(&logits, 0)?;
            let probs = candle_nn::ops::softmax(&logits, 0)?;

            // Entropy = -sum(p * log(p)).  Add a tiny floor to avoid log(0).
            let eps = Tensor::new(1e-12f64, &self.device)?.broadcast_as(probs.shape())?;
            let safe_probs = probs.maximum(&eps)?;
            let log_probs = safe_probs.log()?;
            let entropy = (&safe_probs * log_probs)?.neg()?.sum_all()?;
            entropies.push(entropy);
        }

        let total = Tensor::stack(&entropies, 0)?.sum_all()?;
        let n = Tensor::new(self.blocks.len() as f64, &self.device)?;
        Ok((total / n)?)
    }

    /// Forward evaluation: evaluate the tree on a batch of inputs `x`.
    ///
    /// `x` must be a 1-D tensor of shape `[batch]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.rank() != 1 {
            Err(candle_core::Error::Msg(format!(
                "expected 1-D input tensor, got rank {}",
                x.rank()
            )))?;
        }

        let ones = Tensor::ones(x.shape(), DType::F64, &self.device)?;
        let mut prev_outputs: Vec<Tensor> = Vec::new();

        for layer in 1..=self.depth {
            let n_nodes = 1usize << (self.depth - layer);
            let mut layer_outputs = Vec::with_capacity(n_nodes);

            for node in 0..n_nodes {
                let left = self.compute_input(layer, node, true, x, &ones, &prev_outputs)?;
                let right = self.compute_input(layer, node, false, x, &ones, &prev_outputs)?;
                let out = eml_tensor(&left, &right)?;
                layer_outputs.push(out);
            }

            prev_outputs = layer_outputs;
        }

        Ok(prev_outputs.into_iter().next().expect("tree has a root"))
    }

    /// Compute a single input to an `eml` node as a softmax-weighted mixture
    /// of its allowed sources.
    fn compute_input(
        &self,
        layer: usize,
        node: usize,
        is_left: bool,
        x: &Tensor,
        ones: &Tensor,
        prev_outputs: &[Tensor],
    ) -> Result<Tensor> {
        let (_, _, _, start, n_logits) = self
            .blocks
            .iter()
            .find(|(l, n, left, _, _)| *l == layer && *n == node && *left == is_left)
            .expect("parameter block exists");

        // Extract logits for this input.
        let mut logits = Vec::with_capacity(*n_logits);
        for i in 0..*n_logits {
            logits.push(self.params.as_tensor().i(*start + i)?);
        }
        let logits = Tensor::stack(&logits, 0)?;
        let probs = candle_nn::ops::softmax(&logits, 0)?;

        // Build source tensor stack [n_logits, batch].
        let sources = if layer == 1 {
            // Sources: 1, x
            Tensor::stack(
                &[
                    ones.clone(),
                    x.clone(),
                ],
                0,
            )?
        } else {
            // Sources: 1, x, child output
            let child_idx = 2 * node + if is_left { 0 } else { 1 };
            let f = &prev_outputs[child_idx];
            Tensor::stack(&[ones.clone(), x.clone(), f.clone()], 0)?
        };

        // Weighted sum: probs[logits] * sources[logits, batch] -> [batch]
        let probs = probs.unsqueeze(1)?.broadcast_as(sources.shape())?;
        let weighted = (&probs * sources)?;
        Ok(weighted.sum(0)?)
    }

    /// Convert the trained tree into a discrete `Expr` by snapping each
    /// input selector to its largest logit.
    pub fn snap(&self) -> Result<Expr> {
        let param_values: Vec<f64> = self.params.as_tensor().to_vec1()?;

        // We build expressions bottom-up.  prev_exprs holds the Expr for each
        // node in the previous layer.
        let mut prev_exprs: Vec<Expr> = Vec::new();

        for layer in 1..=self.depth {
            let n_nodes = 1usize << (self.depth - layer);
            let mut layer_exprs = Vec::with_capacity(n_nodes);

            for node in 0..n_nodes {
                let left = self.snap_input(layer, node, true, &param_values, &prev_exprs)?;
                let right = self.snap_input(layer, node, false, &param_values, &prev_exprs)?;
                layer_exprs.push(Expr::Eml(Box::new(left), Box::new(right)));
            }

            prev_exprs = layer_exprs;
        }

        Ok(prev_exprs.into_iter().next().expect("tree has a root"))
    }

    fn snap_input(
        &self,
        layer: usize,
        node: usize,
        is_left: bool,
        param_values: &[f64],
        prev_exprs: &[Expr],
    ) -> Result<Expr> {
        let (_, _, _, start, n_logits) = self
            .blocks
            .iter()
            .find(|(l, n, left, _, _)| *l == layer && *n == node && *left == is_left)
            .expect("parameter block exists");

        let mut best_idx = 0;
        let mut best_val = param_values[*start];
        for i in 1..*n_logits {
            let v = param_values[*start + i];
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }

        Ok(match (layer, best_idx) {
            (1, 0) => Expr::Const(1.0),
            (1, 1) => Expr::Var,
            (_, 0) => Expr::Const(1.0),
            (_, 1) => Expr::Var,
            (_, 2) => {
                let child_idx = 2 * node + if is_left { 0 } else { 1 };
                prev_exprs[child_idx].clone()
            }
            _ => unreachable!(),
        })
    }
}

/// Numerically stable EML operator over tensors.
fn eml_tensor(x: &Tensor, y: &Tensor) -> Result<Tensor> {
    // Clamp exp argument to avoid overflow.
    let x_clamped = x.clamp(-50.0, 50.0)?;
    let exp_x = x_clamped.exp()?;

    // Clamp |y| away from zero for ln.
    let y_abs = y.abs()?;
    let y_clamped = y_abs.clamp(1e-12, f64::INFINITY)?;
    let ln_y = y_clamped.log()?;

    Ok((exp_x - ln_y)?)
}
