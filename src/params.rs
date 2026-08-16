//! Parameter layout and initialization for an EML tree.

use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Total number of trainable logits for a full binary EML tree of the given
/// depth.
///
/// A depth-`d` tree has `d` layers of `eml` nodes.  Layer 1 (the leaves)
/// inputs select only from `{1, x}` (2 logits each).  Layers `2..=d`
/// select from `{1, x, child_output}` (3 logits each).  This matches the
/// paper's count `5·2^d − 6`.
pub fn n_params(depth: usize) -> usize {
    if depth == 0 {
        return 0;
    }
    (1..=depth)
        .map(|layer| {
            let nodes = 1usize << (depth - layer);
            let logits_per_input = if layer == 1 { 2 } else { 3 };
            nodes * 2 * logits_per_input
        })
        .sum()
}

/// Initialize logits so that the softmax is roughly uniform with a small
/// amount of randomness.  Uniform logits make training start from a
/// balanced mixture of `1`, `x`, and child outputs.
pub fn init_uniform<R: Rng>(depth: usize, rng: &mut R) -> Vec<f64> {
    let n = n_params(depth);
    let dist = Normal::new(0.0, 0.1).expect("valid normal distribution");
    dist.sample_iter(rng).take(n).collect()
}

/// Initialize logits with a small bias toward using `x` at the leaves and
/// child outputs at higher layers.  This often helps gradient descent find
/// simple formulas.
pub fn init_biased<R: Rng>(depth: usize, rng: &mut R) -> Vec<f64> {
    let mut params = Vec::new();
    let dist = Normal::new(0.0, 0.05).expect("valid normal distribution");

    for layer in 1..=depth {
        let nodes = 1usize << (depth - layer);
        for _ in 0..nodes {
            for _ in 0..2 {
                // For each input: bias the source we hope to use.
                if layer == 1 {
                    // [1, x] -> bias x
                    params.push(dist.sample(rng));
                    params.push(0.5 + dist.sample(rng));
                } else {
                    // [1, x, f] -> bias f
                    params.push(dist.sample(rng));
                    params.push(dist.sample(rng));
                    params.push(0.5 + dist.sample(rng));
                }
            }
        }
    }
    params
}

/// Return the parameter block for every tree input.
///
/// Each entry is `(layer, node, is_left, start_index, n_logits)` where
/// `n_logits` is `2` for layer 1 and `3` otherwise.
pub fn input_blocks(depth: usize) -> Vec<(usize, usize, bool, usize, usize)> {
    let mut blocks = Vec::new();
    let mut start = 0usize;
    for layer in 1..=depth {
        let nodes = 1usize << (depth - layer);
        let logits = if layer == 1 { 2 } else { 3 };
        for node in 0..nodes {
            blocks.push((layer, node, true, start, logits));
            blocks.push((layer, node, false, start + logits, logits));
            start += 2 * logits;
        }
    }
    blocks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn param_counts_match_paper() {
        assert_eq!(n_params(1), 4);
        assert_eq!(n_params(2), 14);
        assert_eq!(n_params(3), 34);
        assert_eq!(n_params(4), 74);
    }
}
