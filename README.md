# eml-sr

Trainable EML (Exp-Minus-Log) trees for gradient-based symbolic regression in Rust.

This project implements the trainable-circuit architecture from
[Odrzywołek, *All elementary functions from a single operator*, arXiv:2603.21852](https://arxiv.org/abs/2603.21852):

- Every internal node is the single binary operator

  ```text
  eml(x, y) = exp(x) - ln(y)
  ```

- Each node input is a learnable softmax-weighted mixture of three sources:
  the constant `1`, the input variable `x`, and the output of a child sub-tree.
- Parameters are optimized with AdamW via [Candle](https://github.com/huggingface/candle).
- After training, selectors are snapped to one-hot values and the tree is
  pretty-printed as a symbolic expression.

> **Scope note:** This implementation uses real-valued arithmetic.  The paper
> uses complex intermediates to derive trigonometric functions; that extension
> is left for future work.  The current crate demonstrates the core
> gradient-based symbolic-regression pipeline on real elementary functions.

## Quick start

```bash
# Fit a tree to a CSV with columns x and y
cargo run --bin eml-sr-cli -- train data.csv --depth 2 --epochs 3000 --seeds 5

# Or run an example
cargo run --example recover_exp
cargo run --example recover_e_minus_log
cargo run --example recover_constant
```

## Library API

```rust
use candle_core::Device;
use eml_sr::Trainer;

let xs: Vec<f64> = vec![-1.0, 0.0, 1.0];
let ys: Vec<f64> = xs.iter().map(|x| x.exp()).collect();

let result = Trainer::new(2)?
    .lr(1e-2)
    .epochs(3000)
    .seeds(5)
    .device(Device::Cpu)
    .fit(&xs, &ys)?;

println!("{}", result.expr);
```

## Project layout

```text
src/
  lib.rs          public API
  tree.rs         EML tree, parameter layout, forward evaluation
  params.rs       parameter counts and initialization
  train.rs        AdamW training loop and hardening
  expr.rs         snapped expression AST, simplification, display
  bin/
    eml-sr-cli.rs CSV-driven command-line tool
```

## Tests

```bash
cargo test
```

The integration tests verify that the trainable circuit fits several shallow
EML targets with low MSE.  Exact symbolic recovery from random initialization
is probabilistic; the paper reports ~100% success at depth 2 and ~25% at
depths 3–4.

## License

MIT OR Apache-2.0
