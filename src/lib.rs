//! Trainable EML (Exp-Minus-Log) trees for gradient-based symbolic regression.
//!
//! Every internal node in the tree applies the single binary operator
//!
//! ```text
//! eml(x, y) = exp(x) - ln(y)
//! ```
//!
//! over real-valued tensors.  Each node input is a learnable convex
//! combination of three sources: the constant `1`, the input variable `x`,
//! and the output of a child sub-tree.  After training the continuous
//! weights are snapped to one-hot selectors, yielding a discrete symbolic
//! expression.

pub mod error;
pub mod expr;
pub mod params;
pub mod train;
pub mod tree;

pub use error::{Error, Result};
pub use expr::Expr;
pub use train::{Trainer, TrainingResult};
pub use tree::EmlTree;
