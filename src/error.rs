use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("invalid tree depth {0}; depth must be at least 1")]
    InvalidDepth(usize),

    #[error("dataset is empty")]
    EmptyDataset,

    #[error("snapped expression does not reproduce training data (mse={mse})")]
    SnapFailed { mse: f64 },
}

pub type Result<T> = std::result::Result<T, Error>;
