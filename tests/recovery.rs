use candle_core::Device;
use eml_sr::Trainer;

fn make_data<F>(start: f64, end: f64, n: usize, f: F) -> (Vec<f64>, Vec<f64>)
where
    F: Fn(f64) -> f64,
{
    let xs: Vec<f64> = (0..n)
        .map(|i| start + (end - start) * (i as f64) / ((n - 1).max(1) as f64))
        .collect();
    let ys = xs.iter().copied().map(f).collect();
    (xs, ys)
}

/// The EML master formula is a complete search space, but exact symbolic
/// recovery from a random initialization is probabilistic (the paper reports
/// ~100% success at depth 2, ~25% at depths 3–4).  These integration tests
/// therefore verify that the trainable circuit *fits* representative shallow
/// EML targets with low MSE, which is the prerequisite for snapping to an
/// exact formula.

#[test]
fn fit_exponential() {
    // eml(x, 1) = exp(x)
    let (xs, ys) = make_data(-1.0, 1.0, 30, |x| x.exp());
    let result = Trainer::new(2)
        .unwrap()
        .lr(1e-2)
        .epochs(3000)
        .seeds(3)
        .device(Device::Cpu)
        .fit(&xs, &ys)
        .expect("training should succeed");

    assert!(result.train_mse < 5e-2, "train mse = {}", result.train_mse);
}

#[test]
fn fit_e_minus_log() {
    // eml(1, x) = e - ln(x), a depth-1 EML identity.
    let (xs, ys) = make_data(0.5, 2.0, 30, |x| std::f64::consts::E - x.ln());
    let result = Trainer::new(1)
        .unwrap()
        .lr(1e-2)
        .epochs(3000)
        .seeds(5)
        .device(Device::Cpu)
        .fit(&xs, &ys)
        .expect("training should succeed");

    assert!(result.train_mse < 1e-2, "train mse = {}", result.train_mse);
}

#[test]
fn fit_constant() {
    // eml(1, 1) = e, the simplest constant EML expression.
    let (xs, ys) = make_data(-1.0, 1.0, 30, |_| std::f64::consts::E);
    let result = Trainer::new(1)
        .unwrap()
        .lr(1e-2)
        .epochs(2000)
        .seeds(3)
        .device(Device::Cpu)
        .fit(&xs, &ys)
        .expect("training should succeed");

    assert!(result.train_mse < 1e-2, "train mse = {}", result.train_mse);
}
