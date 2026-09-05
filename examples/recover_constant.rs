use candle_core::Device;
use eml_sr::Trainer;

fn main() -> anyhow::Result<()> {
    // Generate y = e (the constant) for varied x.
    // This is exactly eml(1, 1), a depth-1 EML tree.
    let n = 50;
    let xs: Vec<f64> = (0..n)
        .map(|i| -1.0 + 2.0 * (i as f64) / ((n - 1) as f64))
        .collect();
    let ys: Vec<f64> = xs.iter().map(|_| std::f64::consts::E).collect();

    let result = Trainer::new(1)?
        .lr(1e-2)
        .epochs(2000)
        .seeds(3)
        .device(Device::Cpu)
        .fit(&xs, &ys)?;

    println!("epochs:       {}", result.epochs);
    println!("train mse:    {:.6e}", result.train_mse);
    println!("snapped mse:  {:.6e}", result.snapped_mse);
    println!("expression:   {}", result.expr);

    Ok(())
}
