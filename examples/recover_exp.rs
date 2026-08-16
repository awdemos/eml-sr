use candle_core::Device;
use eml_sr::Trainer;

fn main() -> anyhow::Result<()> {
    // Generate y = e^x on [-1, 1].
    let n = 50;
    let xs: Vec<f64> = (0..n).map(|i| -1.0 + 2.0 * (i as f64) / ((n - 1) as f64)).collect();
    let ys: Vec<f64> = xs.iter().map(|x| x.exp()).collect();

    let result = Trainer::new(2)?
        .lr(1e-2)
        .epochs(3000)
        .seeds(3)
        .device(Device::Cpu)
        .fit(&xs, &ys)?;

    println!("epochs:       {}", result.epochs);
    println!("train mse:    {:.6e}", result.train_mse);
    println!("snapped mse:  {:.6e}", result.snapped_mse);
    println!("expression:   {}", result.expr);

    Ok(())
}
