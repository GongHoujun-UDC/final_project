use ndarray::{Array2, Axis};
use rand::Rng;

pub fn train_multitask_linear_regression(
    x: &Array2<f64>,
    y: &Array2<f64>,
    learning_rate: f64,
    epochs: usize,
    l2_penalty: f64,
    gradient_clip: f64,
    verbose: bool,
) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    let mut weights = Array2::from_shape_fn((x.shape()[1], y.shape()[1]), |_| rng.gen_range(-0.1..0.1));

    for epoch in 0..epochs {
        // Prediction: y_hat = X * weights
        let predictions = x.dot(&weights);

        // Error: y_hat - y
        let errors = &predictions - y;

        // Gradient Descent with L2 Regularization
        let gradients = x.t().dot(&errors) / x.shape()[0] as f64 + l2_penalty * &weights;

        // Gradient Clipping
        let clipped_gradients = gradients.mapv(|v| v.min(gradient_clip).max(-gradient_clip));
        weights = weights - clipped_gradients * learning_rate;

        if verbose && epoch % 100 == 0 {
            let mse = (&predictions - y)
                .mapv(|v| v * v)
                .mean_axis(Axis(0))
                .unwrap();
            println!("Epoch {}: MSE = {:?}", epoch, mse);
        }
    }

    weights
}

pub fn predict(x: &Array2<f64>, weights: &Array2<f64>) -> Array2<f64> {
    x.dot(weights)
}

pub fn evaluate(y_true: &Array2<f64>, y_pred: &Array2<f64>) {
    let mse = (y_true - y_pred)
        .mapv(|v| v * v)
        .mean_axis(Axis(0))
        .unwrap();
    let mae = (y_true - y_pred)
        .mapv(|v| v.abs())
        .mean_axis(Axis(0))
        .unwrap();

    println!("Evaluation Metrics:");
    println!("MSE: {:?}", mse);
    println!("MAE: {:?}", mae);
}
