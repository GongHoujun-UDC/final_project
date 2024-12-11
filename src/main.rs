mod utils;
mod model;
mod data;

use data::load_data;
use model::{train_multitask_linear_regression, predict, evaluate};

fn main() {
    let file_path = "games.csv"; // Replace with the actual file path
    let (features, targets) = load_data(file_path);

    // Split data into training and testing sets (80/20 split)
    let train_size = (0.8 * features.shape()[0] as f64) as usize;
    let (x_train, x_test) = features.view().split_at(ndarray::Axis(0), train_size);
    let (y_train, y_test) = targets.view().split_at(ndarray::Axis(0), train_size);

    // Train the model
    let learning_rate = 0.01;
    let epochs = 2000;
    let l2_penalty = 0.1;
    let gradient_clip = 1.0;
    let verbose = true;
    let weights = train_multitask_linear_regression(
        &x_train.to_owned(),
        &y_train.to_owned(),
        learning_rate,
        epochs,
        l2_penalty,
        gradient_clip,
        verbose,
    );

    // Make predictions
    let predictions = predict(&x_test.to_owned(), &weights);

    // Evaluate the model
    evaluate(&y_test.to_owned(), &predictions);

    // Print model weights
    println!("Final model weights: {:?}", weights);
}
