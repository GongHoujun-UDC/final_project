mod data;
mod model;

use crate::data::{parse_csv, normalize, split_data, select_features, generate_polynomial_features};
use crate::model::{LinearRegression, LogisticRegression, mean_squared_error, accuracy, k_fold_cross_validation};
use std::fs::File;
use std::io::{BufReader, BufRead};

fn main() {
    // Load dataset from a CSV file
    let file_path = "winequality-white.csv"; // Update this path based on your CSV location
    let dataset = read_csv(file_path).expect("Failed to read the CSV file!");

    // Parse dataset into features and targets
    let (features, targets) = parse_csv(&dataset);

    // Feature Engineering
    let features = generate_polynomial_features(&features, 2); // Add polynomial features up to degree 2
    let selected_features = select_features(&features, &targets, 0.1); // Select features with correlation > 0.1

    // Normalize features
    let normalized_features = normalize(&selected_features);

    // Split into training and testing
    let (train_x, train_y, test_x, test_y) = split_data(&normalized_features, &targets, 0.8);

    // Linear Regression with momentum
    let mut lin_reg = LinearRegression::new(train_x[0].len());
    lin_reg.train_with_momentum(&train_x, &train_y, 0.01, 1000, 0.9);
    let predictions = lin_reg.predict(&test_x);
    println!("Linear Regression MSE: {}", mean_squared_error(&predictions, &test_y));

    // Logistic Regression with learning rate scheduler
    let binary_targets: Vec<f64> = targets.iter().map(|&y| if y > 5.0 { 1.0 } else { 0.0 }).collect();
    let (train_x_bin, train_y_bin, test_x_bin, test_y_bin) = split_data(&normalized_features, &binary_targets, 0.8);
    let mut log_reg = LogisticRegression::new(train_x_bin[0].len());
    log_reg.train_with_scheduler(&train_x_bin, &train_y_bin, 0.01, 1000, 0.95);
    let predictions_bin = log_reg.predict(&test_x_bin);
    println!("Logistic Regression Accuracy: {}", accuracy(&predictions_bin, &test_y_bin));

    // Run K-Fold Cross Validation
    println!("Running K-Fold Cross Validation...");
    let cv_mse = k_fold_cross_validation(&normalized_features, &targets, 5);
    println!("Cross-Validation Mean MSE: {}", cv_mse);
}

/// Reads the CSV file and returns its content as a vector of strings
fn read_csv(file_path: &str) -> Result<Vec<String>, std::io::Error> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let mut lines = Vec::new();
    for line in reader.lines() {
        lines.push(line?);
    }
    Ok(lines)
}
