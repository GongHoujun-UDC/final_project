use rand::seq::SliceRandom;

pub struct LinearRegression {
    weights: Vec<f64>,
    bias: f64,
}

pub struct LogisticRegression {
    weights: Vec<f64>,
    bias: f64,
}

// Linear Regression Implementation
impl LinearRegression {
    pub fn new(n_features: usize) -> Self {
        Self {
            weights: vec![0.0; n_features],
            bias: 0.0,
        }
    }

    pub fn train_with_momentum(
        &mut self,
        x: &Vec<Vec<f64>>,
        y: &Vec<f64>,
        lr: f64,
        epochs: usize,
        momentum: f64,
    ) {
        let mut velocity = vec![0.0; self.weights.len()];
        for _ in 0..epochs {
            let predictions = self.predict(x);
            let errors: Vec<f64> = predictions.iter().zip(y.iter()).map(|(&p, &t)| p - t).collect();

            for j in 0..self.weights.len() {
                let gradient = errors.iter().zip(x.iter()).map(|(&e, row)| e * row[j]).sum::<f64>() / x.len() as f64;
                velocity[j] = momentum * velocity[j] - lr * gradient;
                self.weights[j] += velocity[j];
            }
            self.bias -= lr * errors.iter().sum::<f64>() / x.len() as f64;
        }
    }

    pub fn predict(&self, x: &Vec<Vec<f64>>) -> Vec<f64> {
        x.iter()
            .map(|row| {
                row.iter().zip(self.weights.iter()).map(|(&xi, &wj)| xi * wj).sum::<f64>() + self.bias
            })
            .collect()
    }
}

// Logistic Regression Implementation
impl LogisticRegression {
    pub fn new(n_features: usize) -> Self {
        Self {
            weights: vec![0.0; n_features],
            bias: 0.0,
        }
    }

    pub fn train_with_scheduler(
        &mut self,
        x: &Vec<Vec<f64>>,
        y: &Vec<f64>,
        initial_lr: f64,
        epochs: usize,
        decay_rate: f64,
    ) {
        let mut lr = initial_lr;

        for epoch in 0..epochs {
            let predictions = self.predict(x);
            let errors: Vec<f64> = predictions.iter().zip(y.iter()).map(|(&p, &t)| p - t).collect();

            for j in 0..self.weights.len() {
                let gradient = errors.iter().zip(x.iter()).map(|(&e, row)| e * row[j]).sum::<f64>() / x.len() as f64;
                self.weights[j] -= lr * gradient; // Update weights
            }

            let bias_gradient = errors.iter().sum::<f64>() / x.len() as f64;
            self.bias -= lr * bias_gradient; // Update bias

            // Update learning rate
            lr *= decay_rate;

            // Debugging: Print loss, gradients, weights, and bias
            if epoch % 100 == 0 {
                let _loss = -y
                    .iter()
                    .zip(predictions.iter())
                    .map(|(&t, &p)| {
                        if p == 0.0 || p == 1.0 {
                            0.0 // Prevent `ln(0)` issues
                        } else {
                            t * p.ln() + (1.0 - t) * (1.0 - p).ln()
                        }
                    })
                    .sum::<f64>()
                    / y.len() as f64;
            }
        }
    }

    pub fn predict(&self, x: &Vec<Vec<f64>>) -> Vec<f64> {
        x.iter()
            .map(|row| {
                let linear = row
                    .iter()
                    .zip(self.weights.iter())
                    .map(|(&xi, &wj)| xi * wj)
                    .sum::<f64>()
                    + self.bias;

                let clamped = linear.clamp(-15.0, 15.0); // Prevents overflow in exp()
                1.0 / (1.0 + (-clamped).exp())
            })
            .collect()
    }
}

// Evaluation Metrics
pub fn mean_squared_error(predictions: &Vec<f64>, targets: &Vec<f64>) -> f64 {
    predictions.iter().zip(targets.iter()).map(|(&p, &t)| (p - t).powi(2)).sum::<f64>() / predictions.len() as f64
}

pub fn accuracy(predictions: &Vec<f64>, targets: &Vec<f64>) -> f64 {
    let correct = predictions.iter().zip(targets.iter())
        .filter(|(&p, &t)| (p >= 0.5 && t == 1.0) || (p < 0.5 && t == 0.0)).count();
    correct as f64 / predictions.len() as f64
}

// K-Fold Cross Validation
pub fn k_fold_cross_validation(features: &Vec<Vec<f64>>, targets: &Vec<f64>, k: usize) -> f64 {
    assert!(features.len() == targets.len(), "Features and targets must have the same length.");
    assert!(features.len() >= k, "Number of folds cannot exceed dataset size.");

    let mut combined: Vec<(Vec<f64>, f64)> = features.iter().cloned().zip(targets.iter().cloned()).collect();
    let mut rng = rand::thread_rng();
    combined.shuffle(&mut rng);

    let fold_size = features.len() / k;
    let mut metric_sum = 0.0;
    let mut valid_folds = 0; // Count valid folds to prevent dividing by zero

    for i in 0..k {
        let test_start = i * fold_size;
        let test_end = (test_start + fold_size).min(features.len()); // Avoid out-of-bound slicing

        let test_set: Vec<_> = combined[test_start..test_end].to_vec();
        let train_set: Vec<_> = [&combined[..test_start], &combined[test_end..]].concat();

        if test_set.is_empty() || train_set.is_empty() {
            continue; // Skip this fold if either set is empty
        }

        let train_features: Vec<_> = train_set.iter().map(|(x, _)| x.clone()).collect();
        let train_targets: Vec<_> = train_set.iter().map(|(_, y)| *y).collect();
        let test_features: Vec<_> = test_set.iter().map(|(x, _)| x.clone()).collect();
        let test_targets: Vec<_> = test_set.iter().map(|(_, y)| *y).collect();

        let mut model: Box<dyn Fn(&Vec<Vec<f64>>) -> Vec<f64>>;

        if targets.iter().all(|&t| t == 0.0 || t == 1.0) {
            // Classification
            let mut log_reg = LogisticRegression::new(train_features[0].len());
            log_reg.train_with_scheduler(&train_features, &train_targets, 0.1, 2000, 0.95);
            model = Box::new(move |x| log_reg.predict(x));
        } else {
            // Regression
            let mut lin_reg = LinearRegression::new(train_features[0].len());
            lin_reg.train_with_momentum(&train_features, &train_targets, 0.01, 1000, 0.9);
            model = Box::new(move |x| lin_reg.predict(x));
        }

        let predictions = model(&test_features);

        if targets.iter().all(|&t| t == 0.0 || t == 1.0) {
            // Classification metric: Accuracy
            let fold_accuracy = accuracy(&predictions, &test_targets);
            metric_sum += fold_accuracy;
        } else {
            // Regression metric: MSE
            let mse = mean_squared_error(&predictions, &test_targets);
            metric_sum += mse;
        }

        valid_folds += 1; // Count only valid folds
    }

    if valid_folds == 0 {
        panic!("No valid folds found for cross-validation.");
    }

    metric_sum / valid_folds as f64
}


// Tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_regression_training() {
        let features = vec![
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![3.0, 4.0],
            vec![4.0, 5.0],
        ];
        let targets = vec![3.0, 5.0, 7.0, 9.0];

        let mut lin_reg = LinearRegression::new(features[0].len());
        lin_reg.train_with_momentum(&features, &targets, 0.01, 1000, 0.9);

        let predictions = lin_reg.predict(&features);
        let mse = mean_squared_error(&predictions, &targets);

        assert!(mse < 0.01, "MSE should be less than 0.01, but got {}", mse);
    }

    #[test]
    fn test_logistic_regression_training() {
        let features = vec![
            vec![1.0, 1.0],
            vec![2.0, 1.0],
            vec![1.0, 2.0],
            vec![2.0, 2.0],
        ];
        let targets = vec![0.0, 0.0, 1.0, 1.0]; // Linearly separable data

        let mut log_reg = LogisticRegression::new(features[0].len());
        log_reg.train_with_scheduler(&features, &targets, 0.1, 5000, 0.99); // Increased epochs and decay rate

        let predictions = log_reg.predict(&features);
        let accuracy_score = accuracy(&predictions, &targets);

        assert!(
            accuracy_score > 0.9,
            "Accuracy should be greater than 0.9, but got {}",
            accuracy_score
        );
    }

    #[test]
    fn test_k_fold_cross_validation_regression() {
        let features = vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
            vec![6.0],
            vec![7.0],
            vec![8.0],
            vec![9.0],
            vec![10.0],
        ];
        let targets = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

        let avg_mse = k_fold_cross_validation(&features, &targets, 5);

        assert!(!avg_mse.is_nan(), "MSE should not be NaN");
        assert!(avg_mse < 0.1, "Average MSE should be less than 0.1, but got {}", avg_mse);
    }

    #[test]
    fn test_k_fold_cross_validation_classification() {
        let features = vec![
            vec![1.0, 2.0],
            vec![2.0, 1.0],
            vec![2.0, 3.0],
            vec![3.0, 2.0],
        ];
        let targets = vec![0.0, 0.0, 1.0, 1.0]; // Linearly separable
        let avg_mse = k_fold_cross_validation(&features, &targets, 4); // Change `5` to `4`

        assert!(avg_mse < 0.01, "Average MSE should be less than 0.01, but got {}", avg_mse);
    }
}
