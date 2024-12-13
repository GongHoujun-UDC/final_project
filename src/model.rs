use rand::seq::SliceRandom;

pub struct LinearRegression {
    weights: Vec<f64>,
    bias: f64,
}

impl LinearRegression {
    pub fn new(n_features: usize) -> Self {
        Self {
            weights: vec![0.0; n_features],
            bias: 0.0,
        }
    }

    pub fn train_with_momentum(&mut self, x: &Vec<Vec<f64>>, y: &Vec<f64>, lr: f64, epochs: usize, momentum: f64) {
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
        x.iter().map(|row| {
            row.iter().zip(self.weights.iter()).map(|(&xi, &wj)| xi * wj).sum::<f64>() + self.bias
        }).collect()
    }
}

pub struct LogisticRegression {
    weights: Vec<f64>,
    bias: f64,
}

impl LogisticRegression {
    pub fn new(n_features: usize) -> Self {
        Self {
            weights: vec![0.0; n_features],
            bias: 0.0,
        }
    }
}

pub fn mean_squared_error(predictions: &Vec<f64>, targets: &Vec<f64>) -> f64 {
    predictions.iter().zip(targets.iter()).map(|(&p, &t)| (p - t).powi(2)).sum::<f64>() / predictions.len() as f64
}

pub fn accuracy(predictions: &Vec<f64>, targets: &Vec<f64>) -> f64 {
    let correct = predictions.iter().zip(targets.iter())
        .filter(|(&p, &t)| (p >= 0.5 && t == 1.0) || (p < 0.5 && t == 0.0)).count();
    correct as f64 / predictions.len() as f64
}

pub fn k_fold_cross_validation(features: &Vec<Vec<f64>>, targets: &Vec<f64>, k: usize) -> f64 {
    let mut combined: Vec<(Vec<f64>, f64)> = features.iter().cloned().zip(targets.iter().cloned()).collect();
    let mut rng = rand::thread_rng();
    combined.shuffle(&mut rng);

    let fold_size = features.len() / k;
    let mut mse_sum = 0.0;

    for i in 0..k {
        let test_start = i * fold_size;
        let test_end = test_start + fold_size;
        let test_set: Vec<_> = combined[test_start..test_end].to_vec();
        let train_set: Vec<_> = [&combined[..test_start], &combined[test_end..]].concat();

        let train_features: Vec<_> = train_set.iter().map(|(x, _)| x.clone()).collect();
        let train_targets: Vec<_> = train_set.iter().map(|(_, y)| *y).collect();
        let test_features: Vec<_> = test_set.iter().map(|(x, _)| x.clone()).collect();
        let test_targets: Vec<_> = test_set.iter().map(|(_, y)| *y).collect();

        let mut model = LinearRegression::new(train_features[0].len());
        model.train_with_momentum(&train_features, &train_targets, 0.01, 1000, 0.9);

        let predictions = model.predict(&test_features);
        mse_sum += mean_squared_error(&predictions, &test_targets);
    }

    mse_sum / k as f64
}
