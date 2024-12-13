pub fn parse_csv(data: &Vec<String>) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut features = Vec::new();
    let mut targets = Vec::new();
    for line in data.iter().skip(1) { // Skip the header row
        let values: Vec<f64> = line.split(';').map(|v| v.parse().unwrap()).collect();
        features.push(values[..values.len() - 1].to_vec());
        targets.push(values[values.len() - 1]);
    }
    (features, targets)
}

pub fn normalize(data: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut normalized = data.clone();
    let n_features = data[0].len();
    for j in 0..n_features {
        let column: Vec<f64> = data.iter().map(|row| row[j]).collect();
        let mean = column.iter().sum::<f64>() / column.len() as f64;
        let std = (column.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / column.len() as f64).sqrt();
        for i in 0..data.len() {
            normalized[i][j] = (data[i][j] - mean) / std;
        }
    }
    normalized
}

pub fn split_data(
    features: &Vec<Vec<f64>>, targets: &Vec<f64>, train_ratio: f64,
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
    let train_size = (features.len() as f64 * train_ratio).round() as usize;
    let (train_x, test_x) = features.split_at(train_size);
    let (train_y, test_y) = targets.split_at(train_size);
    (train_x.to_vec(), train_y.to_vec(), test_x.to_vec(), test_y.to_vec())
}

pub fn select_features(features: &Vec<Vec<f64>>, targets: &Vec<f64>, threshold: f64) -> Vec<Vec<f64>> {
    let mut selected_features = Vec::new();
    let n_features = features[0].len();
    for j in 0..n_features {
        let column: Vec<f64> = features.iter().map(|row| row[j]).collect();
        let mean_x = column.iter().sum::<f64>() / column.len() as f64;
        let mean_y = targets.iter().sum::<f64>() / targets.len() as f64;

        let numerator: f64 = column
            .iter()
            .zip(targets.iter())
            .map(|(&x, &y)| (x - mean_x) * (y - mean_y))
            .sum();

        let denominator: f64 = (column.iter().map(|&x| (x - mean_x).powi(2)).sum::<f64>()
            * targets.iter().map(|&y| (y - mean_y).powi(2)).sum::<f64>())
            .sqrt();

        let correlation = numerator / denominator;
        if correlation.abs() > threshold {
            selected_features.push(column);
        }
    }
    transpose(&selected_features)
}

pub fn generate_polynomial_features(features: &Vec<Vec<f64>>, degree: usize) -> Vec<Vec<f64>> {
    let mut poly_features = features.clone();
    for d in 2..=degree {
        for row in features {
            let mut new_row = Vec::new();
            for value in row {
                new_row.push(value.powi(d as i32));
            }
            poly_features.push(new_row);
        }
    }
    poly_features
}

fn transpose(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n_rows = matrix.len();
    let n_cols = matrix[0].len();
    let mut transposed = vec![vec![0.0; n_rows]; n_cols];
    for i in 0..n_rows {
        for j in 0..n_cols {
            transposed[j][i] = matrix[i][j];
        }
    }
    transposed
}
