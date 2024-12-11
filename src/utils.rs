use ndarray::{Array2};
use ndarray_stats::SummaryStatisticsExt; // Import for statistics

pub fn normalize_features(features: &mut Array2<f64>) {
    for mut column in features.columns_mut() {
        // Compute mean
        let mean = column.mean().unwrap_or_else(|| panic!("Cannot compute mean for an empty column!"));
        let std = column.std(0.0);

        // Skip normalization if standard deviation is zero
        if std == 0.0 {
            eprintln!("Warning: Column has zero variance. Skipping normalization for this column.");
            continue;
        }

        // Normalize column values
        column.mapv_inplace(|x| (x - mean) / std);
    }
}


pub fn one_hot_encode(features: &mut Vec<Vec<f64>>, column_index: usize, unique_values: &[f64]) {
    for row in features.iter_mut() {
        let value = row[column_index];
        row.extend(unique_values.iter().map(|&v| if v == value { 1.0 } else { 0.0 }));
    }

    for row in features.iter_mut() {
        row.remove(column_index); // Remove original column
    }
}
