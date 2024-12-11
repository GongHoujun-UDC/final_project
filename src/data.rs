use csv::ReaderBuilder;
use ndarray::{Array2};
use crate::utils::{normalize_features, one_hot_encode};

pub fn load_data(file_path: &str) -> (Array2<f64>, Array2<f64>) {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(file_path)
        .expect("Failed to open CSV file");

    let mut features = Vec::new();
    let mut targets = Vec::new();

    let mut unique_victory_status = Vec::new();

    for result in reader.records() {
        let record = result.expect("Failed to read record");

        let turns: f64 = record[4].parse().unwrap();
        let opening_ply: f64 = record[15].parse().unwrap();
        let white_rating: f64 = record[9].parse().unwrap();
        let black_rating: f64 = record[11].parse().unwrap();
        let victory_status = match &record[5] {
            "mate" => 1.0,
            "resign" => 2.0,
            "draw" => 3.0,
            _ => 0.0,
        };

        if !unique_victory_status.contains(&victory_status) {
            unique_victory_status.push(victory_status);
        }

        features.push(vec![1.0, turns, opening_ply, victory_status]);
        targets.push(vec![white_rating, black_rating]);
    }

    one_hot_encode(&mut features, 3, &unique_victory_status);

    let mut features = Array2::from_shape_vec((features.len(), features[0].len()), features.concat())
        .expect("Failed to create features array");
    normalize_features(&mut features);

    let targets = Array2::from_shape_vec((targets.len(), 2), targets.concat())
        .expect("Failed to create targets array");

    (features, targets)
}