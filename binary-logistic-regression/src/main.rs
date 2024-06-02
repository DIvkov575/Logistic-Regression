#![allow(non_snake_case)]
mod logistic_regression;

use std::fs::File;
use std::path::PathBuf;
use ndarray::{array, Array1, Array2, Ix2, Shape};
use logistic_regression::*;
use csv;



fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (X_train, y_train) = load("train2.csv");
    let (X_test, _) = load("test2.csv");

    let mut lr = LR::empty();
    lr.fit(X_train, y_train.unwrap());
    let results = lr.predict(X_test);

    println!("{:?}", results);




    Ok(())
}



fn load(path: &str) -> (Array2<f64>, Option<Array1<f64>>) {
    let path = std::path::Path::new(path);
    let mut rdr = csv::ReaderBuilder::new()
        .from_reader(File::options().read(true).open(path).unwrap());

    let mut x_vec = Vec::new();
    let mut y_vec: Vec<f64> = Vec::new();
    for (index, record_result) in rdr.records().enumerate() {
        let record: Vec<f64> = record_result.unwrap().iter().map(|x| x.replace(" ", "").parse::<f64>().unwrap()).collect();
        x_vec.extend(record[0..record.len()-1].to_vec());
        y_vec.push(record[record.len()-1]);
    }

    let num_records = y_vec.len();
    let num_features = x_vec.len() / num_records;

    let X: Array2<f64> = Array1::from_iter(x_vec).into_shape((num_records, num_features)).unwrap();
    let y: Array1<f64> = Array1::from_iter(y_vec);

    (X, Some(y))
}