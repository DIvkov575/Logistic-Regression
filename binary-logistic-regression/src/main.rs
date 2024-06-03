#![allow(non_snake_case)]

use std::fs::File;
use std::str::FromStr;

use ndarray::{Array1, Array2};
use polars::prelude::*;
use crate::logistic_regression::LR;
mod logistic_regression;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut data = CsvReader::new(File::open("data1.csv")?).finish()?;

    let mut X: Array2<f64> = DataFrame::new(data.columns([ "a", "b"])?.iter().map(|x| x.to_owned().to_owned().to_owned()).collect())?.to_ndarray::<Float64Type>(IndexOrder::C)?;
    let mut y: Array1<f64> = data.column("y").cloned()?.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?.t().iter().map(|x| x.to_owned()).collect();



    center(&mut X);

    let mut lr = LR::empty();
    lr.fit(X.clone(), y.clone());
    let y_pred= lr.predict(X);

    println!("{}", data);
    println!("{}", lr.weights);
    println!("{:?}", y_pred);
    println!("{:?}", y);

    Ok(())
}




fn center(X: &mut Array2<f64>) {
    for i in 0..X.shape()[1] {
        let mean = X.column(i.into()).mean().unwrap();
        X.column_mut(i.into()).iter_mut().for_each(|x| *x -= mean);
    }
}