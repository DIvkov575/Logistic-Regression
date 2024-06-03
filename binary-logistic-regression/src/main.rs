#![allow(non_snake_case)]

use std::fs::File;
use std::path::Component::ParentDir;
use std::path::Path;
use std::process::exit;
use ndarray::{Array1, Array2};
use polars::export::arrow::legacy::kernels::take_agg::take_agg_no_null_primitive_iter_unchecked;

use polars::prelude::*;
use polars::prelude::FillNullStrategy::Mean;
use polars::prelude::Label::DataPoint;
use crate::logistic_regression::LR;

mod logistic_regression;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // let test_raw = CsvReader::new(File::open("test.csv")?).finish()?;
    // let train_raw = CsvReader::new(File::open("train.csv")?).finish()?;
    //
    //
    // let mut test = DataFrame::new(
    //     test_raw.columns(vec!["Pclass", "Age",])?.iter().map(|x| x.to_owned().to_owned()).collect()
    // )?;
    // test.with_column(
    //     test_raw.column("Sex")?
    //         .str()?
    //         .iter()
    //         .map(|x| if x.unwrap().to_lowercase() == "female" {0i64} else {1i64})
    //         .collect::<Series>()
    //         .rename("Sex")
    //         .to_owned()
    // )?;
    // test = test.fill_null(Mean)?;
    //
    // let mut train = DataFrame::new(
    //     train_raw.columns(vec!["Pclass", "Age", "Survived"])?.iter().map(|x| x.to_owned().to_owned()).collect()
    // )?;
    // train.with_column(
    //     train_raw.column("Sex")?
    //         .str()?
    //         .iter()
    //         .map(|x| if x.unwrap().to_lowercase() == "female" {0i64} else {1i64})
    //         .collect::<Series>()
    //         .rename("Sex")
    //         .to_owned()
    // )?;
    // train = train.drop_nulls::<&str>(None)?;
    // println!("{:?}", train);
    //
    // let output_labels= test_raw.column("PassengerId")?;
    // let y_train: Array1<f64> = train.column("Survived").cloned()?.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?.t().iter().map(|x| x.to_owned()).collect();
    // train = train.drop("Survived")?;
    // let X_test = test.to_ndarray::<Float64Type>(IndexOrder::C)?;
    // let X_train = train.to_ndarray::<Float64Type>(IndexOrder::C)?;
    //
    // println!("{:?}", y_train);


    let data = CsvReader::new(File::open("data1.csv")?).finish()?;
    let mut X: Array2<f64> = DataFrame::new(data.columns([ "a"])?.iter().map(|x| x.to_owned().to_owned().to_owned()).collect())?.iter().to_ndarray::<Float64Type>(IndexOrder::C)?;
    // let mut X: Array2<f64> = DataFrame::new(data.columns([ "a"])?.iter().map(|x| x.to_owned().to_owned().to_owned()).collect())?.to_ndarray::<Float64Type>(IndexOrder::C)?;
    // let mut y: Array1<f64> = data.column("y").cloned()?.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?.t().iter().map(|x| x.to_owned()).collect();

    // center(&mut X);
    // let mut lr = LR::empty();
    // lr.fit(X.clone(), y.clone());
    // let y_pred= lr.predict(X);
    //
    // println!("{}", data);
    // println!("{}", lr.weights);
    // println!("{:?}", y_pred);
    // println!("{:?}", y);

    // let mut results = DataFrame::new(vec![output_labels.to_owned(), Series::from_iter(y_pred)])?;
    // let mut writer = CsvWriter::new(File::options().write(true).truncate(true).create(true).open(Path::new("out.csv"))?);
    // writer.finish(&mut results)?;

    Ok(())
}




fn center(X: &mut Array2<f64>) {
    for i in 0..X.shape()[1] {
        let mean = X.column(i.into()).mean().unwrap();
        X.column_mut(i.into()).iter_mut().for_each(|x| *x -= mean);
    }
}