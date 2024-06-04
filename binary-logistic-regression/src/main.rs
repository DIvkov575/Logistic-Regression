#![allow(non_snake_case)]

use std::fs::File;
use std::str::FromStr;

use ndarray::{Array1, Array2};
use polars::prelude::*;
use polars::prelude::FillNullStrategy::Mean;
use crate::logistic_regression::LR;
mod logistic_regression;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let test_raw = CsvReader::new(File::open("test.csv")?).finish()?;
    let train_raw = CsvReader::new(File::open("train.csv")?).finish()?;

    let mut test = DataFrame::new(test_raw.columns(vec!["Parch", "Pclass", "Age",])?.iter().map(|x| x.to_owned().to_owned()).collect())?;
    test.with_column(
        test_raw.column("Sex")?
            .str()?
            .iter()
            .map(|x| if x.unwrap().to_lowercase() == "female" {0i64} else {1i64})
            .collect::<Series>()
            .rename("Sex")
            .to_owned()
    )?;
    test = test.fill_null(Mean)?;

    let mut train = DataFrame::new(train_raw.columns(vec!["Parch", "Pclass", "Age", "Survived"])?.iter().map(|x| x.to_owned().to_owned()).collect())?;
    train.with_column(
        train_raw.column("Sex")?
            .str()?
            .iter()
            .map(|x| if x.unwrap().to_lowercase() == "female" {0i64} else {1i64})
            .collect::<Series>()
            .rename("Sex")
            .to_owned()
    )?;
    train = train.drop_nulls::<&str>(None)?;

    let output_labels= test_raw.column("PassengerId")?;
    let y_train: Array1<f64> = train.column("Survived").cloned()?.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?.t().iter().map(|x| x.to_owned()).collect();
    train = train.drop("Survived")?;
    let mut X_test = test.to_ndarray::<Float64Type>(IndexOrder::C)?;
    let mut X_train = train.to_ndarray::<Float64Type>(IndexOrder::C)?;
    center(&mut X_test);
    center(&mut X_train);



    let mut logistic_regession = LR::empty();
    logistic_regession.fit(X_train, y_train);
    let y_pred= logistic_regession.predict(X_test);


    let mut results = DataFrame::new(vec![output_labels.to_owned(), Series::from_iter(y_pred)])?;
    let mut writer = CsvWriter::new(File::options().write(true).truncate(true).create(true).open(std::path::Path::new("out.csv"))?);
    writer.finish(&mut results)?;

    Ok(())
}




fn center(X: &mut Array2<f64>) {
    for i in 0..X.shape()[1] {
        let mean = X.column(i.into()).mean().unwrap();
        X.column_mut(i.into()).iter_mut().for_each(|x| *x -= mean);
    }
}