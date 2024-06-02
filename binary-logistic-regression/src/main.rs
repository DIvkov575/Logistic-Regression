#![allow(non_snake_case)]

use std::fs::File;
use std::path::Path;
use ndarray::Array1;

use polars::prelude::*;
use polars::prelude::FillNullStrategy::Mean;
use polars::prelude::Label::DataPoint;
use crate::logistic_regression::LR;

mod logistic_regression;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let test_raw = CsvReader::new(File::open("test.csv")?).finish()?;
    let train_raw = CsvReader::new(File::open("train.csv")?).finish()?;


    let mut test = DataFrame::new(
        test_raw.columns(vec!["Pclass", "Age", "SibSp"])?.iter().map(|x| x.to_owned().to_owned()).collect()
    )?;
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

    let mut train = DataFrame::new(
        train_raw.columns(vec!["Pclass", "Age", "SibSp", "Survived"])?.iter().map(|x| x.to_owned().to_owned()).collect()
    )?;
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
    let X_test = test.to_ndarray::<Float64Type>(IndexOrder::C)?;
    let X_train = train.to_ndarray::<Float64Type>(IndexOrder::C)?;

    // println!("{:?}", y_train);


    let mut lr = LR::empty();
    lr.fit(X_train, y_train);
    let y_pred= lr.predict(X_test);

    let mut results = DataFrame::new(vec![output_labels.to_owned(), Series::from_iter(y_pred)])?;
    let mut writer = CsvWriter::new(File::options().write(true).truncate(true).create(true).open(Path::new("out.csv"))?);
    writer.finish(&mut results)?;

    Ok(())
}