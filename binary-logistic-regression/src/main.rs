#![allow(non_snake_case)]
mod logistic_regression;

use std::fs::File;
use std::path::{Path, PathBuf};
use ndarray::{array, Array1, Array2, ArrayBase, Ix2, Shape};
use logistic_regression::*;
use polars::prelude::*;


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let test = CsvReader::new(File::open("test2.csv")?).finish()?;
    let train = CsvReader::new(File::open("train2.csv")?).finish()?;

    let X_test = test.to_ndarray::<Float64Type>(IndexOrder::C)?;
    let X_train = train.drop("Survived")?.to_ndarray::<Float64Type>(IndexOrder::C)?;
    let y_train: Array1<f64> = train.column("Survived").cloned()?.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?.t().iter().map(|x| x.to_owned()).collect();

    let mut lr = LR::empty();
    lr.fit(X_train, y_train);
    let y_pred= lr.predict(X_test);

    // let mut results = Series::from_iter(y_pred.into_iter());
    let mut results = DataFrame::from
    let mut writer = CsvWriter::new(
        File::options().write(true).truncate(true).open(Path::new("out.csv"))?
    );
    writer.finish(&mut results)?;





    Ok(())
}
