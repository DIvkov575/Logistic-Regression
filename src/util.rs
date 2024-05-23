use std::fs::File;
use anyhow::Result;
use polars::prelude::*;



pub fn load_data() -> DataFrame {
    // let file = File::open("inputs.csv").expect("could not open file");
    let file = File::open("25:(1+e^-x).csv").expect("could not open file");
    CsvReader::new(file)
        .with_dtypes(Some(Arc::new(Schema::from_iter(vec![
            Field::new("x", DataType::Float64),
            Field::new("y", DataType::Float64)
        ]))))
        .has_header(true)
        .finish()
        .unwrap()

}


