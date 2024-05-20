use std::fs::File;
use anyhow::Result;
use polars::prelude::*;


pub mod constants {
    pub const MAX_ITERATIONS: usize = 10000;
}

pub fn load_data() -> DataFrame {
    let file = File::open("inputs.csv").expect("could not open file");
    CsvReader::new(file)
        .with_dtypes(Some(Arc::new(Schema::from_iter(vec![
            Field::new("x", DataType::Float64),
            Field::new("y", DataType::Float64)
        ]))))
        .has_header(true)
        .finish()?

}


#[derive(Clone)]
pub struct Weights {
    pub m: f64, // population max
    pub a: f64, //
    pub b: f64,
}

impl Weights {
    pub fn empty() -> Self {
        Self { m: 25f64, a: 1f64, b: 1f64 }
    }
    pub fn opt(df: &DataFrame) -> Self {
        let mut weights = Self::empty();
        weights.m = df.column("y")?.max()?.unwrap();
        weights
    }
}