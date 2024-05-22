use std::fs::File;
use anyhow::Result;
use polars::prelude::*;


pub mod constants {
    pub const MAX_ITERATIONS: usize = 100;
}

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


#[derive(Debug, Clone)]
pub struct Weights {
    pub m: f64, // population max
    pub a: f64, //
    pub b: f64,
}

impl Weights {
    pub fn empty() -> Self {
        Self { m: 10f64, a: 1f64, b: 1f64 }
    }
    pub fn opt(df: &DataFrame) -> Self {
        let mut weights = Self::empty();
        weights.m = df.column("y").unwrap().max().unwrap().unwrap();
        weights
    }
}

#[derive(Clone, Debug)]
pub struct Pair {
    pub weights: Weights,
    pub score: f64,
}

pub struct Stack {
    pub data: [Pair; 3]
}
impl Stack {
    pub fn empty() -> Self {
        Self {
            data: [
                Pair {
                    weights: Weights::empty(),
                    score: f64::MAX,
                },
                Pair {
                    weights: Weights::empty(),
                    score: f64::MAX,
                },
                Pair {
                    weights: Weights::empty(),
                    score: f64::MAX,
                },
            ],
        }
    }
    pub fn change1(&self) -> f64 { self.data[1].score - self.data[0].score }
    pub fn change2(&self) -> f64 { self.data[2].score - self.data[1].score }
    pub fn push(&mut self, weights: Weights, score: f64) {
        self.data[0] = self.data[1].clone();
        self.data[1] = self.data[2].clone();
        self.data[2] = Pair { weights, score };
    }
}