mod util;

use anyhow::Result;
use polars::prelude::*;
use std::f64::consts::E;
use std::process::exit;
use polars::export::num::{pow, Pow};
use util::{
    load_data,
};

// https://en.wikipedia.org/wiki/Cross-entropy#Cross-entropy_loss_function_and_logistic_regression
// https://medium.com/@koushikkushal95/logistic-regression-from-scratch-dfb8527a4226
fn main() -> Result<()> {
    let mut df = load_data();
    let mut lr = LogisticRegression::new();

    lr.opt(&mut df);






    Ok(())
}


#[derive(Debug, Clone)]
pub struct LogisticRegression {
    pub m: f64, // population max
    pub a: f64, //
    pub b: f64,
    pub scores: Vec<f64>,
}

impl LogisticRegression {
    pub fn new() -> Self {
        Self { m: 5f64, a: 1f64, b: 1f64, scores: vec![f64::MAX] }
    }
    fn opt(&mut self, df: &mut DataFrame) {
        let mut sd: f64;
        let mut delta = -1_f64;

        self.scores.push(self.mse(&df));

        for i in 0..25  {

            self.m += delta;
            self.scores.push(self.mse(&df));
            sd = self.sd();

            if sd < 0f64 {}
            else if sd > 0f64 {
                delta *= -1_f64
            } else {
                unimplemented!()
            }
        }

    }
    fn sd(&self) ->f64 {
        self.scores[self.scores.len()-1] - self.scores[self.scores.len() - 2]
    }
    fn logistic(&self, x: f64) -> f64 {
        let m = self.m;
        let a = self.a;
        let b = self.b;

        m / (1f64 + a * E.powf((-x)*b))
    }
    fn _real_logistic(&self, x: f64) -> f64 {
        let m = 25f64;
        let a = 1f64;
        let b = 1f64;

        m / (1f64 + a * E.powf((-x)*b))
    }
    fn mse(&self, df: &DataFrame) -> f64 {
        let fx_agg: f64 = df.column("x").unwrap()
            .f64().unwrap()
            .apply(|x| Some(self.logistic(x?)))
            .sum().unwrap();
        let y_agg: f64 = df.column("y").unwrap()
            .sum().unwrap();

        // println!("y_agg {}", y_agg);
        // println!("fx_agg {}", fx_agg);
        (y_agg.pow(2) - 2f64*y_agg*fx_agg + fx_agg.pow(2)) / (df.height() as f64)

    }
}


