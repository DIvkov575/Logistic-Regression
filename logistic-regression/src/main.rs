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
    pub m: f64,
    pub a: f64,
    pub b: f64,
    pub scores_m: Vec<f64>,
    pub scores_a: Vec<f64>,
    pub scores_b: Vec<f64>,
}

impl LogisticRegression {
    pub fn new() -> Self {
        Self {
        m: 50f64,
        a: 50f64,
        b: 50f64,
        scores_m: vec![f64::MAX],
        scores_a: vec![f64::MAX],
        scores_b: vec![f64::MAX],
        }
    }
    fn opt(&mut self, df: &mut DataFrame) {
        let mut sd: f64;
        let mut delta_m = 0.5_f64;
        let mut delta_a = -0.5_f64;
        let mut delta_b = -0.5_f64;
        self.scores_m.push(self.mse(&df));
        for i in 0..500  {
            // m
            self.m += delta_m;
            self.scores_m.push(self.mse(&df));
            sd = self.scores_m[self.scores_m.len()-1] - self.scores_m[self.scores_m.len() - 2];

            if sd < 0f64 {}
            else if sd > 0f64 { delta_m *= -1_f64 }
            else { println!("exit m! m: {}; a: {}; b: {}", self.m, self.a, self.b);exit(0); }


            // a
            self.a += delta_a;
            self.scores_a.push(self.mse(&df));
            sd = self.scores_a[self.scores_a.len()-1] - self.scores_a[self.scores_a.len() - 2];

            if sd < 0f64 {}
            else if sd > 0f64 { delta_a *= -1_f64 }
            else { println!("exit a! m: {}; a: {}; b: {}", self.m, self.a, self.b);exit(0); }


            // b
            self.b += delta_b;
            self.scores_b.push(self.mse(&df));
            sd = self.scores_b[self.scores_b.len()-1] - self.scores_b[self.scores_b.len() - 2];

            if sd < 0f64 {}
            else if sd > 0f64 { delta_b *= -1_f64 }
            else { println!("exit b! m: {}; a: {}; b: {}", self.m, self.a, self.b);exit(0); }

            println!("scores: {} {} {}", self.scores_m[self.scores_m.len()-1], self.scores_a[self.scores_a.len()-1], self.scores_b[self.scores_b.len()-1]);
            println!("m: {}; a: {}; b: {}", self.m, self.a, self.b);
        }


    }
    fn logistic(&self, x: f64) -> f64 {
        let m = self.m;
        let a = self.a;
        let b = self.b;

        m / (1f64 + a * E.powf((-x)*b))
    }
    fn mse(&self, df: &DataFrame) -> f64 {
        let fx_agg: f64 = df.column("x").unwrap()
            .f64().unwrap()
            .apply(|x| Some(self.logistic(x?)))
            .sum().unwrap();
        let y_agg: f64 = df.column("y").unwrap()
            .sum().unwrap();

        (y_agg.pow(2) - 2f64*y_agg*fx_agg + fx_agg.pow(2)) / (df.height() as f64)
    }
}


