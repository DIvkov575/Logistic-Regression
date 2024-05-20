mod util;

use anyhow::Result;
use polars::prelude::*;
use std::f64::consts::E;
use polars::export::num::{pow, Pow};
use util::{
    load_data,
    Weights,
    constants::{
        MAX_ITERATIONS as MI
    }
};




fn main() -> Result<()> {
    let df = load_data();
    let mut weights = Weights::opt(&df);

    for i in 0..MI {


    }





    Ok(())
}

fn logistic(x: f64, w: &Weights ) -> f64 {
    let m = w.m;
    let a = w.a;
    let b = w.b;

    m / (1f64 + a * E.powf((-x)*b))
}

fn mse(df: DataFrame, weights: &Weights) -> f64 {
    df.column("x")?
        .f64()?
        .apply(|x| Some((x? - logistic(x?, &weights)).pow(2)))
        .into_series()
        .sum()? / df.height()
}