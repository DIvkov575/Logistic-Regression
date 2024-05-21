mod util;

use anyhow::Result;
use polars::prelude::*;
use std::f64::consts::E;
use polars::export::num::{pow, Pow};
use util::{
    load_data,
    Weights,
    Stack,
    constants::{
        MAX_ITERATIONS as MI
    }
};

// https://en.wikipedia.org/wiki/Cross-entropy#Cross-entropy_loss_function_and_logistic_regression
// https://medium.com/@koushikkushal95/logistic-regression-from-scratch-dfb8527a4226
fn main() -> Result<()> {
    let df = load_data();
    let mut weights = Weights::opt(&df);
    let mut stack = Stack::empty();

    stack.push(weights.clone(), mse(&df, &weights));
    weights.m += 0.1;
    stack.push(weights.clone(), mse(&df, &weights));

    if stack.change2() == 0f64 {
        println!("erm");
    } else if stack.change2() < 0f64 {
        weights.m += 0.1
    } else if stack.change2() > 0f64 {
        weights.m -= 0.1
    }

    for i in 0..MI {

        stack.push(weights.clone(), mse(&df, &weights));

        if stack.change2() == 0f64 {
            println!("erm");
        } else if stack.change2() < 0f64 {
            weights.m += 0.1
        } else if stack.change2() > 0f64 {
            weights.m -= 0.1
        }



    println!("{:?}", weights);
    }






    Ok(())
}

fn logistic(x: f64, w: &Weights ) -> f64 {
    let m = w.m;
    let a = w.a;
    let b = w.b;

    m / (1f64 + a * E.powf((-x)*b))
}

fn mse(df: &DataFrame, weights: &Weights) -> f64 {
    df.column("x").unwrap()
        .f64().unwrap()
        .apply(|x| Some((x? - logistic(x?, &weights)).pow(2)))
        .into_series()
        .sum::<f64>().unwrap() / df.height() as f64
}