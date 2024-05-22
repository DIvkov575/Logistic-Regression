mod util;

use anyhow::Result;
use polars::prelude::*;
use std::f64::consts::E;
use std::process::exit;
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
    let mut df = load_data();
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

    let mut delta= 0.2;
    for i in 0..1000 {

        stack.push(weights.clone(), mse(&df, &weights));

        if stack.change2() == 0f64 {
            println!("erm");
        } else if stack.change2() < 0f64 {

        }else if stack.change2() > 0f64 {
            delta *= -1f64
        }
        weights.m += delta;

        println!("delta: {:?}", delta);
        println!("change: {:?}", stack.change2());
        println!("score: {:?}", stack.data[2]);

        if stack.change1() == -stack.change2() {
            println!("premature stop at {i}");
            let df1 = df.with_column(
                df.column("x")?
                    .f64()?
                    .apply(|x| Some(logistic(x?, &weights)))
                    .into_series()
                    .rename("y2")
                    .to_owned()
            )?;
            // let mut iters = df.columns(["x", "y", "y2"])?
            //     .iter().map(|s| Ok(s.f64()?.into_iter())).collect::<Result<Vec<_>>>()?;
            //
            // for row in 0..df.height() {
            //     for iter in &mut iters {
            //         let value = iter.next().expect("should have as many iterations as rows");
            //         print!("{} ", value.unwrap());
            //     }
            //     println!("");
            // }
            // exit(0);

        }




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
        .mean().unwrap()
}