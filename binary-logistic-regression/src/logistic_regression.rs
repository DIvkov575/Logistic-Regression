#![allow(non_snake_case, dead_code)]

use std::f64;
use ndarray::{array, Array1, Array2, Ix1};
use polars::datatypes::Int64Type;

pub struct LR {
    pub learn_rate: f64,
    pub weights: Array1<f64>,
    pub bias: f64,
    pub losses: Vec<f64>,
}

impl LR {
    pub fn empty() -> Self {
        Self {
            learn_rate: 0.001,
            weights: array![],
            bias: 0f64,
            losses: Vec::new(),
        }
    }

    pub fn sigmoid(input: Array1<f64>) -> Array1<f64> {
        let mut output: Array1<f64> = Array1::default(Ix1(input.len()));
        for (index, x) in input.iter().enumerate() {
            output[index] = 1. / (1. + f64::consts::E.powf(-x));
        }
        output // keyword "return" can be omitted in rust during return statements if last line
    }


    pub fn fit(&mut self, X: Array2<f64>, y: Array1<f64>) {
        let mut dw;
        let mut db;
        let n_samples= X.shape()[0] as f64;
        let n_features= X.shape()[1];

        self.weights = Array1::zeros(Ix1(n_features));
        self.bias = 0f64;

        for _ in 0..1750{
            let pred = Self::sigmoid(X.clone().dot(&self.weights));

            // gradient descent ( using derivative of BCE loss function to adjust weights)
            let dz: Array1<f64> = pred.clone() - y.clone(); // derivative of sigmoid and bce X.T*(A-y)
            dw = (1. / n_samples) * X.t().dot(&dz.clone());
            db = (1. / n_samples) * (pred.clone().sum() - y.clone().sum());

            // adjust weights based on GD
            for i in 0..dw.len() { self.weights[i] -= self.learn_rate * dw[i]; }
            self.bias -= self.learn_rate * db;
        }
    }

    pub fn predict(&self, X: Array2<f64>) -> Array1<&str> {
        // dot product inputs w/ weights
        let y_hat = X.dot(&self.weights) + self.bias;
        let y_predicted = Self::sigmoid(y_hat);
        // classify as true if probability is above threshold (0.5)
        let y_predicted_cls = y_predicted.iter().map(|x| if x > &0.5 {"1"} else {"0"}).collect();
        y_predicted_cls
    }
}