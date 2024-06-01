use std::f64;
use ndarray::{array, Array, Array1, Array2, ArrayView, Ix1, stack_new_axis};


fn main() {
    // https://medium.com/@koushikkushal95/logistic-regression-from-scratch-dfb8527a4226
    let lr = LR::empty();


}


pub struct LR {
    pub learn_rate: f64,
    pub n_iters: usize,
    pub weights: Array1<f64>,
    pub bias: f64,
    pub losses: Vec<f64>,
}

impl LR {
    pub fn empty() -> Self {
        Self {
            learn_rate: 0.001,
            n_iters: 1000,
            weights: array![],
            bias: 0f64,
            losses: Vec::new(),

        }
    }

    pub fn _sigmoid(input: Array1<f64>) -> Array1<f64> {
        let mut output: Array1<f64> = Array1::default(Ix1(input.len()));
        for (index, x) in input.iter().enumerate() {
                output[index] = 1. / (1. + f64::consts::E.powf(-x));
        }
        output
    }

    pub fn compute_loss(&self, y_true: Array1<f64>, y_pred: Array1<f64>) -> f64 {
        // binary cross entropy loss

        let mut y1 = 0f64;
        let mut y2 = 0f64;

        for index in 0..y_true.len() {
            y1 += y_true[index] * (y_pred[index] + f64::EPSILON).ln();
            y2 += (1f64 - y_true[index]) * (1f64 - y_pred[index] + f64::EPSILON).ln();
        }

        -(y1 + y2) / y_true.len() as f64
    }

    pub fn feed_forward(&self, X: Array2<f64>) -> Array1<f64> {
        let z = X.dot(&self.weights);
        Self::_sigmoid(z)
    }

    pub fn fit(&mut self, X: Array2<f64>, y: Array1<f64>) {
            let mut dw;
            let mut db;
            let n_samples= X.shape()[0] as f64;
            let n_features= X.shape()[1];

            self.weights = Array1::zeros(Ix1(n_features));
            self.bias = 0f64;

            for _ in 0..self.n_iters {
                let A = self.feed_forward(X.clone());
                let x = self.compute_loss(y.clone(), A.clone());
                self.losses.push(x);
                let dz: Array1<f64> = A.clone() - y.clone(); // derivative of sigmoid and bce X.T*(A-y)

                dw = (1. / n_samples) * X.t().dot(&dz.clone());
                db = (1. / n_samples) * (A.clone().sum() - y.clone().sum());
                assert_eq!(self.weights.len(), dw.len());
                for i in 0..dw.len() {
                    self.weights[i] -= self.learn_rate * dw[i];
                }
                self.bias -= self.learn_rate * db;
            }
    }

    pub fn predict(&self, X: Array2<f64>) -> Array1<usize> {
        let y_hat = X.dot(&self.weights) + self.bias;
        let y_predicted = Self::_sigmoid(y_hat);
        let y_predicted_cls = y_predicted.map(|x| if (x > &0.5) {1} else {1});
        y_predicted_cls
    }
}
