use std::f64;


fn main() {
    // let a = vec![
    //     vec![4f64, 3f64],
    //     vec![2f64, 1f64],
    // ];
    let b = vec![
        vec![1f64],
        vec![2f64],
    ];
    let c = vec![
        vec![1f64, 2f64],
        vec![3f64, 4f64],
    ];

    // let b = vec![1f64,2f64];


    // dot(c, b);

    // https://medium.com/@koushikkushal95/logistic-regression-from-scratch-dfb8527a4226



    // let weights = *vec![0f64; 5];
    // println!("{}", weights);

}


pub struct LR {
    pub lr: f64,
    pub n_iters: usize,
    pub weights: Vec<f64>,
    pub bias: f64,
    pub losses: Vec<f64>,
}

impl LR {
    pub fn empty() -> Self {
        Self {
            lr: 0.001,
            n_iters: 1000,
            weights: Vec::new(),
            bias: 0f64,
            losses: Vec::new(),

        }
    }

    pub fn _sigmoid(x: f64) -> f64 {
        1f64 / (1f64 + f64::consts::E.powf(-x))
    }
    pub fn bce(&self, y_true: Vec<f64>, y_pred: Vec<f64>) -> f64 {
        // binary cross entropy loss
        let mut y1 = 0f64;
        let mut y2 = 0f64;

        for index in 0..y_true.len() {
            y1 += y_true[index] * (y_pred[index] + f64::EPSILON).ln();
            y2 += (1f64 - y_true[index]) * (1f64 - y_pred[index] + f64::EPSILON).ln();
        }

        -(y1 + y2) / y_true.len() as f64
    }

    pub fn feed_forward(&self, X: Vec<Vec<f64>>) {
        let z = dot(X, transpose(Vec::from_iter([self.weights.clone()])));
        let A = Self::sigmoid(z);
    }

    // pub fn fit(&mut self, X: &[&[f64]], y: &[f64]) {
    //     let dw;
    //     let db;
    //     let n_samples = X.len();
    //     let n_features = X[0].len();
    //     self.weights = vec![0f64; n_features];
    //     self.bias = 0f64;
    //
    //     for _ in 0..self.n_iters {
    //         let A = self.feed_forward(X);
    //         let dz = A - y; // derivative of sigmoid and bce X.T*(A-y)
    //
    //         dw = (1 / n_samples) * A.iter().zip(dz.iter()).map(|(x, y)| x * y).sum();
    //         db = (1 / n_samples) * A.iter().zip(dz.iter()).map(|(x, y)| x - y).sum();
    //
    //         self.weights -= self.lr * dw;
    //         self.bias -= self.lr * db;
    //     }
    // }

    // pub fn predict(&self, X: &[&[f64]]) -> Vec<f64> {
    //     let threshold = 0.5;
    //     let y_hat = X.iter().zip(self.weights.iter()).map(|(x, y)| x * y).sum() + self.bias; // todo
    //     // let y_
    // }
}



#[allow(non_snake_case)]
fn dot(A: Vec<Vec<f64>>, B: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    assert_eq!(A[0].len(), B.len());
    let mut output: Vec<Vec<f64>> = vec![vec![0f64; B[0].len()]; A[0].len()];

    let j = 0;

    for h in 0..A.len() {
        for i in 0..A[0].len() {
            output[h][j] += A[h][i] * B[i][j];
        }
    }

    for row in &output {
        println!("{:?}", row);
    }


    output
}

#[allow(non_snake_case)]
fn transpose(X: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut Y: Vec<Vec<f64>> = vec![vec![0f64; X.len()]; X[0].len()];
    for i in 0..X.len() {
        for j in 0..X[0].len() {
            Y[i][j] = X[j][i];
        }
    }
    Y
}