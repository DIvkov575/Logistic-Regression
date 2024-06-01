#![allow(non_snake_case)]
mod logistic_regression;

use ndarray::{array, Array1, Array2, Ix2, Shape};
use logistic_regression::*;
use csv;



fn main() {
    let X = array![-4.50,-4.00,-3.50,-3.00,-2.50,-2.00,-1.50,-1.00,-0.50,0.00,0.50,1.00,1.50,2.00,2.50,3.00,3.50,4.00,4.50].into_shape((19, 1)).unwrap();
    let y: Array1<f64> = array![0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,1.,1.,1.,1.,1.,1.,1.,].into();

    let mut lr = LR::empty();
    lr.fit(X.clone(), y);

    let results = lr.predict(X);
    println!("{:?}", results);





}


