#![cfg(test)]
extern crate test;

use crate::bfgs;
use ndarray::Array1;
use test::Bencher;

#[bench]
fn test_x_fourth_p_1000(bencher: &mut Bencher) {
    let p = 1_000;
    let x0 = Array1::from_elem(p, 2.0);
    let f = |x: &Array1<f64>| x.iter().map(|xx| xx.powi(4)).sum();
    let g = |x: &Array1<f64>| x.iter().map(|xx| 4.0 * xx.powi(3)).collect();
    bencher.iter(|| {
        let x_min = bfgs(x0.clone(), f, g);
        assert_eq!(x_min, Ok(Array1::zeros(p)));
    })
}
