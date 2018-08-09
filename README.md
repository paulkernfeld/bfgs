# bfgs

This package contains an implementation of
[BFGS](https://en.wikipedia.org/w/index.php?title=BFGS_method), an algorithm for minimizing
convex twice-differentiable functions.

In this example, we minimize a 2d function:

```rust
extern crate bfgs;
extern crate ndarray;

use ndarray::prelude::*;

fn main() {
    let x0 = Array::from_vec(vec![8.888, 1.234]);  // Chosen arbitrarily
    let f = |x: &Array1<f64>| x.dot(x);
    let g = |x: &Array1<f64>| 2.0 * x;
    let x_min = bfgs::bfgs(x0, f, g);
    assert_eq!(x_min, Ok(Array::from_vec(vec![0.0, 0.0])));
}
```

License: MIT/Apache-2.0
