# bfgs

This package contains an implementation of
[BFGS](https://en.wikipedia.org/w/index.php?title=BFGS_method), an algorithm for minimizing
convex twice-differentiable functions.

BFGS is explained at a high level in
[the blog post](https://paulkernfeld.com/2018/08/06/rust-needs-bfgs.html) introducing this
package.

In this example, we minimize a 2d function:

```rust
extern crate bfgs;
extern crate ndarray;

use ndarray::{Array, Array1};

fn main() {
    let x0 = Array::from_vec(vec![8.888, 1.234]);  // Chosen arbitrarily
    let f = |x: &Array1<f64>| x.dot(x);
    let g = |x: &Array1<f64>| 2.0 * x;
    let x_min = bfgs::bfgs(x0, f, g);
    assert_eq!(x_min, Ok(Array::from_vec(vec![0.0, 0.0])));
}
```

This project uses [cargo-make](https://sagiegurari.github.io/cargo-make/) for builds; to build,
run `cargo make all`.

License: MIT/Apache-2.0
