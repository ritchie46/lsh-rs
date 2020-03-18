#![cfg(feature = "stats")]
use statrs::{
    consts::SQRT_2PI,
    distribution::{Normal, Univariate},
};

/// Assumes R normalized data points. So R = 1.
/// Compute 𝑃1 if c = 1.
/// Compute 𝑃2 if c = c
///
/// # Arguments
/// * `r` - Parameter of l2 hash function (also noted as `w`)
/// * `c` - Approximation factor. cR.
pub fn l2_ph(r: f64, c: f64) -> f64 {
    let norm = Normal::new(0., 1.).unwrap();
    1. - 2. * norm.cdf(-r / c)
        - 2. / (SQRT_2PI * r / c) * (1. - (-(r.powf(2.) / (2. * c.powf(2.)))).exp())
}

///
/// Return NN w/ probability 1 - δ
///
/// # Arguments
/// * `delta` - Prob. not returned NN.
/// * `p_1` - P1 in literature.
/// * `k` - Number of hash projections.
pub fn estimate_l(delta: f64, p_1: f64, k: usize) -> usize {
    (delta.ln() / (1 - p_1.powf(k as f64).ln())) as usize
}