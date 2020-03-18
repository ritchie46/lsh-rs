#![cfg(feature = "stats")]
use crate::table::DataPoint;
use crate::LSH;
use statrs::{
    consts::SQRT_2PI,
    distribution::{Normal, Univariate},
};
use std::time::{Duration, Instant};

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
/// * `p1` - P1 in literature.
/// * `k` - Number of hash projections.
pub fn estimate_l(delta: f64, p1: f64, k: usize) -> usize {
    (delta.ln() / (1. - p1.powf(k as f64)).ln()).round() as usize
}

pub fn optimize_l2_params(delta: f64, dim: usize, vs: &[DataPoint]) {
    let mut results = vec![];

    // assume R normalized data so c = 1
    for r in (4..20).step_by(2) {
        let p1 = l2_ph(r as f64, 1.);
        for k in 8..15 {
            let l = estimate_l(delta, p1, k as usize);
            let mut lsh = LSH::new(k, l, dim).only_index().l2(r as f32);
            lsh.store_vecs(vs);
            let mut q_time = 0.;
            for v in vs {
                let t0 = Instant::now();
                lsh.query_bucket(v);
                let duration = t0.elapsed();
                q_time += duration.as_secs_f64();
            }
            results.push((k, l, q_time))
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_l2_ph() {
        // tested w/ numpy
        let r = 2.0;
        let c = 1.0;
        assert_eq!(0.609548422215397, l2_ph(r, c) as f32);
    }

    #[test]
    fn test_estimate_l() {
        let delta = 0.2;
        let p1 = 0.6;
        let k = 5;
        assert_eq!(20, estimate_l(delta, p1, k));
    }
}
