//! Some utilities to help choose LSH parameters.
use crate::dist::l2_norm;
use crate::prelude::*;
use fnv::FnvHashSet;
use ndarray::aview1;
use rayon::prelude::*;
use statrs::{
    consts::SQRT_2PI,
    distribution::{Normal, Univariate},
};
use std::f64::consts::PI;
use std::time::Instant;

/// Hash collision probability for L2 distance.
///
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

/// Hash collision probability for Sign Random Projections
/// # Arguments
/// * `cosine_sim` - Cosine similarity.
pub fn srp_ph(cosine_sim: f64) -> f64 {
    1. - cosine_sim.acos() / PI
}

///
/// Return NN w/ probability 1 - δ. Generic formula.
///
/// # Arguments
/// * `delta` - Prob. not returned NN.
/// * `p1` - P1 in literature.
/// * `k` - Number of hash projections.
pub fn estimate_l(delta: f64, p1: f64, k: usize) -> usize {
    (delta.ln() / (1. - p1.powf(k as f64)).ln()).round() as usize
}

#[derive(Debug)]
pub struct OptRes {
    pub k: usize,
    pub l: usize,
    pub search_time: f64,
    pub hash_time: f64,
    pub min_len: usize,
    pub max_len: usize,
    pub avg_len: f32,
    pub unique_hash_values: FnvHashSet<i32>,
}

fn lsh_to_result<H: 'static + VecHash<f32, i8> + Send + Sync + Clone>(
    lsh: LshMem<H, f32, i8>,
    vs: &[Vec<f32>],
    k: usize,
    l: usize,
) -> Result<OptRes> {
    let mut lsh = lsh;
    lsh.store_vecs(vs)?;
    let mut search_time = 0.;
    let mut hash_time = 0.;
    let mut bucket_lengths = Vec::with_capacity(vs.len());
    for v in vs {
        let th = Instant::now();
        let mut bucket_ids = lsh.query_bucket_ids(v)?;
        hash_time += th.elapsed().as_secs_f64();

        bucket_lengths.push(bucket_ids.len());

        let t0 = Instant::now();
        let q = aview1(&v);
        bucket_ids.par_sort_by_key(|&idx| {
            let p = &vs[idx as usize];
            let dist = &aview1(&p) - &q;
            let l2 = l2_norm(dist.as_slice().unwrap());
            (l2 * 1e5) as i32
        });
        let duration = t0.elapsed();
        search_time += duration.as_secs_f64();
    }
    let min = *bucket_lengths.iter().min().unwrap_or(&(0 as usize));
    let max = *bucket_lengths.iter().max().unwrap_or(&(0 as usize));
    let avg = bucket_lengths.iter().sum::<usize>() as f32 / bucket_lengths.len() as f32;
    let unique_hash_values = lsh.hash_tables.unwrap().get_unique_hash_int();
    Ok(OptRes {
        k,
        l,
        search_time,
        hash_time,
        min_len: min,
        max_len: max,
        avg_len: avg,
        unique_hash_values,
    })
}

/// Does a grid search over parameter *K* where *L* is determined by the `estimate_l` function.
///
/// # Arguments
/// * `delta` - Probability of not returning NN. P(NN) = 1 - δ
/// * `cosine_sim` - Cosine similarity distance within which the nearest neighbor should exist.
/// * `dim` - Dimension of the data points.
/// * `vs` - Data points.
pub fn optimize_srp_params(
    delta: f64,
    cosine_sim: f64,
    dim: usize,
    k: &[usize],
    vs: &[Vec<f32>],
) -> Result<Vec<OptRes>> {
    let mut params = vec![];
    let p1 = srp_ph(cosine_sim);
    for _k in k {
        let l = estimate_l(delta, p1, *_k);
        params.push((*_k, l))
    }
    let result = params
        .par_iter()
        .map(|&(k, l)| {
            let lsh = LshMem::new(k, l, dim).srp()?;
            lsh_to_result(lsh, vs, k, l)
        })
        .collect();
    result
}

/// Does a grid search over parameter *K* where *L* is determined by the `estimate_l` function.
/// Note that the data already should be normalized by dividing the data points
/// by the query distance *R*.
///
/// # Arguments
/// * `delta` - Probability of not returning NN. P(NN) = 1 - δ
/// * `dim` - Dimension of the data points.
/// * `vs` - Data points.
pub fn optimize_l2_params(
    delta: f64,
    dim: usize,
    k: &[usize],
    vs: &[Vec<f32>],
) -> Result<Vec<OptRes>> {
    let mut params = vec![];
    let r = 4.0;
    let p1 = l2_ph(r as f64, 1.);
    for _k in k {
        let l = estimate_l(delta, p1, *_k as usize);
        params.push((r, *_k, l))
    }
    let result = params
        .par_iter()
        .map(|&(r, k, l)| {
            let lsh = LshMem::new(k, l, dim).l2(r as f32)?;
            lsh_to_result(lsh, vs, k, l)
        })
        .collect();
    result
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
