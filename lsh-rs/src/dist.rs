use crate::DataPoint;
use ndarray::prelude::*;
use rayon::prelude::*;

/// L2 norm of a single vector.
///
/// # Examples
///
/// ```
/// use lsh_rs::dist::l2_norm;
/// let a = vec![1., -1.];
/// let norm_a = l2_norm(&a);
///
/// // norm between two vectors
/// let b = vec![0.2, 1.2];
/// let c: Vec<f32> = a.iter().zip(b).map(|(ai, bi)| ai - bi).collect();
/// let norm_ab = l2_norm(&c);
/// ```
pub fn l2_norm(x: &[f32]) -> f32 {
    let x = aview1(x);
    x.dot(&x).sqrt()
}

/// Dot product between two vectors.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
///
/// # Examples
///
/// ```
/// use lsh_rs::dist::inner_prod;
/// let a = vec![1., -1.];
/// let b = vec![0.2, 1.2];
/// let prod = inner_prod(&a, &b);
/// ```
pub fn inner_prod(a: &[f32], b: &[f32]) -> f32 {
    aview1(a).dot(&aview1(b))
}

/// Cosine similarity between two vectors.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
///
/// # Examples
///
/// ```
/// use lsh_rs::dist::cosine_sim;
/// let a = vec![1., -1.];
/// let b = vec![0.2, 1.2];
/// let sim = cosine_sim(&a, &b);
/// ```
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    inner_prod(a, b) / (l2_norm(a) * l2_norm(b))
}

pub fn cdist(q: &[f32], vs: &[DataPoint], distance_f: &str) -> Vec<f32> {
    let f = match distance_f {
        "cosine" => cosine_sim,
        "inner-prod" => inner_prod,
        "l2" => {
            return vs
                .into_iter()
                .map(|v| {
                    let c = &aview1(q) - &aview1(v);
                    l2_norm(c.as_slice().unwrap())
                })
                .collect()
        }
        _ => panic!("distance function not defined"),
    };

    vs.into_par_iter().map(|v| f(v, q)).collect()
}

pub fn sort_by_distance(q: &[f32], vs: &[DataPoint], distance_f: &str) -> (Vec<usize>, Vec<f32>) {
    let dist = cdist(q, vs, distance_f);
    let mut intermed: Vec<(usize, f32)> = dist.into_iter().enumerate().collect();
    intermed.sort_unstable_by_key(|(_idx, v)| (v * 1e3) as i64);
    let (idx, dist): (Vec<_>, Vec<_>) = intermed.into_iter().unzip();
    (idx, dist)
}

pub fn sort_by_distances(
    qs: &[DataPoint],
    vs: &[DataPoint],
    distance_f: &str,
) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
    // (Vec<usize>, Vec<f32>)
    qs.par_iter()
        .map(|q| sort_by_distance(q, vs, distance_f))
        .unzip()
}
