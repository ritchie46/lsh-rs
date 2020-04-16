use ndarray::prelude::*;
use pyo3::prelude::*;

pub fn l2_norm(x: ArrayView1<f32>) -> f32 {
    x.dot(&x).sqrt()
}

pub fn cdist(q: ArrayView1<f32>, vs: &[ArrayView1<f32>], distance_f: &str) -> Vec<f32> {
    match distance_f {
        "l2" | "euclidean" => {
            return vs
                .into_iter()
                .map(|&v| {
                    let c = &q - &v;
                    l2_norm(c.view())
                })
                .collect()
        }
        _ => panic!("distance function not defined"),
    };
}

pub fn sort_by_distance(
    q: ArrayView1<f32>,
    vs: &[ArrayView1<f32>],
    distance_f: &str,
    top_k: usize,
) -> (Vec<usize>, Vec<f32>) {
    let dist = cdist(q, vs, distance_f);
    let mut intermed: Vec<(usize, f32)> = dist.into_iter().enumerate().collect();
    intermed.sort_unstable_by_key(|(_idx, v)| (v * 1e3) as i64);
    let (idx, dist): (Vec<_>, Vec<_>) = intermed.into_iter().take(top_k).unzip();
    (idx, dist)
}
