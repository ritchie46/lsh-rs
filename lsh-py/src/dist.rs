use ndarray::prelude::*;
use pyo3::prelude::*;

fn l2_dist(a: ArrayView1<f32>, b: ArrayView1<f32>) -> f32 {
    let x = &a - &b;
    l2_norm(x.view())
}

fn l2_norm(x: ArrayView1<f32>) -> f32 {
    x.dot(&x).sqrt()
}

fn cosine_sim(a: ArrayView1<f32>, b: ArrayView1<f32>) -> f32 {
    &a.dot(&b) / (l2_norm(a) * l2_norm(b))
}

pub fn cdist(q: ArrayView1<f32>, vs: &[ArrayView1<f32>], distance_f: &str) -> Vec<f32> {
    let dist_fn = match distance_f {
        "l2" | "euclidean" => l2_dist,
        "cosine" => cosine_sim,
        _ => panic!("distance function not defined"),
    };
    vs.into_iter().map(|&v| dist_fn(q, v)).collect()
}

pub fn sort_by_distance(
    q: ArrayView1<f32>,
    vs: &[ArrayView1<f32>],
    distance_f: &str,
    top_k: usize,
) -> (Vec<usize>, Vec<f32>) {
    let dist = cdist(q, vs, distance_f);
    let mut intermed: Vec<(usize, f32)> = dist.into_iter().enumerate().collect();
    intermed.sort_unstable_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
    let (idx, dist): (Vec<_>, Vec<_>) = intermed.into_iter().take(top_k).unzip();
    (idx, dist)
}
