use crate::utils::{create_rng, l2_norm, rand_unit_vec};
use ndarray::prelude::*;
use ndarray::{aview1, Array, Array1, Array2, Axis};
use ndarray_rand::rand_distr::{StandardNormal, Uniform};
use ndarray_rand::RandomExt;
use rand::Rng;

pub type Hash = String;
type HyperPlanes = Array2<f64>;

pub trait VecHash {
    fn hash_vec(&self, v: &[f64]) -> Hash;
}

/// Also called SimHash.
/// An LSH for the cosine similarity
/// # Arguments
/// * `hyperplanes` - Unit vectors that creates buckets of the data points.
pub struct SignRandomProjections {
    hyperplanes: HyperPlanes,
}

impl SignRandomProjections {
    ///
    /// # Arguments
    ///
    /// * `k` - Number of hyperplanes used for determining the hash.
    /// This will also be the hash length.
    pub fn new(k: usize, dim: usize, seed: u64) -> SignRandomProjections {
        let mut rng = create_rng(seed);
        let hp = Array::random_using((dim, k), StandardNormal, &mut rng);

        SignRandomProjections { hyperplanes: hp }
    }
}

impl VecHash for SignRandomProjections {
    fn hash_vec(&self, v: &[f64]) -> Hash {
        let mut hash: Vec<char> = vec!['0'; self.hyperplanes.len_of(Axis(1))];

        let v = aview1(v);

        for (i, ai) in self.hyperplanes.t().dot(&v).iter().enumerate() {
            if ai > &0.0 {
                hash[i] = '1'
            }
        }
        hash.into_iter().collect()
    }
}

pub struct L2 {
    // https://arxiv.org/pdf/1411.3787.pdf
    a: Array2<f64>,
    r: f64,
    b: Array1<f64>,
    n_projections: usize,
}

impl L2 {
    pub fn new(dim: usize, r: f64, n_projections: usize, seed: u64) -> L2 {
        let mut rng = create_rng(seed);
        let a = Array::random_using((n_projections, dim), StandardNormal, &mut rng);
        let uniform_dist = Uniform::new(0., r);
        let b = Array::random_using((n_projections), uniform_dist, &mut rng);

        L2 {
            a,
            r,
            b,
            n_projections,
        }
    }
}

impl VecHash for L2 {
    fn hash_vec(&self, v: &[f64]) -> Hash {
        let h = (self.a.t().dot(&aview1(v)) + &self.b) / self.r;
        let h = h.map(|x| x.floor() as i32);

        let mut s = String::with_capacity(h.len() * 3);
        for x in h.iter() {
            s.push_str(&x.to_string())
        }
        s
    }
}

struct MIPS {
    // https://papers.nips.cc/paper/5329-asymmetric-lsh-alsh-for-sublinear-time-maximum-inner-product-search-mips.pdf
    U: f64,
    M: f64,
    m: usize,
    dim: usize,
    hasher: L2,
}

impl MIPS {
    pub fn new(dim: usize, r: f64, U: f64, m: usize, n_projections: usize, seed: u64) -> MIPS {
        let l2 = L2::new(dim + m, r, n_projections, seed);
        MIPS {
            U,
            M: 0.,
            m,
            dim,
            hasher: l2,
        }
    }

    pub fn fit(&mut self, v: &[f64]) {
        let mut max_l2 = 0.;
        for x in v.chunks(self.dim) {
            let a = aview1(x);
            let l2 = l2_norm(a);
            if l2 > max_l2 {
                max_l2 = l2
            }
        }
        self.M = max_l2
    }

    pub fn tranform_put(&self, x: &[f64]) -> Vec<f64> {
        let mut x_new = Vec::with_capacity(x.len() + self.m);

        if self.M == 0. {
            panic!("MIPS is not fitted")
        }

        // shrink norm such that l2 norm < U < 1.
        for x_i in x {
            x_new.push(x_i / self.M * self.U)
        }

        let norm_sq = l2_norm(aview1(&x_new)).powf(2.);
        for i in 1..(self.m + 1) {
            x_new.push(norm_sq.powf(i as f64))
        }
        x_new
    }

    pub fn transform_query(&self, x: &[f64]) -> Vec<f64> {
        let mut x_new = Vec::with_capacity(x.len() + self.m);

        x_new.extend_from_slice(x);

        for _ in 0..self.m {
            x_new.push(0.5)
        }
        x_new
    }

    fn hash_vec_query(&self, v: &[f64]) -> Hash {
        let q = self.transform_query(v);
        self.hasher.hash_vec(&q)
    }

    fn hash_vec_put(&self, v: &[f64]) -> Hash {
        let p = self.tranform_put(v);
        self.hasher.hash_vec(&p)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_l2() {
        // Only test if it runs
        let l2 = L2::new(5, 2.2, 5, 1);
        // two close vector
        let h1 = l2.hash_vec(&[1., 2., 3., 1., 3.]);
        let h2 = l2.hash_vec(&[1.1, 2., 3., 1., 3.1]);

        // a distant vec
        let h3 = l2.hash_vec(&[100., 100., 100., 100., 100.1]);

        println!("close: {:?} distant: {}", (&h1, &h2), &h3);
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }
}
