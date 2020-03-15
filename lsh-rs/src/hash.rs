use crate::utils::{create_rng, l2_norm};
use ndarray::prelude::*;
use ndarray_rand::rand_distr::{StandardNormal, Uniform};
use ndarray_rand::RandomExt;
use rand::Rng;
use serde::{Deserialize, Serialize};

pub type Hash = Vec<i32>;

pub trait VecHash {
    fn hash_vec_query(&self, v: &[f32]) -> Hash;
    fn hash_vec_put(&self, v: &[f32]) -> Hash;
}

/// Also called SimHash.
/// A family of hashers for the cosine similarity.
#[derive(Serialize, Deserialize)]
pub struct SignRandomProjections {
    ///  Random unit vectors that will lead to the bits of the hash.
    hyperplanes: Array2<f32>,
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

    fn hash_vec(&self, v: &[f32]) -> Hash {
        let mut hash: Vec<i32> = vec![0; self.hyperplanes.len_of(Axis(1))];

        let v = aview1(v);

        for (i, ai) in self.hyperplanes.t().dot(&v).iter().enumerate() {
            if ai > &0.0 {
                hash[i] = 1
            }
        }
        hash.into_iter().collect()
    }
}

impl VecHash for SignRandomProjections {
    fn hash_vec_query(&self, v: &[f32]) -> Hash {
        self.hash_vec(v)
    }

    fn hash_vec_put(&self, v: &[f32]) -> Hash {
        self.hash_vec(v)
    }
}

/// L2 Hasher family. [Read more.](https://arxiv.org/pdf/1411.3787.pdf)
#[derive(Serialize, Deserialize)]
pub struct L2 {
    a: Array2<f32>,
    r: f32,
    b: Array1<f32>,
    n_projections: usize,
}

impl L2 {
    pub fn new(dim: usize, r: f32, n_projections: usize, seed: u64) -> L2 {
        let mut rng = create_rng(seed);
        let a = Array::random_using((n_projections, dim), StandardNormal, &mut rng);
        let uniform_dist = Uniform::new(0., r);
        let b = Array::random_using(n_projections, uniform_dist, &mut rng);

        L2 {
            a,
            r,
            b,
            n_projections,
        }
    }

    fn hash_vec(&self, v: &[f32]) -> Hash {
        let h = (self.a.dot(&aview1(v)) + &self.b) / self.r;
        let h = h.map(|x| x.floor() as i32);
        h.iter().copied().collect()
    }
}

impl VecHash for L2 {
    fn hash_vec_query(&self, v: &[f32]) -> Hash {
        self.hash_vec(v)
    }

    fn hash_vec_put(&self, v: &[f32]) -> Hash {
        self.hash_vec(v)
    }
}

/// Maximum Inner Product Search. [Read more.](https://papers.nips.cc/paper/5329-asymmetric-lsh-alsh-for-sublinear-time-maximum-inner-product-search-mips.pdf)
#[derive(Serialize, Deserialize)]
pub struct MIPS {
    U: f32,
    M: f32,
    m: usize,
    dim: usize,
    hasher: L2,
}

impl MIPS {
    pub fn new(dim: usize, r: f32, U: f32, m: usize, n_projections: usize, seed: u64) -> MIPS {
        let l2 = L2::new(dim + m, r, n_projections, seed);
        MIPS {
            U,
            M: 0.,
            m,
            dim,
            hasher: l2,
        }
    }

    pub fn fit(&mut self, v: &[f32]) {
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

    pub fn tranform_put(&self, x: &[f32]) -> Vec<f32> {
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
            x_new.push(norm_sq.powf(i as f32))
        }
        x_new
    }

    pub fn transform_query(&self, x: &[f32]) -> Vec<f32> {
        let mut x_new = Vec::with_capacity(x.len() + self.m);

        // normalize query to have l2 == 1.
        let l2 = l2_norm(aview1(x));
        for x_i in x {
            x_new.push(x_i / l2)
        }

        for _ in 0..self.m {
            x_new.push(0.5)
        }
        x_new
    }
}

impl VecHash for MIPS {
    fn hash_vec_query(&self, v: &[f32]) -> Hash {
        let q = self.transform_query(v);
        self.hasher.hash_vec_query(&q)
    }

    fn hash_vec_put(&self, v: &[f32]) -> Hash {
        let p = self.tranform_put(v);
        self.hasher.hash_vec_query(&p)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_l2() {
        // Only test if it runs
        let l2 = L2::new(5, 2.2, 7, 1);
        // two close vector
        let h1 = l2.hash_vec_query(&[1., 2., 3., 1., 3.]);
        let h2 = l2.hash_vec_query(&[1.1, 2., 3., 1., 3.1]);

        // a distant vec
        let h3 = l2.hash_vec_query(&[100., 100., 100., 100., 100.1]);

        println!("close: {:?} distant: {:?}", (&h1, &h2), &h3);
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }
}
