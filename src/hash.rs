use crate::utils::{create_rng, l2_norm, rand_unit_vec};
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
    a: Array1<f64>,
    r: f64,
    b: f64,
}

impl L2 {
    pub fn new(dim: usize, r: f64, seed: u64) -> L2 {
        let mut rng = create_rng(seed);
        let a = Array::random_using(dim, StandardNormal, &mut rng);
        let b = rng.sample(Uniform::new(0., r)) as f64;

        L2 { a, r, b }
    }
}

impl VecHash for L2 {
    fn hash_vec(&self, v: &[f64]) -> Hash {
        let h: f64 = ((self.a.t().dot(&aview1(v)) + self.b) / self.r).floor();
        format!("{}", h)
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
    pub fn new(dim: usize, r: f64, U: f64, m: usize, seed: u64) -> MIPS {
        let l2 = L2::new(dim, r, seed);
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

        // shrink norm such that l2 norm < U < 1.
        for x_i in x {
            x_new.push(x_i / self.M * self.U)
        }

        let norm_sq = l2_norm(aview1(x)).powf(2.);
        for i in 1..(self.m + 1) {
            x_new.push(norm_sq.powf(i as f64))
        }
        x_new
    }

    pub fn transform_query(&self, x: &[f64]) -> Vec<f64> {
        let mut x_new = Vec::with_capacity(x.len() + self.m);
        x_new.extend_from_slice(x);
        for i in 0..self.m {
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
        self.hasher.hash_vec(&v)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use num::abs;

    #[test]
    fn test_l2() {
        // Only test if it runs
        let h = L2::new(5, 0.09, 12);
        // two close vector
        let hash1 = h.hash_vec(&[1., 2., 3., 1., 3.]);
        let hash2 = h.hash_vec(&[1.1, 2., 3., 1., 3.1]);

        let h1: i32 = hash1.parse().unwrap();
        let h2: i32 = hash2.parse().unwrap();

        // a distant vec
        let hash3 = h.hash_vec(&[100., 100., 100., 100., 100.1]);
        let h3: i32 = hash3.parse().unwrap();

        println!("close: {:?} distant: {}", (h1, h2), h3);
        assert!(abs(h1 - h2) < abs(h1 - h3));
        assert!(abs(h1 - h2) < abs(h2 - h3));
    }
}
