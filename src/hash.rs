use crate::utils::{dot_prod, rand_unit_vec};
use ndarray::{aview1, Array, Array1, Array2, Axis};
use ndarray_rand::rand_distr::{StandardNormal, Uniform};
use ndarray_rand::RandomExt;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

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
        let mut rng = SmallRng::seed_from_u64(seed);
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
        let mut rng = SmallRng::seed_from_u64(seed);
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
