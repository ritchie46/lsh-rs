use crate::utils::{dot_prod, rand_unit_vec};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

pub type Hash = Vec<u8>;
type HyperPlanes = Vec<Vec<f64>>;

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
        let mut hp = Vec::with_capacity(k);

        for i in 0..k {
            let rng = SmallRng::seed_from_u64(i as u64 + seed);
            hp.push(rand_unit_vec(dim, rng))
        }

        SignRandomProjections { hyperplanes: hp }
    }
}

impl VecHash for SignRandomProjections {
    fn hash_vec(&self, v: &[f64]) -> Hash {
        let mut hash: Vec<u8> = vec![0; self.hyperplanes.len()];
        for (i, plane) in self.hyperplanes.iter().enumerate() {
            if dot_prod(plane, v) >= 0. {
                hash[i] = 1
            };
        }
        hash
    }
}
