use crate::utils::{dot_prod, rand_unit_vec};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
type HyperPlanes = Vec<Vec<f64>>;

/// Also called SimHash.
/// An LSH for the cosine similarity
/// # Arguments
/// * `hyperplanes` - Unit vectors that creates buckets of the data points.
struct RandomProjection {
    hyperplanes: HyperPlanes,
}

impl RandomProjection {
    ///
    /// # Arguments
    ///
    /// * `k` - Number of hyperplanes used for determining the hash.
    /// This will also be the hash length.
    fn new(k: usize, dim: usize, seed: u64) -> RandomProjection {
        let mut hp = Vec::with_capacity(k);

        for i in 0..k {
            let rng = SmallRng::seed_from_u64(i as u64 + seed);
            hp.push(rand_unit_vec(dim, rng))
        }

        RandomProjection { hyperplanes: hp }
    }

    fn hash_vec(&self, v: &[f64]) -> Vec<u8> {
        let mut hash: Vec<u8> = vec![0; self.hyperplanes.len()];
        for (i, plane) in self.hyperplanes.iter().enumerate() {
            if dot_prod(plane, v) >= 0. {
                hash[i] = 1
            };
        }
        hash
    }
}

///
/// # Arguments
///
/// * `n_hyperplanes` - Number of hyperplanes `K` used to create lsh.
/// * `n_ht` - Number of hashing tables `L`.
struct LSH {
    n_hyperplanes: usize,
    n_ht: usize,
    projections: Vec<RandomProjection>,
    dim: usize,
}

impl LSH {
    pub fn new(n_hyperplanes: usize, n_hashing_tables: usize, dim: usize, seed: u64) -> LSH {
        let mut projections = Vec::with_capacity(n_hashing_tables);
        for i in 0..n_hashing_tables {
            let mut rng = SmallRng::seed_from_u64(i as u64 + seed);
            let seed = rng.gen();
            let proj = RandomProjection::new(n_hyperplanes, dim, seed);
            projections.push(proj);
        }

        LSH {
            n_hyperplanes,
            n_ht: n_hashing_tables,
            projections,
            dim,
        }
    }

    pub fn store_vec(&self, v: &[f64]) {
        for proj in self.projections {
            let hash = proj.hash_vec(v);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_simhash() {
        // Only test if it runs
        let h = RandomProjection::new(5, 3, 1);
        assert_eq!(h.hash_vec(&[2., 3., 4.]), [0, 0, 1, 1, 1]);
        // close input similar hash
        assert_eq!(h.hash_vec(&[2.1, 3.2, 4.5]), [0, 0, 1, 1, 1]);
        // distant input different hash
        assert_ne!(h.hash_vec(&[-2., -3., -4.]), [0, 0, 1, 1, 1]);
    }
}
