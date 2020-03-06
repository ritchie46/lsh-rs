use crate::table::{HashTables, MemoryTable};
use crate::utils::{dot_prod, rand_unit_vec};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

pub type Hash = Vec<u8>;
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

///
/// # Arguments
///
/// * `n_hyperplanes` - Number of hyperplanes `K` used to create lsh. This is the hash length.
/// * `n_ht` - Number of hashing tables `L`.
struct LSH<T: HashTables> {
    n_hyperplanes: usize,
    n_ht: usize,
    projections: Vec<RandomProjection>,
    dim: usize,
    hash_tables: T,
}

impl LSH<MemoryTable> {
    pub fn new(
        n_hyperplanes: usize,
        n_hash_tables: usize,
        dim: usize,
        seed: u64,
    ) -> LSH<MemoryTable> {
        let mut projections = Vec::with_capacity(n_hash_tables);
        for i in 0..n_hash_tables {
            let mut rng = SmallRng::seed_from_u64(i as u64 + seed);
            let seed = rng.gen();
            let proj = RandomProjection::new(n_hyperplanes, dim, seed);
            projections.push(proj);
        }

        LSH {
            n_hyperplanes,
            n_ht: n_hash_tables,
            projections,
            dim,
            hash_tables: MemoryTable::new(n_hash_tables),
        }
    }

    pub fn store_vec(&mut self, v: &[f64]) {
        for (i, proj) in self.projections.iter().enumerate() {
            let hash = proj.hash_vec(v);
            self.hash_tables.put(hash, v.to_vec(), i);
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

    #[test]
    fn test_hash_table_put() {
        let mut lhs = LSH::new(5, 3, 3, 1);
        lhs.store_vec(&[2., 3., 4.]);
        println!("{:?}", lhs.hash_tables)
    }
}
