use crate::hash::{Hash, SignRandomProjections, VecHash};
use crate::table::{Bucket, DataPoint, DataPointSlice, HashTableError, HashTables, MemoryTable};
use fnv::FnvHashSet;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

///
/// # Arguments
///
/// * `n_hyperplanes` - Number of hyperplanes `K` used to create lsh. This is the hash length.
/// * `n_ht` - Number of hashing tables `L`.
struct LSH<T: HashTables> {
    n_hyperplanes: usize,
    n_ht: usize,
    projections: Vec<SignRandomProjections>,
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
        // Every new projection is a new hasher.
        let mut projections = Vec::with_capacity(n_hash_tables);
        for i in 0..n_hash_tables {
            let mut rng = SmallRng::seed_from_u64(i as u64 + seed);
            let seed = rng.gen();
            let proj = SignRandomProjections::new(n_hyperplanes, dim, seed);
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

    pub fn store_vec(&mut self, v: &DataPointSlice) {
        for (i, proj) in self.projections.iter().enumerate() {
            let hash = proj.hash_vec(v);
            match self.hash_tables.put(hash, v.to_vec(), i) {
                Ok(_) => (),
                Err(_) => panic!("Could not store vec"),
            }
        }
    }

    pub fn store_vecs(&mut self, vs: &[DataPoint]) {
        for d in vs {
            self.store_vec(d)
        }
    }

    /// Query all buckets in the hash tables. The data points in the hash tables
    /// are added to one Vector. The final result may have (probably has) duplicates.
    ///
    /// # Arguments
    /// `v` - Query vector
    /// `dedup` - Deduplicate bucket. This requires a sort and then deduplicate. O(n log n)
    pub fn query_bucket(&self, v: &DataPointSlice, dedup: bool) -> Vec<&DataPoint> {
        let mut merged_bucket: Vec<&DataPoint> = vec![];

        for (i, proj) in self.projections.iter().enumerate() {
            let hash = proj.hash_vec(v);
            match self.hash_tables.query_bucket(&hash, i) {
                Err(HashTableError::NotFound) => (),
                Ok(bucket) => {
                    for dp in bucket {
                        merged_bucket.push(dp)
                    }
                }
                _ => panic!("Unexpected query result"),
            }
        }
        if dedup {
            merged_bucket
                .sort_unstable_by_key(|d| d.iter().fold(0, |acc, x| acc + x.powf(2.0) as u64));
            merged_bucket.dedup();
        }
        merged_bucket
    }

    pub fn delete_vec(&mut self, v: &DataPointSlice) {
        for (i, proj) in self.projections.iter().enumerate() {
            let hash = proj.hash_vec(v);
            self.hash_tables.delete(hash, v, i).unwrap_or_default();
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_simhash() {
        // Only test if it runs
        let h = SignRandomProjections::new(5, 3, 1);
        assert_eq!(h.hash_vec(&[2., 3., 4.]), [0, 0, 1, 1, 1]);
        // close input similar hash
        assert_eq!(h.hash_vec(&[2.1, 3.2, 4.5]), [0, 0, 1, 1, 1]);
        // distant input different hash
        assert_ne!(h.hash_vec(&[-2., -3., -4.]), [0, 0, 1, 1, 1]);
    }

    #[test]
    fn test_hash_table() {
        let mut lhs = LSH::new(5, 10, 3, 1);
        let v1 = &[2., 3., 4.];
        let v2 = &[-1., -1., 1.];
        let v3 = &[0.2, -0.2, 0.2];
        lhs.store_vec(v1);
        lhs.store_vec(v2);
        assert!(lhs.query_bucket(v2, false).len() > 0);

        let bucket_len_before = lhs.query_bucket(v1, true).len();
        lhs.delete_vec(v1);
        let bucket_len_before_after = lhs.query_bucket(v1, true).len();
        assert!(bucket_len_before > bucket_len_before_after);
    }
}
