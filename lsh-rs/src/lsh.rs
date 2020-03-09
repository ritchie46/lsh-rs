use crate::hash::{Hash, SignRandomProjections, VecHash, L2, MIPS};
use crate::table::{Bucket, DataPoint, DataPointSlice, HashTableError, HashTables, MemoryTable};
use crate::utils::create_rng;
use fnv::FnvHashSet;
use rand::{Rng, SeedableRng};

///
/// # Arguments
///
/// * `n_ht` - Number of hashing tables `L`.
pub struct LSH<T: HashTables, H: VecHash> {
    n_ht: usize,
    hashers: Vec<H>,
    dim: usize,
    hash_tables: T,
}

impl LSH<MemoryTable, SignRandomProjections> {
    pub fn new_srp(
        n_projections: usize,
        n_hash_tables: usize,
        dim: usize,
        seed: u64,
    ) -> LSH<MemoryTable, SignRandomProjections> {
        let mut hashers = Vec::with_capacity(n_hash_tables);
        let mut rng = create_rng(seed);

        for _ in 0..n_hash_tables {
            let seed = rng.gen();
            let hasher = SignRandomProjections::new(n_projections, dim, seed);
            hashers.push(hasher);
        }

        LSH {
            n_ht: n_hash_tables,
            hashers,
            dim,
            hash_tables: MemoryTable::new(n_hash_tables),
        }
    }
}

impl LSH<MemoryTable, L2> {
    pub fn new_l2(
        n_projections: usize,
        n_hash_tables: usize,
        dim: usize,
        r: f64,
        seed: u64,
    ) -> LSH<MemoryTable, L2> {
        let mut hashers = Vec::with_capacity(n_hash_tables);
        let mut rng = create_rng(seed);

        for _ in 0..n_hash_tables {
            let seed = rng.gen();
            let hasher = L2::new(dim, r, n_projections, seed);
            hashers.push(hasher);
        }

        LSH {
            n_ht: n_hash_tables,
            hashers,
            dim,
            hash_tables: MemoryTable::new(n_hash_tables),
        }
    }
}

impl LSH<MemoryTable, MIPS> {
    pub fn new_mips(
        n_projections: usize,
        n_hash_tables: usize,
        dim: usize,
        r: f64,
        U: f64,
        m: usize,
        seed: u64,
    ) -> LSH<MemoryTable, MIPS> {
        let mut hashers = Vec::with_capacity(n_hash_tables);
        let mut rng = create_rng(seed);

        for _ in 0..n_hash_tables {
            let seed = rng.gen();
            let hasher = MIPS::new(dim, r, U, m, n_projections, seed);
            hashers.push(hasher);
        }
        LSH {
            n_ht: n_hash_tables,
            hashers,
            dim,
            hash_tables: MemoryTable::new(n_hash_tables),
        }
    }
}

impl<H: VecHash> LSH<MemoryTable, H> {
    pub fn store_vec(&mut self, v: &DataPointSlice) {
        for (i, proj) in self.hashers.iter().enumerate() {
            let hash = proj.hash_vec_put(v);
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

        for (i, proj) in self.hashers.iter().enumerate() {
            let hash = proj.hash_vec_query(v);
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
        for (i, proj) in self.hashers.iter().enumerate() {
            let hash = proj.hash_vec_query(v);
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
    }

    #[test]
    fn test_hash_table() {
        let mut lhs: LSH<MemoryTable, SignRandomProjections> = LSH::new_srp(5, 10, 3, 1);
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
