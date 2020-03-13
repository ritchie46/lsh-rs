use crate::hash::{SignRandomProjections, VecHash, L2, MIPS};
use crate::table::{DataPoint, DataPointSlice, HashTableError, HashTables, MemoryTable};
use crate::utils::create_rng;
use fnv::FnvHashSet as HashSet;
use rand::Rng;

/// Wrapper for LSH functionality.
pub struct LSH<T: HashTables, H: VecHash> {
    n_hash_tables: usize,
    n_projections: usize,
    hashers: Vec<H>,
    dim: usize,
    hash_tables: T,
    _seed: u64,
    // store only indexes and no data points.
    only_index_storage: bool,
}

impl LSH<MemoryTable, SignRandomProjections> {
    /// Create a new SignRandomProjections LSH
    pub fn srp(&mut self) -> Self {
        let mut rng = create_rng(self._seed);
        let mut hashers = Vec::with_capacity(self.n_hash_tables);

        for _ in 0..self.n_hash_tables {
            let seed = rng.gen();
            let hasher = SignRandomProjections::new(self.n_projections, self.dim, seed);
            hashers.push(hasher);
        }
        LSH {
            n_hash_tables: self.n_hash_tables,
            n_projections: self.n_projections,
            hashers,
            dim: self.dim,
            hash_tables: MemoryTable::new(self.n_hash_tables, self.only_index_storage),
            _seed: self._seed,
            only_index_storage: self.only_index_storage,
        }
    }
}

impl LSH<MemoryTable, L2> {
    /// Create a new L2 LSH
    ///
    /// See hash function:
    /// https://www.cs.princeton.edu/courses/archive/spring05/cos598E/bib/p253-datar.pdf
    /// in paragraph 3.2
    ///
    /// h(v) = floor(a^Tv + b / r)
    ///
    /// # Arguments
    ///
    /// * `r` - Parameter of hash function.
    pub fn l2(&mut self, r: f32) -> Self {
        let mut rng = create_rng(self._seed);
        let mut hashers = Vec::with_capacity(self.n_hash_tables);
        for _ in 0..self.n_hash_tables {
            let seed = rng.gen();
            let hasher = L2::new(self.dim, r, self.n_projections, seed);
            hashers.push(hasher);
        }
        LSH {
            n_hash_tables: self.n_hash_tables,
            n_projections: self.n_projections,
            hashers,
            dim: self.dim,
            hash_tables: MemoryTable::new(self.n_hash_tables, self.only_index_storage),
            _seed: self._seed,
            only_index_storage: self.only_index_storage,
        }
    }
}

impl LSH<MemoryTable, MIPS> {
    /// Create a new MIPS LSH
    ///
    /// Async hasher
    ///
    /// See hash function:
    /// https://www.cs.rice.edu/~as143/Papers/SLIDE_MLSys.pdf
    ///
    /// # Arguments
    ///
    /// * `r` - Parameter of hash function.
    /// * `U` - Parameter of hash function.
    /// * `m` - Parameter of hash function.
    pub fn mips(&mut self, r: f32, U: f32, m: usize) -> Self {
        let mut rng = create_rng(self._seed);
        let mut hashers = Vec::with_capacity(self.n_hash_tables);

        for _ in 0..self.n_hash_tables {
            let seed = rng.gen();
            let hasher = MIPS::new(self.dim, r, U, m, self.n_projections, seed);
            hashers.push(hasher);
        }
        LSH {
            n_hash_tables: self.n_hash_tables,
            n_projections: self.n_projections,
            hashers,
            dim: self.dim,
            hash_tables: MemoryTable::new(self.n_hash_tables, self.only_index_storage),
            _seed: self._seed,
            only_index_storage: self.only_index_storage,
        }
    }
}

impl<H: VecHash> LSH<MemoryTable, H> {
    /// Create a new Base LSH
    ///
    /// # Arguments
    ///
    /// * `n_projections` - Hash length. Every projections creates an hashed integer
    /// * `n_hash_tables` - Increases the chance of finding the closest but has a performance and space cost.
    /// * `dim` - Dimensions of the data points.

    pub fn new(n_projections: usize, n_hash_tables: usize, dim: usize) -> Self {
        LSH {
            n_hash_tables,
            n_projections,
            hashers: Vec::with_capacity(0),
            dim,
            hash_tables: MemoryTable::new(n_hash_tables, true),
            _seed: 0,
            only_index_storage: false,
        }
    }

    /// Set seed of LSH
    /// # Arguments
    /// * `seed` - Seed for the RNG's if 0, RNG's are seeded randomly.
    pub fn seed(&mut self, seed: u64) -> &mut Self {
        self._seed = seed;
        self
    }

    /// Only store indexes of data points. The mapping of data point to indexes is done outside
    /// of the LSH struct.
    pub fn only_index(&mut self) -> &mut Self {
        self.only_index_storage = true;
        self
    }
}

impl<H: VecHash> LSH<MemoryTable, H> {
    /// Store a single vector in storage. Returns id.
    ///
    /// # Arguments
    /// * `v` - Data point.
    ///
    /// # Examples
    /// ```
    ///let mut lsh = LSH::new(5, 10, 3).srp();
    ///let v = &[2., 3., 4.];
    ///let id = lsh.store_vec(v);
    /// ```
    pub fn store_vec(&mut self, v: &DataPointSlice) -> u32 {
        let mut idx = 0;
        for (i, proj) in self.hashers.iter().enumerate() {
            let hash = proj.hash_vec_put(v);
            idx = match self.hash_tables.put(hash, v, i) {
                Ok(i) => i,
                Err(_) => panic!("Could not store vec"),
            }
        }
        idx
    }

    /// Store multiple vectors in storage. Before storing the storage capacity is possibly
    /// increased to match the data points.
    ///
    /// # Arguments
    /// * `vs` - Array of data points.
    ///
    /// # Examples
    ///```
    ///let mut lsh = LSH::new(5, 10, 3).srp();
    ///let vs = &[[2., 3., 4.],
    ///           [-1., -1., 1.]];
    ///let ids = lsh.store_vecs(vs);
    /// ```
    pub fn store_vecs(&mut self, vs: &[DataPoint]) -> Vec<u32> {
        self.hash_tables.increase_storage(vs.len());
        vs.iter().map(|x| self.store_vec(x)).collect()
    }

    fn query_bucket_union(&self, v: &DataPointSlice) -> HashSet<u32> {
        let mut bucket_union = HashSet::default();

        for (i, proj) in self.hashers.iter().enumerate() {
            let hash = proj.hash_vec_query(v);
            match self.hash_tables.query_bucket(&hash, i) {
                Err(HashTableError::NotFound) => (),
                Ok(bucket) => {
                    bucket_union = bucket_union.union(bucket).copied().collect();
                }
                _ => panic!("Unexpected query result"),
            }
        }
        bucket_union
    }

    /// Query all buckets in the hash tables. The union of the matching buckets over the `L`
    /// hash tables is returned
    ///
    /// # Arguments
    /// * `v` - Query vector
    pub fn query_bucket(&self, v: &DataPointSlice) -> Vec<&DataPoint> {
        let bucket_union = self.query_bucket_union(v);

        bucket_union
            .iter()
            .map(|&idx| self.hash_tables.idx_to_datapoint(idx))
            .collect()
    }

    pub fn query_bucket_ids(&self, v: &DataPointSlice) -> Vec<u32> {
        let bucket_union = self.query_bucket_union(v);
        bucket_union.iter().copied().collect()
    }

    /// Delete data point from storage. This does not free memory as the storage vector isn't resized.
    ///
    /// # Arguments
    /// * `v` - Data point
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
    fn test_hash_table() {
        let mut lsh = LSH::new(5, 10, 3).seed(1).srp();
        let v1 = &[2., 3., 4.];
        let v2 = &[-1., -1., 1.];
        lsh.store_vec(v1);
        lsh.store_vec(v2);
        assert!(lsh.query_bucket(v2).len() > 0);

        let bucket_len_before = lsh.query_bucket(v1).len();
        lsh.delete_vec(v1);
        let bucket_len_before_after = lsh.query_bucket(v1).len();
        assert!(bucket_len_before > bucket_len_before_after);
    }

    #[test]
    fn test_index_only() {
        // Test if vec storage is increased
        let mut lsh = LSH::new(5, 9, 3).seed(1).l2(2.);
        let v1 = &[2., 3., 4.];
        lsh.store_vec(v1);
        assert_eq!(lsh.hash_tables.vec_store.map.len(), 1);

        // Test if vec storage is empty
        let mut lsh = LSH::new(5, 9, 3).seed(1).only_index().l2(2.);
        lsh.store_vec(v1);
        assert_eq!(lsh.hash_tables.vec_store.map.len(), 0)
    }
}
