use crate::hash::{Hash, SignRandomProjections, VecHash, L2, MIPS};
use crate::multi_probe::create_hash_permutation;
use crate::table::{
    general::{HashTableError, HashTables},
    mem::MemoryTable,
};
use crate::utils::create_rng;
use crate::{DataPoint, DataPointSlice};
use fnv::FnvHashSet as HashSet;
use rand::Rng;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

/// Wrapper for LSH functionality.
pub struct LSH<T: HashTables, H: VecHash> {
    /// Number of hash tables. `M` in literature.
    n_hash_tables: usize,
    /// Number of hash functions. `K` in literature.
    n_projections: usize,
    /// Hash functions.
    hashers: Vec<H>,
    /// Dimensions of p and q
    dim: usize,
    /// Storage data structure
    hash_tables: T,
    /// seed for hash functions. If 0, randomness is seeded from the os.
    _seed: u64,
    /// store only indexes and no data points.
    only_index_storage: bool,
    _multi_probe: bool,
    /// Number of (optional) changing bits per hash.
    _multi_probe_n_perturbations: usize,
    /// Length of probing sequence
    _multi_probe_n_probes: usize,
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
            _multi_probe: self._multi_probe,
            _multi_probe_n_perturbations: self._multi_probe_n_perturbations,
            _multi_probe_n_probes: self._multi_probe_n_probes,
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
            _multi_probe: self._multi_probe,
            _multi_probe_n_perturbations: self._multi_probe_n_perturbations,
            _multi_probe_n_probes: self._multi_probe_n_probes,
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
            _multi_probe: self._multi_probe,
            _multi_probe_n_perturbations: self._multi_probe_n_perturbations,
            _multi_probe_n_probes: self._multi_probe_n_probes,
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
            _multi_probe: false,
            _multi_probe_n_perturbations: 3,
            _multi_probe_n_probes: 16,
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

    /// Enable multi-probing LSH and set multi-probing parameters.
    ///
    /// # Arguments
    /// * `n_probes` - The length of the probing sequence.
    /// * `n_permutations` - The upper bounds of bits that may shift per hash.
    pub fn multi_probe(&mut self, n_probes: usize, n_permutations: usize) -> &mut Self {
        self._multi_probe = true;
        self._multi_probe_n_perturbations = n_permutations;
        self._multi_probe_n_probes = n_probes;
        self
    }

    pub fn increase_storage(&mut self, upper_bound: usize) -> &mut Self {
        self.hash_tables.increase_storage(upper_bound);
        self
    }

    pub fn describe(&self) {
        self.hash_tables.describe();
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
            let mut hash = proj.hash_vec_put(v);
            hash.shrink_to_fit();
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
        if self._multi_probe {
            return self.multi_probe_bucket_union(v);
        }

        let mut bucket_union = HashSet::default();

        for (i, proj) in self.hashers.iter().enumerate() {
            let hash = proj.hash_vec_query(v);
            self.process_bucket_union_result(&hash, i, &mut bucket_union)
        }
        bucket_union
    }

    /// Query all buckets in the hash tables. The union of the matching buckets over the `L`
    /// hash tables is returned
    ///
    /// # Arguments
    /// * `v` - Query vector
    pub fn query_bucket(&self, v: &DataPointSlice) -> Vec<&DataPoint> {
        if self.only_index_storage {
            panic!("cannot query bucket, use query_bucket_ids")
        }
        let bucket_union = self.query_bucket_union(v);

        bucket_union
            .iter()
            .map(|&idx| self.hash_tables.idx_to_datapoint(idx).unwrap())
            .collect()
    }

    /// Query all buckets in the hash tables and return the data point indexes. The union of the
    /// matching buckets of `L` hash tables is returned.
    ///
    /// # Arguments
    /// * `v` - Query vector
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

    fn process_bucket_union_result(
        &self,
        hash: &Hash,
        hash_table_idx: usize,
        bucket_union: &mut HashSet<u32>,
    ) {
        match self.hash_tables.query_bucket(hash, hash_table_idx) {
            Err(HashTableError::NotFound) => (),
            Ok(bucket) => {
                *bucket_union = bucket_union.union(bucket).copied().collect();
            }
            _ => panic!("Unexpected query result"),
        };
    }

    fn multi_probe_bucket_union(&self, v: &DataPointSlice) -> HashSet<u32> {
        let mut probing_seq = HashSet::with_capacity_and_hasher(
            self._multi_probe_n_probes,
            fnv::FnvBuildHasher::default(),
        );
        for _ in 0..self._multi_probe_n_probes {
            probing_seq.insert(create_hash_permutation(
                self.n_projections,
                self._multi_probe_n_perturbations,
            ));
        }

        let mut bucket_union = HashSet::default();
        for (i, proj) in self.hashers.iter().enumerate() {
            // fist process the original query
            let original_hash = proj.hash_vec_query(v);
            self.process_bucket_union_result(&original_hash, i, &mut bucket_union);

            for pertub in &probing_seq {
                let hash = original_hash
                    .iter()
                    .zip(pertub)
                    .map(|(&a, &b)| a + b)
                    .collect();
                self.process_bucket_union_result(&hash, i, &mut bucket_union);
            }
        }
        bucket_union
    }
}

/// Intermediate data structure for serialization. Only contains the absolute
/// necessities for reproducible results.
#[derive(Serialize, Deserialize)]
struct IntermediatBlob {
    hash_tables: Vec<u8>,
    hashers: Vec<u8>,
    n_hash_tables: usize,
    n_projections: usize,
    dim: usize,
    _seed: u64,
}

impl<H> LSH<MemoryTable, H>
where
    H: Serialize + DeserializeOwned + VecHash + std::marker::Sync,
{
    pub fn load<P: AsRef<Path>>(&mut self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let mut f = File::open(path)?;
        let mut buf: Vec<u8> = vec![];
        f.read_to_end(&mut buf)?;

        let ib: IntermediatBlob = bincode::deserialize(&buf)?;
        self.hashers = bincode::deserialize(&ib.hashers)?;
        self.hash_tables = bincode::deserialize(&ib.hash_tables)?;
        self.n_hash_tables = ib.n_hash_tables;
        self.n_projections = ib.n_projections;
        self.dim = ib.dim;
        self._seed = ib._seed;

        Ok(())
    }
    pub fn dump<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let hash_tables = bincode::serialize(&self.hash_tables)?;
        let hashers = bincode::serialize(&self.hashers)?;

        let ib = IntermediatBlob {
            hash_tables,
            hashers,
            n_hash_tables: self.n_hash_tables,
            n_projections: self.n_projections,
            dim: self.dim,
            _seed: self._seed,
        };
        let mut f = File::create(path)?;
        let blob = bincode::serialize(&ib)?;
        f.write(&blob)?;
        Ok(())
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
        assert_eq!(lsh.hash_tables.vec_store.map.len(), 0);
        lsh.query_bucket_ids(v1);
    }

    #[test]
    fn test_serialization() {
        let mut lsh = LSH::new(5, 9, 3).seed(1).l2(2.);
        let v1 = &[2., 3., 4.];
        lsh.store_vec(v1);
        let mut tmp = std::env::temp_dir();
        tmp.push("lsh");
        std::fs::create_dir(&tmp);
        tmp.push("serialized.cbor");
        assert!(lsh.dump(&tmp).is_ok());

        // load from file
        let res = lsh.load(&tmp);
        println!("{:?}", res);
        assert!(res.is_ok());
        println!("{:?}", lsh.hash_tables)
    }
}
