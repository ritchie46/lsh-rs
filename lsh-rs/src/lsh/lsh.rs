use crate::utils::create_rng;
use crate::{
    hash::{Hash, SignRandomProjections, VecHash, L2, MIPS},
    multi_probe::create_hash_permutation,
    table::{general::HashTables, mem::MemoryTable, sqlite_mem::SqlTableMem},
    Error, Result,
};
use crate::{DataPoint, DataPointSlice, SqlTable};
use fnv::FnvHashSet as HashSet;
use rand::Rng;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

pub type LshSql<H> = LSH<SqlTable, H>;
pub type LshSqlMem<H> = LSH<SqlTableMem, H>;
pub type LshMem<H> = LSH<MemoryTable, H>;

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
    pub hash_tables: Option<T>,
    /// seed for hash functions. If 0, randomness is seeded from the os.
    _seed: u64,
    /// store only indexes and no data points.
    only_index_storage: bool,
    _multi_probe: bool,
    /// Number of (optional) changing bits per hash.
    _multi_probe_n_perturbations: usize,
    /// Length of probing sequence
    _multi_probe_n_probes: usize,
    _db_dir: String,
}

/// Create a new LSH instance. Used in the builder pattern
fn lsh_from_lsh<T: HashTables, H: VecHash + Serialize + DeserializeOwned>(
    lsh: &mut LSH<T, H>,
    hashers: Vec<H>,
) -> Result<LSH<T, H>> {
    let mut ht = *T::new(lsh.n_hash_tables, lsh.only_index_storage, &lsh._db_dir)?;

    // Load hashers if store hashers fails. (i.e. exists)
    let hashers = match ht.store_hashers(&hashers) {
        Ok(_) => hashers,
        Err(_) => match ht.load_hashers() {
            Err(e) => panic!(format!("could not load hashers: {}", e)),
            Ok(hashers) => hashers,
        },
    };
    let lsh = LSH {
        n_hash_tables: lsh.n_hash_tables,
        n_projections: lsh.n_projections,
        hashers,
        dim: lsh.dim,
        hash_tables: Some(ht),
        _seed: lsh._seed,
        only_index_storage: lsh.only_index_storage,
        _multi_probe: lsh._multi_probe,
        _multi_probe_n_perturbations: lsh._multi_probe_n_perturbations,
        _multi_probe_n_probes: lsh._multi_probe_n_probes,
        _db_dir: lsh._db_dir.clone(),
    };
    Ok(lsh)
}

impl<T: HashTables> LSH<T, SignRandomProjections> {
    /// Create a new SignRandomProjections LSH
    pub fn srp(&mut self) -> Result<Self> {
        let mut rng = create_rng(self._seed);
        let mut hashers = Vec::with_capacity(self.n_hash_tables);

        for _ in 0..self.n_hash_tables {
            let seed = rng.gen();
            let hasher = SignRandomProjections::new(self.n_projections, self.dim, seed);
            hashers.push(hasher);
        }
        lsh_from_lsh(self, hashers)
    }
}

impl<T: HashTables> LSH<T, L2> {
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
    pub fn l2(&mut self, r: f32) -> Result<Self> {
        let mut rng = create_rng(self._seed);
        let mut hashers = Vec::with_capacity(self.n_hash_tables);
        for _ in 0..self.n_hash_tables {
            let seed = rng.gen();
            let hasher = L2::new(self.dim, r, self.n_projections, seed);
            hashers.push(hasher);
        }
        lsh_from_lsh(self, hashers)
    }
}

impl<T: HashTables> LSH<T, MIPS> {
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
    pub fn mips(&mut self, r: f32, U: f32, m: usize) -> Result<Self> {
        let mut rng = create_rng(self._seed);
        let mut hashers = Vec::with_capacity(self.n_hash_tables);

        for _ in 0..self.n_hash_tables {
            let seed = rng.gen();
            let hasher = MIPS::new(self.dim, r, U, m, self.n_projections, seed);
            hashers.push(hasher);
        }
        lsh_from_lsh(self, hashers)
    }
}

impl<H: VecHash, T: HashTables> LSH<T, H> {
    /// Create a new Base LSH
    ///
    /// # Arguments
    ///
    /// * `n_projections` - Hash length. Every projections creates an hashed integer
    /// * `n_hash_tables` - Increases the chance of finding the closest but has a performance and space cost.
    /// * `dim` - Dimensions of the data points.

    pub fn new(n_projections: usize, n_hash_tables: usize, dim: usize) -> Self {
        let lsh = LSH {
            n_hash_tables,
            n_projections,
            hashers: Vec::with_capacity(0),
            dim,
            hash_tables: None,
            _seed: 0,
            only_index_storage: false,
            _multi_probe: false,
            _multi_probe_n_perturbations: 3,
            _multi_probe_n_probes: 16,
            _db_dir: ".".to_string(),
        };
        lsh
    }

    fn validate_vec(&self, v: &DataPointSlice) -> Result<()> {
        if !(v.len() == self.dim) {
            return Err(Error::Failed(
                "data point is not valid, are the dimensions correct?".to_string(),
            ));
        };
        Ok(())
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

    pub fn increase_storage(&mut self, upper_bound: usize) -> Result<&mut Self> {
        self.hash_tables
            .as_mut()
            .unwrap()
            .increase_storage(upper_bound);
        Ok(self)
    }

    pub fn set_database_directory(&mut self, path: &str) -> &mut Self {
        self._db_dir = path.to_string();
        self
    }

    pub fn describe(&self) -> Result<String> {
        self.hash_tables.as_ref().unwrap().describe()
    }

    /// Store a single vector in storage. Returns id.
    ///
    /// # Arguments
    /// * `v` - Data point.
    ///
    /// # Examples
    /// ```
    /// use lsh_rs::LshSql;
    /// let mut lshd = LshSql::new(5, 10, 3).srp();
    /// let v = &[2., 3., 4.];
    /// let id = lsh.store_vec(v);
    /// ```
    pub fn store_vec(&mut self, v: &DataPointSlice) -> Result<u32> {
        self.validate_vec(v)?;

        let mut idx = 0;
        let mut ht = self.hash_tables.take().unwrap();
        for (i, proj) in self.hashers.iter().enumerate() {
            let mut hash = proj.hash_vec_put(v);
            idx = ht.put(hash, v, i)?;
        }
        self.hash_tables.replace(ht);
        Ok(idx)
    }

    /// Store multiple vectors in storage. Before storing the storage capacity is possibly
    /// increased to match the data points.
    ///
    /// # Arguments
    /// * `vs` - Array of data points.
    ///
    /// # Examples
    ///```
    /// use lsh_rs::LshSql;
    /// let mut lsh = LshSql::new(5, 10, 3).srp();
    /// let vs = &[vec![2., 3., 4.],
    ///           vec![-1., -1., 1.]];
    /// let ids = lsh.store_vecs(vs);
    /// ```
    pub fn store_vecs(&mut self, vs: &[DataPoint]) -> Result<Vec<u32>> {
        self.validate_vec(&vs[0])?;
        self.hash_tables
            .as_mut()
            .unwrap()
            .increase_storage(vs.len());
        vs.iter().map(|x| self.store_vec(x)).collect()
    }

    fn query_bucket_union(&self, v: &DataPointSlice) -> Result<HashSet<u32>> {
        self.validate_vec(v)?;
        if self._multi_probe {
            return self.multi_probe_bucket_union(v);
        }

        let mut bucket_union = HashSet::default();

        for (i, proj) in self.hashers.iter().enumerate() {
            let hash = proj.hash_vec_query(v);
            self.process_bucket_union_result(&hash, i, &mut bucket_union)?;
        }
        Ok(bucket_union)
    }

    /// Query all buckets in the hash tables. The union of the matching buckets over the `L`
    /// hash tables is returned
    ///
    /// # Arguments
    /// * `v` - Query vector
    pub fn query_bucket(&self, v: &DataPointSlice) -> Result<Vec<&DataPoint>> {
        self.validate_vec(v)?;
        if self.only_index_storage {
            return Err(Error::Failed(
                "cannot query bucket, use query_bucket_ids".to_string(),
            ));
        }
        let bucket_union = self.query_bucket_union(v)?;

        bucket_union
            .iter()
            .map(|&idx| Ok(self.hash_tables.as_ref().unwrap().idx_to_datapoint(idx)?))
            .collect()
    }

    /// Query all buckets in the hash tables and return the data point indexes. The union of the
    /// matching buckets of `L` hash tables is returned.
    ///
    /// # Arguments
    /// * `v` - Query vector
    pub fn query_bucket_ids(&self, v: &DataPointSlice) -> Result<Vec<u32>> {
        self.validate_vec(v)?;
        let bucket_union = self.query_bucket_union(v)?;
        Ok(bucket_union.iter().copied().collect())
    }

    /// Delete data point from storage. This does not free memory as the storage vector isn't resized.
    ///
    /// # Arguments
    /// * `v` - Data point
    pub fn delete_vec(&mut self, v: &DataPointSlice) -> Result<()> {
        self.validate_vec(v)?;
        for (i, proj) in self.hashers.iter().enumerate() {
            let hash = proj.hash_vec_query(v);
            let mut ht = self.hash_tables.take().unwrap();
            ht.delete(hash, v, i).unwrap_or_default();
            self.hash_tables = Some(ht)
        }
        Ok(())
    }

    fn process_bucket_union_result(
        &self,
        hash: &Hash,
        hash_table_idx: usize,
        bucket_union: &mut HashSet<u32>,
    ) -> Result<()> {
        match self
            .hash_tables
            .as_ref()
            .unwrap()
            .query_bucket(hash, hash_table_idx)
        {
            Err(Error::NotFound) => Ok(()),
            Ok(bucket) => {
                *bucket_union = bucket_union.union(&bucket).copied().collect();
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    fn multi_probe_bucket_union(&self, v: &DataPointSlice) -> Result<HashSet<u32>> {
        self.validate_vec(v)?;
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
        Ok(bucket_union)
    }
}

impl<T: VecHash + Serialize> LSH<SqlTable, T> {
    pub fn commit(&mut self) -> Result<()> {
        let ht = self.hash_tables.as_mut().unwrap();
        ht.commit()?;
        Ok(())
    }

    pub fn init_transaction(&mut self) -> Result<()> {
        let ht = self.hash_tables.as_mut().unwrap();
        ht.init_transaction()?;
        Ok(())
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
    H: Serialize + DeserializeOwned + VecHash,
{
    pub fn load<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
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
    pub fn dump<P: AsRef<Path>>(&self, path: P) -> Result<()> {
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
