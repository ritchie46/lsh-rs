use crate::data::Integer;
use crate::table::general::Bucket;
use crate::{data::Numeric, prelude::*, utils::create_rng};
use fnv::FnvHashSet;
use ndarray::prelude::*;
use num::Float;
use rand::Rng;
use rayon::prelude::*;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::marker::PhantomData;
use std::path::Path;

/// Wrapper for LSH functionality.
/// Can be initialized following the Builder pattern.
///
/// # Example
///
/// ```
/// use lsh_rs::prelude::*;
/// let n_projections = 9;
/// let n_hash_tables = 45;
/// let dim = 10;
/// let lsh: LshMem<_, f32> = LshMem::new(n_projections, n_hash_tables, dim)
///     .only_index()
///     .seed(1)
///     .srp().unwrap();
/// ```
/// # Builder pattern methods
/// The following methods can be used to change internal state during object initialization:
/// * [only_index](struct.LSH.html#method.only_index)
/// * [seed](struct.LSH.html#method.seed)
/// * [set_database_file](struct.LSH.html#method.set_database_file)
/// * [multi_probe](struct.LSH.html#method.multi_probe)
/// * [increase_storage](struct.LSH.html#method.increase_storage)
pub struct LSH<H, N, T, K = i8>
where
    N: Numeric,          // data type
    H: VecHash<N, K>,    // hashers
    T: HashTables<N, K>, // hash tables
    K: Integer,          // hashed data type
{
    /// Number of hash tables. `L` in literature.
    pub n_hash_tables: usize,
    /// Number of hash functions. `K` in literature.
    pub n_projections: usize,
    /// Hash functions.
    pub hashers: Vec<H>,
    /// Dimensions of p and q
    pub dim: usize,
    /// Storage data structure
    pub hash_tables: Option<T>,
    /// seed for hash functions. If 0, randomness is seeded from the os.
    _seed: u64,
    /// store only indexes and no data points.
    only_index_storage: bool,
    _multi_probe: bool,
    /// multi probe budget
    pub(crate) _multi_probe_budget: usize,
    _db_path: String,
    phantom: PhantomData<(N, K)>,
}

/// Create a new LSH instance. Used in the builder pattern
fn lsh_from_lsh<
    N: Numeric,
    T: HashTables<N, K>,
    H: VecHash<N, K> + Serialize + DeserializeOwned,
    K: Integer,
>(
    lsh: &mut LSH<H, N, T, K>,
    hashers: Vec<H>,
) -> Result<LSH<H, N, T, K>> {
    let mut ht = *T::new(lsh.n_hash_tables, lsh.only_index_storage, &lsh._db_path)?;

    // Load hashers if store hashers fails. (i.e. exists)
    let hashers = match ht.store_hashers(&hashers) {
        Ok(_) => hashers,
        Err(_) => match ht.load_hashers() {
            Err(e) => panic!("could not load hashers: {}", e),
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
        _multi_probe_budget: lsh._multi_probe_budget,
        _db_path: lsh._db_path.clone(),
        phantom: PhantomData,
    };
    Ok(lsh)
}

impl<N, T> LSH<SignRandomProjections<N>, N, T, i8>
where
    N: Numeric + DeserializeOwned,
    T: HashTables<N, i8>,
{
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

impl<N, T, K> LSH<L2<N, K>, N, T, K>
where
    N: Numeric + Float + DeserializeOwned,
    K: Integer + DeserializeOwned,
    T: HashTables<N, K>,
{
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

impl<N, T, K> LSH<MIPS<N, K>, N, T, K>
where
    N: Numeric + Float + DeserializeOwned,
    K: Integer + DeserializeOwned,
    T: HashTables<N, K>,
{
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
    pub fn mips(&mut self, r: f32, U: N, m: usize) -> Result<Self> {
        let mut rng = create_rng(self._seed);
        let mut hashers = Vec::with_capacity(self.n_hash_tables);

        for _ in 0..self.n_hash_tables {
            let seed = rng.gen();
            let hasher = MIPS::new(self.dim, r, U, m, self.n_projections, seed);
            hashers.push(hasher);
        }
        lsh_from_lsh(self, hashers)
    }

    /// Fit M parameter of the MIPS hasher. This needs to be done before the hasher can be used.
    pub fn fit(&mut self, vs: &[Vec<N>]) -> Result<()> {
        self.hashers.iter_mut().for_each(|h| h.fit(vs));
        Ok(())
    }
}

impl<N, T, K> LSH<MinHash<N, K>, N, T, K>
where
    N: Integer + DeserializeOwned,
    K: Integer + DeserializeOwned,
    T: HashTables<N, K>,
{
    pub fn minhash(&mut self) -> Result<Self> {
        let mut rng = create_rng(self._seed);
        let mut hashers = Vec::with_capacity(self.n_hash_tables);

        for _ in 0..self.n_hash_tables {
            let seed = rng.gen();
            let hasher = MinHash::new(self.n_projections, self.dim, seed);
            hashers.push(hasher);
        }
        lsh_from_lsh(self, hashers)
    }
}

impl<H, N, T, K> LSH<H, N, T, K>
where
    N: Numeric,
    H: VecHash<N, K> + Sync,
    T: HashTables<N, K> + Sync,
    K: Integer,
{
    /// Query bucket collision for a batch of data points in parallel.
    ///
    /// # Arguments
    /// * `vs` - Array of data points.
    pub fn query_bucket_ids_batch_par(&self, vs: &[Vec<N>]) -> Result<Vec<Vec<u32>>> {
        vs.into_par_iter()
            .map(|v| self.query_bucket_ids(v))
            .collect()
    }

    /// Query bucket collision for a batch of data points in parallel.
    ///
    /// # Arguments
    /// * `vs` - Array of data points.
    pub fn query_bucket_ids_batch_arr_par(&self, vs: ArrayView2<N>) -> Result<Vec<Vec<u32>>> {
        vs.axis_iter(Axis(0))
            .into_par_iter()
            .map(|v| self.query_bucket_ids(v.as_slice().unwrap()))
            .collect()
    }
}

impl<H, N, T, K> LSH<H, N, T, K>
where
    H: VecHash<N, K>,
    N: Numeric + Sync,
    T: HashTables<N, K>,
    K: Integer,
{
    /// Store multiple vectors in storage. Before storing the storage capacity is possibly
    /// increased to match the data points.
    ///
    /// # Arguments
    /// * `vs` - Array of data points.
    ///
    /// # Examples
    ///```
    /// use lsh_rs::prelude::*;
    /// let mut lsh = LshSql::new(5, 10, 3).srp().unwrap();
    /// let vs = &[vec![2., 3., 4.],
    ///            vec![-1., -1., 1.]];
    /// let ids = lsh.store_vecs(vs);
    /// ```
    pub fn store_vecs(&mut self, vs: &[Vec<N>]) -> Result<Vec<u32>> {
        self.validate_vec(&vs[0])?;
        self.hash_tables
            .as_mut()
            .unwrap()
            .increase_storage(vs.len());

        let mut ht = self.hash_tables.take().unwrap();
        let mut insert_idx = Vec::with_capacity(vs.len());
        for (i, proj) in self.hashers.iter().enumerate() {
            for v in vs.iter() {
                let hash = proj.hash_vec_put(v);
                match (ht.put(hash, v, i), i) {
                    // only for the first hash table save the index as it will be the same for all
                    (Ok(idx), 0) => insert_idx.push(idx),
                    (Err(e), _) => return Err(e),
                    _ => {}
                }
            }
        }
        self.hash_tables.replace(ht);
        Ok(insert_idx)
    }

    /// Store a 2D array in storage. Before storing the storage capacity is possibly
    /// increased to match the data points.
    ///
    /// # Arguments
    /// * `vs` - Array of data points.
    ///
    /// # Examples
    ///```
    /// use lsh_rs::prelude::*;
    /// use ndarray::prelude::*;
    /// let mut lsh = LshMem::new(5, 10, 3).srp().unwrap();
    /// let vs = array![[1., 2., 3.], [4., 5., 6.]];
    /// let ids = lsh.store_array(vs.view());
    /// ```
    pub fn store_array(&mut self, vs: ArrayView2<N>) -> Result<Vec<u32>> {
        self.validate_vec(vs.slice(s![0, ..]).as_slice().unwrap())?;
        self.hash_tables
            .as_mut()
            .unwrap()
            .increase_storage(vs.len());

        let mut ht = self.hash_tables.take().unwrap();
        let mut insert_idx = Vec::with_capacity(vs.len());
        for (i, proj) in self.hashers.iter().enumerate() {
            for v in vs.axis_iter(Axis(0)) {
                let hash = proj.hash_vec_put(v.as_slice().unwrap());
                match (ht.put(hash, v.as_slice().unwrap(), i), i) {
                    // only for the first hash table save the index as it will be the same for all
                    (Ok(idx), 0) => insert_idx.push(idx),
                    (Err(e), _) => return Err(e),
                    _ => {}
                }
            }
        }
        self.hash_tables.replace(ht);
        Ok(insert_idx)
    }
}

impl<H, N, T, K> LSH<H, N, T, K>
where
    N: Numeric,
    H: VecHash<N, K>,
    T: HashTables<N, K>,
    K: Integer,
{
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
            _multi_probe_budget: 16,
            _db_path: "./lsh.db3".to_string(),
            phantom: PhantomData,
        };
        lsh
    }

    pub(crate) fn validate_vec<A>(&self, v: &[A]) -> Result<()> {
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
    /// * `budget` - The number of probes (close hashes) will be executed per query.
    pub fn multi_probe(&mut self, budget: usize) -> &mut Self {
        self._multi_probe = true;
        self._multi_probe_budget = budget;
        self
    }

    pub fn base(&mut self) -> &mut Self {
        self._multi_probe = false;
        self
    }

    /// Increase storage of the `hash_tables` backend. This can reduce system calls.
    ///
    /// # Arguments
    /// * `upper_bound` - The maximum storage capacity required.
    pub fn increase_storage(&mut self, upper_bound: usize) -> Result<&mut Self> {
        self.hash_tables
            .as_mut()
            .unwrap()
            .increase_storage(upper_bound);
        Ok(self)
    }

    /// Location where the database file should be written/ can be found.
    /// This only has effect with the `SqlTable` backend.
    ///
    /// # Arguments
    /// * `path` - File path.
    pub fn set_database_file(&mut self, path: &str) -> &mut Self {
        self._db_path = path.to_string();
        self
    }

    /// Collects statistics of the buckets in the `hash_tables`.
    /// # Statistics
    /// * average bucket length
    /// * minimal bucket length
    /// * maximum bucket length
    /// * bucket lenght standard deviation
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
    /// use lsh_rs::prelude::*;
    /// let mut lsh = LshMem::new(5, 10, 3).srp().unwrap();
    /// let v = &[2., 3., 4.];
    /// let id = lsh.store_vec(v);
    /// ```
    pub fn store_vec(&mut self, v: &[N]) -> Result<u32> {
        self.validate_vec(v)?;

        let mut idx = 0;
        let mut ht = self.hash_tables.take().unwrap();
        for (i, proj) in self.hashers.iter().enumerate() {
            let hash = proj.hash_vec_put(v);
            idx = ht.put(hash, &v, i)?;
        }
        self.hash_tables.replace(ht);
        Ok(idx)
    }

    /// Update a data point in the `hash_tables`.
    ///
    /// # Arguments
    /// * `idx` - Id of the hash that needs to be updated.
    /// * `new_v` - New data point that needs to be hashed.
    /// * `old_v` - Old data point. Needed to remove the old hash.
    pub fn update_by_idx(&mut self, idx: u32, new_v: &[N], old_v: &[N]) -> Result<()> {
        let mut ht = self.hash_tables.take().unwrap();
        for (i, proj) in self.hashers.iter().enumerate() {
            let new_hash = proj.hash_vec_put(new_v);
            let old_hash = proj.hash_vec_put(old_v);
            ht.update_by_idx(&old_hash, new_hash, idx, i)?;
        }
        self.hash_tables.replace(ht);
        Ok(())
    }

    fn query_bucket_union(&self, v: &[N]) -> Result<Bucket> {
        self.validate_vec(v)?;
        if self._multi_probe {
            return self.multi_probe_bucket_union(v);
        }

        let mut bucket_union = FnvHashSet::default();

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
    pub fn query_bucket(&self, v: &[N]) -> Result<Vec<&Vec<N>>> {
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
    pub fn query_bucket_ids(&self, v: &[N]) -> Result<Vec<u32>> {
        self.validate_vec(v)?;
        let bucket_union = self.query_bucket_union(v)?;
        Ok(bucket_union.iter().copied().collect())
    }

    /// Query bucket collision for a batch of data points.
    ///
    /// # Arguments
    /// * `vs` - Array of data points.
    pub fn query_bucket_ids_batch(&self, vs: &[Vec<N>]) -> Result<Vec<Vec<u32>>> {
        vs.iter().map(|v| self.query_bucket_ids(v)).collect()
    }

    /// Query bucket collision for a batch of data points.
    ///
    /// # Arguments
    /// * `vs` - Array of data points.
    pub fn query_bucket_ids_batch_arr(&self, vs: ArrayView2<N>) -> Result<Vec<Vec<u32>>> {
        vs.axis_iter(Axis(0))
            .map(|v| self.query_bucket_ids(v.as_slice().unwrap()))
            .collect()
    }

    /// Delete data point from storage. This does not free memory as the storage vector isn't resized.
    ///
    /// # Arguments
    /// * `v` - Data point
    pub fn delete_vec(&mut self, v: &[N]) -> Result<()> {
        self.validate_vec(v)?;
        for (i, proj) in self.hashers.iter().enumerate() {
            let hash = proj.hash_vec_query(v);
            let mut ht = self.hash_tables.take().unwrap();
            ht.delete(&hash, v, i).unwrap_or_default();
            self.hash_tables = Some(ht)
        }
        Ok(())
    }

    pub(crate) fn process_bucket_union_result(
        &self,
        hash: &[K],
        hash_table_idx: usize,
        bucket_union: &mut Bucket,
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
}

#[cfg(feature = "sqlite")]
impl<N, H, K> LSH<H, N, SqlTable<N, K>, K>
where
    N: Numeric,
    H: VecHash<N, K> + Serialize,
    K: Integer,
{
    /// Commit SqlTable backend
    pub fn commit(&mut self) -> Result<()> {
        let ht = self.hash_tables.as_mut().unwrap();
        ht.commit()?;
        Ok(())
    }

    /// Init transaction of SqlTable backend.
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

impl<H, N, K> LSH<H, N, MemoryTable<N, K>, K>
where
    H: Serialize + DeserializeOwned + VecHash<N, K>,
    N: Numeric + DeserializeOwned,
    K: Integer + DeserializeOwned,
{
    /// Deserialize MemoryTable backend
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

    /// Serialize MemoryTable backend
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
