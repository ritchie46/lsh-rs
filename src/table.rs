use crate::hash::Hash;
use fnv::FnvHashMap as HashMap;

pub type DataPoint = Vec<f64>;
pub type Bucket = Vec<DataPoint>;
pub enum HashTableError {
    Failed,
}

/// Hashtable consisting of `L` Hash tables.
pub trait HashTables {
    /// # Arguments
    ///
    /// * `hash` - hashed vector.
    /// * `d` - Vector to store in the buckets.
    /// * `hash_table` - Number of the hash_table to store the vector. Ranging from 0 to L.
    fn put(&mut self, hash: Hash, d: DataPoint, hash_table: usize) -> Result<(), HashTableError>;

    /// Query the whole bucket
    fn query_bucket(&self) -> Result<Bucket, HashTableError>;

    /// Query the most similar
    fn query(&self, distance_fn: &dyn Fn(DataPoint) -> f64) -> Result<DataPoint, HashTableError>;
}

pub struct MemoryTable {
    hash_tables: Vec<HashMap<Hash, Bucket>>,
    n_hash_tables: usize,
}

impl MemoryTable {
    pub fn new(n_hash_tables: usize) -> MemoryTable {
        // TODO: Check the average number of vectors in the buckets.
        // this way the capacity can be approximated by the number of DataPoints that will
        // be stored.
        let hash_tables = vec![HashMap::default(); n_hash_tables];
        MemoryTable {
            hash_tables,
            n_hash_tables,
        }
    }
}

impl HashTables for MemoryTable {
    fn put(
        &mut self,
        hash: Hash,
        d: DataPoint,
        hash_table_idx: usize,
    ) -> Result<(), HashTableError> {
        let tbl = &mut self.hash_tables[hash_table_idx];
        let bucket = tbl.entry(hash).or_insert_with(|| Vec::new());
        bucket.push(d);
        Ok(())
    }

    /// Query the whole bucket
    fn query_bucket(&self) -> Result<Bucket, HashTableError> {
        Err(HashTableError::Failed)
    }

    /// Query the most similar
    fn query(&self, distance_fn: &dyn Fn(DataPoint) -> f64) -> Result<DataPoint, HashTableError> {
        Err(HashTableError::Failed)
    }
}

impl std::fmt::Debug for MemoryTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "hash_tables:\nhash, \t buckets\n");
        for ht in self.hash_tables.iter() {
            write!(f, "{:?}\n", ht);
        }
        Ok(())
    }
}
