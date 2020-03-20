use crate::hash::Hash;
use crate::{DataPoint, DataPointSlice};
use fnv::FnvHashSet as HashSet;

/// Bucket contains indexes to VecStore
pub type Bucket = HashSet<u32>;

#[derive(Debug)]
pub enum HashTableError {
    Failed,
    NotFound,
    TableNotExist,
    NotImplemented,
}

/// Hashtable consisting of `L` Hash tables.
pub trait HashTables {
    fn new(n_hash_tables: usize, only_index_storage: bool, dump_path: &str) -> Self;

    /// # Arguments
    ///
    /// * `hash` - hashed vector.
    /// * `d` - Vector to store in the buckets.
    /// * `hash_table` - Number of the hash_table to store the vector. Ranging from 0 to L.
    fn put(
        &mut self,
        hash: Hash,
        d: &DataPointSlice,
        hash_table: usize,
    ) -> Result<u32, HashTableError>;

    fn delete(
        &mut self,
        hash: Hash,
        d: &DataPointSlice,
        hash_table: usize,
    ) -> Result<(), HashTableError>;

    /// Query the whole bucket
    fn query_bucket(&self, hash: &Hash, hash_table: usize) -> Result<Bucket, HashTableError>;

    fn idx_to_datapoint(&self, idx: u32) -> Result<&DataPoint, HashTableError>;

    fn increase_storage(&mut self, size: usize);

    fn describe(&self);
}
