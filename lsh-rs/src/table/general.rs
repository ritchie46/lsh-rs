use crate::hash::Hash;
use crate::{DataPoint, DataPointSlice, VecHash};
use fnv::FnvHashSet as HashSet;
use serde::de::DeserializeOwned;
use serde::Serialize;

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
    fn new(n_hash_tables: usize, only_index_storage: bool, db_dir: &str) -> Self;

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

    fn increase_storage(&mut self, size: usize) {}

    fn describe(&self) {}

    // Should fail if hashers already stored.
    fn store_hashers<H: VecHash + Serialize>(
        &mut self,
        hashers: &[H],
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    // If store_hashers fails, load_hasher can be executed
    fn load_hashers<H: VecHash + DeserializeOwned>(
        &self,
    ) -> Result<(Vec<H>), Box<dyn std::error::Error>> {
        Err(Box::new(std::fmt::Error))
    }
}
