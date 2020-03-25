use crate::hash::{Hash, HashPrimitive};
use crate::{DataPoint, DataPointSlice, VecHash};
use fnv::{FnvHashSet as HashSet, FnvHashSet};
use serde::de::DeserializeOwned;
use serde::Serialize;
use thiserror::Error;

/// Bucket contains indexes to VecStore
pub type Bucket = HashSet<u32>;

#[derive(Debug, Error)]
pub enum HashTableError {
    #[error("Something went wrong")]
    Failed(String),
    #[error("Vector not found")]
    NotFound,
    #[error("Table does not exist")]
    TableNotExist,
    #[error("Not implemented")]
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
    fn load_hashers<H: VecHash + DeserializeOwned>(&self) -> anyhow::Result<(Vec<H>)> {
        // just chose an error to make a default trait implementation
        Err(anyhow::Error::new(HashTableError::NotImplemented))
    }

    fn get_unique_hash_int(&self) -> FnvHashSet<HashPrimitive>;
}
