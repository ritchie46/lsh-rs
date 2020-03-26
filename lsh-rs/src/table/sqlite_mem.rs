use super::sqlite::SqlTable;
use crate::{
    hash::{Hash, HashPrimitive},
    table::general::Bucket,
    DataPoint, DataPointSlice, Error, HashTables, Result,
};
use fnv::FnvHashSet;
use std::ops::{Deref, DerefMut};

pub struct SqlTableMem {
    sql_table: SqlTable,
}

impl Deref for SqlTableMem {
    type Target = SqlTable;

    fn deref(&self) -> &SqlTable {
        &self.sql_table
    }
}

impl DerefMut for SqlTableMem {
    fn deref_mut(&mut self) -> &mut SqlTable {
        &mut self.sql_table
    }
}

impl HashTables for SqlTableMem {
    fn new(n_hash_tables: usize, only_index_storage: bool, db_dir: &str) -> Result<Box<Self>> {
        let sql_table = *SqlTable::new(n_hash_tables, only_index_storage, db_dir)?;
        Ok(Box::new(SqlTableMem { sql_table }))
    }

    /// # Arguments
    ///
    /// * `hash` - hashed vector.
    /// * `d` - Vector to store in the buckets.
    /// * `hash_table` - Number of the hash_table to store the vector. Ranging from 0 to L.
    fn put(&mut self, hash: Hash, d: &DataPointSlice, hash_table: usize) -> Result<u32> {
        self.sql_table.put(hash, d, hash_table)
    }

    fn delete(&mut self, hash: Hash, d: &DataPointSlice, hash_table: usize) -> Result<()> {
        self.sql_table.delete(hash, d, hash_table)
    }

    /// Query the whole bucket
    fn query_bucket(&self, hash: &Hash, hash_table: usize) -> Result<Bucket> {
        self.sql_table.query_bucket(hash, hash_table)
    }

    fn idx_to_datapoint(&self, idx: u32) -> Result<&DataPoint> {
        self.sql_table.idx_to_datapoint(idx)
    }

    fn describe(&self) {
        self.sql_table.describe()
    }

    fn get_unique_hash_int(&self) -> FnvHashSet<HashPrimitive> {
        self.sql_table.get_unique_hash_int()
    }
}
