#![cfg(feature = "sqlite")]
use super::sqlite::SqlTable;
use crate::data::Integer;
use crate::prelude::*;
use crate::{data::Numeric, table::general::Bucket, HashTables};
use fnv::FnvHashSet;
use std::ops::{Deref, DerefMut};
use std::path::Path;

/// In memory Sqlite backend for [LSH](struct.LSH.html).
pub struct SqlTableMem<N, K>
where
    N: Numeric,
    K: Integer,
{
    sql_table: SqlTable<N, K>,
}

impl<N, K> SqlTableMem<N, K>
where
    N: Numeric,
    K: Integer,
{
    pub fn to_db<P: AsRef<Path>>(&mut self, db_path: P) -> Result<()> {
        let mut new_con = rusqlite::Connection::open(db_path)?;
        {
            let backup = rusqlite::backup::Backup::new(&self.conn, &mut new_con)?;
            backup.step(-1)?;
        }
        self.conn = new_con;
        self.committed.set(true);
        Ok(())
    }
}

impl<N, K> Deref for SqlTableMem<N, K>
where
    N: Numeric,
    K: Integer,
{
    type Target = SqlTable<N, K>;

    fn deref(&self) -> &SqlTable<N, K> {
        &self.sql_table
    }
}

impl<N, K> DerefMut for SqlTableMem<N, K>
where
    N: Numeric,
    K: Integer,
{
    fn deref_mut(&mut self) -> &mut SqlTable<N, K> {
        &mut self.sql_table
    }
}

impl<N, K> HashTables<N, K> for SqlTableMem<N, K>
where
    N: Numeric,
    K: Integer,
{
    fn new(n_hash_tables: usize, only_index_storage: bool, _db_path: &str) -> Result<Box<Self>> {
        let conn = rusqlite::Connection::open_in_memory()?;
        let sql_table = SqlTable::init_from_conn(n_hash_tables, only_index_storage, conn)?;
        Ok(Box::new(SqlTableMem { sql_table }))
    }

    /// # Arguments
    ///
    /// * `hash` - hashed vector.
    /// * `d` - Vector to store in the buckets.
    /// * `hash_table` - Number of the hash_table to store the vector. Ranging from 0 to L.
    fn put(&mut self, hash: Vec<K>, d: &[N], hash_table: usize) -> Result<u32> {
        self.sql_table.put(hash, d, hash_table)
    }

    fn delete(&mut self, hash: &[K], d: &[N], hash_table: usize) -> Result<()> {
        self.sql_table.delete(hash, d, hash_table)
    }

    /// Query the whole bucket
    fn query_bucket(&self, hash: &[K], hash_table: usize) -> Result<Bucket> {
        self.sql_table.query_bucket(hash, hash_table)
    }

    fn idx_to_datapoint(&self, idx: u32) -> Result<&Vec<N>> {
        self.sql_table.idx_to_datapoint(idx)
    }

    fn describe(&self) -> Result<String> {
        self.sql_table.describe()
    }

    fn get_unique_hash_int(&self) -> FnvHashSet<i32> {
        self.sql_table.get_unique_hash_int()
    }
}
