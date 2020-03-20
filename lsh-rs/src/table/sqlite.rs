use super::general::{Bucket, HashTableError, HashTables};
use crate::hash::{Hash, HashPrimitive};
use crate::{DataPoint, DataPointSlice};
use rusqlite::{params, Connection};
use std::mem;

fn hash_to_blob(hash: &[i32]) -> &[u8] {
    let data = hash.as_ptr() as *const u8;
    unsafe { std::slice::from_raw_parts(data, hash.len() * std::mem::size_of::<HashPrimitive>()) }
}

fn blob_to_hash(blob: &[u8]) -> &[i32] {
    let data = blob.as_ptr() as *const i32;
    unsafe { std::slice::from_raw_parts(data, blob.len() / std::mem::size_of::<HashPrimitive>()) }
}

///
/// Requirement on Debian: libsqlite3-dev
pub struct SqlTable {
    n_hash_tables: usize,
    only_index_storage: bool, // for now only supported
    counter: u32,
    conn: Connection,
    table_names: Vec<String>,
}

impl SqlTable {
    pub fn new(n_hash_tables: usize, only_index_storage: bool) -> Self {
        let conn = Connection::open_in_memory().expect("could not open sqlite");

        let mut table_names = Vec::with_capacity(n_hash_tables);
        for idx in 0..n_hash_tables {
            table_names.push(format!("hash_table_{}", idx));
            conn.execute(
                &format!(
                    "CREATE TABLE hash_table_{} (
             hash       BLOB PRIMARY KEY,
             id         INTEGER
            )
                ",
                    idx
                ),
                params![],
            )
            .expect("could not create table");
        }
        SqlTable {
            n_hash_tables,
            only_index_storage,
            counter: 0,
            conn,
            table_names,
        }
    }
}

impl HashTables for SqlTable {
    fn put(
        &mut self,
        hash: Hash,
        _d: &DataPointSlice,
        hash_table: usize,
    ) -> Result<u32, HashTableError> {
        // the unique id of the unique vector
        let idx = self.counter;

        // Get the table name to store this id
        let opt = self.table_names.get(hash_table);
        let tbl_name = if let Some(tbl_name) = opt {
            tbl_name
        } else {
            return Err(HashTableError::TableNotExist);
        };

        let blob = hash_to_blob(&hash);
        let r = self.conn.execute(
            &format!(
                "
INSERT INTO {} (hash, id)
VALUES (?1, ?2)
        ",
                tbl_name
            ),
            params![blob, idx],
        );

        // Once we've traversed the last table we increment the id counter.
        if hash_table == self.n_hash_tables - 1 {
            self.counter += 1
        };
        match r {
            Ok(_) => return Ok(idx),
            Err(e) => panic!(format!("could not insert in db: {:?}", e)),
        }
    }

    fn delete(
        &mut self,
        hash: Hash,
        d: &DataPointSlice,
        hash_table: usize,
    ) -> Result<(), HashTableError> {
        Ok(())
    }

    /// Query the whole bucket
    fn query_bucket(&self, hash: &Hash, hash_table: usize) -> Result<&Bucket, HashTableError> {
        Err(HashTableError::NotImplemented)
    }

    fn idx_to_datapoint(&self, idx: u32) -> Result<&DataPoint, HashTableError> {
        Err(HashTableError::NotImplemented)
    }

    fn increase_storage(&mut self, size: usize) {}

    fn describe(&self) {}
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_sql_table_init() {
        let sql = SqlTable::new(1, true);
        let mut stmt = sql
            .conn
            .prepare(&format!("SELECT * FROM {}", sql.table_names[0]))
            .expect("query failed");
        let r = stmt.query(params![]).expect("query failed");
    }

    #[test]
    fn test_sql_crud() {
        let mut sql = SqlTable::new(1, true);
        let v = vec![1., 2.];
        let hash: Hash = vec![0, 2];
        sql.put(hash, &v, 0);
    }

    #[test]
    fn test_blob_hash_casting() {
        for hash in vec![
            &vec![2, 3, 4],
            &vec![-200, 687, 1245],
            &vec![1, 2, 3, 4, 5, 6],
            &vec![-8979875, -2, -3, 1, 2, 3, 4, 5, 6],
        ] {
            let hash = &hash[..];
            let blob = hash_to_blob(hash);
            let hash_back = blob_to_hash(blob);
            assert_eq!(hash, hash_back)
        }
    }
}
