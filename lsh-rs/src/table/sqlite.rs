#![cfg(feature = "sqlite")]
use super::general::Bucket;
use crate::constants::DESCRIBE_MAX;
use crate::data::{Integer, Numeric};
use crate::prelude::*;
use fnv::FnvHashSet;
use rusqlite::{params, Connection, NO_PARAMS};
use serde::de::DeserializeOwned;
use serde::export::PhantomData;
use serde::Serialize;
use std::cell::Cell;

fn vec_to_blob<T>(hash: &[T]) -> &[u8] {
    let data = hash.as_ptr() as *const u8;
    unsafe { std::slice::from_raw_parts(data, hash.len() * std::mem::size_of::<T>()) }
}

fn blob_to_vec<T>(blob: &[u8]) -> &[T] {
    let data = blob.as_ptr() as *const T;
    unsafe { std::slice::from_raw_parts(data, blob.len() / std::mem::size_of::<T>()) }
}

fn query_bucket(blob: &[u8], table_name: &str, connection: &Connection) -> Result<Bucket> {
    let mut stmt = connection.prepare_cached(&format!(
        "
SELECT (id) FROM {}
WHERE hash = ?
        ",
        table_name
    ))?;
    let mut rows = stmt.query(params![blob])?;

    let mut bucket = FnvHashSet::default();
    while let Some(row) = rows.next()? {
        bucket.insert(row.get(0)?);
    }
    Ok(bucket)
}

fn make_table(table_name: &str, connection: &Connection) -> Result<()> {
    connection.execute_batch(&format!(
        "CREATE TABLE IF NOT EXISTS {} (
             hash       BLOB,
             id         INTEGER
            )
                ",
        table_name
    ))?;
    Ok(())
}

fn insert_table<K>(
    table_name: &str,
    hash: &Vec<K>,
    idx: u32,
    connection: &Connection,
) -> Result<usize> {
    let blob = vec_to_blob(hash);
    let mut stmt = connection.prepare_cached(&format!(
        "
INSERT INTO {} (hash, id)
VALUES (?1, ?2)
        ",
        table_name
    ))?;
    let idx = stmt.execute(params![blob, idx])?;
    Ok(idx)
}

fn hash_table_stats(
    table_name: &str,
    limit: u32,
    conn: &Connection,
) -> Result<(f64, f64, u32, u32)> {
    let mut stmt = conn.prepare_cached(&format!(
        "
SELECT
	avg(c) as mean,
	avg(c * c) - avg(c) * avg(c) as variance,
	min(c) as minimum,
	max(c) as maximum

FROM (
	SELECT count(id) as c
	FROM {}
	GROUP BY hash
	LIMIT ?
);
    ",
        table_name
    ))?;
    let out = stmt.query_row(params![limit], |row| {
        let mean: f64 = row.get(0)?;
        let variance: f64 = row.get(1)?;
        let stdev = variance.powf(0.5);
        let minimum: u32 = row.get(2)?;
        let maximum: u32 = row.get(3)?;
        Ok((mean, stdev, minimum, maximum))
    })?;
    Ok(out)
}

/// Sqlite backend for [LSH](struct.LSH.html).
///
/// State will be save during sessions. The database is automatically
/// loaded if [LSH](struct.LSH.html) can find the database file (defaults to `./lsh.db3`.
pub struct SqlTable<N, K>
where
    N: Numeric,
    K: Integer,
{
    n_hash_tables: usize,
    only_index_storage: bool, // for now only supported
    counter: u32,
    pub conn: Connection,
    table_names: Vec<String>,
    pub committed: Cell<bool>,
    phantom: PhantomData<(N, K)>,
}

fn fmt_table_name(hash_table: usize) -> String {
    format!("hash_table_{}", hash_table)
}

fn get_table_names(n_hash_tables: usize) -> Vec<String> {
    let mut table_names = Vec::with_capacity(n_hash_tables);
    for idx in 0..n_hash_tables {
        let table_name = fmt_table_name(idx);
        table_names.push(table_name);
    }
    table_names
}

fn get_unique_hash_int(n_hash_tables: usize, conn: &Connection) -> Result<FnvHashSet<i32>> {
    let mut hash_numbers = FnvHashSet::default();
    for table_name in get_table_names(n_hash_tables) {
        let mut stmt = conn.prepare(&format!["SELECT hash FROM {} LIMIT 100;", table_name])?;
        let mut rows = stmt.query(NO_PARAMS)?;

        while let Some(r) = rows.next()? {
            let blob: Vec<u8> = r.get(0)?;
            let hash = blob_to_vec(&blob);
            hash.iter().for_each(|&v| {
                hash_numbers.insert(v);
            })
        }
    }
    Ok(hash_numbers)
}

fn init_table(conn: &Connection, table_names: &[String]) -> Result<()> {
    for table_name in table_names {
        make_table(&table_name, &conn)?;
    }
    Ok(())
}

fn init_db_setttings(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "PRAGMA journal_mode = OFF;
    PRAGMA synchronous = OFF;
    PRAGMA cache_size = 100000;
    PRAGMA main.locking_mode=EXCLUSIVE;",
    )?;
    Ok(())
}

impl<N, K> SqlTable<N, K>
where
    N: Numeric,
    K: Integer,
{
    fn get_table_name_put(&self, hash_table: usize) -> Result<&str> {
        let opt = self.table_names.get(hash_table);
        match opt {
            Some(tbl_name) => Ok(&tbl_name[..]),
            None => Err(Error::TableNotExist),
        }
    }

    pub fn init_from_conn(
        n_hash_tables: usize,
        only_index_storage: bool,
        conn: Connection,
    ) -> Result<Self> {
        let table_names = get_table_names(n_hash_tables);
        init_db_setttings(&conn)?;
        init_table(&conn, &table_names)?;
        let sql = SqlTable {
            n_hash_tables,
            only_index_storage,
            counter: 0,
            conn,
            table_names,
            committed: Cell::new(false),
            phantom: PhantomData,
        };
        sql.init_transaction()?;
        Ok(sql)
    }

    pub fn commit(&self) -> Result<()> {
        if !self.committed.replace(true) {
            self.conn.execute_batch("COMMIT TRANSACTION;")?;
        }
        Ok(())
    }

    pub fn init_transaction(&self) -> Result<()> {
        self.committed.set(false);
        self.conn.execute_batch("BEGIN TRANSACTION;")?;
        Ok(())
    }

    pub fn to_mem(&mut self) -> Result<()> {
        let mut new_con = rusqlite::Connection::open_in_memory()?;
        {
            let backup = rusqlite::backup::Backup::new(&self.conn, &mut new_con)?;
            backup.step(-1)?;
        }
        self.conn = new_con;
        self.committed.set(true);
        Ok(())
    }

    pub fn index_hash(&self) -> Result<()> {
        self.commit()?;
        for tbl_name in get_table_names(self.n_hash_tables) {
            self.conn.execute_batch(&format!(
                "
                CREATE INDEX hash_index_{}
                ON {} (hash);",
                tbl_name, tbl_name
            ))?;
        }
        Ok(())
    }
}

impl<N, K> HashTables<N, K> for SqlTable<N, K>
where
    N: Numeric,
    K: Integer,
{
    fn new(n_hash_tables: usize, only_index_storage: bool, db_path: &str) -> Result<Box<Self>> {
        let path = std::path::Path::new(db_path);
        let conn = Connection::open(path)?;
        SqlTable::init_from_conn(n_hash_tables, only_index_storage, conn).map(|tbl| Box::new(tbl))
    }

    fn put(&mut self, hash: Vec<K>, _d: &[N], hash_table: usize) -> Result<u32> {
        // the unique id of the unique vector
        let idx = self.counter;

        // Get the table name to store this id
        let table_name = self.get_table_name_put(hash_table)?;
        let r = insert_table(&table_name, &hash, idx, &self.conn);

        // Once we've traversed the last table we increment the id counter.
        if hash_table == self.n_hash_tables - 1 {
            self.counter += 1
        };

        match r {
            Ok(_) => Ok(idx),
            Err(Error::SqlFailure(_)) => Ok(idx), // duplicates
            Err(e) => Err(Error::Failed(format!("{:?}", e))),
        }
    }

    /// Query the whole bucket
    fn query_bucket(&self, hash: &[K], hash_table: usize) -> Result<Bucket> {
        self.commit()?;
        let table_name = fmt_table_name(hash_table);
        let blob = vec_to_blob(hash);
        let res = query_bucket(blob, &table_name, &self.conn);

        match res {
            Ok(bucket) => Ok(bucket),
            Err(e) => Err(Error::Failed(format!("{:?}", e))),
        }
    }

    fn describe(&self) -> Result<String> {
        let mut stmt = self.conn.prepare(
            r#"SELECT count(*) FROM sqlite_master
WHERE type='table' AND type LIKE '%hash%';"#,
        )?;

        let row: String = stmt.query_row(NO_PARAMS, |row| {
            let i: i64 = row.get_unwrap(0);
            Ok(i.to_string())
        })?;
        let mut out = String::from(format!("No. of tables: {}\n", row));

        out.push_str("Unique hash values:\n");
        let hv = get_unique_hash_int(self.n_hash_tables, &self.conn).unwrap();
        out.push_str(&format!("{:?}", hv));

        let tables = get_table_names(self.n_hash_tables);
        let mut avg = Vec::with_capacity(self.n_hash_tables);
        let mut std_dev = Vec::with_capacity(self.n_hash_tables);
        let mut min = Vec::with_capacity(self.n_hash_tables);
        let mut max = Vec::with_capacity(self.n_hash_tables);

        // maximum 3 tables will be used in stats
        let i = std::cmp::min(3, self.n_hash_tables);
        for table_name in &tables[..i] {
            let stats = hash_table_stats(&table_name, DESCRIBE_MAX, &self.conn)?;
            avg.push(stats.0);
            std_dev.push(stats.1);
            min.push(stats.2);
            max.push(stats.3);
        }
        out.push_str("\nHash collisions:\n");
        out.push_str(&format!("avg:\t{:?}\n", avg));
        out.push_str(&format!("std-dev:\t{:?}\n", std_dev));
        out.push_str(&format!("min:\t{:?}\n", min));
        out.push_str(&format!("max:\t{:?}\n", max));
        Ok(out)
    }

    fn store_hashers<H: VecHash<N, K> + Serialize>(&mut self, hashers: &[H]) -> Result<()> {
        let buf: Vec<u8> = bincode::serialize(hashers)?;

        // fails if already exists
        self.conn.execute_batch(
            "CREATE TABLE state (
            hashers     BLOB
        )",
        )?;
        let mut stmt = self
            .conn
            .prepare("INSERT INTO state (hashers) VALUES (?1)")?;

        // unlock database by committing any running transaction.
        self.commit()?;
        stmt.execute(params![buf])?;
        self.init_transaction()?;
        Ok(())
    }

    fn load_hashers<H: VecHash<N, K> + DeserializeOwned>(&self) -> Result<Vec<H>> {
        let mut stmt = self.conn.prepare("SELECT * FROM state;")?;
        let buf: Vec<u8> = stmt.query_row(NO_PARAMS, |row| {
            let v: Vec<u8> = row.get_unwrap(0);
            Ok(v)
        })?;
        let hashers: Vec<H> = bincode::deserialize(&buf)?;
        Ok(hashers)
    }

    fn get_unique_hash_int(&self) -> FnvHashSet<i32> {
        get_unique_hash_int(self.n_hash_tables, &self.conn).unwrap()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::table::sqlite_mem::SqlTableMem;

    #[test]
    fn test_sql_table_init() {
        let sql = SqlTableMem::<f32, i8>::new(1, true, ".").unwrap();
        let mut stmt = sql
            .conn
            .prepare(&format!("SELECT * FROM {}", sql.table_names[0]))
            .expect("query failed");
        stmt.query(NO_PARAMS).expect("query failed");
    }

    #[test]
    fn test_sql_crud() {
        let mut sql = *SqlTableMem::new(1, true, ".").unwrap();
        let v = vec![1., 2.];
        for hash in &[vec![1, 2], vec![2, 3]] {
            sql.put(hash.clone(), &v, 0).unwrap();
        }
        // make one hash collision by repeating one hash
        let hash = vec![1, 2];
        sql.put(hash.clone(), &v, 0).unwrap();
        let bucket = sql.query_bucket(&hash, 0);
        println!("{:?}", &bucket);
        match bucket {
            Ok(b) => assert!(b.contains(&0)),
            _ => assert!(false),
        }
    }

    #[test]
    fn test_blob_hash_casting() {
        for hash in vec![
            &vec![2, 3, 4],
            &vec![-124, 32, 89],
            &vec![1, 2, 3, 4, 5, 6],
            &vec![-12, -2, -3, 1, 2, 3, 4, 5, 6],
        ] {
            let hash = &hash[..];
            let blob = vec_to_blob(hash);
            let hash_back: &[i32] = blob_to_vec(blob);
            assert_eq!(hash, hash_back)
        }
    }

    #[test]
    fn test_in_mem_to_disk() {
        let mut sql = *SqlTableMem::<f32, i8>::new(1, true, ".").unwrap();
        let v = vec![1., 2.];
        for hash in &[vec![1, 2], vec![2, 3]] {
            sql.put(hash.clone(), &v, 0).unwrap();
        }
        sql.commit().unwrap();
        let p = "./delete.db3";
        sql.to_db(p).unwrap();

        let mut sql = SqlTable::<f32, i8>::new(1, true, p).unwrap();
        sql.to_mem().unwrap();
        assert_eq!(sql.query_bucket(&vec![1, 2], 0).unwrap().take(&0), Some(0));
        std::fs::remove_file(p).unwrap();
    }
}
