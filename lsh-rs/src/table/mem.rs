use crate::data::Integer;
use crate::{
    constants::DESCRIBE_MAX,
    data::Numeric,
    prelude::*,
    table::general::{Bucket, HashTables},
    utils::{all_eq, increase_capacity},
};
use fnv::{FnvHashMap as HashMap, FnvHashSet};
use serde::{Deserialize, Serialize};
use std::iter::FromIterator;

/// Indexible vector storage.
/// indexes will be stored in hashtables. The original vectors can be looked up in this data structure.
#[derive(Debug, Deserialize, Serialize)]
pub struct VecStore<N> {
    pub map: Vec<Vec<N>>,
}

impl<N: Numeric> VecStore<N> {
    fn push(&mut self, d: Vec<N>) -> u32 {
        self.map.push(d);
        (self.map.len() - 1) as u32
    }

    fn position(&self, d: &[N]) -> Option<u32> {
        self.map.iter().position(|x| all_eq(x, d)).map(|x| x as u32)
    }

    fn get(&self, idx: u32) -> &Vec<N> {
        &self.map[idx as usize]
    }

    fn increase_storage(&mut self, size: usize) {
        increase_capacity(size, &mut self.map);
    }
}

/// In memory backend for [LSH](struct.LSH.html).
#[derive(Deserialize, Serialize)]
pub struct MemoryTable<N, K>
where
    N: Numeric,
    K: Integer,
{
    hash_tables: Vec<HashMap<Vec<K>, Bucket>>,
    n_hash_tables: usize,
    pub vec_store: VecStore<N>,
    only_index_storage: bool,
    counter: u32,
}

impl<N, K> MemoryTable<N, K>
where
    N: Numeric,
    K: Integer,
{
    fn remove_idx(&mut self, idx: u32, hash: &[K], hash_table: usize) -> Result<()> {
        let tbl = &mut self.hash_tables[hash_table];
        let bucket = tbl.get_mut(hash);
        match bucket {
            None => return Err(Error::NotFound),
            Some(bucket) => {
                bucket.remove(&idx);
                Ok(())
            }
        }
    }
    fn insert_idx(&mut self, idx: u32, hash: Vec<K>, hash_table: usize) {
        debug_assert!(hash_table < self.n_hash_tables);
        let tbl = unsafe { self.hash_tables.get_unchecked_mut(hash_table) };
        let bucket = tbl.entry(hash).or_insert_with(|| FnvHashSet::default());
        bucket.insert(idx);
    }
}

impl<N, K> HashTables<N, K> for MemoryTable<N, K>
where
    N: Numeric,
    K: Integer,
{
    fn new(n_hash_tables: usize, only_index_storage: bool, _: &str) -> Result<Box<Self>> {
        // TODO: Check the average number of vectors in the buckets.
        // this way the capacity can be approximated by the number of DataPoints that will
        // be stored.
        let hash_tables = vec![HashMap::default(); n_hash_tables];
        let vector_store = VecStore { map: vec![] };
        let m = MemoryTable {
            hash_tables,
            n_hash_tables,
            vec_store: vector_store,
            only_index_storage,
            counter: 0,
        };
        Ok(Box::new(m))
    }

    fn put(&mut self, hash: Vec<K>, d: &[N], hash_table: usize) -> Result<u32> {
        // Store hash and id/idx
        let idx = self.counter;
        self.insert_idx(idx, hash, hash_table);

        // There are N hash_tables per unique vector. So we only store
        // the unique v hash_table 0 and increment the counter (the id)
        // after we've update the last (N) hash_table.
        if (hash_table == 0) && (!self.only_index_storage) {
            self.vec_store.push(d.to_vec());
        } else if hash_table == self.n_hash_tables - 1 {
            self.counter += 1
        }
        Ok(idx)
    }

    /// Expensive operation we need to do a linear search over all datapoints
    fn delete(&mut self, hash: &[K], d: &[N], hash_table: usize) -> Result<()> {
        // First find the data point in the VecStore
        let idx = match self.vec_store.position(d) {
            None => return Ok(()),
            Some(idx) => idx,
        };
        // Note: data point remains in VecStore as shrinking the vector would mean we need to
        // re-hash all datapoints.
        self.remove_idx(idx, &hash, hash_table)
    }

    fn update_by_idx(
        &mut self,
        old_hash: &[K],
        new_hash: Vec<K>,
        idx: u32,
        hash_table: usize,
    ) -> Result<()> {
        self.remove_idx(idx, old_hash, hash_table)?;
        self.insert_idx(idx, new_hash, hash_table);
        Ok(())
    }

    /// Query the whole bucket
    fn query_bucket(&self, hash: &[K], hash_table: usize) -> Result<Bucket> {
        let tbl = &self.hash_tables[hash_table];
        match tbl.get(hash) {
            None => Err(Error::NotFound),
            Some(bucket) => Ok(bucket.clone()),
        }
    }

    fn idx_to_datapoint(&self, idx: u32) -> Result<&Vec<N>> {
        Ok(self.vec_store.get(idx))
    }

    fn increase_storage(&mut self, size: usize) {
        increase_capacity(size, &mut self.hash_tables);
        self.vec_store.increase_storage(size);
    }

    fn describe(&self) -> Result<String> {
        let mut lengths = vec![];
        let mut max_len = 0;
        let mut min_len = 1000000;
        let mut set: FnvHashSet<i32> = FnvHashSet::default();
        // iterator over hash tables 0..L
        for map in self.hash_tables.iter() {
            // iterator over all hashes
            // zip to truncate at the describe maximum
            for ((k, v), _) in map.iter().zip(0..DESCRIBE_MAX) {
                let len = v.len();
                let hash_values: FnvHashSet<i32> =
                    FnvHashSet::from_iter(k.iter().map(|&k| k.to_i32().unwrap()));
                set = set.union(&hash_values).copied().collect();
                lengths.push(len);
                if len > max_len {
                    max_len = len
                }
                if len < min_len {
                    min_len = len
                }
            }
        }

        let avg = lengths.iter().sum::<usize>() as f32 / lengths.len() as f32;
        let var = lengths
            .iter()
            .map(|&v| (avg - v as f32).powf(2.))
            .sum::<f32>()
            / lengths.len() as f32;
        let std_dev = var.powf(0.5);

        let mut out = String::from(&format!("No. of tables: {}\n", self.n_hash_tables));
        out.push_str(&format!("Unique hash values:\n{:?}\n", set));
        out.push_str("\nHash collisions:\n");
        out.push_str(&format!("avg:\t{:?}\n", avg));
        out.push_str(&format!("std-dev:\t{:?}\n", std_dev));
        out.push_str(&format!("min:\t{:?}\n", min_len));
        out.push_str(&format!("max:\t{:?}\n", max_len));

        Ok(out)
    }

    fn get_unique_hash_int(&self) -> FnvHashSet<i32> {
        let mut hash_numbers = FnvHashSet::default();

        for ht in &self.hash_tables {
            for ((hash, _), _i) in ht.iter().zip(0..100) {
                for &v in hash {
                    hash_numbers.insert(v.to_i32().unwrap());
                }
            }
        }
        hash_numbers
    }
}

impl<N, K> std::fmt::Debug for MemoryTable<N, K>
where
    N: Numeric,
    K: Integer,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "hash_tables:\nhash, \t buckets\n")?;
        for ht in self.hash_tables.iter() {
            write!(f, "{:?}\n", ht)?;
        }
        Ok(())
    }
}
