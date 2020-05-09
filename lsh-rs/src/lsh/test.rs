#![cfg(test)]
use crate::prelude::*;

#[test]
fn test_hash_table() {
    let mut lsh = LshMem::new(5, 10, 3).seed(1).srp().unwrap();
    let v1 = &[2., 3., 4.];
    let v2 = &[-1., -1., 1.];
    lsh.store_vec(v1).unwrap();
    lsh.store_vec(v2).unwrap();
    assert!(lsh.query_bucket(v2).unwrap().len() > 0);

    let bucket_len_before = lsh.query_bucket(v1).unwrap().len();
    lsh.delete_vec(v1).unwrap();
    let bucket_len_before_after = lsh.query_bucket(v1).unwrap().len();
    assert!(bucket_len_before > bucket_len_before_after);
}

#[test]
fn test_index_only() {
    // Test if vec storage is increased
    let mut lsh = hi8::LshMem::new(5, 9, 3).seed(1).l2(2.).unwrap();
    let v1 = &[2., 3., 4.];
    lsh.store_vec(v1).unwrap();
    assert_eq!(lsh.hash_tables.unwrap().vec_store.map.len(), 1);

    // Test if vec storage is empty
    let mut lsh = hi8::LshMem::new(5, 9, 3)
        .seed(1)
        .only_index()
        .l2(2.)
        .unwrap();
    lsh.store_vec(v1).unwrap();
    assert_eq!(lsh.hash_tables.as_ref().unwrap().vec_store.map.len(), 0);
    lsh.query_bucket_ids(v1).unwrap();
}

#[test]
fn test_serialization() {
    let mut lsh = hi8::LshMem::new(5, 9, 3).seed(1).l2(2.).unwrap();
    let v1 = &[2., 3., 4.];
    lsh.store_vec(v1).unwrap();
    let mut tmp = std::env::temp_dir();
    tmp.push("lsh");
    std::fs::create_dir(&tmp).unwrap_or_default();
    tmp.push("serialized.bincode");
    assert!(lsh.dump(&tmp).is_ok());

    // load from file
    let res = lsh.load(&tmp);
    println!("{:?}", res);
    assert!(res.is_ok());
    println!("{:?}", lsh.hash_tables)
}

#[test]
#[cfg(feature = "sqlite")]
fn test_db() {
    let v1 = &[2., 3., 4.];
    {
        let mut lsh = hi8::LshSql::new(5, 2, 3).seed(2).srp().unwrap();
        lsh.store_vec(v1).unwrap();
        assert!(lsh.query_bucket_ids(v1).unwrap().contains(&0));
        lsh.commit().unwrap();
        lsh.describe().unwrap();
    }

    // tests if the same db is reused.
    let lsh2 = hi8::LshSql::new(5, 2, 3).srp().unwrap();
    lsh2.describe().unwrap();
    assert!(lsh2.query_bucket_ids(v1).unwrap().contains(&0));
}

#[test]
#[cfg(feature = "sqlite")]
fn test_mem_db() {
    let v1 = &[2., 3., 4.];
    let mut lsh = hi8::LshSqlMem::new(5, 2, 3).seed(2).srp().unwrap();
    lsh.store_vec(v1).unwrap();
    assert!(lsh.query_bucket_ids(v1).unwrap().contains(&0));
    lsh.describe().unwrap();
}
