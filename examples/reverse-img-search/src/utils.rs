use lsh_rs::{MemoryTable, L2, LSH};
use std::fs;

pub fn file_iter(vec_folder: &str) -> Box<dyn Iterator<Item = Vec<f32>>> {
    let a = fs::read_dir(vec_folder).unwrap().map(|entry| {
        let entry = entry.unwrap();
        let f = fs::File::open(entry.path()).unwrap();
        let mut v: Vec<f32> = serde_cbor::from_reader(f).unwrap();
        v
    });
    Box::new(a)
}

pub fn load_lsh(serialize_folder: &str) -> LSH<MemoryTable, L2> {
    let mut lsh = LSH::new(1, 1, 1);
    lsh.load(format!("{}/save.bincode", serialize_folder))
        .expect("loading failed");
    lsh.describe();
    lsh
}
