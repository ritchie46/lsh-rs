use crate::constants::DISTANCE_R;
use lsh_rs::{MemoryTable, L2, LSH};
use ndarray::prelude::*;
use std::fs;
use std::fs::DirEntry;
use std::io::Read;
use std::path::PathBuf;

pub fn scale_vec(v: &[f32]) -> Vec<f32> {
    let v = aview1(&v);
    let v = &v / DISTANCE_R;
    v.to_vec()
}

pub fn read_vec(path: &str) -> Vec<f32> {
    let mut f = fs::File::open(path).unwrap();
    let mut buf: Vec<u8> = vec![];
    f.read_to_end(&mut buf);
    bincode::deserialize(&buf).unwrap()
}

pub fn sorted_paths(folder: &str) -> Vec<PathBuf> {
    let mut entries = fs::read_dir(folder)
        .unwrap()
        .map(|res| res.unwrap().path())
        .collect::<Vec<_>>();
    entries.sort();
    entries
}

pub fn load_lsh(serialize_folder: &str) -> LSH<MemoryTable, L2> {
    let mut lsh = LSH::new(1, 1, 1);
    lsh.load(format!("{}/save.bincode", serialize_folder))
        .expect("loading failed");
    lsh.describe();
    lsh
}
