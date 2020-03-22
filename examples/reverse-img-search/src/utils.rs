use crate::constants::DISTANCE_R;
use lsh_rs::{MemoryTable, L2, LSH};
use ndarray::prelude::*;
use rayon::prelude::*;
use rusqlite::{named_params, Connection, Result as DbResult};
use std::fs;
use std::fs::DirEntry;
use std::io::Read;
use std::path::PathBuf;

pub fn scale_vec(v: &[f32]) -> Vec<f32> {
    let v = aview1(&v);
    let v = &v / DISTANCE_R;
    v.to_vec()
}

pub fn select_vec_by_row_ids(
    lower: usize,
    upper: usize,
    conn: &Connection,
) -> DbResult<Vec<Vec<u8>>> {
    let mut stmt = conn.prepare_cached(
        "
SELECT vec FROM vecs
WHERE ROWID BETWEEN :lower AND :upper;
    ",
    )?;
    let mut rows = stmt
        .query_named(named_params! {":lower": (lower + 1) as u32, ":upper": (upper + 1) as u32 })?;

    let mut vecs = Vec::with_capacity(upper - lower);
    while let Some(row) = rows.next()? {
        let v: Vec<u8> = row.get(0)?;
        vecs.push(v);
    }
    Ok(vecs)
}

pub fn select_and_scale_vecs(
    lower: usize,
    higher: usize,
    conn: &Connection,
) -> DbResult<Vec<Vec<f32>>> {
    let vs: Vec<Vec<f32>> = select_vec_by_row_ids(lower, higher, conn)?
        .par_iter()
        .map(|v| {
            let v: Vec<f32> = v.iter().map(|&x| x as f32).collect();
            let v = &aview1(&v) / DISTANCE_R;
            v.to_vec()
        })
        .collect();
    Ok(vs)
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
