use crate::constants::{BREAK_100, DISTANCE_R, IMG_HEIGHT, IMG_WIDTH, N_TOTAL};
use crate::utils::{read_vec, select_and_scale_vecs, select_vec_by_row_ids, sorted_paths};
use image::{GenericImage, GenericImageView, ImageResult};
use lsh_rs::{
    stats::{estimate_l, l2_ph, optimize_l2_params},
    utils::l2_norm,
    SqlTable, LSH,
};
use ndarray::prelude::*;
use rayon::prelude::*;
use rusqlite::{named_params, params, Connection, Result as DbResult};
use std::fs;
use std::fs::{DirEntry, ReadDir};
use std::io::Write;
use std::path::PathBuf;

pub fn convert_img<P>(path: P) -> ImageResult<Vec<u8>>
where
    P: AsRef<std::path::Path>,
{
    let img = image::open(path)?;
    let img = img.thumbnail_exact(IMG_WIDTH as u32, IMG_HEIGHT as u32);
    let v: Vec<u8> = img.to_bytes();
    Ok(v)
}

pub fn create_img_vecs(folder: &str, conn: &Connection) -> Result<(), Box<dyn std::error::Error>> {
    let files = fs::read_dir(folder)?;
    let files: Vec<DirEntry> = files.map(|e| e.unwrap()).collect();
    conn.execute_batch("BEGIN TRANSACTION;")?;
    let mut stmt = conn.prepare(
        "
    INSERT INTO vecs (path, vec) VALUES (:path, :vec)
    ",
    )?;

    let mut c = 0;
    let chunk_size = 10000;

    files.chunks(chunk_size).for_each(|chunk| {
        let vecs: Vec<(PathBuf, Vec<u8>)> = chunk
            .par_iter()
            .map(|entry| {
                let path = entry.path();
                let v = match convert_img(&path) {
                    Ok(v) => v,
                    Err(_) => panic!("cold not read image."),
                };
                (path, v)
            })
            .collect();
        c += chunk_size;
        println!("{:?}", c);

        vecs.iter().for_each(|(path, v)| {
            stmt.execute_named(
                named_params! {":path": path.as_path().to_str().unwrap(), ":vec": v},
            )
            .expect("failing insert");
        })
    });

    conn.execute_batch("COMMIT TRANSACTION;")?;
    Ok(())
}

pub fn sample_params(
    n: usize,
    delta: f64,
    conn: &Connection,
) -> Result<(), Box<dyn std::error::Error>> {
    let vs = select_and_scale_vecs(0, n, conn).expect("could not get vecs");
    let dim = vs[0].len();
    let mut results = optimize_l2_params(delta, dim, &vs)?;

    // now only ran on a sample n of N.
    // search_time is expected to increase by N/n (due to duplicates)
    let search_time_factor = N_TOTAL as f64 / n as f64;
    results.sort_unstable_by_key(|opt_res| {
        let t = opt_res.hash_time + opt_res.search_time * search_time_factor;
        t as i32
    });
    for opt_res in results {
        let t = opt_res.hash_time + opt_res.search_time * search_time_factor;
        println!("{:?}, total time: {}", opt_res, t);
    }
    Ok(())
}

pub fn describe_vecs(conn: &Connection, n: usize) -> Result<(), std::io::Error> {
    let mut l2_norms = Vec::with_capacity(n);
    let mut c = 0;
    let vs = select_vec_by_row_ids(0, n, conn).expect("could not get vecs from db");

    for v in vs {
        let v: Vec<f32> = v.iter().map(|&x| x as f32).collect();
        let l2 = l2_norm(aview1(&v));
        l2_norms.push(l2);
        if c > 100 && BREAK_100 {
            break;
        }
    }
    println!(
        "L2 norms: min: {}, max: {}, avg: {}",
        l2_norms.iter().copied().fold(0. / 0., f32::min),
        l2_norms.iter().copied().fold(0. / 0., f32::max),
        l2_norms.iter().sum::<f32>() / l2_norms.len() as f32
    );
    Ok(())
}

pub fn make_lsh(
    n_projections: usize,
    n_hash_tables: usize,
    dim: usize,
    seed: u64,
    r: f32,
    chunk_size: usize,
    conn: &Connection,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut stmt = conn.prepare_cached("SELECT count(*) FROM vecs;").unwrap();
    let n_total: i32 = stmt.query_row(params![], |row| row.get(0)).unwrap();
    let n_total = n_total as usize;

    let mut lsh: LSH<SqlTable, _> = LSH::new(n_projections, n_hash_tables, dim)
        .seed(seed)
        .only_index()
        .l2(r)?;

    let mut prev_i = 0;
    for i in (chunk_size..n_total).step_by(chunk_size) {
        println!("{} until {}", prev_i, i);
        let vs = select_and_scale_vecs(prev_i, i, conn).expect("could not get vecs from db");
        prev_i = i;
        lsh.store_vecs(&vs)?;
    }
    println!("indexing...");
    lsh.commit()?;
    Ok(())
}
