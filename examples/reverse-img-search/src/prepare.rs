use crate::utils::{read_vec, sorted_paths};
use crate::{BREAK_100, DISTANCE_R, N_TOTAL};
use image::{GenericImage, GenericImageView};
use lsh_rs::{
    stats::{estimate_l, l2_ph, optimize_l2_params},
    utils::l2_norm,
    LSH,
};
use ndarray::prelude::*;
use rayon::prelude::*;
use std::fs;
use std::fs::{DirEntry, ReadDir};
use std::io::Write;

pub fn create_img_vecs(folder: &str, out_folder: &str) -> Result<(), Box<dyn std::error::Error>> {
    let files = fs::read_dir(folder)?;
    let files: Vec<DirEntry> = files.map(|e| e.unwrap()).collect();

    files.par_iter().for_each(|entry| {
        let img = image::open(entry.path()).unwrap();
        let img = img.thumbnail_exact(90, 90);
        let v: Vec<f32> = img.to_bytes().iter().map(|&x| (x as f32) / 255.).collect();

        let original_name = entry.file_name();
        let new_name = original_name.to_str().unwrap().split('.').next().unwrap();

        let mut f = fs::File::create(format!("{}/{}", out_folder, new_name)).unwrap();
        let buf = serde_cbor::to_vec(&v).unwrap();
        f.write(&buf).unwrap();
        println!("{:?}", new_name)
    });
    Ok(())
}

pub fn optimize_params(vec_folder: &str, n: usize) -> Result<(), Box<dyn std::error::Error>> {
    let delta = 0.10;

    let vs: Vec<Vec<f32>> = sorted_paths(vec_folder)
        .iter()
        .zip(0..n)
        .map(|(path, i)| {
            let v = read_vec(path.to_str().unwrap());
            let v = aview1(&v);
            let v = &v / DISTANCE_R;
            v.to_vec()
        })
        .collect();

    let dim = vs[0].len();

    let mut results = optimize_l2_params(delta, dim, &vs);

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

pub fn describe_vecs(folder: &str) -> Result<(), std::io::Error> {
    let mut l2_norms = vec![];
    let mut c = 0;

    for p in sorted_paths(folder) {
        let v = read_vec(p.to_str().unwrap());
        print!("{}\r", c);
        std::io::stdout().flush().unwrap();
        c += 1;
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
    folder: &str,
    serialize_folder: &str,
    n_projections: usize,
    n_hash_tables: usize,
    dim: usize,
    seed: u64,
    r: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut lsh = LSH::new(n_projections, n_hash_tables, dim)
        .seed(seed)
        .increase_storage(30000)
        .only_index()
        .l2(r);

    let mut c = 0;

    for p in sorted_paths(folder) {
        let v = read_vec(p.to_str().unwrap());
        c += 1;
        print!("{}\r", c);
        std::io::stdout().flush().unwrap();

        let v = aview1(&v);
        let v = &v / DISTANCE_R;

        lsh.store_vec(&v.to_vec());
    }
    lsh.describe();
    lsh.dump(format!("{}/save.bincode", serialize_folder));

    Ok(())
}