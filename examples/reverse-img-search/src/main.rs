extern crate image;
#[macro_use]
extern crate ndarray;
mod img_prep;

use crate::img_prep::create_img_vecs;
use lsh_rs::{
    stats::{estimate_l, l2_ph, optimize_l2_params},
    utils::l2_norm,
    LSH,
};
use ndarray::prelude::*;
use rayon::prelude::*;
use std::fs;
use std::io::Write;

const BREAK_100: bool = true;
const DISTANCE_R: f32 = 20.;
const N_TOTAL: u32 = 30000;

fn file_iter(vec_folder: &str) -> Box<dyn Iterator<Item = Vec<f32>>> {
    let a = fs::read_dir(vec_folder).unwrap().map(|entry| {
        let entry = entry.unwrap();
        let f = fs::File::open(entry.path()).unwrap();
        let mut v: Vec<f32> = serde_cbor::from_reader(f).unwrap();
        v
    });
    Box::new(a)
}

fn optimize_params(vec_folder: &str, n: usize) -> Result<(), Box<dyn std::error::Error>> {
    let delta = 0.10;

    let vs: Vec<Vec<f32>> = file_iter(vec_folder)
        .zip(0..n)
        .map(|(v, i)| {
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

fn make_lsh(
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

    for v in file_iter(folder) {
        c += 1;
        print!("{}\r", c);
        std::io::stdout().flush().unwrap();

        let v = aview1(&v);
        let v = &v / DISTANCE_R;

        lsh.store_vec(&v.to_vec());
        if c > 100 && BREAK_100 {
            break;
        }
    }
    lsh.describe();
    lsh.dump(format!("{}/save.cbor", serialize_folder));

    Ok(())
}

fn describe_vecs(folder: &str) -> Result<(), std::io::Error> {
    let mut l2_norms = vec![];
    let mut c = 0;
    for v in file_iter(folder) {
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

fn show_usage_msg() {
    println!(
        "Reverse image search

Usage:
    RUN [SUBCOMMAND]

Subcommands:
    prepare-vec
        "
    );
    std::process::exit(0);
}

fn main() {
    let folder = std::env::var("IMG_FOLDER").expect("IMG_FOLDER not set");
    let vec_folder = std::env::var("VEC_FOLDER").expect("VEC_FOLDER not set");
    let ser_folder = std::env::var("SERIALIZE_FOLDER").expect("SERIALIZE_FOLDER not set");

    let args: Vec<String> = std::env::args().collect();

    if args.len() == 1 {
        show_usage_msg();
        std::process::exit(0);
    }
    match &args[1][..] {
        "prepare-vec" => {
            create_img_vecs(&folder, &vec_folder);
        }
        "describe" => {
            describe_vecs(&vec_folder);
        }
        "make-lsh" => {
            make_lsh(&vec_folder, &ser_folder, 19, 150, 90 * 90 * 3, 12, 4.);
        }
        "opt" => {
            optimize_params(&vec_folder, 250);
        }
        _ => {
            show_usage_msg();
        }
    };
    println!("{:?}", args);
    // create_img_vecs(&folder, &out_folder);
}
