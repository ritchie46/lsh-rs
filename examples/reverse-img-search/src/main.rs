extern crate image;
use image::{GenericImage, GenericImageView};
use lsh_rs::{
    stats::{estimate_l, l2_ph},
    LSH,
};
use rayon::prelude::*;
use std::fs;
use std::fs::DirEntry;
#[macro_use]
extern crate ndarray;
use ndarray::prelude::*;
use ndarray::Zip;
use std::io::Write; // <--- bring flush() into scope

const BREAK_100: bool = true;

fn optimize_params() {
    let delta = 0.2;
    for r in (2..20).step_by(2) {
        let p1 = l2_ph(r as f64, 1.);

        for k in 5..20 {
            let l = estimate_l(delta, p1, k as usize);
            println!("r: {} k: {} L: {} p1: {}", r, k, l, p1)
        }
    }
}

fn create_img_vecs(folder: &str, out_folder: &str) -> Result<(), Box<dyn std::error::Error>> {
    let files = fs::read_dir(folder)?;
    let files: Vec<DirEntry> = files.map(|e| e.unwrap()).collect();

    files.par_iter().for_each(|entry| {
        let img = image::open(entry.path()).unwrap();
        let img = img.thumbnail_exact(90, 90);
        let v: Vec<f32> = img.to_bytes().iter().map(|&x| (x as f32) / 255.).collect();

        let original_name = entry.file_name();
        let new_name = original_name.to_str().unwrap().split('.').next().unwrap();

        let f = fs::File::create(format!("{}/{}", out_folder, new_name)).unwrap();
        serde_cbor::to_writer(f, &v);
        println!("{:?}", new_name)
    });
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
    for entry in fs::read_dir(folder)? {
        c += 1;
        print!("{}\r", c);
        std::io::stdout().flush().unwrap();
        let entry = entry?;
        let f = fs::File::open(entry.path())?;
        let v: Vec<f32> = serde_cbor::from_reader(f).unwrap();
        lsh.store_vec(&v);
        if c > 100 && BREAK_100 {
            break;
        }
    }
    lsh.describe();
    lsh.dump(format!("{}/save.cbor", serialize_folder));

    Ok(())
}

fn describe_vecs(folder: &str) -> Result<(), std::io::Error> {
    let mut entries = fs::read_dir(folder)?;
    let mut prev_entry = entries.next().expect("no files in directory?")?;

    let mut l2_norms = vec![];
    let mut c = 0;
    for entry in entries {
        print!("{}\r", c);
        std::io::stdout().flush().unwrap();
        c += 1;
        let entry = entry?;

        let f = fs::File::open(entry.path())?;
        let v: Vec<f32> = serde_cbor::from_reader(f).unwrap();

        let f = fs::File::open(prev_entry.path())?;
        let v_prev: Vec<f32> = serde_cbor::from_reader(f).unwrap();

        let mut ar: Array1<f32> = Array1::zeros(v.len());
        Zip::from(&mut ar)
            .and(aview1(&v))
            .and(&aview1(&v_prev))
            .apply(|r, &a, &b| {
                let mut x = a - b;
                x = (x * x).powf(0.5);
                *r = x
            });
        let l2 = ar.mean().unwrap();
        l2_norms.push(l2);
        prev_entry = entry;
        if c > 100 {
            break;
        }
    }
    println!(
        "L2 norms: min: {}, max:{} ,avg:{}",
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
    let out_folder = std::env::var("VEC_FOLDER").expect("VEC_FOLDER not set");
    let ser_folder = std::env::var("SERIALIZE_FOLDER").expect("SERIALIZE_FOLDER not set");

    let args: Vec<String> = std::env::args().collect();

    if args.len() == 1 {
        show_usage_msg();
        std::process::exit(0);
    }
    match &args[1][..] {
        "prepare-vec" => {
            create_img_vecs(&folder, &out_folder);
        }
        "describe" => {
            describe_vecs(&out_folder);
        }
        "make-lsh" => {
            make_lsh(&out_folder, &ser_folder, 12, 20, 90 * 90 * 3, 12, 4.);
        }
        "opt" => {
            optimize_params();
        }
        _ => {
            show_usage_msg();
        }
    };
    println!("{:?}", args);
    // create_img_vecs(&folder, &out_folder);
}
