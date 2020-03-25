extern crate image;
#[macro_use]
extern crate ndarray;
mod constants;
mod prepare;
mod query;
mod utils;
use crate::constants::{IMG_HEIGHT, IMG_WIDTH, L};
use crate::prepare::{create_img_vecs, describe_vecs, make_lsh, sample_params};
use crate::query::query_image;
use crate::utils::load_lsh;
use ndarray::prelude::*;
use rusqlite::Connection;
use std::io::Write;
use std::process::exit;

fn show_usage_msg() {
    println!(
        "Reverse image search

Usage:
    RUN [SUBCOMMAND]

Subcommands:
    prepare-intermediate-vectors
    describe <no. of samples>
    make-lsh <chunk-size>
    sample-params
    query <img-path> <img-viewer>
        Example:
            query /home/johndoe/images/holiday.jpg firefox
        "
    );
    std::process::exit(0);
}

fn main() {
    let img_folder = std::env::var("IMG_FOLDER").expect("IMG_FOLDER not set");
    let vec_folder = std::env::var("VEC_FOLDER").expect("VEC_FOLDER not set");
    let ser_folder = std::env::var("SERIALIZE_FOLDER").expect("SERIALIZE_FOLDER not set");

    let args: Vec<String> = std::env::args().collect();

    let mut db_file = std::fs::canonicalize(".").unwrap();
    db_file.push("ris.db3");
    let conn = Connection::open(db_file).expect("could not open db");
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS vecs (
        path        TEXT PRIMARY KEY,
        vec         BLOB
    )",
    );

    if args.len() == 1 {
        show_usage_msg();
        std::process::exit(0);
    }
    match &args[1][..] {
        "prepare-intermediate-vectors" => {
            create_img_vecs(&img_folder, &conn);
        }
        "describe" => {
            let n: usize = args[2].parse().expect("n not properly defined");
            describe_vecs(&conn, n);
        }
        "make-lsh" => {
            let chunk_size = match args.get(2) {
                None => 5000,
                Some(i) => i.parse().expect("chunk size not properly defined"),
            };
            make_lsh(19, L, IMG_WIDTH * IMG_HEIGHT * 3, 12, 4., chunk_size, &conn);
        }
        "sample-params" => {
            sample_params(250, 0.1, &conn);
        }
        "query" => {
            let img = args.get(2).expect("expected image path.");
            let default = "firefox".to_string();
            let img_viewer = args.get(3).unwrap_or(&default);
            println!("{:?}", query_image(img, &img_viewer, &conn));
        }
        _ => {
            show_usage_msg();
        }
    };
    println!("{:?}", args);
}
