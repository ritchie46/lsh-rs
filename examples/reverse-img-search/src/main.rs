extern crate image;
#[macro_use]
extern crate ndarray;
mod constants;
mod prepare;
mod query;
mod utils;
use crate::constants::{IMG_HEIGHT, IMG_WIDTH};
use crate::prepare::{create_img_vecs, describe_vecs, make_lsh, optimize_params};
use crate::query::query_image;
use crate::utils::load_lsh;
use ndarray::prelude::*;
use std::io::Write;

fn show_usage_msg() {
    println!(
        "Reverse image search

Usage:
    RUN [SUBCOMMAND]

Subcommands:
    prepare-intermediate-vectors
    describe
    make-lsh
    sample-params
    load
    query
        "
    );
    std::process::exit(0);
}

fn main() {
    let img_folder = std::env::var("IMG_FOLDER").expect("IMG_FOLDER not set");
    let vec_folder = std::env::var("VEC_FOLDER").expect("VEC_FOLDER not set");
    let ser_folder = std::env::var("SERIALIZE_FOLDER").expect("SERIALIZE_FOLDER not set");

    let args: Vec<String> = std::env::args().collect();

    if args.len() == 1 {
        show_usage_msg();
        std::process::exit(0);
    }
    match &args[1][..] {
        "prepare-intermediate-vectors" => {
            create_img_vecs(&img_folder, &vec_folder);
        }
        "describe" => {
            describe_vecs(&vec_folder);
        }
        "make-lsh" => {
            make_lsh(
                &vec_folder,
                &ser_folder,
                19,
                150,
                IMG_WIDTH * IMG_HEIGHT * 3,
                12,
                4.,
            );
        }
        "sample-params" => {
            optimize_params(&vec_folder, 250);
        }
        "load" => {
            load_lsh(&ser_folder);
        }
        "query" => {
            query_image(&vec_folder, &ser_folder, &img_folder);
        }
        _ => {
            show_usage_msg();
        }
    };
    println!("{:?}", args);
}
