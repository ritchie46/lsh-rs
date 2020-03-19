extern crate image;
#[macro_use]
extern crate ndarray;
mod prepare;
mod utils;
use crate::prepare::{create_img_vecs, describe_vecs, make_lsh, optimize_params};
use ndarray::prelude::*;
use std::io::Write;

pub const BREAK_100: bool = true;
pub const DISTANCE_R: f32 = 20.;
pub const N_TOTAL: u32 = 30000;

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
}
