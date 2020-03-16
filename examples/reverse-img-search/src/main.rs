extern crate image;
use image::{GenericImage, GenericImageView};
use rayon::prelude::*;
use std::fs;
use std::fs::DirEntry;

fn create_img_vecs(folder: &str, out_folder: &str) -> Result<(), Box<dyn std::error::Error>> {
    let files = fs::read_dir(folder).unwrap();
    let files: Vec<DirEntry> = files.map(|e| e.unwrap()).collect();

    files.par_iter().for_each(|entry| {
        let img = image::open(entry.path()).unwrap();
        let img = img.thumbnail_exact(90, 90);
        let v = img.to_bytes();

        let original_name = entry.file_name();
        let new_name = original_name.to_str().unwrap().split('.').next().unwrap();

        let f = fs::File::create(format!("{}/{}", out_folder, new_name)).unwrap();
        serde_cbor::to_writer(f, &v);
        println!("{:?}", new_name)
    });
    Ok(())
}

fn main() {}
