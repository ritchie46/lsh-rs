use crate::utils::sorted_paths;
use crate::{
    query,
    utils::{load_lsh, read_vec, scale_vec},
};
use lsh_rs::utils::l2_norm;
use ndarray::aview1;
use std::fs;
use std::path::Path;

pub fn query_image(vec_folder: &str, serialize_folder: &str, img_folder: &str) {
    let vec_basename = Path::new(vec_folder)
        .file_stem()
        .unwrap()
        .to_owned()
        .into_string()
        .unwrap();
    let img_basename = Path::new(img_folder)
        .file_stem()
        .unwrap()
        .to_owned()
        .into_string()
        .unwrap();

    let mut lsh = load_lsh(serialize_folder);

    let file_names: Vec<String> = sorted_paths(vec_folder)
        .iter()
        .map(|p| p.to_str().unwrap().to_owned())
        .collect();

    for file_name in &file_names[20..] {
        let q = read_vec(&file_name);
        let q_scaled = scale_vec(&q);
        let bucket = lsh.query_bucket_ids(&q_scaled);

        let mut results = bucket
            .iter()
            .map(|&id| {
                let vec_file = &file_names[id as usize];
                let p = read_vec(vec_file);

                let dist = &aview1(&p) - &aview1(&q);
                let l2 = l2_norm(dist.view());
                let img_file = vec_file.replace(&vec_basename, &img_basename);
                (l2, img_file)
            })
            .collect::<Vec<_>>();
        results.sort_unstable_by_key(|(l2, _)| (l2 * 1e3) as i32);

        let mut c = 0;
        for (l2, img) in &results {
            c += 1;
            println!("firefox {}.jpg", img);
            if c > 4 {
                break;
            }
        }
        c = 0;
        for (l2, img) in results {
            c += 1;
            println!("l2 {}", l2);
            if c > 4 {
                break;
            }
        }
        break;
    }
}
