use crate::prepare::convert_img;
use crate::utils::sorted_paths;
use crate::{
    constants, query,
    utils::{load_lsh, read_vec, scale_vec},
};
use lsh_rs::utils::l2_norm;
use lsh_rs::{SqlTable, LSH};
use ndarray::aview1;
use rusqlite::{params, types::Value, vtab::array, Connection};
use std::fs;
use std::path::Path;
use std::process::Command;
use std::rc::Rc;

fn select_row_id(
    row_ids: &[i64],
    conn: &Connection,
) -> rusqlite::Result<(Vec<String>, Vec<Vec<u8>>)> {
    // https://github.com/jgallagher/rusqlite/blob/master/src/vtab/array.rs#L180

    array::load_module(conn).unwrap();
    let values: Vec<Value> = row_ids.iter().map(|&i| Value::from(i)).collect();
    let ptr = Rc::new(values);
    let mut stmt = conn.prepare("SELECT path, vec FROM vecs WHERE ROWID IN rarray(?);")?;
    let mut rows = stmt.query(&[&ptr])?;

    let len = row_ids.len();
    let mut vs: Vec<Vec<u8>> = Vec::with_capacity(len);
    let mut paths: Vec<String> = Vec::with_capacity(len);
    while let Some(r) = rows.next()? {
        paths.push(r.get(0)?);
        vs.push(r.get(1)?)
    }
    Ok((paths, vs))
}

pub fn query_image(
    query_img_path: &str,
    img_viewer: &str,
    conn: &Connection,
) -> Result<(), Box<dyn std::error::Error>> {
    let v: Vec<f32> = convert_img(query_img_path)?
        .iter()
        .map(|&x| x as f32)
        .collect();
    let q = scale_vec(&v);
    let mut lsh: LSH<SqlTable, _> = LSH::new(1, 1, 1).l2(1.);
    let row_ids: Vec<i64> = lsh
        .query_bucket_ids(&q)
        .iter()
        .map(|&x| (x + 1) as i64)
        .take(constants::QUERY_L_FACT_UPPER_BOUND * constants::L)
        .collect();

    let (paths, vs) = select_row_id(&row_ids, conn).expect("could not select by row id");
    let mut scores: Vec<(f32, String)> = vs
        .iter()
        .zip(paths)
        .map(|(v, path)| {
            let v: Vec<f32> = v.iter().map(|&x| x as f32).collect();
            let p = scale_vec(&v);
            let dist = &aview1(&q) - &aview1(&p);
            let l2 = l2_norm(dist.view());
            (l2, path)
        })
        .collect();
    scores.sort_unstable_by_key(|(l2, _)| (l2 * 1e3) as i32);

    println!("Top 3:");
    for (l2, path) in &scores[..5] {
        println!("Score: {}\t file: {}", l2, path);
        let mut cmd = Command::new(img_viewer);
        cmd.arg(path);
        cmd.status();
    }
    Ok(())
}
