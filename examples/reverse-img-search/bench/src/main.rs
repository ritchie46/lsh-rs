#![feature(test)]
extern crate test;
use reverse_img_search::prepare::convert_img;
use std::fs::canonicalize;
use std::io::Write;
use test::Bencher;

#[bench]
fn bench_img_convert(b: &mut Bencher) {
    let mut p = canonicalize(".").unwrap();
    p.push("static/img.jpg");
    let mut stdout = std::io::stdout();
    stdout.write(&format!("{:?}", p).as_bytes());
    b.iter(|| convert_img(&p))
}
