#![feature(test)]
extern crate test;
use lsh::{lsh::LSH, utils::rand_unit_vec, MemoryTable};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use test::Bencher;

fn store_n(n: usize, dim: usize) -> LSH<MemoryTable> {
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let rng = SmallRng::seed_from_u64(i as u64);
        v.push(rand_unit_vec(dim, rng))
    }
    let mut lsh = LSH::new_srp(20, 7, 100, 1);
    lsh.store_vecs(&v);
    lsh
}

#[bench]
fn bench_storing(b: &mut Bencher) {
    b.iter(|| store_n(100, 100))
}

#[bench]
fn bench_query(b: &mut Bencher) {
    let lsh = store_n(100, 100);

    let mut seed = 295;
    let rng = SmallRng::seed_from_u64(seed);
    let q = rand_unit_vec(100, rng);
    b.iter(|| {
        let rng = SmallRng::seed_from_u64(seed);
        let q = rand_unit_vec(100, rng);
        lsh.query_bucket(&q, false);
        seed += 1;
    });
}
