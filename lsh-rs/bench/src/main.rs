#![feature(test)]
extern crate test;
use lsh_rs::{prelude::*, utils::rand_unit_vec};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use test::Bencher;

fn prep_vecs(n: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let rng = SmallRng::seed_from_u64(i as u64);
        v.push(rand_unit_vec(dim, rng))
    }
    v
}

fn store_n(n: usize, dim: usize, index_only: bool) -> LshMem<SignRandomProjections<f32>, f32, i8> {
    let v = prep_vecs(n, dim);
    let mut lsh;
    if index_only {
        lsh = LSH::new(20, 7, 100).seed(1).only_index().srp().unwrap();
    } else {
        lsh = LSH::new(20, 7, 100).seed(1).srp().unwrap();
    }
    lsh.store_vecs(&v);
    lsh
}

#[bench]
fn bench_storing(b: &mut Bencher) {
    b.iter(|| store_n(1000, 100, false))
}

#[bench]
fn bench_storing_index_only(b: &mut Bencher) {
    b.iter(|| store_n(1000, 100, true))
}

#[bench]
fn bench_storing_sqlite_mem(b: &mut Bencher) {
    let mut lsh = hi8::LshSqlMem::new(20, 7, 100).seed(1).l2(4.).unwrap();
    b.iter(|| {
        let v = prep_vecs(1000, 100);
        lsh.store_vecs(&v);
    })
}

#[bench]
fn bench_query(b: &mut Bencher) {
    let lsh = store_n(100, 100, false);

    let mut seed = 295;
    let rng = SmallRng::seed_from_u64(seed);
    let q = rand_unit_vec(100, rng);
    b.iter(|| {
        let rng = SmallRng::seed_from_u64(seed);
        let q = rand_unit_vec(100, rng);
        lsh.query_bucket(&q);
        seed += 1;
    });
}

#[bench]
fn bench_sqlite(b: &mut Bencher) {
    let mut sql = SqlTableMem::new(1, true, ".").unwrap();
    let v = vec![1., 2.];
    let hash = vec![1, 2];
    b.iter(|| {
        sql.put(hash.clone(), &v, 0);
    })
}

mod srp {
    use super::*;

    #[bench]
    fn bench_srp(b: &mut Bencher) {
        let srp = SignRandomProjections::new(15, 100, 0);
        let v = [1.; 100];
        b.iter(|| srp.hash_vec_query(&v))
    }
}

mod l2 {
    use super::*;

    #[bench]
    fn bench_l2(b: &mut Bencher) {
        let l2: L2<f64, i8> = L2::new(100, 4., 15, 0);
        let v = [1.; 100];
        b.iter(|| l2.hash_vec_query(&v))
    }
}
