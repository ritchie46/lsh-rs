use crate::data::Integer;
use crate::multi_probe::StepWiseProbe;
use crate::{data::Numeric, dist::l2_norm, multi_probe::QueryDirectedProbe, utils::create_rng};
use ndarray::prelude::*;
use ndarray_rand::rand_distr::{StandardNormal, Uniform};
use ndarray_rand::RandomExt;
use num::traits::NumCast;
use num::{Float, Zero};
use serde::export::PhantomData;
use serde::{Deserialize, Serialize};

/// Implement this trait to create your own custom hashers.
/// In case of a symmetrical hash function, only `hash_vec_query` needs to be implemented.
pub trait VecHash<N, K> {
    /// Create a hash for a query data point.
    fn hash_vec_query(&self, v: &[N]) -> Vec<K>;
    /// Create a hash for a data point that is being stored.
    fn hash_vec_put(&self, v: &[N]) -> Vec<K> {
        self.hash_vec_query(v)
    }

    /// If the hasher implements the QueryDirectedProbe trait it should return Some(self)
    fn as_query_directed_probe(&self) -> Option<&dyn QueryDirectedProbe<N, K>> {
        None
    }
    /// If the hasher implements the StepWiseProbe trait it should return Some(self)
    fn as_step_wise_probe(&self) -> Option<&dyn StepWiseProbe<N, K>> {
        None
    }
}

/// A family of hashers for the cosine similarity.
#[derive(Serialize, Deserialize, Clone)]
pub struct SignRandomProjections<N: Numeric> {
    ///  Random unit vectors that will lead to the bits of the hash.
    hyperplanes: Array2<N>,
}

impl<N: Numeric> SignRandomProjections<N> {
    ///
    /// # Arguments
    ///
    /// * `k` - Number of hyperplanes used for determining the hash.
    /// This will also be the hash length.
    pub fn new(k: usize, dim: usize, seed: u64) -> Self {
        let mut rng = create_rng(seed);
        let hp: Array2<f32> = Array::random_using((dim, k), StandardNormal, &mut rng);
        let hp = hp.mapv(|v| N::from_f32(v).unwrap());

        SignRandomProjections { hyperplanes: hp }
    }

    fn hash_vec(&self, v: &[N]) -> Vec<i8> {
        let v = aview1(v);
        self.hyperplanes
            .t()
            .dot(&v)
            .mapv(|ai| if ai > Zero::zero() { 1 } else { 0 })
            .to_vec()
    }
}

impl<N: Numeric> VecHash<N, i8> for SignRandomProjections<N> {
    fn hash_vec_query(&self, v: &[N]) -> Vec<i8> {
        self.hash_vec(v)
    }
}

/// L2 Hasher family. [Read more.](https://arxiv.org/pdf/1411.3787.pdf)
#[derive(Serialize, Deserialize, Clone)]
pub struct L2<N = f32, K = i32> {
    pub a: Array2<N>,
    pub r: N,
    pub b: Array1<N>,
    n_projections: usize,
    phantom: PhantomData<K>,
}

impl<N, K> L2<N, K>
where
    N: Numeric + Float,
    K: Integer,
{
    pub fn new(dim: usize, r: f32, n_projections: usize, seed: u64) -> Self {
        let mut rng = create_rng(seed);
        let a = Array::random_using((n_projections, dim), StandardNormal, &mut rng);
        let uniform_dist = Uniform::new(0., r);
        let b = Array::random_using(n_projections, uniform_dist, &mut rng);

        // cast to generic
        let a = a.mapv(|v| N::from_f32(v).unwrap());
        let b = b.mapv(|v| N::from_f32(v).unwrap());
        let r = N::from_f32(r).unwrap();

        L2 {
            a,
            r,
            b,
            n_projections,
            phantom: PhantomData,
        }
    }

    pub(crate) fn hash_vec(&self, v: &[N]) -> Array1<N> {
        ((self.a.dot(&aview1(v)) + &self.b) / self.r).mapv(|x| x.floor())
    }

    fn hash_and_cast_vec(&self, v: &[N]) -> Vec<K> {
        // not DRY. we don't call hash_vec to save function call.
        ((self.a.dot(&aview1(v)) + &self.b) / self.r)
            .mapv(|x| {
                let hp = NumCast::from(x.floor())
                    .expect("Hash value doesnt fit in the Hash number type i8");
                hp
            })
            .to_vec()
    }
}

impl<N, K> VecHash<N, K> for L2<N, K>
where
    N: Numeric + Float,
    K: Integer,
{
    fn hash_vec_query(&self, v: &[N]) -> Vec<K> {
        self.hash_and_cast_vec(v)
    }

    fn as_query_directed_probe(&self) -> Option<&dyn QueryDirectedProbe<N, K>> {
        Some(self)
    }
}

/// Maximum Inner Product Search. [Read more.](https://papers.nips.cc/paper/5329-asymmetric-lsh-alsh-for-sublinear-time-maximum-inner-product-search-mips.pdf)
#[derive(Serialize, Deserialize, Clone)]
pub struct MIPS<N, K> {
    U: N,
    M: N,
    m: usize,
    dim: usize,
    hasher: L2<N, K>,
}

impl<N, K> MIPS<N, K>
where
    N: Numeric + Float,
    K: Integer,
{
    pub fn new(dim: usize, r: f32, U: N, m: usize, n_projections: usize, seed: u64) -> Self {
        let l2 = L2::new(dim + m, r, n_projections, seed);
        MIPS {
            U,
            M: Zero::zero(),
            m,
            dim,
            hasher: l2,
        }
    }

    pub fn fit(&mut self, v: &[N]) {
        // TODO: add fit to vechash trait?
        let mut max_l2 = Zero::zero();
        for x in v.chunks(self.dim) {
            let l2 = l2_norm(x);
            if l2 > max_l2 {
                max_l2 = l2
            }
        }
        self.M = max_l2
    }

    pub fn tranform_put(&self, x: &[N]) -> Vec<N> {
        let mut x_new = Vec::with_capacity(x.len() + self.m);

        if self.M == Zero::zero() {
            panic!("MIPS is not fitted")
        }

        // shrink norm such that l2 norm < U < 1.
        for x_i in x.iter().cloned() {
            x_new.push(x_i / self.M * self.U)
        }

        let norm_sq = l2_norm(&x_new).powf(N::from_f32(2.).unwrap());
        for i in 1..(self.m + 1) {
            x_new.push(norm_sq.powf(N::from_usize(i).unwrap()))
        }
        x_new
    }

    pub fn transform_query(&self, x: &[N]) -> Vec<N> {
        let mut x_new = Vec::with_capacity(x.len() + self.m);

        // normalize query to have l2 == 1.
        let l2 = l2_norm(x);
        for x_i in x.iter().cloned() {
            x_new.push(x_i / l2)
        }

        let half = N::from_f32(0.5).unwrap();
        for _ in 0..self.m {
            x_new.push(half)
        }
        x_new
    }
}

impl<N, K> VecHash<N, K> for MIPS<N, K>
where
    N: Numeric + Float,
    K: Integer,
{
    fn hash_vec_query(&self, v: &[N]) -> Vec<K> {
        let q = self.transform_query(v);
        self.hasher.hash_vec_query(&q)
    }

    fn hash_vec_put(&self, v: &[N]) -> Vec<K> {
        let p = self.tranform_put(v);
        self.hasher.hash_vec_query(&p)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_l2() {
        // Only test if it runs
        let l2 = <L2>::new(5, 2.2, 7, 1);
        // two close vector
        let h1 = l2.hash_vec_query(&[1., 2., 3., 1., 3.]);
        let h2 = l2.hash_vec_query(&[1.1, 2., 3., 1., 3.1]);

        // a distant vec
        let h3 = l2.hash_vec_query(&[10., 10., 10., 10., 10.1]);

        println!("close: {:?} distant: {:?}", (&h1, &h2), &h3);
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }
}
