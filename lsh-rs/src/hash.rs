use crate::{
    data::Numeric, dist::l2_norm, multi_probe::QueryDirectedProbe, utils::create_rng,
    DataPointSlice, FloatSize,
};
use ndarray::prelude::*;
use ndarray_rand::rand_distr::{StandardNormal, Uniform};
use ndarray_rand::RandomExt;
use num::traits::NumCast;
use num::{Float, Zero};
use serde::{Deserialize, Serialize};

pub type HashPrimitive = i8;
pub type Hash = Vec<HashPrimitive>;

pub trait VecHash<N> {
    fn hash_vec_query(&self, v: &[N]) -> Hash;
    fn hash_vec_put(&self, v: &[N]) -> Hash;

    fn as_query_directed_probe(&self) -> Option<&dyn QueryDirectedProbe<N>> {
        None
    }
}

/// Also called SimHash.
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

    fn hash_vec(&self, v: &[N]) -> Hash {
        let v = aview1(v);
        self.hyperplanes
            .t()
            .dot(&v)
            .mapv(|ai| if ai > Zero::zero() { 1 } else { 0 })
            .to_vec()
    }
}

impl<N: Numeric> VecHash<N> for SignRandomProjections<N> {
    fn hash_vec_query(&self, v: &[N]) -> Hash {
        self.hash_vec(v)
    }

    fn hash_vec_put(&self, v: &[N]) -> Hash {
        self.hash_vec(v)
    }
}

/// L2 Hasher family. [Read more.](https://arxiv.org/pdf/1411.3787.pdf)
#[derive(Serialize, Deserialize, Clone)]
pub struct L2<N: Numeric> {
    pub a: Array2<N>,
    pub r: N,
    pub b: Array1<N>,
    n_projections: usize,
}

impl<N: Numeric + Float> L2<N> {
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
        }
    }

    pub(crate) fn hash_vec(&self, v: &[N]) -> Array1<N> {
        ((self.a.dot(&aview1(v)) + &self.b) / self.r).mapv(|x| x.floor())
    }

    fn hash_and_cast_vec(&self, v: &[N]) -> Hash {
        // not DRY. we don't call hash_vec to save function call.
        ((self.a.dot(&aview1(v)) + &self.b) / self.r)
            .mapv(|x| {
                let a: HashPrimitive = NumCast::from(x.floor()).unwrap();
                a
            })
            .to_vec()
    }
}

impl<N: Numeric + Float> VecHash<N> for L2<N> {
    fn hash_vec_query(&self, v: &[N]) -> Hash {
        self.hash_and_cast_vec(v)
    }

    fn hash_vec_put(&self, v: &[N]) -> Hash {
        self.hash_and_cast_vec(v)
    }

    fn as_query_directed_probe(&self) -> Option<&dyn QueryDirectedProbe<N>> {
        Some(self)
    }
}

// /// Maximum Inner Product Search. [Read more.](https://papers.nips.cc/paper/5329-asymmetric-lsh-alsh-for-sublinear-time-maximum-inner-product-search-mips.pdf)
// #[derive(Serialize, Deserialize, Clone)]
// pub struct MIPS {
//     U: f32,
//     M: f32,
//     m: usize,
//     dim: usize,
//     hasher: L2,
// }
//
// impl MIPS {
//     pub fn new(dim: usize, r: f32, U: f32, m: usize, n_projections: usize, seed: u64) -> MIPS {
//         let l2 = L2::new(dim + m, r, n_projections, seed);
//         MIPS {
//             U,
//             M: 0.,
//             m,
//             dim,
//             hasher: l2,
//         }
//     }
//
//     pub fn fit(&mut self, v: &[f32]) {
//         let mut max_l2 = 0.;
//         for x in v.chunks(self.dim) {
//             let l2 = l2_norm(x);
//             if l2 > max_l2 {
//                 max_l2 = l2
//             }
//         }
//         self.M = max_l2
//     }
//
//     pub fn tranform_put(&self, x: &[f32]) -> Vec<f32> {
//         let mut x_new = Vec::with_capacity(x.len() + self.m);
//
//         if self.M == 0. {
//             panic!("MIPS is not fitted")
//         }
//
//         // shrink norm such that l2 norm < U < 1.
//         for x_i in x {
//             x_new.push(x_i / self.M * self.U)
//         }
//
//         let norm_sq = l2_norm(&x_new).powf(2.);
//         for i in 1..(self.m + 1) {
//             x_new.push(norm_sq.powf(i as f32))
//         }
//         x_new
//     }
//
//     pub fn transform_query(&self, x: &[f32]) -> Vec<f32> {
//         let mut x_new = Vec::with_capacity(x.len() + self.m);
//
//         // normalize query to have l2 == 1.
//         let l2 = l2_norm(x);
//         for x_i in x {
//             x_new.push(x_i / l2)
//         }
//
//         for _ in 0..self.m {
//             x_new.push(0.5)
//         }
//         x_new
//     }
// }
//
// impl VecHash for MIPS {
//     fn hash_vec_query(&self, v: &[f32]) -> Hash {
//         let q = self.transform_query(v);
//         self.hasher.hash_vec_query(&q)
//     }
//
//     fn hash_vec_put(&self, v: &[f32]) -> Hash {
//         let p = self.tranform_put(v);
//         self.hasher.hash_vec_query(&p)
//     }
// }
//
// #[cfg(test)]
// mod test {
//     use super::*;
//
//     #[test]
//     fn test_l2() {
//         // Only test if it runs
//         let l2 = L2::new(5, 2.2, 7, 1);
//         // two close vector
//         let h1 = l2.hash_vec_query(&[1., 2., 3., 1., 3.]);
//         let h2 = l2.hash_vec_query(&[1.1, 2., 3., 1., 3.1]);
//
//         // a distant vec
//         let h3 = l2.hash_vec_query(&[100., 100., 100., 100., 100.1]);
//
//         println!("close: {:?} distant: {:?}", (&h1, &h2), &h3);
//         assert_eq!(h1, h2);
//         assert_ne!(h1, h3);
//     }
// }
