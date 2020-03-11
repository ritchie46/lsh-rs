use ndarray::ArrayView1;
use num::Zero;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use std::ops::{Add, Mul};

pub fn l2_norm(x: ArrayView1<f32>) -> f32 {
    x.dot(&x).sqrt()
}

pub fn create_rng(seed: u64) -> SmallRng {
    // TODO: if seed == 0, use random seeded rng
    if seed == 0 {
        SmallRng::from_entropy()
    }
    else {
        SmallRng::seed_from_u64(seed)
    }
}

pub fn rand_unit_vec<RNG: Rng>(size: usize, rng: RNG) -> Vec<f32> {
    rng.sample_iter(StandardNormal).take(size).collect()
}

/// TODO: Use blas implemented dot product.
pub fn dot_prod<T>(u: &[T], v: &[T]) -> T
where
    T: Add<T, Output = T> + Mul<T, Output = T> + Zero + Copy,
{
    let mut sum = T::zero();
    for i in 0..u.len() {
        sum = sum + u[i] * v[i];
    }
    sum
}

pub fn all_eq<T>(u: &[T], v: &[T]) -> bool
where
    T: PartialEq,
{
    if u.len() != v.len() {
        return false;
    }
    for (u_, v_) in u.iter().zip(v) {
        if u_ != v_ {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_dot() {
        let a = dot_prod(&[1, 2, 3], &[1, 2, 3]);
        assert_eq!(a, 14);
    }

    #[test]
    fn test_all_eq() {
        assert!(all_eq(&[1, 2], &[1, 2]));
        assert!(all_eq(&[1., 2.], &[1., 2.]));
        assert!(!all_eq(&[1.1, -1.], &[1., 2.]));
    }
}
