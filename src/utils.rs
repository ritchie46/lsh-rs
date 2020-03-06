use num::Zero;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use std::ops::{Add, Mul};

pub fn rand_unit_vec<rng: Rng>(size: usize, rng: rng) -> Vec<f64> {
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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_seed_vec() {
        let rng = SmallRng::seed_from_u64(1);
        let v = rand_unit_vec(3, rng).iter().sum::<f64>();
        assert_eq!(v, -0.17196687602505994);
    }

    #[test]
    fn test_dot() {
        let a = dot_prod(&[1, 2, 3], &[1, 2, 3]);
        assert_eq!(a, 14);
    }
}
