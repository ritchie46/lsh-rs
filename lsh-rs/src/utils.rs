use rand::rngs::SmallRng;
use rand::{thread_rng, Rng, SeedableRng};
use rand_distr::StandardNormal;

pub fn increase_capacity<T>(size: usize, container: &mut Vec<T>) {
    if container.capacity() < size {
        let diff = size - container.capacity();
        container.reserve(diff)
    }
}

pub fn create_rng(seed: u64) -> SmallRng {
    // TODO: if seed == 0, use random seeded rng
    if seed == 0 {
        match SmallRng::from_rng(thread_rng()) {
            Ok(rng) => rng,
            Err(_) => SmallRng::from_entropy(),
        }
    } else {
        SmallRng::seed_from_u64(seed)
    }
}

pub fn rand_unit_vec<RNG: Rng>(size: usize, rng: RNG) -> Vec<f32> {
    rng.sample_iter(StandardNormal).take(size).collect()
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
    fn test_all_eq() {
        assert!(all_eq(&[1, 2], &[1, 2]));
        assert!(all_eq(&[1., 2.], &[1., 2.]));
        assert!(!all_eq(&[1.1, -1.], &[1., 2.]));
    }
}
