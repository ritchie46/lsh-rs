use crate::utils::create_rng;
use crate::HashPrimitive;
use rand::distributions::Uniform;
use rand::Rng;

pub fn create_hash_permutation(hash_len: usize, n: usize) -> Vec<HashPrimitive> {
    let mut permut = vec![0; hash_len];
    let rng = create_rng(0);
    let shifts = rng.sample_iter(Uniform::new(-1, 2));

    let rng = create_rng(0);
    let candidate_idx = rng.sample_iter(Uniform::new(0, hash_len));

    for (v, i) in shifts.zip(candidate_idx).take(n) {
        permut[i] += v
    }
    permut
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_permutation() {
        let permut = create_hash_permutation(5, 3);
        println!("{:?}", permut);
    }
}
