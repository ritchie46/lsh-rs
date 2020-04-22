use crate::utils::create_rng;
use crate::HashPrimitive;
use rand::distributions::Uniform;
use rand::seq::SliceRandom;
use rand::Rng;

fn uniform_without_replacement<T: Copy>(bucket: &mut [T], n: usize) -> Vec<T> {
    // https://stackoverflow.com/questions/196017/unique-non-repeating-random-numbers-in-o1#196065
    let mut max_idx = bucket.len() - 1;
    let mut rng = create_rng(0);

    let mut samples = Vec::with_capacity(n);

    for _ in 0..n {
        let idx = rng.sample(Uniform::new(0, max_idx));
        samples.push(bucket[idx]);
        bucket.swap(idx, max_idx);
        max_idx -= 1;
    }
    samples
}

pub fn create_hash_permutation(hash_len: usize, n: usize) -> Vec<HashPrimitive> {
    let mut permut = vec![0; hash_len];
    let shift_options = [-1i8, 1];

    let mut idx: Vec<usize> = (0..hash_len).collect();
    let candidate_idx = uniform_without_replacement(&mut idx, n);

    let mut rng = create_rng(0);
    for i in candidate_idx {
        let v = *shift_options.choose(&mut rng).unwrap();
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
