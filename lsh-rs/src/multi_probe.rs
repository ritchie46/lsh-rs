use crate::utils::create_rng;
use crate::Hash;
use crate::HashPrimitive;
use itertools::Itertools;
use rand::distributions::Uniform;
use rand::seq::SliceRandom;
use rand::Rng;
use statrs::function::factorial::binomial;

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

pub fn query_directed_probing(hash_len: usize, budget: usize, w: f32) {
    // https://www.cs.princeton.edu/cass/papers/mplsh_vldb07.pdf
    // https://www.youtube.com/watch?v=c5DHtx5VxX8
}

/// Retrieve perturbation indexes. Every index in a hash can be perturbed by +1 or -1.
///
/// # Arguments
/// * - `hash_length` The hash length is used to determine all the combinations of indexes that can be shifted.
/// * - `n_perturbation` The number of indexes allowed to be changed. We generally first deplete
///     all hashes where 1 index is changed. Then all combinations where two indexes are changed etc.
fn step_wise_perturb(
    hash_length: usize,
    n_perturbations: usize,
) -> Box<dyn Iterator<Item = Vec<(usize, HashPrimitive)>>> {
    // TODO: later opt in for impl return type
    //       https://stackoverflow.com/questions/27646925/how-do-i-return-a-filter-iterator-from-a-function
    let idx = 0..hash_length * 2;
    let switchpoint = hash_length - 1;
    let a = idx.combinations(n_perturbations).map(move |comb| {
        // return of comb are indexes and perturbations (-1 or +1).
        // where idx are the indexes that are perturbed.
        // if n_perturbations is 2 output could be:
        // comb -> [(0, -1), (3, 1)]
        // if n_perturbations is 4 output could be:
        // comb -> [(1, -1), (9, -1), (4, 1), (3, -1)]
        comb.iter()
            .map(|&i| if i > switchpoint { (i / 2, -1) } else { (i, 1) })
            .collect_vec()
    });
    Box::new(a)
}

/// Generates new hashes by step wise shifting one indexes.
/// First all one index shifts are returned (these are closer to the original hash)
/// then the two index shifts, three index shifts etc.
///
/// This is done until the budget is depleted.
pub fn step_wise_probing(hash_len: usize, budget: usize) -> Vec<Vec<HashPrimitive>> {
    let mut hash_perturbs = Vec::with_capacity(budget);

    let n = hash_len as u64;
    // number of combinations (indexes we allow to perturb)
    let mut k = 1;
    let mut budget = budget as f64;
    while budget > 0. && k <= n {
        // binomial coefficient
        // times two as we have -1 and +1.
        let n_combinations = binomial(n, k) * 2.;

        step_wise_perturb(n as usize, k as usize)
            .take(budget as usize)
            .for_each(|v| {
                let mut new_perturb = vec![0; hash_len];
                v.iter().for_each(|(idx, shift)| new_perturb[*idx] += *shift);
                hash_perturbs.push(new_perturb)
            });
        k += 1;
        budget -= n_combinations;
    }
    hash_perturbs
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_permutation() {
        let permut = create_hash_permutation(5, 3);
        println!("{:?}", permut);
    }

    #[test]
    fn test_step_wise_perturb() {
        let a = step_wise_perturb(4, 2);
        assert_eq!(
            vec![vec![(0, 1), (1, 1)], vec![(0, 1), (2, 1)]],
            a.take(2).collect_vec()
        );
    }

    #[test]
    fn test_step_wise_probe() {
        let a = step_wise_probing(4, 20);
        assert_eq!(vec![1, 0, 0, 0], a[0]);
        assert_eq!(vec![0, 1, 0, -1], a[a.len() - 1]);
    }
}
