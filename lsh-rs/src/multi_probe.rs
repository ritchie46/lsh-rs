use crate::utils::create_rng;
use crate::{DataPointSlice, FloatSize, Hash, HashPrimitive, HashTables, Result, VecHash, L2, LSH};
use fnv::FnvHashSet;
use itertools::Itertools;
use ndarray::prelude::*;
use ndarray::stack;
use rand::distributions::Uniform;
use rand::seq::SliceRandom;
use rand::Rng;
use statrs::function::factorial::binomial;
use std::any::Any;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Implementation of paper:
///
/// Liv, Q., Josephson, W., Whang, L., Charikar, M., & Li, K. (n.d.).
/// Multi-Probe LSH: Efficient Indexing for High-Dimensional Similarity Search
/// Retrieved from https://www.cs.princeton.edu/cass/papers/mplsh_vldb07.pdf

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
    let switchpoint = hash_length;
    let a = idx.combinations(n_perturbations).map(move |comb| {
        // return of comb are indexes and perturbations (-1 or +1).
        // where idx are the indexes that are perturbed.
        // if n_perturbations is 2 output could be:
        // comb -> [(0, -1), (3, 1)]
        // if n_perturbations is 4 output could be:
        // comb -> [(1, -1), (9, -1), (4, 1), (3, -1)]
        comb.iter()
            .map(|&i| {
                if i >= switchpoint {
                    (i - switchpoint, -1)
                } else {
                    (i, 1)
                }
            })
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
                v.iter()
                    .for_each(|(idx, shift)| new_perturb[*idx] += *shift);
                hash_perturbs.push(new_perturb)
            });
        k += 1;
        budget -= n_combinations;
    }
    hash_perturbs
}

#[derive(PartialEq, Clone)]
struct PerturbState<'a> {
    // original sorted zj
    z: &'a [usize],
    // original xi(delta)
    distances: &'a [FloatSize],
    // selection of zjs
    // We start with the first one, as this is the lowest score.
    selection: Vec<usize>,
    switchpoint: usize,
    original_hash: Option<Hash>,
}

impl<'a> PerturbState<'a> {
    fn new(z: &'a [usize], distances: &'a [FloatSize], switchpoint: usize, hash: Hash) -> Self {
        PerturbState {
            z,
            distances,
            selection: vec![0],
            switchpoint,
            original_hash: Some(hash),
        }
    }

    fn score(&self) -> FloatSize {
        let mut score = 0.;
        for &index in self.selection.iter() {
            let zj = self.z[index];
            score += self.distances[zj];
        }
        score
    }

    // map zj value to (i, delta) as in paper
    fn i_delta(&self) -> Vec<(usize, HashPrimitive)> {
        let mut out = Vec::with_capacity(self.z.len());
        for &idx in self.selection.iter() {
            let zj = self.z[idx];
            let delta;
            let index;
            if zj >= self.switchpoint {
                delta = 1;
                index = zj - self.switchpoint;
            } else {
                delta = -1;
                index = zj;
            }
            out.push((index, delta))
        }
        out
    }

    fn shift(&mut self) {
        let max = self.selection.pop().unwrap();
        self.selection.push(max + 1);
    }

    fn expand(&mut self) {
        let max = self.selection[self.selection.len() - 1];
        self.selection.push(max + 1)
    }

    fn gen_hash(&mut self) -> Hash {
        let mut hash = self.original_hash.take().expect("hash already taken");
        for (i, delta) in self.i_delta() {
            let ptr = &mut hash[i];
            *ptr += delta
        }
        hash
    }
}

// implement ordering so that we can create a min heap
impl Ord for PerturbState<'_> {
    fn cmp(&self, other: &PerturbState) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl PartialOrd for PerturbState<'_> {
    fn partial_cmp(&self, other: &PerturbState) -> Option<Ordering> {
        other.score().partial_cmp(&self.score())
    }
}

impl Eq for PerturbState<'_> {}

impl L2 {
    /// Computes the distance between the query hash and the boundary of the slot r (W in the paper)
    ///
    /// As stated by Multi-Probe LSH paper:
    /// For δ ∈ {−1, +1}, let xi(δ) be the distance of q from the boundary of the slot
    fn distance_to_bound(
        &self,
        q: &DataPointSlice,
        hash: Option<&Hash>,
    ) -> (Array1<FloatSize>, Array1<FloatSize>) {
        let hash = match hash {
            None => self.hash_vec(q).to_vec(),
            Some(h) => h.iter().map(|&v| v as FloatSize).collect_vec(),
        };
        let f = self.a.dot(&aview1(q)) + &self.b;
        let xi_min1 = f - &aview1(&hash) * self.r;
        let xi_plus1: Array1<FloatSize> = self.r - &xi_min1;
        (xi_min1, xi_plus1)
    }

    pub fn query_directed_probing(&self, q: &DataPointSlice, budget: usize) -> Vec<Hash> {
        // https://www.cs.princeton.edu/cass/papers/mplsh_vldb07.pdf
        // https://www.youtube.com/watch?v=c5DHtx5VxX8
        let hash = self.hash_vec_query(q);
        let (xi_min, xi_plus) = self.distance_to_bound(q, Some(&hash));
        // >= this point = +1
        // < this point = -1
        let switchpoint = xi_min.len();

        let distances: Vec<FloatSize> = stack!(Axis(0), xi_min, xi_plus).to_vec();

        // indexes of the least scores to the highest
        // all below is an argsort
        let z = distances.clone();
        let mut z = z.iter().enumerate().collect::<Vec<_>>();
        z.sort_unstable_by(|(_idx_a, a), (_idx_b, b)| a.partial_cmp(b).unwrap());
        let z = z.iter().map(|(idx, _)| *idx).collect::<Vec<_>>();

        let mut hashes = Vec::with_capacity(budget + 1);
        hashes.push(hash.clone());
        // Algorithm 1 from paper
        let mut heap = BinaryHeap::new();
        let a0 = PerturbState::new(&z, &distances, switchpoint, hash);
        heap.push(a0);
        for _ in 0..budget {
            let mut ai = heap.pop().unwrap();
            let mut a_s = ai.clone();
            let mut a_e = ai.clone();
            a_s.shift();
            heap.push(a_s);
            a_e.expand();
            heap.push(a_e);

            hashes.push(ai.gen_hash())
        }
        hashes
    }
}

pub trait MultiProbe {
    fn multi_probe_bucket_union(&self, v: &DataPointSlice) -> Result<FnvHashSet<u32>>;
}

impl<H: VecHash + Sync + Any, T: HashTables> MultiProbe for LSH<T, H> {
    fn multi_probe_bucket_union(&self, v: &DataPointSlice) -> Result<FnvHashSet<u32>> {
        // uses dynamic typing through runtime reflection.
        // TODO:
        //  use impl specialization once stable
        //  autoref specialization was tried but did not succeed.
        //  https://github.com/dtolnay/case-studies/blob/master/autoref-specialization/README.md
        self.validate_vec(v)?;
        let mut bucket_union = FnvHashSet::default();

        let value_any = &self.hashers as &dyn Any;
        match value_any.downcast_ref::<Vec<L2>>() {
            Some(l2_hashers) => {
                for (i, hasher) in l2_hashers.iter().enumerate() {
                    let hashes = hasher.query_directed_probing(v, self._multi_probe_budget);
                    for hash in hashes {
                        self.process_bucket_union_result(&hash, i, &mut bucket_union)?
                    }
                }
            }
            None => {
                let probing_seq = step_wise_probing(self.n_projections, self._multi_probe_budget);
                for (i, proj) in self.hashers.iter().enumerate() {
                    // fist process the original query
                    let original_hash = proj.hash_vec_query(v);
                    self.process_bucket_union_result(&original_hash, i, &mut bucket_union)?;

                    for pertub in &probing_seq {
                        let hash = original_hash
                            .iter()
                            .zip(pertub)
                            .map(|(&a, &b)| a + b)
                            .collect();
                        self.process_bucket_union_result(&hash, i, &mut bucket_union)?;
                    }
                }
            }
        }
        Ok(bucket_union)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::LshMem;

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
        assert_eq!(vec![0, 1, -1, 0], a[a.len() - 1]);
    }

    #[test]
    fn test_l2_xi_distances() {
        let l2 = L2::new(4, 4., 3, 1);
        let (xi_min, xi_plus) = l2.distance_to_bound(&[1., 2., 3., 1.], None);
        assert_eq!(xi_min, arr1(&[2.0210547, 1.9154847, 0.89937115]));
        assert_eq!(xi_plus, arr1(&[1.9789453, 2.0845153, 3.1006289]));
    }

    #[test]
    fn test_perturbstate() {
        let distances = [1., 0.1, 3., 2., 9., 4., 0.8, 5.];
        // argsort
        let z = vec![1, 6, 0, 3, 2, 5, 7, 4];
        let switchpoint = 4;
        let mut a0 = PerturbState::new(&z, &distances, switchpoint, vec![0, 0, 0, 0]);
        // initial selection is the first zj [0]
        // This leads to:
        //   distance/score:    0.1
        //   index:             1
        //   delta:             -1
        assert_eq!(a0.clone().gen_hash(), [0, -1, 0, 0]);
        assert_eq!(a0.score(), 0.1);
        assert_eq!(a0.selection, [0]);

        // after expansion operation selection is [0, 1]
        // This leads to:
        //   distance/ score:   0.1 + 0.8
        //   index:             [1, 2]
        //   delta:             [-1, 1]

        let mut ae = a0.clone();
        ae.expand();
        assert_eq!(ae.gen_hash(), [0, -1, 1, 0]);
        assert_eq!(ae.score(), 0.1 + 0.8);
        assert_eq!(ae.selection, [0, 1]);

        // after shift operation selection is [1]
        // This leads to:
        //   distance/ score:   0.8
        //   index:             2
        //   delta:             1
        let mut a_s = a0.clone();
        a_s.shift();
        assert_eq!(a_s.gen_hash(), [0, 0, 1, 0]);
        assert_eq!(a_s.score(), 0.8);
        assert_eq!(a_s.selection, [1]);
    }

    #[test]
    fn test_query_directed_probe() {
        let l2 = L2::new(4, 4., 3, 1);
        let hashes = l2.query_directed_probing(&[1., 2., 3., 1.], 4);
        println!("{:?}", hashes)
    }
}
