//! Multi probe LSH
use crate::data::{Integer, Numeric};
use crate::{prelude::*, utils::create_rng};
use fnv::FnvHashSet;
use itertools::Itertools;
use ndarray::prelude::*;
use ndarray::stack;
use num::{Float, One, Zero};
use rand::distributions::Uniform;
use rand::seq::SliceRandom;
use rand::Rng;
use statrs::function::factorial::binomial;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Query directed probing
///
/// Implementation of paper:
///
/// Liv, Q., Josephson, W., Whang, L., Charikar, M., & Li, K. (n.d.).
/// Multi-Probe LSH: Efficient Indexing for High-Dimensional Similarity Search
/// Retrieved from https://www.cs.princeton.edu/cass/papers/mplsh_vldb07.pdf

pub trait QueryDirectedProbe<N, K> {
    fn query_directed_probe(&self, q: &[N], budget: usize) -> Result<Vec<Vec<K>>>;
}

/// Step wise probing
pub trait StepWiseProbe<N, K>: VecHash<N, K> {
    fn step_wise_probe(&self, q: &[N], budget: usize, hash_len: usize) -> Result<Vec<Vec<K>>>;
}

impl<N> StepWiseProbe<N, i8> for SignRandomProjections<N>
where
    N: Numeric,
{
    fn step_wise_probe(&self, q: &[N], budget: usize, hash_len: usize) -> Result<Vec<Vec<i8>>> {
        let probing_seq = step_wise_probing(hash_len, budget, false);
        let original_hash = self.hash_vec_query(q);

        let a = probing_seq
            .iter()
            .map(|pertub| {
                original_hash
                    .iter()
                    .zip(pertub)
                    .map(
                        |(&original, &shift)| {
                            if shift == 1 {
                                original * -1
                            } else {
                                original
                            }
                        },
                    )
                    .collect_vec()
            })
            .collect_vec();
        Ok(a)
    }
}

fn uniform_without_replacement<T: Copy>(bucket: &mut [T], n: usize) -> Vec<T> {
    // https://stackoverflow.com/questions/196017/unique-non-repeating-random-numbers-in-o1#196065
    let mut max_idx = bucket.len() - 1;
    let mut rng = create_rng(0);

    let mut samples = Vec::with_capacity(n);

    for _ in 0..n {
        let idx = rng.sample(Uniform::new(0, max_idx));
        debug_assert!(idx < bucket.len());
        unsafe {
            samples.push(*bucket.get_unchecked(idx));
        };
        bucket.swap(idx, max_idx);
        max_idx -= 1;
    }
    samples
}

fn create_hash_permutation(hash_len: usize, n: usize) -> Vec<i8> {
    let mut permut = vec![0; hash_len];
    let shift_options = [-1i8, 1];

    let mut idx: Vec<usize> = (0..hash_len).collect();
    let candidate_idx = uniform_without_replacement(&mut idx, n);

    let mut rng = create_rng(0);
    for i in candidate_idx {
        debug_assert!(i < permut.len());
        let v = *shift_options.choose(&mut rng).unwrap();
        // bounds check not needed as i cannot be larger than permut
        unsafe { *permut.get_unchecked_mut(i) += v }
    }
    permut
}

/// Retrieve perturbation indexes. Every index in a hash can be perturbed by +1 or -1.
///
/// First retrieve all hashes where 1 index is changed,
/// then all combinations where two indexes are changed etc.
///
/// # Arguments
/// * - `hash_length` The hash length is used to determine all the combinations of indexes that can be shifted.
/// * - `n_perturbation` The number of indexes allowed to be changed. We generally first deplete
/// * - `two_shifts` If true every index is changed by +1 and -1, else only by +1.
fn step_wise_perturb(
    hash_length: usize,
    n_perturbations: usize,
    two_shifts: bool,
) -> impl Iterator<Item = Vec<(usize, i8)>> {
    let multiply;
    if two_shifts {
        multiply = 2
    } else {
        multiply = 1
    }

    let idx = 0..hash_length * multiply;
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
    a
}

/// Generates new hashes by step wise shifting one indexes.
/// First all one index shifts are returned (these are closer to the original hash)
/// then the two index shifts, three index shifts etc.
///
/// This is done until the budget is depleted.
fn step_wise_probing(hash_len: usize, mut budget: usize, two_shifts: bool) -> Vec<Vec<i8>> {
    let mut hash_perturbs = Vec::with_capacity(budget);

    let n = hash_len as u64;
    // number of combinations (indexes we allow to perturb)
    let mut k = 1;
    while budget > 0 && k <= n {
        // binomial coefficient
        // times two as we have -1 and +1.
        let multiply;
        if two_shifts {
            multiply = 2
        } else {
            multiply = 1
        }
        let n_combinations = binomial(n, k) as usize * multiply;

        step_wise_perturb(n as usize, k as usize, two_shifts)
            .take(budget as usize)
            .for_each(|v| {
                let mut new_perturb = vec![0; hash_len];
                v.iter().for_each(|(idx, shift)| {
                    debug_assert!(*idx < new_perturb.len());
                    let v = unsafe { new_perturb.get_unchecked_mut(*idx) };
                    *v += *shift;
                });
                hash_perturbs.push(new_perturb)
            });
        k += 1;
        budget -= n_combinations;
    }
    hash_perturbs
}

#[derive(PartialEq, Clone)]
struct PerturbState<'a, N, K>
where
    N: Numeric + Float + Copy,
{
    // original sorted zj
    z: &'a [usize],
    // original xi(delta)
    distances: &'a [N],
    // selection of zjs
    // We start with the first one, as this is the lowest score.
    selection: Vec<usize>,
    switchpoint: usize,
    original_hash: Option<Vec<K>>,
}

impl<'a, N, K> PerturbState<'a, N, K>
where
    N: Numeric + Float,
    K: Integer,
{
    fn new(z: &'a [usize], distances: &'a [N], switchpoint: usize, hash: Vec<K>) -> Self {
        PerturbState {
            z,
            distances,
            selection: vec![0],
            switchpoint,
            original_hash: Some(hash),
        }
    }

    fn score(&self) -> N {
        let mut score = Zero::zero();
        for &index in self.selection.iter() {
            debug_assert!(index < self.z.len());
            let zj = unsafe { *self.z.get_unchecked(index) };
            debug_assert!(zj < self.distances.len());
            unsafe { score += self.distances.get_unchecked(zj).clone() };
        }
        score
    }

    // map zj value to (i, delta) as in paper
    fn i_delta(&self) -> Vec<(usize, K)> {
        let mut out = Vec::with_capacity(self.z.len());
        for &idx in self.selection.iter() {
            debug_assert!(idx < self.z.len());
            let zj = unsafe { *self.z.get_unchecked(idx) };
            let delta;
            let index;
            if zj >= self.switchpoint {
                delta = One::one();
                index = zj - self.switchpoint;
            } else {
                delta = K::from_i8(-1).unwrap();
                index = zj;
            }
            out.push((index, delta))
        }
        out
    }

    fn check_bounds(&mut self, max: usize) -> Result<()> {
        if max == self.z.len() - 1 {
            Err(Error::Failed("Out of bounds".to_string()))
        } else {
            self.selection.push(max + 1);
            Ok(())
        }
    }

    fn shift(&mut self) -> Result<()> {
        let max = self.selection.pop().unwrap();
        self.check_bounds(max)
    }

    fn expand(&mut self) -> Result<()> {
        let max = self.selection[self.selection.len() - 1];
        self.check_bounds(max)
    }

    fn gen_hash(&mut self) -> Vec<K> {
        let mut hash = self.original_hash.take().expect("hash already taken");
        for (i, delta) in self.i_delta() {
            debug_assert!(i < hash.len());
            let ptr = unsafe { hash.get_unchecked_mut(i) };
            *ptr += delta
        }
        hash
    }
}

// implement ordering so that we can create a min heap
impl<N, K> Ord for PerturbState<'_, N, K>
where
    N: Numeric + Float,
    K: Integer,
{
    fn cmp(&self, other: &PerturbState<N, K>) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<N, K> PartialOrd for PerturbState<'_, N, K>
where
    N: Numeric + Float,
    K: Integer,
{
    fn partial_cmp(&self, other: &PerturbState<N, K>) -> Option<Ordering> {
        other.score().partial_cmp(&self.score())
    }
}

impl<N, K> Eq for PerturbState<'_, N, K>
where
    N: Numeric + Float,
    K: Integer,
{
}

macro_rules! impl_query_directed_probe {
    ($vechash:ident) => {
        impl<N, K> $vechash<N, K>
        where
            N: Numeric + Float,
            K: Integer,
        {
            /// Computes the distance between the query hash and the boundary of the slot r (W in the paper)
            ///
            /// As stated by Multi-Probe LSH paper:
            /// For δ ∈ {−1, +1}, let xi(δ) be the distance of q from the boundary of the slot
            fn distance_to_bound(&self, q: &[N], hash: Option<&Vec<K>>) -> (Array1<N>, Array1<N>) {
                let hash = match hash {
                    None => self.hash_vec(q).to_vec(),
                    Some(h) => h.iter().map(|&k| N::from(k).unwrap()).collect_vec(),
                };
                let f = self.a.dot(&aview1(q)) + &self.b;
                let xi_min1 = f - &aview1(&hash) * self.r;
                let xi_plus1: Array1<N> = xi_min1.map(|x| self.r - *x);
                (xi_min1, xi_plus1)
            }
        }

        impl<N, K> QueryDirectedProbe<N, K> for $vechash<N, K>
        where
            N: Numeric + Float,
            K: Integer,
        {
            fn query_directed_probe(&self, q: &[N], budget: usize) -> Result<Vec<Vec<K>>> {
                // https://www.cs.princeton.edu/cass/papers/mplsh_vldb07.pdf
                // https://www.youtube.com/watch?v=c5DHtx5VxX8
                let hash = self.hash_vec_query(q);
                let (xi_min, xi_plus) = self.distance_to_bound(q, Some(&hash));
                // >= this point = +1
                // < this point = -1
                let switchpoint = xi_min.len();

                let distances: Vec<N> = stack!(Axis(0), xi_min, xi_plus).to_vec();

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
                    let mut ai = match heap.pop() {
                        Some(ai) => ai,
                        None => {
                            return Err(Error::Failed(
                                "All query directed probing combinations depleted".to_string(),
                            ))
                        }
                    };
                    let mut a_s = ai.clone();
                    let mut a_e = ai.clone();
                    if a_s.shift().is_ok() {
                        heap.push(a_s);
                    }
                    if a_e.expand().is_ok() {
                        heap.push(a_e);
                    }
                    hashes.push(ai.gen_hash())
                }
                Ok(hashes)
            }
        }
    };
}
impl_query_directed_probe!(L2);
impl_query_directed_probe!(MIPS);

impl<N, K, H, T> LSH<H, N, T, K>
where
    N: Numeric,
    K: Integer,
    H: VecHash<N, K>,
    T: HashTables<N, K>,
{
    pub fn multi_probe_bucket_union(&self, v: &[N]) -> Result<FnvHashSet<u32>> {
        self.validate_vec(v)?;
        let mut bucket_union = FnvHashSet::default();

        // Check if hasher has implemented this trait. If so follow this more specialized path.
        // Only L2 should have implemented it. This is the trick to choose a different function
        // path for the L2 struct.
        let h0 = &self.hashers[0];
        if h0.as_query_directed_probe().is_some() {
            for (i, hasher) in self.hashers.iter().enumerate() {
                if let Some(h) = hasher.as_query_directed_probe() {
                    let hashes = h.query_directed_probe(v, self._multi_probe_budget)?;
                    for hash in hashes {
                        self.process_bucket_union_result(&hash, i, &mut bucket_union)?
                    }
                }
            }
        } else if h0.as_step_wise_probe().is_some() {
            for (i, hasher) in self.hashers.iter().enumerate() {
                if let Some(h) = hasher.as_step_wise_probe() {
                    let hashes =
                        h.step_wise_probe(v, self._multi_probe_budget, self.n_projections)?;
                    for hash in hashes {
                        self.process_bucket_union_result(&hash, i, &mut bucket_union)?
                    }
                }
            }
        } else {
            unimplemented!()
        }
        Ok(bucket_union)
    }
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
        let a = step_wise_perturb(4, 2, true);
        assert_eq!(
            vec![vec![(0, 1), (1, 1)], vec![(0, 1), (2, 1)]],
            a.take(2).collect_vec()
        );
    }

    #[test]
    fn test_step_wise_probe() {
        let a = step_wise_probing(4, 20, true);
        assert_eq!(vec![1, 0, 0, 0], a[0]);
        assert_eq!(vec![0, 1, -1, 0], a[a.len() - 1]);
    }

    #[test]
    fn test_l2_xi_distances() {
        let l2 = L2::<f32>::new(4, 4., 3, 1);
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
        let a0 = PerturbState::new(&z, &distances, switchpoint, vec![0, 0, 0, 0]);
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
        ae.expand().unwrap();
        assert_eq!(ae.gen_hash(), [0, -1, 1, 0]);
        assert_eq!(ae.score(), 0.1 + 0.8);
        assert_eq!(ae.selection, [0, 1]);

        // after shift operation selection is [1]
        // This leads to:
        //   distance/ score:   0.8
        //   index:             2
        //   delta:             1
        let mut a_s = a0.clone();
        a_s.shift().unwrap();
        assert_eq!(a_s.gen_hash(), [0, 0, 1, 0]);
        assert_eq!(a_s.score(), 0.8);
        assert_eq!(a_s.selection, [1]);
    }

    #[test]
    fn test_query_directed_probe() {
        let l2 = <L2>::new(4, 4., 3, 1);
        let hashes = l2.query_directed_probe(&[1., 2., 3., 1.], 4).unwrap();
        println!("{:?}", hashes)
    }

    #[test]
    fn test_query_directed_bounds() {
        // if shift and expand operation have reached the end of the vecs an error should be returned
        let mut lsh = hi8::LshMem::new(2, 1, 1).multi_probe(1000).l2(4.).unwrap();
        lsh.store_vec(&[1.]).unwrap();
        assert!(lsh.query_bucket_ids(&[1.]).is_err())
    }
}
