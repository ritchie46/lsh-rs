use crate::activations::Activation;
use crate::{activations, loss::Loss};
use fnv::FnvHashMap;
use lsh_rs::{DataPoint, DataPointSlice, LshMem, SignRandomProjections};
use ndarray::prelude::*;
use ndarray_rand::rand_distr::{StandardNormal, Uniform};
use ndarray_rand::RandomExt;

type Perceptron = Array1<f32>;

struct MemArena {
    pool: Vec<Perceptron>,
    // Freed indexes will be added to the free buffer.
    free: Vec<usize>,
}

impl MemArena {
    fn new() -> Self {
        MemArena {
            pool: vec![],
            free: vec![],
        }
    }

    fn add(&mut self, p: Perceptron) -> usize {
        match self.free.pop() {
            Some(idx) => {
                self.pool.insert(idx, p);
                idx
            }
            None => {
                self.pool.push(p);
                self.pool.len() - 1
            }
        }
    }

    fn get(&self, idx: &[usize]) -> Vec<&Perceptron> {
        idx.iter()
            .map(|&idx| self.pool.get(idx).expect("out of bounds idx"))
            .collect()
    }
}

pub struct Network {
    w: Vec<Vec<u32>>,
    // biases for all layers
    lsh2bias: Vec<FnvHashMap<u32, f32>>,
    activations: Vec<Activation>,
    lsh_store: Vec<LshMem<SignRandomProjections>>,
    n_layers: usize,
    pool: MemArena,
    lsh2pool: Vec<FnvHashMap<u32, usize>>,
    dimensions: Vec<usize>,
}

impl Network {
    ///   Example of one hidden layer with
    ///         - 2 inputs
    ///         - 3 hidden nodes
    ///         - 3 outputs
    ///
    ///         layers -->    [0,        1,          2]
    ///         ----------------------------------------
    ///         dimensions =  (2,     3,          3)
    ///         activations = (      ReLU,      Sigmoid)
    pub fn new(
        dimensions: Vec<usize>,
        activations: Vec<Activation>,
        n_projections: usize,
        n_hash_tables: usize,
    ) -> Self {
        let n_layers = dimensions.len();
        let mut w = Vec::with_capacity(n_layers);
        let mut pool = MemArena::new();
        let mut lsh2pool = Vec::with_capacity(n_layers);
        let mut lsh2bias = Vec::with_capacity(n_layers);
        let mut lsh_store = Vec::with_capacity(n_layers);

        for i in 0..(n_layers - 1) {
            let mut lsh2pool_i = FnvHashMap::default();
            let mut lsh2bias_i = FnvHashMap::default();

            let in_size = dimensions[i];
            let out_size = dimensions[i + 1];
            let n_perceptrons = in_size * out_size;
            let mut w_idx = Vec::with_capacity(n_perceptrons);

            let mut lsh = LshMem::new(n_projections, n_hash_tables, in_size)
                .srp()
                .unwrap();

            for _ in 0..out_size {
                let p = Array1::random(in_size, StandardNormal);
                let p = p / (in_size as f32).powf(0.5);

                let lsh_idx = lsh.store_vec(p.as_slice().unwrap()).unwrap();
                let pool_idx = pool.add(p);
                lsh2pool_i.insert(lsh_idx, pool_idx);
                lsh2bias_i.insert(lsh_idx, 0.);
                w_idx.push(lsh_idx);
            }

            lsh2pool.push(lsh2pool_i);
            lsh2bias.push(lsh2bias_i);
            lsh_store.push(lsh);
            w.push(w_idx);
        }

        Network {
            w,
            lsh2bias,
            activations,
            lsh_store,
            n_layers,
            pool,
            lsh2pool,
            dimensions,
        }
    }

    // fn get_perceptrons(&self, idx: &[u32]) -> Vec<&Perceptron> {
    //     let pool_idx: Vec<usize> = idx
    //         .iter()
    //         .map(|idx| *self.lsh2pool.get(idx).expect("out of bounds idx"))
    //         .collect();
    //     self.pool.get(&pool_idx)
    // }

    fn get_biases(&self, layer: usize, idx: &[u32]) -> Vec<f32> {
        let lsh2bias = self.lsh2bias.get(layer).expect("Could not get bias layer");
        idx.iter()
            .map(|idx| *lsh2bias.get(idx).expect("Could not get bias"))
            .collect()
    }

    fn apply_layer(&self, i: usize, input: &[f32]) -> Vec<Computation> {
        let lsh = &self.lsh_store[i];
        let activation = &self.activations[i];
        let idx_j = lsh.query_bucket_ids(input).unwrap();
        let bias = self.get_biases(i, &idx_j);

        // index of the vectors in the pool
        let lsh2pool_i = &self.lsh2pool[i];
        let k: Vec<usize> = idx_j
            .iter()
            .map(|idx| *(lsh2pool_i.get(idx).unwrap()))
            .collect();
        let ps = self.pool.get(&k);

        ps.iter()
            .zip(bias)
            .zip(idx_j)
            .zip(k)
            .map(|(((&p, b), j), k)| {
                let j = j as usize;
                let z = aview1(input).dot(p) + b;
                let a = activation.activate(z);
                Computation { i, j, z, a, k }
            })
            .collect()
    }

    pub fn forward(&self, x: &[f32]) -> Vec<Vec<Computation>> {
        let mut comp = Vec::with_capacity(self.n_layers);

        // first layer
        let prev_comp = self.apply_layer(0, x);
        comp.push(prev_comp);

        for i in 1..self.n_layers - 1 {
            let prev_comp = comp.last().unwrap();
            let input = make_input_next_layer(prev_comp, self.dimensions[i]);
            comp.push(self.apply_layer(i, &input))
        }
        comp
    }

    pub fn backprop(&self, comp: &[Vec<Computation>], y_true: &[f32]) {
        // determine partial derivative and delta for output layer
        let last_activation = &self.activations[self.activations.len() - 1];

        // iter only over the activations of the last layer
        // the loop is over all the perceptrons in one layer.
        for c in &comp[comp.len() - 1] {
            let delta = Loss::MSE(last_activation).delta(y_true[c.i], c.a);
            let dw = c.a * delta;
            // TODO: update params

            // Per perceptron we traverse back all the layers (except the input)
            for i in ((self.n_layers - 1)..1).step_by(2) {}
        }
    }
}

#[derive(Debug)]
pub struct Computation {
    // the ith layer in the network
    i: usize,
    // the jth perceptron in the layer (same as lsh idx)
    j: usize,
    // the kth layer in the memory pool
    k: usize,
    z: f32,
    a: f32,
}

fn make_input_next_layer(prev_comp: &[Computation], layer_size: usize) -> Vec<f32> {
    // The previous layer had only a few of all possible activation.
    // create a new zero filled vector where only the activations are filled.

    let mut layer = vec![0.; layer_size];
    prev_comp.iter().for_each(|c| layer[c.j] = c.a);
    layer
}
