extern crate blas_src;
extern crate minifb;
extern crate mnist;
extern crate ndarray;
use minifb::{Window, WindowOptions};
pub mod activations;
pub mod loss;
mod network;
mod test;
use crate::activations::Activation;
use crate::loss::Loss;
use crate::network::Network;
use mnist::{Mnist, MnistBuilder};
use ndarray::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;
use std::sync::Mutex;

enum Error {
    StopIter,
}

type Result<T> = std::result::Result<T, Error>;

const LABELS: usize = 10;
const N_PIXELS: usize = 784;
const WIDTH: usize = 28;
const BATCH_SIZE: usize = 64;

fn one_hot_encode(idx: usize, n: usize) -> Vec<u8> {
    let mut ohe = vec![0_u8; n];
    ohe[idx] = 1;
    ohe
}

fn get_argmax<T>(a: &[T]) -> usize
where
    T: std::cmp::PartialOrd,
{
    let (argmax, _val) = a
        .iter()
        .enumerate()
        .fold((0, &a[0]), |(max_idx, max_val), (i, v)| {
            if v > max_val {
                (i, v)
            } else {
                (max_idx, max_val)
            }
        });
    argmax
}

struct DataSet {
    x: Vec<f32>,
    y_ohe: Vec<u8>,
    idx: Vec<usize>,
    batch_size: usize,
    counter: usize,
    n_total: usize,
}

impl DataSet {
    fn new(x: Vec<u8>, y: Vec<u8>, batch_size: usize) -> Self {
        let n_total = x.len() / N_PIXELS;
        let x = x.iter().map(|&v| v as f32 / 255. - 0.5).collect();
        let idx: Vec<usize> = (0..n_total).collect();

        let y_ohe: Vec<u8> = y
            .iter()
            .flat_map(|&y_i| one_hot_encode(y_i as usize, LABELS))
            .collect();

        DataSet {
            x,
            y_ohe,
            idx,
            batch_size,
            counter: 0,
            n_total,
        }
    }

    fn to_shuf_idx(&self, idx: &[usize]) -> Vec<usize> {
        idx.iter().map(|&i| self.idx[i]).collect()
    }

    fn get_tpl_pairs(&self, idx: &[usize]) -> Vec<(&[f32], &[u8])> {
        let idx = self.to_shuf_idx(idx);
        let xy: Vec<(&[f32], &[u8])> = idx
            .iter()
            .map(|&i| {
                let x_off = i * N_PIXELS;
                let x = &self.x[x_off..x_off + N_PIXELS];
                let y_off = i * LABELS;
                let y = &self.y_ohe[y_off..y_off + LABELS];
                (x, y)
            })
            .collect();
        xy
    }

    fn shuffle(&mut self) {
        self.idx.shuffle(&mut thread_rng())
    }

    fn get_batch(&mut self) -> Result<Vec<(&[f32], &[u8])>> {
        let idx: Vec<usize> = (self.counter..self.counter + self.batch_size).collect();
        self.counter += self.batch_size;
        if self.counter > self.n_total {
            self.counter = 0;
            return Err(Error::StopIter);
        }
        Ok(self.get_tpl_pairs(&idx))
    }

    fn show_image(&self, idx: usize) {
        let mut window = Window::new("", WIDTH, WIDTH, WindowOptions::default()).unwrap();
        let xy = self.get_tpl_pairs(&[idx])[0];
        let x: Vec<u32> = xy.0.iter().map(|&v| (v * 255.) as u32).collect();
        let y = xy.1;
        window.limit_update_rate(Some(std::time::Duration::from_micros(16600)));
        println!("{}", get_argmax(y));

        while window.is_open() {
            window.update_with_buffer(&x, WIDTH, WIDTH).unwrap();
        }
    }
}

fn main() {
    let (trn_size, rows, cols) = (50_000, WIDTH, WIDTH);
    let mut p = std::fs::canonicalize(".").unwrap();
    p.push("data");
    println!("mnist path: {:?}", &p);

    // Deconstruct the returned Mnist struct.
    let Mnist {
        trn_img, trn_lbl, ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(trn_size)
        .validation_set_length(0)
        .test_set_length(0)
        .base_path(&p.to_str().unwrap())
        .finalize();

    let mut ds = DataSet::new(trn_img, trn_lbl, BATCH_SIZE);

    let mut m = Network::new(
        vec![N_PIXELS, 512, 10],
        vec![Activation::ReLU, Activation::Sigmoid],
        8,
        50,
        0.01 / BATCH_SIZE as f32,
        0,
        "nll",
    );

    for epoch in 0..50 {
        println!("epoch {}", epoch);
        ds.shuffle();

        let mut c = 0;
        let mut correct = 0;

        while let Ok(xy) = ds.get_batch() {
            c += 1;

            let mut inputs_neurons_batch = Mutex::new(Vec::with_capacity(BATCH_SIZE));

            let loss: f32 = xy
                .par_iter()
                .enumerate()
                .map(|(i, (x, y))| {
                    let (mut neurons, input) = m.forward(x);
                    let mut lock = inputs_neurons_batch.lock().unwrap();

                    let loss = m.backprop(&mut neurons, &y);
                    lock.push((input, neurons, y));
                    loss
                })
                .sum::<f32>()
                / BATCH_SIZE as f32;
            let inputs_neurons_batch = inputs_neurons_batch.into_inner().unwrap();

            for (input, neurons, _) in inputs_neurons_batch.iter() {
                for (n, input) in neurons.iter().zip(input) {
                    m.update_param(input, n)
                }
            }

            if c % 5 == 0 {
                m.rehash();
            }

            if inputs_neurons_batch.len() == 0 {
                continue;
            }
            let (_, neurons, y) = &inputs_neurons_batch[inputs_neurons_batch.len() - 1];
            // let neurons = &inputs_neurons_batch[inputs_neurons_batch.len() - 1].1;
            let output_layer = &neurons[neurons.len() - 1];

            if output_layer.len() > 0 {
                let activations: Vec<f32> = output_layer.iter().map(|c| c.a).collect();
                let y_pred_idx = get_argmax(&activations);
                let y_pred = output_layer[y_pred_idx].j;
                let y_true = get_argmax(y);
                if y_true == y_pred {
                    correct += 1
                }
                let hidden_layer = &neurons[0];
                if c % 10 == 0 {
                    println!(
                        "loss: {} y_pred {} y_true: {} accuracy: {} n_activations_hidden_layer: {}",
                        loss,
                        y_pred,
                        y_true,
                        correct as f32 / c as f32,
                        hidden_layer.len()
                    )
                }
            }
        }
        m.lr *= 0.99
    }
}
