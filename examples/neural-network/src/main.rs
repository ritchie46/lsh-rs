extern crate blas_src;
extern crate mnist;
extern crate ndarray;
pub mod activations;
pub mod loss;
mod network;
mod test;
use crate::activations::Activation;
use crate::network::Network;
use mnist::{Mnist, MnistBuilder};
use ndarray::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;

enum Error {
    StopIter,
}

type Result<T> = std::result::Result<T, Error>;

const LABELS: usize = 10;
const PIXEL_OFFSET: usize = 784;

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
        let n_total = x.len() / 784;
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
                let x_off = i * PIXEL_OFFSET;
                let x = &self.x[x_off..x_off + PIXEL_OFFSET];
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
}

fn main() {
    let (trn_size, rows, cols) = (50_000, 28, 28);
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

    let mut ds = DataSet::new(trn_img, trn_lbl, 64);
    // Start with an LSH that collides everything. This way we can check if the model learns first.
    let mut m = Network::new(
        vec![PIXEL_OFFSET, 256, 10],
        vec![Activation::ReLU, Activation::Sigmoid],
        9,
        50,
        0.001,
        0,
    );

    for epoch in 0..10 {
        println!("epoch {}", epoch);
        ds.shuffle();

        let mut c = 0;
        while let Ok(xy) = ds.get_batch() {
            c += 1;
            let mut r = vec![];
            let mut loss = 0.;

            // TODO: Utilize batch? Store results?
            for (x, y) in &xy {
                let (r_, _) = m.forward(x);
                r = r_;
                loss += m.backprop(&mut r, &y);
            }
            if c % 3 == 0 {
                m.rehash_all();
            }

            let output_layer = &r[r.len() - 1];
            if output_layer.len() > 0 && c % 10 == 0 {
                let (_, y) = xy[xy.len() - 1];

                let activations: Vec<f32> = output_layer.iter().map(|c| c.a).collect();
                let y_pred_idx = get_argmax(&activations);
                let y_pred = output_layer[y_pred_idx].j;
                let y_true = get_argmax(y);

                println!("{:?}", (loss, y_pred, y_true, activations))
            }
        }
    }

    println!("Hello, world! {:?}", ds.get_tpl_pairs(&[1, 2])[0].1);
}
