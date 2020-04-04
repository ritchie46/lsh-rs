#![cfg(test)]
use crate::activations::Activation;
use crate::network::Network;
use ndarray::prelude::*;

fn get_model(output_size: usize) -> Network {
    let dim = vec![2, 3, output_size];
    let act = vec![Activation::ReLU, Activation::None];
    // increase the probability of all neurons being selected by having many hash_tables
    // and the simples hash value, i.e. a bit.
    let m = Network::new(dim, act, 1, 100, 0.01, 1);
    m
}

#[test]
fn test_shapes() {
    let m = get_model(4);
    assert_eq!(m.w[0].len(), 3);
    assert_eq!(m.w[1].len(), 4);
}

#[test]
fn test_flow() {
    // check if tensors can flow
    let mut m = get_model(4);
    let input = &[0.2, -0.2];
    let (mut comp, _) = m.forward(input);

    let w_before = m.get_weight(0, 0).clone();
    let y_true = &[0, 1, 0, 0];
    m.backprop(&mut comp, y_true);
    let w_after = m.get_weight(0, 0).clone();
    assert![w_before[0] != w_after[0]];
}

#[test]
fn test_gradients() {
    let mut m = get_model(2);
    let pool = &mut m.pool.pool;
    // input dim = 2, hidden layer = 3
    // so the weights matrix is 2 x 3

    let x = &[1., -1.]; //       z   a
    pool[0] = array![1., -1.]; // 1 + 1 = 2   2
    pool[1] = array![2., 2.]; // 2 - 2 =  0   0
    pool[2] = array![4., 3.]; // 4 - 3 =  1   1

    // second layer
    // weight matrix = 3 x 2
    // input = // [2, 0, 1]
    pool[3] = array![1., 0.5, 0.5]; // 2 + 0 + 0.5 = 2.5
    pool[4] = array![0.5, -0.2, 0.2]; // 1 + 0 + 0.2 = 1.2

    // get first layer
    let (mut neurons, inputs) = m.forward(x);
    let layer_1 = &neurons[0];
    let layer_2 = &neurons[1];
    for n in layer_1 {
        match n.j {
            0 => assert_eq!(n.z, 2.),
            1 => assert_eq!(n.z, 0.),
            2 => assert_eq!(n.z, 1.),
            _ => panic!("this neuron was not expected."),
        }
    }

    for n in layer_2 {
        match n.j {
            0 => assert_eq!(n.z, 2.5),
            1 => assert_eq!(n.z, 1.2),
            _ => panic!("this neuron was not expected."),
        }
    }

    m.backprop(&mut neurons, &[0, 1]);
    // j     a      y    δ (y - a)
    // 0    2.5     0   2.5
    // 1    1.2     1   0.2

    let layer_1 = &neurons[0];
    let layer_2 = &neurons[1];
    for n in layer_2 {
        match n.j {
            0 => assert_eq!(n.delta, 2.5),
            1 => assert_eq!((n.delta * 1000.).round() / 1000., 0.2),
            _ => panic!("this neuron was not expected."),
        }
    }
    // now follow the gradient: δj = 2.5 and 0.2
    // δ2 = δj * w2 * relu_prime (1 or 0). In this case 1, otherwise its not an interesting case
    // layer w2 = [1., 0.5, 0.5]
    // 2.5 * [1., 0.5, 0.5] * 1 =   [2.5, 1.25, 1.25]
    // 0.2 * [0.5, -0.2, 0.2] * 1 = [0.1, -0.04, 0.04]
    // sum of above:                ------------------- +
    //                              [2.6,  1.21, 1.29]
    // jth node:                     0,     1,     2
    for n in layer_1 {
        match n.j {
            0 => assert_eq!(n.delta, 2.6),
            1 => assert_eq!(n.delta, 1.21),
            2 => assert_eq!(n.delta, 1.29),
            _ => panic!("this neuron was not expected."),
        }
    }
    println!("{:?}", neurons[1]);
    println!("{:?}", neurons[0]);

    for (n, input) in neurons.iter().zip(&inputs) {
        m.update_param(input, n)
    }
}
