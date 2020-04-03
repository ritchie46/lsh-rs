#![cfg(test)]
use crate::activations::Activation;
use crate::network::Network;

fn get_model() -> Network {
    let dim = vec![2, 3, 4];
    let act = vec![Activation::ReLU, Activation::None];
    let m = Network::new(dim, act, 3, 10, 0.01, 1);
    m
}

#[test]
fn test_shapes() {
    let m = get_model();
    assert_eq!(m.w[0].len(), 3);
    assert_eq!(m.w[1].len(), 4);
}

#[test]
fn test_flow() {
    // check if tensors can flow
    let mut m = get_model();
    let input = &[0.2, -0.2];
    let comp = m.forward(input);

    let w_before = m.get_weight(0, 0).clone();
    let y_true = &[1., -1., 0.5, 1.];
    m.backprop(&comp, y_true);
    let w_after = m.get_weight(0, 0).clone();
    assert![w_before[0] != w_after[0]];
}
