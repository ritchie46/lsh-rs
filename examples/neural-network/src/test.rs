#![cfg(test)]
use crate::activations::Activation;
use crate::network::Network;

fn get_model() -> Network {
    let dim = vec![2, 3, 3];
    let act = vec![Activation::ReLU, Activation::None];
    let m = Network::new(dim, act, 3, 3);
    m
}

#[test]
fn test_forward() {
    // check if tensors can flow
    let m = get_model();
    let input = &[0.2, -0.2];
    let comp = m.forward(input);
}
