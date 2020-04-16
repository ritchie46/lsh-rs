use ndarray::ArrayView1;

pub fn l2_norm(x: ArrayView1<f32>) -> f32 {
    x.dot(&x).sqrt()
}
