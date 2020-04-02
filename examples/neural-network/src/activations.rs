use ndarray::prelude::*;

pub enum Activation {
    ReLU,
    None,
}

impl Activation {
    pub fn activate(&self, z: f32) -> f32 {
        use Activation::*;
        match self {
            ReLU => {
                if z > 0. {
                    z
                } else {
                    0.
                }
            }
            None => z,
        }
    }

    pub fn prime(&self, z: f32) -> f32 {
        use Activation::*;
        match self {
            ReLU => {
                if z > 0. {
                    1.
                } else {
                    0.
                }
            }
            None => 1.,
        }
    }
}
