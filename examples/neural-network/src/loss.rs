use crate::activations::Activation;

pub enum Loss<'a> {
    MSE(&'a Activation),
}

impl<'a> Loss<'a> {
    fn activation_fn(&self) -> &Activation {
        use Loss::*;
        match self {
            MSE(a) => a,
        }
    }

    pub fn activation(&self, z: f32) -> f32 {
        use Loss::*;
        match self {
            MSE(a) => a.activate(z),
        }
    }

    pub fn loss(&self, y_true: f32, y_pred: f32) -> f32 {
        use Loss::*;
        match self {
            MSE(_) => (y_pred - y_true).powf(2.),
        }
    }

    pub fn prime(&self, y_true: f32, y_pred: f32) -> f32 {
        use Loss::*;
        match self {
            MSE(_) => y_pred - y_true,
        }
    }

    pub fn delta(&self, y_true: f32, y_pred: f32) -> f32 {
        self.prime(y_true, y_pred) * self.activation_fn().prime(y_pred)
    }
}
