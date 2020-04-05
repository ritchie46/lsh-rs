use crate::activations::Activation;

pub enum Loss<'a> {
    MSE(&'a Activation),
    NLL(&'a Activation),
}

impl<'a> Loss<'a> {
    fn activation_fn(&self) -> &Activation {
        use Loss::*;
        match self {
            MSE(a) => a,
            NLL(a) => a,
        }
    }

    pub fn loss(&self, y_true: f32, y_pred: f32) -> f32 {
        use Loss::*;
        match self {
            MSE(_) => (y_pred - y_true).powf(2.),
            NLL(_) => {
                if y_true == 1. {
                    -y_pred.ln()
                } else {
                    -(1. - y_pred).ln()
                }
            }
        }
    }

    pub fn prime(&self, y_true: f32, y_pred: f32) -> f32 {
        use Loss::*;
        match self {
            MSE(_) => y_pred - y_true,
            NLL(_) => {
                if y_true == 1. {
                    -1. / y_pred
                } else {
                    1. / (1. - y_pred)
                }
            }
        }
    }

    pub fn delta(&self, y_true: f32, y_pred: f32) -> f32 {
        self.prime(y_true, y_pred) * self.activation_fn().prime(y_pred)
    }
}
