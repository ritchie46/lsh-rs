use ndarray::LinalgScalar;
use num::{NumCast, ToPrimitive};

pub trait Numeric: LinalgScalar + NumCast + ToPrimitive + Send + Sync {}

impl Numeric for f32 {}
