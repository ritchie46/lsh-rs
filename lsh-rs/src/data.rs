use ndarray::{LinalgScalar, ScalarOperand};
use num::{FromPrimitive, NumCast, ToPrimitive};
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize};
use std::any::Any;
use std::cmp::{PartialEq, PartialOrd};
use std::ops::AddAssign;

pub trait Numeric:
    LinalgScalar
    + ScalarOperand
    + NumCast
    + ToPrimitive
    + Send
    + Sync
    + PartialEq
    + PartialOrd
    + FromPrimitive
    + AddAssign
    + Serialize
{
}

impl Numeric for f32 {}
impl Numeric for f64 {}
