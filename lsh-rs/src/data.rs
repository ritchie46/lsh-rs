use ndarray::LinalgScalar;
use num::{FromPrimitive, NumCast, ToPrimitive};
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize};
use std::any::Any;
use std::cmp::{PartialEq, PartialOrd};

pub trait Numeric:
    LinalgScalar
    + NumCast
    + ToPrimitive
    + Send
    + Sync
    + PartialEq
    + PartialOrd
    + FromPrimitive
    + Serialize
{
}

impl Numeric for f32 {}
