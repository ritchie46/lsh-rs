use ndarray::{LinalgScalar, ScalarOperand};
use num::{FromPrimitive, NumCast, ToPrimitive};
use serde::export::fmt::{Debug, Display};
use serde::Serialize;
use std::cmp::{PartialEq, PartialOrd};
use std::hash::Hash;
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
    + Debug
    + Display
{
}

impl Numeric for f32 {}
impl Numeric for f64 {}
impl Numeric for i8 {}
impl Numeric for i16 {}
impl Numeric for i32 {}
impl Numeric for i64 {}
impl Numeric for u8 {}
impl Numeric for u16 {}
impl Numeric for u32 {}
impl Numeric for u64 {}

pub trait Integer:
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
    + Debug
    + Eq
    + Display
    + Hash
{
}
impl Integer for i8 {}
impl Integer for i16 {}
impl Integer for i32 {}
impl Integer for i64 {}
