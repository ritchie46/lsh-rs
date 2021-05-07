//! Generic traits for numeric input and hash outputs.
use ndarray::{LinalgScalar, ScalarOperand};
use num::{FromPrimitive, NumCast, ToPrimitive};
use serde::Serialize;
use std::fmt::{Debug, Display};
use std::cmp::{Ord, PartialEq, PartialOrd};
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

pub trait Integer: Numeric + Ord + Eq + Hash {}
impl Integer for u8 {}
impl Integer for u16 {}
impl Integer for u32 {}
impl Integer for u64 {}

impl Integer for i8 {}
impl Integer for i16 {}
impl Integer for i32 {}
impl Integer for i64 {}
