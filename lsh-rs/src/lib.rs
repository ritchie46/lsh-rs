#![allow(dead_code)]
#[cfg(feature = "blas")]
extern crate blas_src;
extern crate ndarray;
mod hash;
pub mod lsh;
mod table;
pub mod utils;
pub use crate::lsh::LSH;
pub use hash::{SignRandomProjections, L2, MIPS};
pub use table::MemoryTable;
