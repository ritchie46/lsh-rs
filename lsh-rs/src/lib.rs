#![allow(dead_code)]
#[cfg(feature = "blas")]
extern crate blas_src;
extern crate ndarray;
mod hash;
pub mod lsh;
mod table;
pub mod utils;
pub use table::MemoryTable;
