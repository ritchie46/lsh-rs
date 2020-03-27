//! # lsh-rs (Locality Sensitive Hashing)
//!
//! Locality sensitive hashing can help retrieving Approximate Nearest Neighbors in sub-linear time.
//!
//! For more information on the subject see:
//! * [Introduction on LSH](http://people.csail.mit.edu/gregory/annbook/introduction.pdf)
//! * [Section 2. describes the hash families used in this crate](https://arxiv.org/pdf/1411.3787.pdf)
//!
//! ## Hashing implementations
//! * Signed Random Projections (Cosine similarity)
//! * L2 distance
//! * Maximum Inner Product (Dot products)
//!
//! ## Getting started
//!
//! ```rust
//! use lsh_rs::LshSql;
//! // 2 rows w/ dimension 3.
//! let p = &[vec![1., 1.5, 2.],
//!         vec![2., 1.1, -0.3]];
//!
//! // Do one time expensive preprocessing.
//! let n_projections = 9;
//! let n_hash_tables = 30;
//! let dim = 3;
//! let mut lsh = LshSql::new(n_projections, n_hash_tables, dim).srp();
//! lsh.store_vecs(p);
//!
//! // Query in sublinear time.
//! let query = &[1.1, 1.2, 1.2];
//! lsh.query_bucket(query);
//! ```
//!
//! ## Signed Random Projections
//! LSH for maximum cosine similarity search.
//! ```rust
//! let mut lsh = LshSql::new(n_projections, n_hash_tables, dim).srp();
//! ```
//!
//! ## L2
//! LSH for minimal L2 distance search.
//!
//! ```
//! // hyper parameter r in https://arxiv.org/pdf/1411.3787.pdf (eq. 8)
//! let bucket_width = 2.2;
//! let mut lsh = LshSql::new(n_projections, n_hash_tables, dim).l2(bucket_width);
//! ```
//!
//! ## Maximum Inner Product (MIPS)
//! LSH for maximum inner product search.
//! ```rust
//! let bucket_width = 2.2;
//! // l2(x) < U < 1.0
//! let U = 0.83;
//! // number of concatenations
//! let m = 3;
//! let mut lsh = LshSql::new(n_projections, n_hash_tables, dim).mips(r, U, m);
//! ```
//!
//! ## Seed
//! Random projections are used to generate the hash functions. The default seeding of randomness
//! is taken from the system. If you want to have reproducable outcomes, you can set a manual seed.
//!
//! ```rust
//! let mut lsh = LshSql::new(n_projections, n_hash_tables, dim).seed(12).srp();
//! ```
//!
//! ## Unique indexes
//! Instead of storing data points as vectors. Storing `L` copies of the data points (one in every
//! hash table). You can choose to only store unique indexes of the data points. The index ids are
//! assigned in chronological order. This will drastically decrease the required memory.
//! ```rust
//! let mut lsh = LshSql::new(n_projections, n_hash_tables, dim).only_index().srp();
//! ```
//!
//! ## BLAS support
//! Utilizing [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) will heavily increase
//! performance. To make use of BLAS, install `lsh-rs` w/ `"blas"` feature and reinstall `ndarray` w/ `"blas"` support.
//!  <br>
//!  <br>
//! **Cargo.toml:**
//! ```toml
//! lsh-rs = {version ="x.x"}, features=["blas"]}
//! ndarray = {version = "0.13", features=["blas"]}
//! # Or any other blas backend.
//! blas-src = { version = "0.5", defeault-features = false, features = ["openblas"]}
//! ```
#![allow(dead_code, non_snake_case)]
#[cfg(feature = "blas")]
extern crate blas_src;
extern crate ndarray;
mod hash;
mod lsh {
    pub mod lsh;
    mod test;
}
mod multi_probe;
mod table {
    pub mod general;
    pub mod mem;
    pub mod sqlite;
    pub mod sqlite_mem;
}
mod error;
pub mod utils;
pub use crate::lsh::lsh::{LshMem, LshSql, LshSqlMem, LSH};
pub use hash::{Hash, HashPrimitive, SignRandomProjections, VecHash, L2, MIPS};
pub use table::{general::HashTables, mem::MemoryTable, sqlite::SqlTable};
#[cfg(feature = "stats")]
pub mod stats;

pub type DataPoint = Vec<f32>;
pub type DataPointSlice = [f32];
pub use error::Error;
pub type Result<T> = std::result::Result<T, Error>;
