//! # lsh-rs (Locality Sensitive Hashing)
//!
//! Locality sensitive hashing can help retrieving Approximate Nearest Neighbors in sub-linear time.
//!
//! For more information on the subject see:
//! * [Introduction on LSH](http://people.csail.mit.edu/gregory/annbook/introduction.pdf)
//! * [Section 2. describes the hash families used in this crate](https://arxiv.org/pdf/1411.3787.pdf)
//! * [LSH and neural networks](https://www.ritchievink.com/blog/2020/04/07/sparse-neural-networks-and-hash-tables-with-locality-sensitive-hashing/)
//!
//! ###
//!
//! ## Implementations
//!
//! * **Base LSH**
//!     - Signed Random Projections *(Cosine similarity)*
//!     - L2 distance
//!     - MIPS *(Dot products/ Maximum Inner Product Search)*
//!     - MinHash *(Jaccard Similarity)*
//! * **Multi Probe LSH**
//!     - **Step wise probing**
//!         - SRP (only bit shifts)
//!     - **Query directed probing**
//!         - L2
//!         - MIPS
//! * Generic numeric types
//!
//! ## Features
//! * "blas"
//! * "sqlite"
//!
//! ## Getting started
//!
//! ```rust
//! use lsh_rs::prelude::*;
//! // 2 rows w/ dimension 3.
//! let p = &[vec![1., 1.5, 2.],
//!         vec![2., 1.1, -0.3]];
//!
//! // Do one time expensive preprocessing.
//! let n_projections = 9;
//! let n_hash_tables = 30;
//! let dim = 10;
//! let dim = 3;
//! let mut lsh = LshMem::new(n_projections, n_hash_tables, dim)
//!     .srp()
//!     .unwrap();
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
//! # use lsh_rs::prelude::*;
//! # let n_projections = 9;
//! # let n_hash_tables = 30;
//! # let dim = 10;
//! let mut lsh = LshMem::<_, f32>::new(n_projections, n_hash_tables, dim)
//!     .srp()
//!     .unwrap();
//! ```
//!
//! ## L2
//! LSH for minimal L2 distance search.
//!
//! ```
//! // hyper parameter r in https://arxiv.org/pdf/1411.3787.pdf (eq. 8)
//! # use lsh_rs::prelude::*;
//! # let bucket_width = 2.2;
//! # let n_projections = 9;
//! # let n_hash_tables = 10;
//! # let dim = 10;
//! let mut lsh = LshMem::<_, f32>::new(n_projections, n_hash_tables, dim)
//!     .l2(bucket_width)
//!     .unwrap();
//! ```
//!
//! ## Jaccard Index
//! LSH for the Jaccard Index
//! ```rust
//! # use lsh_rs::prelude::*;
//! # let n_projections = 14;
//! // length of the shingles vector
//! let dim = 2500;
//! # let n_hash_tables = 10;
//! let mut lsh = LshSqlMem::<_, u16>::new(n_projections, n_hash_tables, dim)
//!     .minhash()
//!     .unwrap();
//! ```
//!
//! ## Maximum Inner Product (MIPS)
//! LSH for maximum inner product search.
//! ```rust
//! # use lsh_rs::prelude::*;
//! let bucket_width = 2.2;
//! // l2(x) < U < 1.0
//! let U = 0.83;
//! let r = 4.;
//! // number of concatenations
//! let m = 3;
//! let n_projections = 15;
//! let n_hash_tables = 10;
//! let dim = 10;
//! let mut lsh = LshMem::<_, f32>::new(n_projections, n_hash_tables, dim)
//!     .mips(r, U, m)
//!     .unwrap();
//! ```
//!
//! ## Seed
//! Random projections are used to generate the hash functions. The default seeding of randomness
//! is taken from the system. If you want to have reproducable outcomes, you can set a manual seed.
//!
//! ```rust
//! # use lsh_rs::prelude::*;
//! # let n_projections = 9;
//! # let n_hash_tables = 10;
//! # let dim = 10;
//! let mut lsh = LshMem::<_, f32>::new(n_projections, n_hash_tables, dim)
//!     .seed(12)
//!     .srp()
//!     .unwrap();
//! ```
//!
//! ## Unique indexes
//! Instead of storing data points as vectors. Storing `L` copies of the data points (one in every
//! hash table). You can choose to only store unique indexes of the data points. The index ids are
//! assigned in chronological order. This will drastically decrease the required memory.
//! ```rust
//! # use lsh_rs::prelude::*;
//! # let n_projections = 9;
//! # let n_hash_tables = 10;
//! # let dim = 10;
//! let mut lsh = LshMem::<_, f32>::new(n_projections, n_hash_tables, dim)
//!     .only_index()
//!     .srp()
//!     .unwrap();
//! ```
//!
//! ## Builder pattern methods
//! The following methods can be used to change internal state during object initialization:
//! * [only_index](struct.LSH.html#method.only_index)
//! * [seed](struct.LSH.html#method.seed)
//! * [set_database_file](struct.LSH.html#method.set_database_file)
//! * [multi_probe](struct.LSH.html#method.multi_probe)
//! * [increase_storage](struct.LSH.html#method.increase_storage)
//! * [fit (only for MIPS)](struct.MIPS.html#method.fit)
//!
//! ## Backends
//! The [LSH struct](struct.LSH.html) is exposed with multiple backends that store the hashes.
//! * in memory (fastest / can save state with serialization) [LshMem](type.LshMem.html)
//! * SQLite (slower due to disk io, but automatic state preservation between sessions) [LshSql](type.LshSql.html)
//! * in memory SQLite (can backup to SQLite when processing is done) [LshSqlMem](type.LshSqlMem.html)
//!
//! ## Hash primitives
//! The hashers in this crate will produces hashes of type `Vec<T>`. Where `T` should be one of `i8`,
//! `i16`, `i32` or `i64`. This concrete primitive value can be set by choosing on of the utillity types
//! in the following sub-modules:
//! * [hi8](prelude/hi8/index.html)
//! * [hi16](prelude/hi16/index.html)
//! * [hi32](prelude/hi32/index.html)
//! * [hi64](prelude/hi64/index.html)
//!
//! Using smaller primitives for the hash values, will result in less space requirements and greater
//! performance. However this may lead to panics if the hash value doesn't fit the chosen primitive
//! due to buffer overflow.
//!
//! *Note: the hash primitive cannot be set for every Hash family that has implemented
//! [VecHash](trait.VecHash.html). For instance, [SignRandomProjections](struct.SignRandomProjections.html)
//! will allways use `i8` as hash primitive.*
//!
//! ```rust
//! # use lsh_rs::prelude::*;
//! # let n_projections = 9;
//! # let n_hash_tables = 10;
//! # let dim = 10;
//! // use i8 hash values:
//! let lsh_i8 = hi8::LshMem::<_, u8>::new(n_projections, n_hash_tables, dim)
//!     .minhash()
//!     .unwrap();
//! // use i64 hash values:
//! let lhs_i8 = hi64::LshMem::<_, u8>::new(n_projections, n_hash_tables, dim)
//!     .minhash()
//!     .unwrap();
//! ```
//!
//! ## BLAS support
//! Utilizing [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) will heavily increase
//! performance. To make use of BLAS, install `lsh-rs` with `"blas"` feature and reinstall `ndarray` with `"blas"` support.
//!  <br>
//!  <br>
//! **Cargo.toml:**
//! ```toml
//! lsh-rs = {version ="x.x"}, features=["blas"]}
//! ndarray = {version = "0.13", features=["blas"]}
//! # Or any other blas backend.
//! blas-src = { version = "0.6", defeault-features = false, features = ["openblas"]}
//! ```
//!
//! ## Need your own hashers?
//! The LSH struct can easily be extended with your own hashers. Your own hasher structs need
//! to implement [VecHash<N, K>](trait.VecHash.html). `N` and `K` are generic types of the input
//! and output numbers respectively.
//!
//! ## Need you own backend?
//! If you need another backend, you can extend you backend with the [HashTables<N, K>](trait.HashTables.html) trait.
#![allow(dead_code, non_snake_case)]
#[cfg(feature = "blas")]
extern crate blas_src;
extern crate ndarray;
mod hash;
mod lsh {
    pub mod lsh;
    mod test;
}
pub mod dist;
mod multi_probe;
mod table {
    pub mod general;
    pub mod mem;
    pub mod sqlite;
    pub mod sqlite_mem;
}
mod constants;
mod error;
mod utils;
pub use hash::VecHash;
pub use multi_probe::{QueryDirectedProbe, StepWiseProbe};
pub use table::{general::HashTables, mem::MemoryTable, sqlite::SqlTable, sqlite_mem::SqlTableMem};
pub mod data;
pub mod prelude;
pub mod stats;
