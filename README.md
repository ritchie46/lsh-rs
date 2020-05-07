 # lsh-rs (Locality Sensitive Hashing)
[![rust docs](https://docs.rs/lsh-rs/badge.svg)](https://docs.rs/lsh-rs/latest/lsh_rs/)
[![Build Status](https://travis-ci.org/ritchie46/lsh-rs.svg?branch=master)](https://travis-ci.org/ritchie46/lsh-rs)

Locality sensitive hashing can help retrieving Approximate Nearest Neighbors in sub-linear time.

For more information on the subject see:
* [Introduction on LSH](http://people.csail.mit.edu/gregory/annbook/introduction.pdf)
* [Section 2. describes the hash families used in this crate](https://arxiv.org/pdf/1411.3787.pdf)
* [LSH and neural networks](https://www.ritchievink.com/blog/2020/04/07/sparse-neural-networks-and-hash-tables-with-locality-sensitive-hashing/)

[Read the Python docs](https://lsh-rs.readthedocs.io/en/latest/) for the Python bindings.

## Implementations

* **Base LSH**
    - Signed Random Projections *(Cosine similarity)*
    - L2 distance
    - MIPS *(Dot products/ Maximum Inner Product Search)*
    - MinHash *(Jaccard Similarity)*
* **Multi Probe LSH**
    - **Step wise probing**
        - SRP (only bit shifts)
    - **Query directed probing**
        - L2
        - MIPS
* Generic numeric types

## Getting started

```rust
use lsh_rs::LshMem;
// 2 rows w/ dimension 3.
let p = &[vec![1., 1.5, 2.],
        vec![2., 1.1, -0.3]];

// Do one time expensive preprocessing.
let n_projections = 9;
let n_hash_tables = 30;
let dim = 10;
let dim = 3;
let mut lsh = LshMem::new(n_projections, n_hash_tables, dim).srp().unwrap();
lsh.store_vecs(p);

// Query in sublinear time.
let query = &[1.1, 1.2, 1.2];
lsh.query_bucket(query);
```

## Signed Random Projections
LSH for maximum cosine similarity search.
```rust
use lsh_rs::prelude::*;
let n_projections = 9;
let n_hash_tables = 30;
let dim = 10;
let mut lsh = LshMem::<_, f32>::new(n_projections, n_hash_tables, dim).srp();
```

## L2
LSH for minimal L2 distance search.

```rust
// hyper parameter r in https://arxiv.org/pdf/1411.3787.pdf (eq. 8)
use lsh_rs::prelude::*;
let bucket_width = 2.2;
let n_projections = 9;
let n_hash_tables = 10;
let dim = 10;
let mut lsh = LshMem::<_, f32>::new(n_projections, n_hash_tables, dim).l2(bucket_width).unwrap();
```
## Maximum Inner Product (MIPS)
LSH for maximum inner product search.
```rust
use lsh_rs::prelude::*;
let bucket_width = 2.2;
// l2(x) < U < 1.0
let U = 0.83;
let r = 4.;
// number of concatenations
let m = 3;
let n_projections = 15;
let n_hash_tables = 10;
let dim = 10;
let mut lsh: LshMem<_, f32> = LshMem::new(n_projections, n_hash_tables, dim).mips(r, U, m).unwrap();
```

## Seed
Random projections are used to generate the hash functions. The default seeding of randomness
is taken from the system. If you want to have reproducable outcomes, you can set a manual seed.

```rust
use lsh_rs::prelude::*;
let n_projections = 9;
let n_hash_tables = 10;
let dim = 10;
let mut lsh = LshMem::<_, f32>::new(n_projections, n_hash_tables, dim).seed(12).srp();
```

## Unique indexes
Instead of storing data points as vectors. Storing `L` copies of the data points (one in every
hash table). You can choose to only store unique indexes of the data points. The index ids are
assigned in chronological order. This will drastically decrease the required memory.

```rust
use lsh_rs::prelude::*;
let n_projections = 9;
let n_hash_tables = 10;
let dim = 10;
let mut lsh = LshMem::<_, f32>::new(n_projections, n_hash_tables, dim).seed(12).srp();
```

## BLAS support
Utilizing [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) will heavily increase
performance. To make use of BLAS, install `lsh-rs` w/ `"blas"` feature and reinstall `ndarray` w/ `"blas"` support.
 <br>
 <br>
**Cargo.toml:**
```toml
lsh-rs = {version ="x.x", features=["blas"]}
ndarray = {version = "0.13", features=["blas"]}
# Or any other blas backend.
blas-src = { version = "0.6", defeault-features = false, features = ["openblas"]}
```
## Backends
The [LSH struct](struct.LSH.html) is exposed with multiple backends that store the hashes.
* in memory (fastest / can save state with serialization) [LshMem](type.LshMem.html)
* SQLite (slower due to disk io, but automatic state preservation between sessions) [LshSql](type.LshSql.html)
* in memory SQLite (can backup to SQLite when processing is done) [LshSqlMem](type.LshSqlMem.html)

## Python
At the moment, the Python bindings are only compiled for Linux x86_64 systems.

`$ pip install floky`

```python
from floky import SRP
import numpy as np

N = 10000
n = 100
dim = 10

# Generate some random data points
data_points = np.random.randn(N, dim)

# Do a one time (expensive) fit.
lsh = SRP(n_projections=19, n_hash_tables=10)
lsh.fit(data_points)

# Query approximated nearest neigbors in sub-linear time
query = np.random.randn(n, dim)
results = lsh.predict(query)
```
