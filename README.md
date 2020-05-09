 # lsh-rs (Locality Sensitive Hashing)
[![rust docs](https://docs.rs/lsh-rs/badge.svg)](https://docs.rs/lsh-rs/latest/lsh_rs/)
[![Build Status](https://travis-ci.org/ritchie46/lsh-rs.svg?branch=master)](https://travis-ci.org/ritchie46/lsh-rs)

Locality sensitive hashing can help retrieving Approximate Nearest Neighbors in sub-linear time.

For more information on the subject see:
* [Introduction on LSH](http://people.csail.mit.edu/gregory/annbook/introduction.pdf)
* [Section 2. describes the hash families used in this crate](https://arxiv.org/pdf/1411.3787.pdf)
* [LSH and neural networks](https://www.ritchievink.com/blog/2020/04/07/sparse-neural-networks-and-hash-tables-with-locality-sensitive-hashing/)

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
let mut lsh = LshMem::new(n_projections, n_hash_tables, dim)
    .srp()
    .unwrap();
lsh.store_vecs(p);

// Query in sublinear time.
let query = &[1.1, 1.2, 1.2];
lsh.query_bucket(query);
```

## Documentation
* [Read the Rust docs](https://docs.rs/lsh-rs/latest/lsh_rs/).
* [Read the Python docs](https://lsh-rs.readthedocs.io/en/latest/) for the Python bindings.

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
