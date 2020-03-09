use lsh::{MemoryTable, SignRandomProjections, L2, LSH, MIPS};
use pyo3::prelude::*;

macro_rules! methods {
    () => [
       fn store_vec(&mut self, v: Vec<f64>) {
            self.lsh.store_vec(&v)
        }

        pub fn store_vecs(&mut self, vs: Vec<Vec<f64>>) {
            self.lsh.store_vecs(&vs)
        }

        pub fn query_bucket(&self, v: Vec<f64>, dedup: bool) -> PyResult<Vec<Vec<f64>>> {
            let q = self.lsh.query_bucket(&v, dedup);
            let mut result = Vec::with_capacity(q.len());
            for qi in q {
                result.push(qi.clone())
            }
            Ok(result)
        }

        pub fn delete_vec(&mut self, v: Vec<f64>) {
            self.lsh.delete_vec(&v)
        }
    ]
}

#[pymodule]
fn lshpy(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<LSHL2>()?;
    m.add_class::<LSH_MIPS>()?;
    m.add_class::<LSH_SRP>()?;
    Ok(())
}

#[pyclass]
struct LSHL2 {
    lsh: LSH<MemoryTable, L2>,
}

#[pymethods]
impl LSHL2 {
    #[new]
    fn new(n_projections: usize, n_hash_tables: usize, dim: usize, r: f64, seed: u64) -> Self {
        let lsh = LSH::new_l2(n_projections, n_hash_tables, dim, r, seed);
        LSHL2 { lsh }
    }
    methods![];
}

#[pyclass]
struct LSH_MIPS {
    lsh: LSH<MemoryTable, MIPS>,
}

#[pymethods]
impl LSH_MIPS {
    #[new]
    fn new(
        n_projections: usize,
        n_hash_tables: usize,
        dim: usize,
        r: f64,
        U: f64,
        m: usize,
        seed: u64,
    ) -> Self {
        let lsh = LSH::new_mips(n_projections, n_hash_tables, dim, r, U, m, seed);
        LSH_MIPS { lsh }
    }
    methods![];
}

#[pyclass]
struct LSH_SRP {
    lsh: LSH<MemoryTable, SignRandomProjections>,
}

#[pymethods]
impl LSH_SRP {
    #[new]
    fn new(n_projections: usize, n_hash_tables: usize, dim: usize, seed: u64) -> Self {
        let lsh = LSH::new_srp(n_projections, n_hash_tables, dim, seed);
        LSH_SRP { lsh }
    }
    methods![];
}
