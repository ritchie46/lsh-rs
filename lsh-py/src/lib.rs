use lsh::{MemoryTable, SignRandomProjections, L2, LSH, MIPS};
use pyo3::prelude::*;

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
    fn new(n_projections: usize, n_hash_tables: usize, dim: usize, r: f32, seed: u64) -> Self {
        let lsh = LSH::new(n_projections, n_hash_tables, dim).seed(seed).l2(r);
        LSHL2 { lsh }
    }
    fn store_vec(&mut self, v: Vec<f32>) {
        self.lsh.store_vec(&v)
    }

    pub fn store_vecs(&mut self, vs: Vec<Vec<f32>>) {
        self.lsh.store_vecs(&vs)
    }

    fn query_bucket(&self, v: Vec<f32>) -> PyResult<Vec<Vec<f32>>> {
        let q = self.lsh.query_bucket(&v);
        let mut result = Vec::with_capacity(q.len());
        for qi in q {
            result.push(qi.clone())
        }
        Ok(result)
    }

    fn delete_vec(&mut self, v: Vec<f32>) {
        self.lsh.delete_vec(&v)
    }
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
        r: f32,
        U: f32,
        m: usize,
        seed: u64,
    ) -> Self {
        let lsh = LSH::new(n_projections, n_hash_tables, dim)
            .seed(seed)
            .mips(r, U, m);
        LSH_MIPS { lsh }
    }
    fn store_vec(&mut self, v: Vec<f32>) {
        self.lsh.store_vec(&v)
    }

    pub fn store_vecs(&mut self, vs: Vec<Vec<f32>>) {
        self.lsh.store_vecs(&vs)
    }

    fn query_bucket(&self, v: Vec<f32>) -> PyResult<Vec<Vec<f32>>> {
        let q = self.lsh.query_bucket(&v);
        let mut result = Vec::with_capacity(q.len());
        for qi in q {
            result.push(qi.clone())
        }
        Ok(result)
    }

    fn delete_vec(&mut self, v: Vec<f32>) {
        self.lsh.delete_vec(&v)
    }
}

#[pyclass]
struct LSH_SRP {
    lsh: LSH<MemoryTable, SignRandomProjections>,
}

#[pymethods]
impl LSH_SRP {
    #[new]
    fn new(n_projections: usize, n_hash_tables: usize, dim: usize, seed: u64) -> Self {
        let lsh = LSH::new(n_projections, n_hash_tables, dim).seed(seed).srp();
        LSH_SRP { lsh }
    }
    fn store_vec(&mut self, v: Vec<f32>) {
        self.lsh.store_vec(&v)
    }

    pub fn store_vecs(&mut self, vs: Vec<Vec<f32>>) {
        self.lsh.store_vecs(&vs)
    }

    fn query_bucket(&self, v: Vec<f32>) -> PyResult<Vec<Vec<f32>>> {
        let q = self.lsh.query_bucket(&v);
        let mut result = Vec::with_capacity(q.len());
        for qi in q {
            result.push(qi.clone())
        }
        Ok(result)
    }

    fn delete_vec(&mut self, v: Vec<f32>) {
        self.lsh.delete_vec(&v)
    }
}
