use lsh_rs::{MemoryTable, SignRandomProjections, L2, LSH, MIPS};
use pyo3::prelude::*;

#[pymodule]
fn lshpy(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<LshL2>()?;
    m.add_class::<LshMips>()?;
    m.add_class::<LshSrp>()?;
    Ok(())
}

#[pyclass]
struct LshL2 {
    lsh: LSH<MemoryTable, L2>,
}

#[pymethods]
impl LshL2 {
    #[new]
    fn new(n_projections: usize, n_hash_tables: usize, dim: usize, r: f32, seed: u64) -> Self {
        let lsh = LSH::new(n_projections, n_hash_tables, dim).seed(seed).l2(r);
        LshL2 { lsh }
    }
    fn store_vec(&mut self, v: Vec<f32>) {
        self.lsh.store_vec(&v);
    }

    pub fn store_vecs(&mut self, vs: Vec<Vec<f32>>) {
        self.lsh.store_vecs(&vs);
    }

    fn query_bucket(&self, v: Vec<f32>) -> PyResult<Vec<Vec<f32>>> {
        let q = self
            .lsh
            .query_bucket(&v)
            .into_iter()
            .map(|dp| dp.clone())
            .collect();
        Ok(q)
    }

    fn query_bucket_idx(&self, v: Vec<f32>) -> PyResult<Vec<u32>> {
        let q = self.lsh.query_bucket_ids(&v);
        Ok(q)
    }

    fn delete_vec(&mut self, v: Vec<f32>) {
        self.lsh.delete_vec(&v);
    }
}

#[pyclass]
struct LshMips {
    lsh: LSH<MemoryTable, MIPS>,
}

#[pymethods]
impl LshMips {
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
        LshMips { lsh }
    }
    fn store_vec(&mut self, v: Vec<f32>) {
        self.lsh.store_vec(&v);
    }

    pub fn store_vecs(&mut self, vs: Vec<Vec<f32>>) {
        self.lsh.store_vecs(&vs);
    }

    fn query_bucket(&self, v: Vec<f32>) -> PyResult<Vec<Vec<f32>>> {
        let q = self
            .lsh
            .query_bucket(&v)
            .into_iter()
            .map(|dp| dp.clone())
            .collect();
        Ok(q)
    }

    fn query_bucket_idx(&self, v: Vec<f32>) -> PyResult<Vec<u32>> {
        let q = self.lsh.query_bucket_ids(&v);
        Ok(q)
    }

    fn delete_vec(&mut self, v: Vec<f32>) {
        self.lsh.delete_vec(&v)
    }
}

#[pyclass]
struct LshSrp {
    lsh: LSH<MemoryTable, SignRandomProjections>,
}

#[pymethods]
impl LshSrp {
    #[new]
    fn new(n_projections: usize, n_hash_tables: usize, dim: usize, seed: u64) -> Self {
        let lsh = LSH::new(n_projections, n_hash_tables, dim).seed(seed).srp();
        LshSrp { lsh }
    }
    fn store_vec(&mut self, v: Vec<f32>) {
        self.lsh.store_vec(&v);
    }

    pub fn store_vecs(&mut self, vs: Vec<Vec<f32>>) {
        self.lsh.store_vecs(&vs);
    }

    fn query_bucket(&self, v: Vec<f32>) -> PyResult<Vec<Vec<f32>>> {
        let q = self
            .lsh
            .query_bucket(&v)
            .into_iter()
            .map(|dp| dp.clone())
            .collect();
        Ok(q)
    }

    fn query_bucket_idx(&self, v: Vec<f32>) -> PyResult<Vec<u32>> {
        let q = self.lsh.query_bucket_ids(&v);
        Ok(q)
    }

    fn delete_vec(&mut self, v: Vec<f32>) {
        self.lsh.delete_vec(&v)
    }
}
