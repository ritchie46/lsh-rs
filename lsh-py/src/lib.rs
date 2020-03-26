use lsh_rs::{LshMem, LshSql, MemoryTable, SignRandomProjections, VecHash, L2, MIPS};
use pyo3::prelude::*;
use std::ops::{Deref, DerefMut};

// https://github.com/PyO3/pyo3/issues/696

#[pymodule]
fn lshpy(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<LshL2>()?;
    m.add_class::<LshMips>()?;
    m.add_class::<LshSrp>()?;
    Ok(())
}

enum LshTypes {
    L2(LshSql<L2>),
    Mips(LshSql<MIPS>),
    Srp(LshSql<SignRandomProjections>),
    Empty,
}

#[pyclass]
struct Base {
    lsh: LshTypes,
}

#[pymethods]
impl Base {
    #[new]
    fn new() -> Self {
        Base {
            lsh: LshTypes::Empty,
        }
    }

    fn store_vec(&mut self, v: Vec<f32>) {
        match &mut self.lsh {
            LshTypes::L2(lsh) => lsh.store_vec(&v),
            LshTypes::Mips(lsh) => lsh.store_vec(&v),
            LshTypes::Srp(lsh) => lsh.store_vec(&v),
            LshTypes::Empty => panic!("base not initialized"),
        };
    }

    fn store_vecs(&mut self, vs: Vec<Vec<f32>>) {
        match &mut self.lsh {
            LshTypes::L2(lsh) => lsh.store_vecs(&vs),
            LshTypes::Mips(lsh) => lsh.store_vecs(&vs),
            LshTypes::Srp(lsh) => lsh.store_vecs(&vs),
            LshTypes::Empty => panic!("base not initialized"),
        };
    }

    fn query_bucket(&self, v: Vec<f32>) -> PyResult<Vec<Vec<f32>>> {
        let q = match &self.lsh {
            LshTypes::L2(lsh) => lsh
                .query_bucket(&v)
                .into_iter()
                .map(|dp| dp.clone())
                .collect(),
            LshTypes::Mips(lsh) => lsh
                .query_bucket(&v)
                .into_iter()
                .map(|dp| dp.clone())
                .collect(),
            LshTypes::Srp(lsh) => lsh
                .query_bucket(&v)
                .into_iter()
                .map(|dp| dp.clone())
                .collect(),
            LshTypes::Empty => panic!("base not initialized"),
        };
        Ok(q)
    }

    fn query_bucket_idx(&self, v: Vec<f32>) -> PyResult<Vec<u32>> {
        let q = match &self.lsh {
            LshTypes::L2(lsh) => lsh.query_bucket_ids(&v),
            LshTypes::Mips(lsh) => lsh.query_bucket_ids(&v),
            LshTypes::Srp(lsh) => lsh.query_bucket_ids(&v),
            LshTypes::Empty => panic!("base not initialized"),
        };
        Ok(q)
    }

    fn delete_vec(&mut self, v: Vec<f32>) {
        match &mut self.lsh {
            LshTypes::L2(lsh) => lsh.delete_vec(&v),
            LshTypes::Mips(lsh) => lsh.delete_vec(&v),
            LshTypes::Srp(lsh) => lsh.delete_vec(&v),
            LshTypes::Empty => panic!("base not initialized"),
        };
    }

    fn describe(&mut self) {
        match &mut self.lsh {
            LshTypes::L2(lsh) => lsh.describe(),
            LshTypes::Mips(lsh) => lsh.describe(),
            LshTypes::Srp(lsh) => lsh.describe(),
            LshTypes::Empty => panic!("base not initialized"),
        };
    }
}

#[pyclass(extends=Base)]
struct LshL2 {}

#[pymethods]
impl LshL2 {
    #[new]
    fn new(
        n_projections: usize,
        n_hash_tables: usize,
        dim: usize,
        r: f32,
        seed: u64,
    ) -> (Self, Base) {
        let lsh = LshSql::new(n_projections, n_hash_tables, dim)
            .seed(seed)
            .l2(r);

        (
            LshL2 {},
            Base {
                lsh: LshTypes::L2(lsh),
            },
        )
    }
}

#[pyclass(extends=Base)]
struct LshMips {}

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
    ) -> (Self, Base) {
        let lsh = LshSql::new(n_projections, n_hash_tables, dim)
            .seed(seed)
            .mips(r, U, m);
        (
            LshMips {},
            Base {
                lsh: LshTypes::Mips(lsh),
            },
        )
    }
}
#[pyclass(extends=Base)]
struct LshSrp {}

#[pymethods]
impl LshSrp {
    #[new]
    fn new(n_projections: usize, n_hash_tables: usize, dim: usize, seed: u64) -> (Self, Base) {
        let lsh = LshSql::new(n_projections, n_hash_tables, dim)
            .seed(seed)
            .srp();
        (
            LshSrp {},
            Base {
                lsh: LshTypes::Srp(lsh),
            },
        )
    }
}
