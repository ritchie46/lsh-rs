use lsh_rs::{
    Error as LshError, LshMem, LshSql, MemoryTable, Result, SignRandomProjections, VecHash, L2,
    MIPS,
};
use pyo3::exceptions::RuntimeError;
use pyo3::prelude::*;
use std::ops::{Deref, DerefMut};
use thiserror::Error;

// https://github.com/PyO3/pyo3/issues/696

// intermediate
type IntResult<T> = std::result::Result<T, PyLshErr>;

#[derive(Debug, Error)]
enum PyLshErr {
    #[error(transparent)]
    Err(#[from] LshError),
}

impl std::convert::From<PyLshErr> for PyErr {
    fn from(err: PyLshErr) -> PyErr {
        RuntimeError::py_err(format!("{}", err))
    }
}

#[pymodule]
fn floky(_py: Python, m: &PyModule) -> PyResult<()> {
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

impl Base {
    fn _store_vec(&mut self, v: Vec<f32>) -> IntResult<()> {
        match &mut self.lsh {
            LshTypes::L2(lsh) => lsh.store_vec(&v)?,
            LshTypes::Mips(lsh) => lsh.store_vec(&v)?,
            LshTypes::Srp(lsh) => lsh.store_vec(&v)?,
            LshTypes::Empty => panic!("base not initialized"),
        };
        Ok(())
    }

    fn _store_vecs(&mut self, vs: Vec<Vec<f32>>) -> IntResult<()> {
        match &mut self.lsh {
            LshTypes::L2(lsh) => lsh.store_vecs(&vs)?,
            LshTypes::Mips(lsh) => lsh.store_vecs(&vs)?,
            LshTypes::Srp(lsh) => lsh.store_vecs(&vs)?,
            LshTypes::Empty => panic!("base not initialized"),
        };
        Ok(())
    }
    fn _query_bucket_idx(&self, v: Vec<f32>) -> IntResult<Vec<u32>> {
        let q = match &self.lsh {
            LshTypes::L2(lsh) => lsh.query_bucket_ids(&v),
            LshTypes::Mips(lsh) => lsh.query_bucket_ids(&v),
            LshTypes::Srp(lsh) => lsh.query_bucket_ids(&v),
            LshTypes::Empty => panic!("base not initialized"),
        };
        Ok(q?)
    }

    fn _query_bucket(&self, v: Vec<f32>) -> IntResult<Vec<Vec<f32>>> {
        let q = match &self.lsh {
            LshTypes::L2(lsh) => lsh
                .query_bucket(&v)?
                .into_iter()
                .map(|dp| dp.clone())
                .collect(),
            LshTypes::Mips(lsh) => lsh
                .query_bucket(&v)?
                .into_iter()
                .map(|dp| dp.clone())
                .collect(),
            LshTypes::Srp(lsh) => lsh
                .query_bucket(&v)?
                .into_iter()
                .map(|dp| dp.clone())
                .collect(),
            LshTypes::Empty => panic!("base not initialized"),
        };
        Ok(q)
    }

    fn _delete_vec(&mut self, v: Vec<f32>) -> IntResult<()> {
        match &mut self.lsh {
            LshTypes::L2(lsh) => lsh.delete_vec(&v)?,
            LshTypes::Mips(lsh) => lsh.delete_vec(&v)?,
            LshTypes::Srp(lsh) => lsh.delete_vec(&v)?,
            LshTypes::Empty => panic!("base not initialized"),
        };
        Ok(())
    }

    fn _describe(&mut self) -> IntResult<String> {
        let s = match &mut self.lsh {
            LshTypes::L2(lsh) => lsh.describe()?,
            LshTypes::Mips(lsh) => lsh.describe()?,
            LshTypes::Srp(lsh) => lsh.describe()?,
            LshTypes::Empty => panic!("base not initialized"),
        };
        Ok(s)
    }

    fn _commit(&mut self) -> IntResult<()> {
        match &mut self.lsh {
            LshTypes::L2(lsh) => lsh.commit()?,
            LshTypes::Mips(lsh) => lsh.commit()?,
            LshTypes::Srp(lsh) => lsh.commit()?,
            LshTypes::Empty => panic!("base not initialized"),
        };
        Ok(())
    }

    fn _init_transaction(&mut self) -> IntResult<()> {
        match &mut self.lsh {
            LshTypes::L2(lsh) => lsh.init_transaction()?,
            LshTypes::Mips(lsh) => lsh.init_transaction()?,
            LshTypes::Srp(lsh) => lsh.init_transaction()?,
            LshTypes::Empty => panic!("base not initialized"),
        };
        Ok(())
    }

    fn _index(&self) -> IntResult<()> {
        match &self.lsh {
            LshTypes::L2(lsh) => lsh.hash_tables.as_ref().unwrap().index_hash()?,
            LshTypes::Mips(lsh) => lsh.hash_tables.as_ref().unwrap().index_hash()?,
            LshTypes::Srp(lsh) => lsh.hash_tables.as_ref().unwrap().index_hash()?,
            LshTypes::Empty => panic!("base not initialized"),
        };
        Ok(())
    }

    fn _to_mem(&mut self, pages_per_step: usize) -> IntResult<()> {
        match &mut self.lsh {
            LshTypes::L2(lsh) => lsh.hash_tables.as_mut().unwrap().to_mem(pages_per_step)?,
            LshTypes::Mips(lsh) => lsh.hash_tables.as_mut().unwrap().to_mem(pages_per_step)?,
            LshTypes::Srp(lsh) => lsh.hash_tables.as_mut().unwrap().to_mem(pages_per_step)?,
            LshTypes::Empty => panic!("base not initialized"),
        };
        Ok(())
    }
}

#[pymethods]
impl Base {
    #[new]
    fn new() -> Self {
        Base {
            lsh: LshTypes::Empty,
        }
    }

    fn store_vec(&mut self, v: Vec<f32>) -> PyResult<()> {
        self._store_vec(v)?;
        Ok(())
    }

    fn store_vecs(&mut self, vs: Vec<Vec<f32>>) -> PyResult<()> {
        self._store_vecs(vs)?;
        Ok(())
    }

    fn query_bucket(&self, v: Vec<f32>) -> PyResult<Vec<Vec<f32>>> {
        let q = self._query_bucket(v)?;
        Ok(q)
    }

    fn query_bucket_idx(&self, v: Vec<f32>) -> PyResult<Vec<u32>> {
        let q = self._query_bucket_idx(v)?;
        Ok(q)
    }

    fn delete_vec(&mut self, v: Vec<f32>) -> PyResult<()> {
        self._delete_vec(v)?;
        Ok(())
    }

    fn describe(&mut self) -> PyResult<String> {
        let s = self._describe()?;
        Ok(s)
    }

    fn commit(&mut self) -> PyResult<()> {
        self._commit()?;
        Ok(())
    }

    fn init_transaction(&mut self) -> PyResult<()> {
        self._init_transaction()?;
        Ok(())
    }

    fn index(&self) -> PyResult<()> {
        self._index()?;
        Ok(())
    }

    fn to_mem(&mut self, pages_per_step: usize) -> PyResult<()> {
        self._to_mem(pages_per_step)?;
        Ok(())
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
        db_path: String,
    ) -> PyResult<(Self, Base)> {
        let r = LshSql::new(n_projections, n_hash_tables, dim)
            .seed(seed)
            .only_index()
            .set_database_file(&db_path)
            .l2(r);

        let lsh = match r {
            Ok(lsh) => lsh,
            Err(e) => return Err(RuntimeError::py_err(format!("{}", e))),
        };
        Ok((
            LshL2 {},
            Base {
                lsh: LshTypes::L2(lsh),
            },
        ))
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
        db_path: String,
    ) -> PyResult<(Self, Base)> {
        let r = LshSql::new(n_projections, n_hash_tables, dim)
            .seed(seed)
            .only_index()
            .set_database_file(&db_path)
            .mips(r, U, m);
        let lsh = match r {
            Ok(lsh) => lsh,
            Err(e) => return Err(RuntimeError::py_err(format!("{}", e))),
        };

        Ok((
            LshMips {},
            Base {
                lsh: LshTypes::Mips(lsh),
            },
        ))
    }
}
#[pyclass(extends=Base)]
struct LshSrp {}

#[pymethods]
impl LshSrp {
    #[new]
    fn new(
        n_projections: usize,
        n_hash_tables: usize,
        dim: usize,
        seed: u64,
        db_path: String,
    ) -> PyResult<(Self, Base)> {
        let r = LshSql::new(n_projections, n_hash_tables, dim)
            .seed(seed)
            .only_index()
            .set_database_file(&db_path)
            .srp();
        let lsh = match r {
            Ok(lsh) => lsh,
            Err(e) => return Err(RuntimeError::py_err(format!("{}", e))),
        };
        Ok((
            LshSrp {},
            Base {
                lsh: LshTypes::Srp(lsh),
            },
        ))
    }
}
