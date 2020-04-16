mod dist;
use crate::dist::sort_by_distance;
use lsh_rs::{Error as LshError, LshMem, LshSql, SignRandomProjections, L2, MIPS};
use pyo3::exceptions::{RuntimeError, ValueError};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use thiserror::Error;

use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use numpy::PyArray2;
use pyo3::prelude::*;

#[pyfunction]
pub fn sort_by_distances(
    qs: &PyArray2<f32>,
    vs: &PyArray2<f32>,
    distance_f: &str,
    indexes: Vec<Vec<usize>>,
    top_k: usize,
) -> PyResult<(Vec<Vec<usize>>, Vec<Vec<f32>>)> {
    // let gil_guard = Python::acquire_gil();
    // let py = gil_guard.python();
    let distance_f = match distance_f {
        "cosine" => "cosine",
        "l2" | "euclidean" => "l2",
        _ => return Err(PyErr::new::<ValueError, _>("distance function not correct")),
    };

    let vs = vs.as_array();
    // (Vec<usize>, Vec<f32>)
    let r = qs
        .as_array()
        .axis_iter(Axis(0))
        .into_par_iter()
        .zip(indexes)
        .map(|(q, idx)| {
            let vs = idx
                .iter()
                .map(|i| vs.index_axis(Axis(0), *i))
                .collect::<Vec<_>>();
            sort_by_distance(q, &vs, distance_f, top_k)
        })
        .unzip();

    Ok(r)
}

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
    m.add_class::<LshL2Mem>()?;
    m.add_class::<LshSrpMem>()?;
    m.add_wrapped(wrap_pyfunction!(sort_by_distances)).unwrap();
    Ok(())
}

// #[pyfunction]
// fn sort_by_distances(
//     qs: Vec<Vec<f32>>,
//     vs: Vec<Vec<f32>>,
//     distance_f: &str,
//     idx: Vec<Vec<usize>>,
//     top_k: usize,
// ) -> PyResult<(Vec<Vec<usize>>, Vec<Vec<f32>>)> {
//     let gil_guard = Python::acquire_gil();
//     let py = gil_guard.python();
//     let distance_f = match distance_f {
//         "cosine" => "cosine",
//         "l2" => "l2",
//         _ => return Err(PyErr::new::<ValueError, _>("distance function not correct")),
//     };
//     let r = py.allow_threads(move ||
//         _sort_by_distances(&qs, &vs, distance_f, &idx, top_k));
//     Ok(r)
// }

enum LshTypes {
    L2(LshSql<L2>),
    L2Mem(LshMem<L2>),
    Mips(LshSql<MIPS>),
    Srp(LshSql<SignRandomProjections>),
    SrpMem(LshMem<SignRandomProjections>),
    Empty,
}

macro_rules! call_lsh_types {
    ($lsh:expr, $method_call:ident, $value:expr, $($optional:tt),*) => {
        match $lsh {
            LshTypes::L2(lsh) => {lsh.$method_call($value) $($optional),*},
            LshTypes::L2Mem(lsh) => {lsh.$method_call($value)$($optional),*},
            LshTypes::Mips(lsh) => {lsh.$method_call($value)$($optional),*},
            LshTypes::Srp(lsh) => {lsh.$method_call($value)$($optional),*},
            LshTypes::SrpMem(lsh) => {lsh.$method_call($value)$($optional),*},
            LshTypes::Empty => panic!("base not initialized"),
        };
    };

    ($lsh:expr, $method_call:ident) => {
        match $lsh {
            LshTypes::L2(lsh) => lsh.$method_call(),
            LshTypes::L2Mem(lsh) => lsh.$method_call(),
            LshTypes::Mips(lsh) => lsh.$method_call(),
            LshTypes::Srp(lsh) => lsh.$method_call(),
            LshTypes::SrpMem(lsh) => lsh.$method_call(),
            LshTypes::Empty => panic!("base not initialized"),
        };
    };
}

#[pyclass]
struct Base {
    lsh: LshTypes,
}

impl Base {
    fn _store_vec(&mut self, v: Vec<f32>) -> IntResult<()> {
        call_lsh_types!(&mut self.lsh, store_vec, &v,)?;
        Ok(())
    }

    fn _store_vecs(&mut self, vs: Vec<Vec<f32>>) -> IntResult<()> {
        call_lsh_types!(&mut self.lsh, store_vecs, &vs,)?;
        Ok(())
    }
    fn _query_bucket_idx(&self, v: Vec<f32>) -> IntResult<Vec<u32>> {
        let q = call_lsh_types!(&self.lsh, query_bucket_ids, &v,)?;
        Ok(q)
    }

    fn _increase_storage(&mut self, upper_bound: usize) -> IntResult<()> {
        call_lsh_types!(&mut self.lsh, increase_storage, upper_bound, ;);
        Ok(())
    }

    fn _query_batch(&self, vs: Vec<Vec<f32>>) -> IntResult<Vec<Vec<u32>>> {
        let gil_guard = Python::acquire_gil();
        let py = gil_guard.python();
        // allow threads doesn't make a difference on the rust side. But allows other python
        // code to run.
        // https://github.com/PyO3/pyo3/issues/649#issuecomment-546656381
        let q = match &self.lsh {
            LshTypes::L2(lsh) => lsh.query_bucket_ids_batch(&vs),
            LshTypes::L2Mem(lsh) => py.allow_threads(move || lsh.query_bucket_ids_batch_par(&vs)),
            LshTypes::Mips(lsh) => lsh.query_bucket_ids_batch(&vs),
            LshTypes::Srp(lsh) => lsh.query_bucket_ids_batch(&vs),
            LshTypes::SrpMem(lsh) => py.allow_threads(move || lsh.query_bucket_ids_batch_par(&vs)),
            _ => panic!("base not initialized"),
        }?;
        Ok(q)
    }

    fn _query_bucket(&self, v: Vec<f32>) -> IntResult<Vec<Vec<f32>>> {
        let q = match &self.lsh {
            LshTypes::L2(lsh) => lsh
                .query_bucket(&v)?
                .into_iter()
                .map(|dp| dp.clone())
                .collect(),
            LshTypes::L2Mem(lsh) => lsh
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
            LshTypes::SrpMem(lsh) => lsh
                .query_bucket(&v)?
                .into_iter()
                .map(|dp| dp.clone())
                .collect(),
            LshTypes::Empty => panic!("base not initialized"),
        };
        Ok(q)
    }

    fn _delete_vec(&mut self, v: Vec<f32>) -> IntResult<()> {
        call_lsh_types!(&mut self.lsh, delete_vec, &v,)?;
        Ok(())
    }

    fn _describe(&mut self) -> IntResult<String> {
        let s = call_lsh_types!(&mut self.lsh, describe)?;
        Ok(s)
    }

    fn _commit(&mut self) -> IntResult<()> {
        match &mut self.lsh {
            LshTypes::L2(lsh) => lsh.commit()?,
            LshTypes::Mips(lsh) => lsh.commit()?,
            LshTypes::Srp(lsh) => lsh.commit()?,
            _ => panic!("base not initialized"),
        };
        Ok(())
    }

    fn _init_transaction(&mut self) -> IntResult<()> {
        match &mut self.lsh {
            LshTypes::L2(lsh) => lsh.init_transaction()?,
            LshTypes::Mips(lsh) => lsh.init_transaction()?,
            LshTypes::Srp(lsh) => lsh.init_transaction()?,
            _ => panic!("base not initialized"),
        };
        Ok(())
    }

    fn _index(&self) -> IntResult<()> {
        match &self.lsh {
            LshTypes::L2(lsh) => lsh.hash_tables.as_ref().unwrap().index_hash()?,
            LshTypes::Mips(lsh) => lsh.hash_tables.as_ref().unwrap().index_hash()?,
            LshTypes::Srp(lsh) => lsh.hash_tables.as_ref().unwrap().index_hash()?,
            _ => panic!("base not initialized"),
        };
        Ok(())
    }

    fn _to_mem(&mut self) -> IntResult<()> {
        match &mut self.lsh {
            LshTypes::L2(lsh) => lsh.hash_tables.as_mut().unwrap().to_mem()?,
            LshTypes::Mips(lsh) => lsh.hash_tables.as_mut().unwrap().to_mem()?,
            LshTypes::Srp(lsh) => lsh.hash_tables.as_mut().unwrap().to_mem()?,
            _ => panic!("base not initialized"),
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
        let gil_guard = Python::acquire_gil();
        let py = gil_guard.python();
        py.allow_threads(move || self._store_vecs(vs))?;
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

    fn query_bucket_idx_batch(&self, vs: Vec<Vec<f32>>) -> PyResult<Vec<Vec<u32>>> {
        let q = self._query_batch(vs)?;
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

    fn to_mem(&mut self) -> PyResult<()> {
        self._to_mem()?;
        Ok(())
    }

    fn increase_storage(&mut self, upper_bound: usize) -> PyResult<()> {
        self._increase_storage(upper_bound)?;
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
struct LshL2Mem {}

#[pymethods]
impl LshL2Mem {
    #[new]
    fn new(
        n_projections: usize,
        n_hash_tables: usize,
        dim: usize,
        r: f32,
        seed: u64,
        db_path: String,
    ) -> PyResult<(Self, Base)> {
        let r = LshMem::new(n_projections, n_hash_tables, dim)
            .seed(seed)
            .only_index()
            .set_database_file(&db_path)
            .l2(r);

        let lsh = match r {
            Ok(lsh) => lsh,
            Err(e) => return Err(RuntimeError::py_err(format!("{}", e))),
        };
        Ok((
            LshL2Mem {},
            Base {
                lsh: LshTypes::L2Mem(lsh),
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

#[pyclass(extends=Base)]
struct LshSrpMem {}

#[pymethods]
impl LshSrpMem {
    #[new]
    fn new(
        n_projections: usize,
        n_hash_tables: usize,
        dim: usize,
        seed: u64,
        db_path: String,
    ) -> PyResult<(Self, Base)> {
        let r = LshMem::new(n_projections, n_hash_tables, dim)
            .seed(seed)
            .only_index()
            .set_database_file(&db_path)
            .srp();
        let lsh = match r {
            Ok(lsh) => lsh,
            Err(e) => return Err(RuntimeError::py_err(format!("{}", e))),
        };
        Ok((
            LshSrpMem {},
            Base {
                lsh: LshTypes::SrpMem(lsh),
            },
        ))
    }
}
