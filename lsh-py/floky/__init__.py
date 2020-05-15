from .floky import LshL2, LshSrp, LshSrpMem, LshL2Mem, sort_by_distances
from tqdm import tqdm
import numpy as np
import os
from collections import namedtuple
from typing import Union, List


QueryResult = namedtuple(
    "QueryResult", ["index", "vectors", "n_collisions", "distances"]
)


class Base:
    def __init__(
        self,
        lsh: Union[LshL2, LshL2Mem, LshSrp, LshSrpMem],
        n_projections: int,
        n_hash_tables: int,
        dim: int,
        db_path: str,
        seed: int,
        in_mem: bool,
        log: bool,
    ):
        self.n_projection = n_projections
        self.n_hash_tables = n_hash_tables
        self.dim = dim
        self.lsh = lsh
        self.seed = seed
        self.db_path = db_path
        self.data = None
        self.in_mem = in_mem
        self.log = log

    def base(self):
        """
        Toggle base LSH (In contrast to multi-probe LSH)
        """
        self.lsh.base()

    def multi_probe(self, budget: int):
        """
        Toggle multi-probe LSH.

        Parameters
        ----------
        budget
            The upper bound on the number of probes.
        """
        self.lsh.multi_probe(budget)

    def describe(self):
        """
        Prints information about the buckets/ hashes.
        """
        print(self.lsh.describe())

    def store_vec(self, v: Union[np.ndarray, List[float]]):
        """
        Hash and store vector.

        Parameters
        ----------
        v
            Shape: (dim, )
            Store data point `v`
        """
        self.lsh.store_vec(v)

    def store_vecs(
        self, vs: Union[np.ndarray, List[List[float]]], chunk_size: int = 250
    ):
        """
        Hash and store multiple vectors.

        Parameters
        ----------
        vs
            Shape: (n, dim)
            Storae data points `vs`
        chunk_size
            How many chunks will be written to the backend at once.
            If an in memory backend is used this can be significantly higher compared to the SQLite backend.
        """

        length = len(vs)
        i = chunk_size
        prev_i = 0

        with tqdm(total=length, disable=not self.log) as pbar:
            while prev_i < length:
                self.lsh.store_vecs(vs[prev_i:i])
                prev_i = i
                i += chunk_size
                pbar.update(chunk_size)
                if not self.in_mem:
                    self.commit()
                    self.init_transaction()
        if not self.in_mem:
            self.commit()
            if self.log:
                print("start indexing...")
            self.index()

    def query_bucket(self, v: Union[np.ndarray, List]) -> List[List[float]]:
        """
        Query union over colliding hash tables.
        Note that fit/ predict probably is faster as there is less data transfer over FFI.

        Parameters
        ----------
        v
            Shape: (dim, )
            Query data point `v`

        Returns
        -------
        List of data points.
        """
        return self.lsh.query_bucket(v)

    def query_bucket_idx(self, v: Union[np.ndarray, List[float]]) -> List[int]:
        """
        Query union over colliding hash tables and return their index/ id.
        This is the fastest way to obtain the index.

        Parameters
        ----------
        v
            Shape: (dim, )
            Query data point `v`

        Returns
        -------
        List of ids/ indexes

        """
        return self.lsh.query_bucket_idx(v)

    def delete_vec(self, v: Union[np.ndarray, List[float]]):
        """
        Delete vector from hash tables. Depending on the backend this may or may not clear memory.

        Parameters
        ----------
        v
            Shape: (dim, )
            Data point `v`

        """
        self.lsh.delete_vec(v)

    def commit(self):
        """
        Commit SQLite backend.
        """
        self.lsh.commit()

    def init_transaction(self):
        """
        Start transaction for SQLite backend
        """
        if not self.in_mem:
            self.lsh.init_transaction()

    def index(self):
        """
        Create an index for the SQLite backend
        """
        self.lsh.index()

    def reset(self, dim: int):
        raise NotImplementedError

    def fit(self, X: Union[np.ndarray, List[List[float]]], chunk_size: int = 250):
        """
        One shot store and hash data points.

        Parameters
        ----------
        X
            Shape: (n, dim)
            The data points that should be hashed and stored
        chunk_size
            How many chunks will be written to the backend at once.
            If an in memory backend is used this can be significantly higher compared to the SQLite backend.
        """
        dim = len(X[0])
        self.reset(dim)
        self.data = np.ascontiguousarray(np.array(X, dtype=np.float32))
        self.lsh.increase_storage(len(X))
        self.store_vecs(self.data, chunk_size)

    def _predict(
        self,
        x: Union[np.ndarray, List[List[float]]],
        distance_f: str,
        only_index: bool,
        top_k: int,
        bound: Union[None, int]
    ) -> List[QueryResult]:
        """

        Parameters
        ----------
        x
            Shape: (n, dim)
            Query data points
        distance_f
            - "cosine"
            - "euclidean"
        only_index
            Only return indexes and not the data points.
        top_k
            Take the k closest

        Returns
        -------
        Named tuples List[QueryResult]
        """
        if self.data is None:
            raise ValueError("data attribute is not set")
        if not isinstance(x, (list, np.ndarray)):
            raise ValueError("x is not an array")
        X = np.ascontiguousarray(np.array(x, dtype=np.float32))
        if len(x.shape) == 1:
            X = np.array([x])
        elif len(x.shape) != 2:
            raise ValueError("x should be a 2d array")

        qrs = []
        idx_batch = self.lsh.query_bucket_idx_batch(X)
        sorted_idx, dist = sort_by_distances(X, self.data, distance_f, idx_batch, top_k, bound)
        for sorted_idx, dist, original_idx in zip(sorted_idx, dist, idx_batch):
            if len(sorted_idx) == 0:
                qrs.append(QueryResult([], [], 0, []))
                continue
            original_idx = np.array(original_idx)
            sorted_idx = np.array(sorted_idx)
            n_collisions = len(original_idx)

            idx = original_idx[sorted_idx]
            if only_index:
                data = None
            else:
                data = self.data[idx]
            qrs.append(QueryResult(idx, data, n_collisions, dist))

        return qrs

    def clean(self):
        """
        Remove database file
        """
        if not self.in_mem:
            os.remove(self.db_path)

    def to_mem(self, pages_per_step: int = 100):
        """
        SQLite disk based backend to SQLite memory backend.

        Parameters
        ----------
        pages_per_step
            Number of pages per step
        """
        if not self.in_mem:
            self.lsh.to_mem(pages_per_step)


class L2(Base):
    def __init__(
        self,
        n_projections: int,
        n_hash_tables: int,
        dim: int = 10,
        r: float = 4.0,
        seed: int = 0,
        db_path: str = "./lsh.db3",
        in_mem: bool = True,
        log: bool = True,
    ):
        """
        L2 LSH. Used to find data points with minimal euclidean distance.

        Parameters
        ----------
        n_projections
            Number of values in the hash; `K` in literature.
        n_hash_tables
            Number of hash tables; `L` in literature.
        dim
            Dimension of the data points.
        r
            Parameter of L2 LSH. Sets the width of the hashing buckets. Sometimes also called `W` in literature.
        seed
            Seed for the hashing functions. If set to zero, the hashing functions are randomly generated.
        db_path
            Path to SQLite database file. Only needed for SQLite backend.
        in_mem
            In memory backend or SQLite backend
        log
            Print fit information to screen
        """
        if in_mem:
            self.lsh_builder = LshL2Mem
        else:
            self.lsh_builder = LshL2

        lsh = self.lsh_builder(n_projections, n_hash_tables, dim, r, seed, db_path)
        self.r = r
        super().__init__(
            lsh, n_projections, n_hash_tables, dim, db_path, seed, in_mem, log
        )

    def reset(self, dim: int):
        self.clean()
        self.dim = dim
        self.lsh = self.lsh_builder(
            self.n_projection,
            self.n_hash_tables,
            self.dim,
            self.r,
            self.seed,
            self.db_path,
        )

    def predict(
        self,
        x: Union[np.ndarray, List[List[float]]],
        only_index: bool = False,
        top_k: int = 5,
        bound: Union[None, int] = None
    ):
        """
        Query data points.

        Parameters
        ----------
        x
            Shape: (n, dim)
            Query data points
        only_index
            Only return indexes and not the data points.
        top_k
            Take the k closest
        bound
            Only take the first 0..bound slice

        Returns
        -------
        Named tuples List[QueryResult]
        """

        return self._predict(x, "euclidean", only_index, top_k, bound)


class SRP(Base):
    def __init__(
        self,
        n_projections: int,
        n_hash_tables: int,
        dim: int = 10,
        seed: int = 0,
        db_path: str = "./lsh.db3",
        in_mem: bool = True,
        log: bool = True,
    ):
        """
        Signed Random Projections. Used to for cosine similarity.

        Parameters
        ----------
        n_projections
            Number of values in the hash; `K` in literature.
        n_hash_tables
            Number of hash tables; `L` in literature.
        dim
            Dimension of the data points.
        seed
            Seed for the hashing functions. If set to zero, the hashing functions are randomly generated.
        db_path
            Path to SQLite database file. Only needed for SQLite backend.
        in_mem
            In memory backend or SQLite backend
        log
            Print fit information to screen
        """
        if in_mem:
            self.lsh_builder = LshSrpMem
        else:
            self.lsh_builder = LshSrp
        lsh = self.lsh_builder(n_projections, n_hash_tables, dim, seed, db_path)
        super().__init__(
            lsh, n_projections, n_hash_tables, dim, db_path, seed, in_mem, log
        )

    def reset(self, dim: int):
        self.clean()
        self.dim = dim
        self.lsh = self.lsh_builder(
            self.n_projection, self.n_hash_tables, self.dim, self.seed, self.db_path
        )

    def predict(
        self,
        x: Union[np.ndarray, List[List[float]]],
        only_index: bool = False,
        top_k: int = 5,
        bound: Union[None, int] = None
    ):
        """
        Query data points.

        Parameters
        ----------
        x
            Shape: (n, dim)
            Query data points
        only_index
            Only return indexes and not the data points.
        top_k
            Take the k closest
        bound
            Only take the first 0..bound slice

        Returns
        -------
        Named tuples List[QueryResult]
        """
        return self._predict(x, "cosine", only_index, top_k, bound)
