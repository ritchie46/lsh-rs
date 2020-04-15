from .floky import LshL2, LshSrp, LshSrpMem, LshL2Mem
from tqdm import tqdm
from scipy.spatial.distance import cdist
import numpy as np
import os
from collections import namedtuple


QueryResult = namedtuple("QueryResult", ["index", "vectors", "n_collisions", "distances"])


class Base:
    def __init__(self, lsh, n_projections, n_hash_tables, dim, db_path, seed, in_mem, log):
        self.n_projection = n_projections
        self.n_hash_tables = n_hash_tables
        self.dim = dim
        self.lsh = lsh
        self.seed = seed
        self.db_path = db_path
        self.data = None
        self.in_mem = in_mem
        self.log = log

    def describe(self):
        print(self.lsh.describe())

    def store_vec(self, v):
        self.lsh.store_vec(v)

    def store_vecs(self, vs, chunk_size=250):
        length = len(vs)
        i = chunk_size
        prev_i = 0

        with tqdm(total=length, disable= not self.log) as pbar:
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

    def query_bucket(self, v):
        return self.lsh.query_bucket(v)

    def query_bucket_idx(self, v):
        return self.lsh.query_bucket_idx(v)

    def delete_vec(self, v):
        self.lsh.delete_vec(v)

    def commit(self):
        self.lsh.commit()

    def init_transaction(self):
        if not self.in_mem:
            self.lsh.init_transaction()

    def index(self):
        self.lsh.index()

    def reset(self):
        raise NotImplementedError

    def fit(self, X, chunk_size=250):
        self.reset()
        self.data = X
        self.store_vecs(X, chunk_size)

    def _predict(self, x, distance_f, bound):
        if self.data is None:
            raise ValueError("data attribute is not set")
        idx = np.array(self.query_bucket_idx(x))
        n_collisions = len(idx)

        step = bound * self.n_hash_tables
        i = 0
        j = step

        while i < n_collisions:
            b_idx = idx[i: j]
            i = j
            j += step
            dist = cdist(x[None, :], self.data[b_idx], metric=distance_f).flatten()

            mask = dist < 1
            if mask.sum() > 0:
                break
        dist = dist[mask]
        sorted_idx = dist.argsort()
        idx = b_idx[mask][sorted_idx]
        distances = dist[sorted_idx]

        return QueryResult(idx, self.data[idx], n_collisions, distances)

    def clean(self):
        if not self.in_mem:
            os.remove(self.db_path)

    def to_mem(self, pages_per_step=100):
        if not self.in_mem:
            self.lsh.to_mem(pages_per_step)


class L2(Base):
    def __init__(
        self, n_projections, n_hash_tables, dim, r=4.0, seed=0, db_path="./lsh.db3", in_mem=False, log=True
    ):
        if in_mem:
            self.lsh_builder = LshL2Mem
        else:
            self.lsh_builder = LshL2

        lsh = self.lsh_builder(n_projections, n_hash_tables, dim, r, seed, db_path)
        self.r = r
        super().__init__(lsh, n_projections, n_hash_tables, dim, db_path, seed, in_mem, log)

    def reset(self):
        self.clean()
        self.lsh = self.lsh_builder(
            self.n_projection,
            self.n_hash_tables,
            self.dim,
            self.r,
            self.seed,
            self.db_path,
        )

    def predict(self, x, bound=3):
        return self._predict(x, 'euclidean', bound)


class CosineSim(Base):
    def __init__(self, n_projections, n_hash_tables, dim, seed=0, db_path="./lsh.db3", in_mem=False, log=True):
        if in_mem:
            self.lsh_builder = LshSrpMem
        else:
            self.lsh_builder = LshSrp
        lsh = self.lsh_builder(n_projections, n_hash_tables, dim, seed, db_path)
        super().__init__(lsh, n_projections, n_hash_tables, dim, db_path, seed, in_mem, log)

    def reset(self):
        self.clean()
        self.lsh = self.lsh_builder(
            self.n_projection, self.n_hash_tables, self.dim, self.db_path, self.seed
        )

    def predict(self, x, bound=3):
        return self._predict(x, 'cosine', bound)
