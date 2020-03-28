from .floky import LshL2 as _LshL2
from tqdm import tqdm


class Base:
    def __init__(self, lsh, n_projections, n_hash_tables, dim, db_dir):
        self.n_projection = (n_projections,)
        self.n_hash_tables = (n_hash_tables,)
        self.dim = dim
        self.lsh = lsh
        self.db_dir = db_dir

    def describe(self):
        self.lsh.describe()

    def store_vec(self, v):
        self.lsh.store_vec(v)

    def store_vecs(self, vs, chunk_size=5000):
        length = len(vs)
        i = chunk_size
        prev_i = 0

        with tqdm(total=length) as pbar:
            while i < length:
                self.lsh.store_vecs(vs[prev_i: i])
                prev_i = i
                i += chunk_size
                pbar.update(chunk_size)
                self.commit()
                self.init_transaction()
        self.commit()

    def query_bucket(self, v):
        self.lsh.query_bucket(v)

    def query_bucket_idx(self, v):
        self.lsh.delete_vecket_idx(v)

    def delete_vec(self, v):
        self.lsh.delete_vec(v)

    def commit(self):
        self.lsh.commit()

    def init_transaction(self):
        self.lsh.init_transaction()


class L2(Base):
    def __init__(self, n_projections, n_hash_tables, dim, r=4.0, seed=0, db_dir="."):
        print("ONLY INDEX")
        lsh = _LshL2(n_hash_tables, n_hash_tables, dim, r, seed, db_dir)
        super().__init__(lsh, n_projections, n_hash_tables, dim, db_dir)
