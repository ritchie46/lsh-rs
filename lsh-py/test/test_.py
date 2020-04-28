from floky import L2, SRP, QueryResult
import numpy as np
from scipy.spatial.distance import cdist
from typing import List


def get_mean_collisions(results: List[QueryResult]):
    return np.mean(list(map(lambda qr: qr.n_collisions, results)))


def test_l2():
    # first check we don't get any error if we don't have results
    np.random.seed(1)
    n = 1
    dim = 10
    arr = np.random.randn(n, dim)
    lsh = L2(n_projections=10, n_hash_tables=1, log=False, seed=1)
    lsh.fit(arr)
    assert lsh.predict(np.random.randn(1, dim))[0] == QueryResult([], [], 0, [])

    N = 10000
    n = 100

    arr = np.random.randn(N, dim)
    dist = cdist(arr[:n], arr, metric="euclidean")
    # get top 4 non trivial results
    top_k = dist.argsort(1)[:, 1:5]
    top_k_dist = dist[np.arange(n)[:, None], top_k]
    # define the distance R to the mean of top_k distances
    R = top_k_dist.mean()

    # use that to rescale the data
    arr /= R

    lsh = L2(n_projections=10, n_hash_tables=1, log=False, seed=1, r=4.0)
    lsh.fit(arr)

    query = np.random.randn(n, dim) / R
    results = lsh.predict(query, only_index=True, top_k=5)
    assert get_mean_collisions(results) == 36.7


def test_srp():
    np.random.seed(1)
    N = 10000
    n = 100
    dim = 10

    arr = np.random.randn(N, dim)
    lsh = SRP(n_projections=10, n_hash_tables=1, log=False, seed=1)
    lsh.fit(arr)
    query = np.random.randn(n, dim)
    results = lsh.predict(query)
    assert get_mean_collisions(results) == 36.21
