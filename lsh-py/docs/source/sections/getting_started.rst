Getting started
===============

:code:`floky` exposes :code:`Python` bindings to the LSH implementation of `lsh-rs <https://docs.rs/lsh-rs/0.2.3/lsh_rs/>`_. An LSH
implementation in :code:`Rust`. :code:`floky` is therefore blazingly fast.

Below shows how you can get started with :code:`floky`.

.. code-block:: python

    from floky import SRP
    import numpy as np

    N = 10000
    n = 100
    dim = 10

    # Generate some random data points
    data_points = np.random.randn(N, dim)

    # Do a one time (expensive) fit.
    lsh = SRP(n_projections=19, n_hash_tables=10)
    lsh.fit(data_points)

    # Query approximated nearest neigbors in sub-linear time
    query = np.random.randn(n, dim)
    results = lsh.predict(query)
