import numba
import numpy as np
from numba.typed import List

from infoweight.stopwords import sample_cdf

offset = 0.1
n_samples = 100
infoweights = np.full(10, 0.5)
column_counts = np.array([10, 20] * 5)
marginals = np.full(10, 0.1).reshape(1, -1)
marginals_entropies = np.array([np.nan])
column_groups = np.broadcast_to(0, 10)
supervision_weight = 0.95


def test_rng():
    rng = np.random.default_rng(seed=42)
    rngs = List([np.random.default_rng(s) for s in rng.spawn(numba.get_num_threads())])
    cdf1 = sample_cdf(
        offset,
        n_samples,
        rngs,
        infoweights,
        column_counts,
        marginals,
        marginals_entropies,
        column_groups,
        None,
        None,
        None,
        supervision_weight,
    )

    rng = np.random.default_rng(seed=42)
    rngs = List([np.random.default_rng(s) for s in rng.spawn(numba.get_num_threads())])
    cdf2 = sample_cdf(
        offset,
        n_samples,
        rngs,
        infoweights,
        column_counts,
        marginals,
        marginals_entropies,
        column_groups,
        None,
        None,
        None,
        supervision_weight,
    )

    assert np.allclose(cdf1, cdf2)
