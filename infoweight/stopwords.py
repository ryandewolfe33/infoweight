import numba
import numpy as np
from numba.typed import List

from .marginal import make_marginals
from .weight import column_weight


@numba.njit(cache=True)
def alias_sampling_setup(probs):
    """
    Set up arrays for Walker's Alias Method.
    """
    n = len(probs)
    # Store q and alias together for better caching later
    result = np.empty((n, 2))
    result[:, 0] = probs * n
    q = result[:, 0]
    alias = result[:, 1]

    # Pre-allocate array stacks
    smaller = np.zeros(n, dtype=np.int32)
    larger = np.zeros(n, dtype=np.int32)
    next_small_idx = 0
    next_large_idx = 0

    # Categorize indices into smaller or larger than 1.0
    for i in range(n):
        if q[i] < 1.0:
            smaller[next_small_idx] = i
            next_small_idx += 1
        else:
            larger[next_large_idx] = i
            next_large_idx += 1

    # Pair small outcomes with large outcomes
    while next_small_idx > 0 and next_large_idx > 0:
        next_small_idx -= 1
        small = smaller[next_small_idx]
        next_large_idx -= 1
        large = larger[next_large_idx]

        alias[small] = large
        q[large] = (q[large] + q[small]) - 1.0

        if q[large] < 1.0:
            smaller[next_small_idx] = large
            next_small_idx += 1
        else:
            larger[next_large_idx] = large
            next_large_idx += 1

    return result


@numba.njit(nogil=True, inline="always")
def alias_sample(alias_array, rng):
    # alias_array contain q and alias vstacked
    # 1. Uniformly pick a column index
    u = rng.random()
    idx = int(u * alias_array.shape[0])

    # 2. Split selection based on the scaled probability threshold
    coin = rng.random()
    if coin >= alias_array[idx, 0]:
        idx = int(alias_array[idx, 1])
    return idx


@numba.njit(nogil=True, cache=True)
def sample_column(
    count,
    alias_array,
    rng,
):
    counts = np.zeros(alias_array.shape[0], dtype=np.int64)
    for _ in range(count):
        idx = alias_sample(alias_array, rng)
        counts[idx] += 1
    count_indices = np.nonzero(counts)[0]
    count_data = counts[count_indices]
    return count_indices, count_data


@numba.njit
def sample_cdf(
    offset,
    n_samples,
    rngs,
    indptr,
    indices,
    data,
    n_rows,
    prior_strength=1e-4,
    supervision_weight=0.95,
    normalize=True,
    target=None,
    column_groups=None,
    uniform=False,
):
    # Uniform marginal does not require column groups
    # Must happen out here to make prange happy
    n_cols = len(indptr) - 1
    if uniform or column_groups is None:
        column_groups = np.zeros(n_cols, dtype=np.int64)
    if not uniform:
        marginal, marginal_entropy, target_marginal, target_marginal_entropy = (
            make_marginals(
                indptr,
                indices,
                data,
                n_rows,
                normalize,
                target,
                column_groups,
            )
        )
    else:
        uniform_indices = np.arange(n_rows, dtype=indices.dtype)
        uniform_indptr = np.array([0, n_rows])
        uniform_data = np.ones(n_rows, dtype=data.dtype)
        marginal, marginal_entropy, target_marginal, target_marginal_entropy = (
            make_marginals(
                uniform_indptr,
                uniform_indices,
                uniform_data,
                n_rows,
                normalize,
                target,
            )
        )

    alias_arrays = np.empty((marginal.shape[0], marginal.shape[1], 2))
    for group in range(marginal.shape[0]):
        alias_arrays[group, :, :] = alias_sampling_setup(marginal[group])

    cdf = np.zeros(n_cols, dtype=np.float64)
    for col in numba.prange(n_cols):
        group = column_groups[col]
        alias_array = alias_arrays[group, :, :]
        count = np.sum(data[indptr[col] : indptr[col + 1]])
        rng = rngs[col]
        iw = column_weight(
            indices[indptr[col] : indptr[col + 1]],
            data[indptr[col] : indptr[col + 1]],
            marginal[group],
            prior_strength,
            supervision_weight,
            marginal_entropy[group],
            target,
            target_marginal[group],
            target_marginal_entropy[group],
        )
        for _ in range(n_samples):
            sample_indices, sample_data = sample_column(
                count,
                alias_array,
                rng,
            )
            sample_weight = column_weight(
                sample_indices,
                sample_data,
                marginal[group],
                prior_strength,
                supervision_weight,
                marginal_entropy[group],
                target,
                target_marginal[group],
                target_marginal_entropy[group],
            )
            if sample_weight < iw - offset:
                cdf[col] += 1
    cdf /= n_samples
    return cdf


def is_stopword(
    p,
    offset,
    n_samples,
    data,
    prior_strength=1e-4,
    supervision_weight=0.95,
    normalize=True,
    target=None,
    column_groups=None,
    method="measured",
    rng=None,
):
    """Determine which columns (if any) are stopwords. A stopword is
    a column that, with probability p, has less information than if it
    were randomly sampled according to the marginal distribution.

    Parameters
    ----------
    p: float
    offset: float
    n_samples: int
    data: sp.sparse_array | sp.sparse_matrix
    prior_strength: float
    supervision_weight: float
    normalize: bool
    target: NDArray[int] | None
    column_groups: NDArray[int] | None
    method="measured"
    rng: np.random.Generator | None
    """
    if method not in ["measured", "uniform", "either"]:
        raise ValueError("method must be one of 'measured', 'uniform', or 'either'.")

    if rng is None:
        rng = np.random.default_rng()
    # Spawn independent generators for each column so we can process in parallel
    rngs = List([np.random.default_rng(s) for s in rng.spawn(data.shape[1])])

    csc_data = data.tocsc()

    if method == "measured" or method == "either":
        measured_cdf = sample_cdf(
            offset,
            n_samples,
            rngs,
            csc_data.indptr,
            csc_data.indices,
            csc_data.data,
            csc_data.shape[0],
            prior_strength,
            supervision_weight,
            normalize,
            target,
            column_groups,
            False,
        )
        measured_stopwords = measured_cdf < (1 - p)
        if method == "measured":
            return measured_stopwords

    if method == "uniform" or method == "either":
        uniform_cdf = sample_cdf(
            offset,
            n_samples,
            rngs,
            csc_data.indptr,
            csc_data.indices,
            csc_data.data,
            csc_data.shape[0],
            prior_strength,
            supervision_weight,
            normalize,
            target,
            column_groups,
            True,
        )
        uniform_stopwords = uniform_cdf < (1 - p)
        if method == "uniform":
            return uniform_stopwords

    return measured_stopwords + uniform_stopwords
