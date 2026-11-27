import numba
import numpy as np
import scipy.sparse as sp
from numba.typed import List
from numpy.typing import NDArray

from .information_theory import entropy, kl_divergence
from .infoweight_ import combine_unsupervised_and_supervised_weight, information_weight
from .marginal import compute_marginal, sum_row_groups


@numba.njit(nogil=True, cache=True)
def alias_sampling_setup(marginals):
    """Set up arrays for each row of marginals to sample via Walker's Alias Method."""
    assert len(marginals.shape) == 2, (
        "marginals must be a 2d array where each row is a distribution."
    )
    n = marginals.shape[1]
    all_qs = marginals * n
    all_aliases = np.empty_like(marginals, dtype=np.int64)

    for row in range(marginals.shape[0]):
        q = all_qs[row, :]
        alias = all_aliases[row, :]
        # Pre-allocate array stacks
        smaller = np.zeros(n, dtype=np.int64)
        larger = np.zeros(n, dtype=np.int64)
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

    return all_qs, all_aliases


@numba.njit(nogil=True, inline="always")
def alias_sample(q, alias, rng):
    # alias_array contain q and alias vstacked
    # 1. Uniformly pick a column index
    u = rng.random()
    idx = int(u * alias.shape[0])

    # 2. Split selection based on the scaled probability threshold
    coin = rng.random()
    if coin >= q[idx]:
        idx = alias[idx]
    return idx


@numba.njit(nogil=True, cache=True)
def sample_column(
    count,
    q,
    alias,
    rng,
):
    n = len(q)
    counts = np.zeros(n, dtype=np.int64)
    for _ in range(count):
        idx = alias_sample(q, alias, rng)
        counts[idx] += 1
    count_indices = np.nonzero(counts)[0]
    count_data = counts[count_indices]
    return count_indices, count_data


@numba.njit(nogil=True, cache=True)
def sum_single_column_row_groups(
    count_indices,
    count_data,
    target,
):
    target_counts = np.zeros(np.max(target) + 1, dtype=count_data.dtype)
    for index, count in zip(count_indices, count_data):
        if target[index] >= 0:
            target_counts[target[index]] += count
    count_indices = np.nonzero(target_counts)[0].astype(count_indices.dtype)
    count_data = target_counts[count_indices]
    return count_indices, count_data


@numba.njit(nogil=True, cache=True)
def column_weight(
    col_indices,
    col_data,
    marginal,
    marginal_entropy=np.nan,
    dtype=np.float64,
    target: NDArray[np.integer] | None = None,
    target_marginal: NDArray[np.floating] | None = None,
    target_marginal_entropy: float | None = None,
    supervision_weight: float = 0.95,
):
    weight = kl_divergence(col_indices, col_data, marginal, dtype)
    if not np.isnan(marginal_entropy):
        weight /= marginal_entropy
    if (
        target is not None
        and target_marginal is not None
        and target_marginal_entropy is not None
    ):
        target_col_indices, target_col_data = sum_single_column_row_groups(
            col_indices, col_data, target
        )
        supervised_weight = kl_divergence(
            target_col_indices, target_col_data, target_marginal, dtype
        )
        if not np.isnan(target_marginal_entropy):
            supervised_weight /= target_marginal_entropy

        weight = combine_unsupervised_and_supervised_weight(
            weight, supervision_weight, supervision_weight
        )
    return weight


@numba.njit(parallel=True, cache=True)
def sample_cdf(
    offset: float,
    n_samples: int,
    rngs: List[np.random.Generator],
    infoweights: NDArray[np.floating],
    column_counts: NDArray[np.integer],
    qs: NDArray[np.floating],
    aliases: NDArray[np.integer],
    marginals: NDArray[np.floating],
    marginals_entropies: NDArray[np.floating],
    column_groups: NDArray[np.integer],
    target: NDArray[np.integer] | None = None,
    target_marginal: NDArray[np.floating] | None = None,
    target_marginal_entropy: NDArray[np.floating] | None = None,
    supervision_weight: float = 0.95,
):
    """Estimate the CDF (via sampling) of each column at the value of that column's
    information weight + the offset.

    Parameters
    ----------
    offset: float
        Estimate the CDF at the column's information weight + the offset.

    n_samples: int
        Number of times to sample each column.

    rngs: List[np.random.Generator]
        Numba List of independent numpy random Generator objects with length equal to the
        number of columns. Indpendent sources of randomness means we can sample each column
        in parallel.

    infoweights: NDArray[np.floating]
        Array of the information weight of each column.

    qs: NDArray[np.floating]
        2d Array of with shape (n_column_groups, n_cols). Each row r is the q array for
        sampling row ids proportion to that column group r's marginal distribution via
        Walker's Alias method.

    aliases: NDArray[np.int64]
        2d Array of with shape (n_column_groups, n_cols). Each row r is the alias array for
        sampling row ids proportion to that column group r's marginal distribution via Walker's
        Alias method.

    marginals: NDArray[np.floating]
        2d Array where row r is the marginal distribution for column group r.

    marginal_entropies: NDArray[np.floating]
        Array of the entropy of the marginal for each column group for normalisation. If the
        entropy is set to np.nan, sampled information weight is not normalised.

    column_groups: NDArray[np.integer]
        Array with length equal to n_cols containing each column's group id.

    target_data: (NDArray[np.integer], NDArray[np.floating], NDArray[np.floating]) | None (optional, default=None)
        Data required to compute the supervised column weight if available. If None, information
        weight is unsupervised. Otherwise, target_data is a tuple containing the target array,
        2d array of target marginals, and 1D array of target marginal entropies.

    supervision_weight: float (optional default=0.95)
        Amount of weight to put on the supervised vs unsupervised information weights (if
        target_data is not None).

    Returns
    -------
    Array of CDF values.
    """
    dtype = infoweights.dtype.type
    n_cols = len(infoweights)
    cdf = np.zeros(n_cols, dtype=np.int32)
    for col in numba.prange(n_cols):
        count = column_counts[col]
        if count == 0:
            continue
        group = column_groups[col]
        rng = rngs[col]

        # Unpack target_marginal and target_marginal_entropy if present
        target_group_marginal = None
        if target_marginal is not None:
            target_group_marginal = target_marginal[group]
        target_group_marginal_entropy = None
        if target_marginal_entropy is not None:
            target_group_marginal_entropy = target_marginal_entropy[group]

        for _ in range(n_samples):
            sample_indices, sample_data = sample_column(
                count,
                qs[group],
                aliases[group],
                rng,
            )
            sample_weight = column_weight(
                sample_indices,
                sample_data,
                marginals[group],
                marginals_entropies[group],
                dtype,
                target,
                target_group_marginal,
                target_group_marginal_entropy,
                supervision_weight,
            )
            if sample_weight < infoweights[col] - offset:
                cdf[col] += 1

    return cdf / n_samples


def is_stopword(
    p,
    offset,
    n_samples,
    data,
    method="measured",
    target=None,
    column_groups=None,
    normalize=True,
    reweight_groups=True,
    supervision_weight=0.95,
    dtype=None,
    rng=None,
    infoweights=None,
):
    """Determine which columns (if any) are stopwords. A stopword is
    a column that, with probability p, has less information than if it
    were randomly sampled according to the marginal distribution.

    Parameters
    ----------
    p: float
        Statistical significance with which to reject the null hypothesis.

    offset: float
        Amount the observed mean must deviate from the null hypothesis.

    n_samples: int
        Number of times to sample each column.

    data: sp.sparse_array | sp.sparse_matrix
        Observed count data.

    method: {"measured", "uniform", "either"} (optional, default="measured")
        Defined the null column distribution. Measured is the observed marginal,
        uniform uses a uniform distribution, and either returns stopwords that are
        stopwords under either the measured or uniform null distribution.

    target: NDArray[int] | None (optional, default=None)
        Labels to pass to information weight computation.

    column_groups: NDArray[int] | None (optional, default=None)
        Column groups to pass to information weight computaiton.

    normalize: bool (optional, default=True)
        Normalize flag to pass to information weight computation.

    reweight_groups: bool (optional, default=True)
        Reweight_groups flag to pass to information weight computation.

    supervision_weight: float
        Supervision weight to pass to information weight computation.

    dtype: np.float32 | np.float64 | None (optional, default=None)
        dtype to compute the infomation weights in.

    rng: np.random.Generator | None (optional, default=None)
        Source of all randomness for repeatable runs.

    infoweights: NDArray[np.floating] | None (optional, default=None)
        Pre-computed information weights for each column. Be sure they were computed
        on the same data using the same flags set here.

    Returns
    -------
    stopword_mask: Array of bools to flag if that column is a stopword.
    """
    if method not in ["measured", "uniform", "either"]:
        raise ValueError("method must be one of 'measured', 'uniform', or 'either'.")
    data = sp.csc_array(data)

    if column_groups is None:
        column_groups = np.broadcast_to(0, data.shape[1])

    if dtype is None:
        if isinstance(data.data, np.float32 | np.int32):
            dtype = np.float32
        else:
            dtype = np.float64

    if rng is None:
        rng = np.random.default_rng()
    # Spawn independent generators for each column so we can process in parallel
    rngs = List([np.random.default_rng(s) for s in rng.spawn(data.shape[1])])

    # Pre-compute each column's information weight if not passed
    if infoweights is None:
        infoweights = information_weight(
            data,
            supervision_weight,
            normalize,
            target,
            column_groups,
            reweight_groups,
            dtype,
        )
    column_counts = data.sum(axis=0)

    if method == "measured" or method == "either":
        marginals = compute_marginal(
            data.indptr, data.indices, data.data, data.shape[0], column_groups, dtype
        )
        marginals_entropies = np.full(marginals.shape[0], np.nan)
        if normalize:
            marginals_entropies = entropy(marginals)

        target_marginals = None
        target_marginals_entropies = None
        if target is not None:
            target_matrix = sum_row_groups(data, target).tocsc()
            target_marginals = compute_marginal(
                target_matrix.indptr,
                target_matrix.indices,
                target_matrix.data,
                target_matrix.shape[0],
                column_groups,
                dtype,
            )
            target_marginals_entropies = np.full(target_marginals.shape[0], np.nan)
            if normalize:
                target_marginals_entropies = entropy(target_marginals)

        # Set up fast sampling of row ids with probabilities proportional to marginal
        qs, aliases = alias_sampling_setup(marginals)

        cdf = sample_cdf(
            offset,
            n_samples,
            rngs,
            infoweights,
            column_counts,
            qs,
            aliases,
            marginals,
            marginals_entropies,
            column_groups,
            target,
            target_marginals,
            target_marginals_entropies,
            supervision_weight,
        )
        measured_stopwords = cdf < (1 - p)
        if method == "measured":
            return measured_stopwords

    if method == "uniform" or method == "either":
        # Column groups not needed for uniform null hypotheses
        dummy_column_groups = np.broadcast_to(0, data.shape[1])
        marginals = np.broadcast_to(dtype(1 / data.shape[0]), (1, data.shape[0]))
        marginals_entropies = entropy(marginals)
        if normalize:
            marginals_entropies = entropy(marginals)

        target_marginals = None
        target_marginals_entropies = None
        if target is not None:
            uniform_one_column_array = sp.csr_array(
                np.full(len(target), 1 / len(target), dtype=dtype)
            ).transpose()
            target_marginals = compute_marginal(
                uniform_one_column_array.indptr,
                uniform_one_column_array.indices,
                uniform_one_column_array.data,
                target.shape[0],
                np.array([0]),
                dtype,
            )
            target_marginals_entropies = np.full(target_marginals.shape[0], np.nan)
            if normalize:
                target_marginals_entropies = entropy(target_marginals)

        # Set up fast sampling of row ids with probabilities proportional to marginal
        qs, aliases = alias_sampling_setup(marginals)

        cdf = sample_cdf(
            offset,
            n_samples,
            rngs,
            infoweights,
            column_counts,
            qs,
            aliases,
            marginals,
            marginals_entropies,
            dummy_column_groups,
            target,
            target_marginals,
            target_marginals_entropies,
            supervision_weight,
        )
        uniform_stopwords = cdf < (1 - p)
        if method == "uniform":
            return uniform_stopwords

    return measured_stopwords + uniform_stopwords
