import numba
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from .information_theory import compute_entropy, kl_divergence
from .marginal import compute_marginal, sum_row_groups


@numba.njit(nogil=True, cache=True, parallel=True)
def column_weights(
    indptr: NDArray,
    indices: NDArray,
    data: NDArray,
    shape: tuple[int, int],
    normalize: bool,
    column_groups: NDArray[np.integer],
    dtype: np.float32 | np.float64 = np.float64,
):
    """Compute the information weight for eac column of a count matrix in parallel.

    Parameters
    ----------
    indptr: NDArray
        indptr of the matrix in csc format.

    indices: NDArray
        indices of the matrix in csc format.

    data: NDArray
        data of the matrix in csc format.

    shape: tuple[int, int],
        shape of the matrix.

    normalize: bool
        Flag for wether to normalize weights by the entropy of the marginal distribution.

    column_groups: NDArray
        Array of group ids for the columns.

    Returns
    -------
    weights: NDArray
        Array of the information weight for each column.

    """
    assert shape[1] == len(indptr) - 1, (
        "shape[1] must match the number of columns in indptr."
    )
    marginals = compute_marginal(
        indptr,
        indices,
        data,
        shape[0],
        column_groups,
        dtype,
    )
    marginal_entropies = compute_entropy(marginals)
    weights = np.empty(shape[1], dtype=dtype)
    for col in numba.prange(shape[1]):
        col_indices = indices[indptr[col] : indptr[col + 1]]
        col_data = data[indptr[col] : indptr[col + 1]]
        group = column_groups[col]
        weight = kl_divergence(col_indices, col_data, marginals[group], dtype)
        if normalize:
            weight /= marginal_entropies[group]
        weights[col] = weight
    return weights


@numba.njit(nogil=True, fastmath=True, cache=True)
def combine_unsupervised_and_supervised_weight(
    unsupervised_weight: NDArray[np.floating] | float,
    supervised_weight: NDArray[np.floating] | float,
    supervision_weight: float,
) -> NDArray[np.floating] | float:
    return (
        unsupervised_weight ** (1 - supervision_weight)
        * supervised_weight**supervision_weight
    )


@numba.njit
def expected_group_weights(
    weights: NDArray[np.floating],
    column_groups: NDArray[np.floating],
    column_marginal: NDArray[np.floating],
):
    n_groups = np.max(column_groups) + 1
    expected_weights = np.zeros(n_groups, dtype=weights.dtype)
    group_total_probs = np.zeros_like(expected_weights)
    for weight, group, prob in zip(weights, column_groups, column_marginal):
        expected_weights[group] += prob * weight
        group_total_probs[group] += prob
    # Some groups might have 0 total probability, so avoid division by 0 error
    expected_weights[group_total_probs > 0] /= group_total_probs[group_total_probs > 0]
    return expected_weights


def compute_group_weights(
    data,
    weights,
    column_groups,
):
    """Reweight groups inplace by the groups' expected infoweight (over the row marginal distribution).

    Parameters
    ----------
    data:
        A sparse array of the raw data, used to compute the column marginal.

    weights:
        Array of weights. Updated inplace.

    column_groups:
        Array of ids for the column group of each column in data.

    Returns
    -------
    group_weights: An array of the expected weight for each group.
    """
    column_marginal = data.sum(axis=0)
    column_marginal = (column_marginal / np.sum(column_marginal)).astype(weights.dtype)
    group_weights = expected_group_weights(weights, column_groups, column_marginal)
    return group_weights


def information_weight(
    data,
    supervision_weight=0.95,
    normalize=True,
    target=None,
    column_groups=None,
    reweight_groups: bool = True,
    dtype: np.float32 | np.float64 | None = None,
):
    """Compute information based weights for columns, and scale, reweight and combine
    based on additional optional inforamtion. The basic information weight for a column
    is the amount of information gained by moving from the row-marginal distribution to
    the column distribution. It is computed as the KL-divergence between distributions.

    Parameters
    ----------
    data: scipy sparse matrix (n_samples, n_features)
        A matrix of count data where rows represent observations and
        columns represent features. Column weightings will be learned
        from this data.

    supervision_weight: float (optional, default=0.95)
        Parameter for combining supervised and unsupervised weights when
        targets are passed. Final weight is supervised_weight**supervision_weight
        * unsupervised_weight**(1-supervision weight). Must be in (0, 1].

    normalize: bool (optional, default=True)
        Normalize the information weight by dividing by the entropy of the marginal distribution.
        This normalizes the 'scale' so weights from different distributions can be combined.

    target: ndarray or None (optional, default=None)
        If supervised target labels are available, these can be used to define distributions
        over the target classes rather than over rows, allowing weights to be
        supervised and target based. If None then unsupervised weighting is used.

    column_groups: ndarray or None (optional, default=None)
        If columns have a natural grouping, i.e. cols 10-15 are a one-hot-encoding of a single
        categorical variable, we should compare the column distribution to the within group
        marginal. If passed None then all columns have the same group.

    reweight_groups: bool (optional, default=True)
        Reweight each column group by the expected (over the column marginal) infomartion
        weight of the group. This scales the weights in each column group so more informative
        groups are more important, even if individuals column are not.

    dtype: np.float32 | np.float64 | None (optional, default=None)
        Numpy floating dtype over which to compute and return the information weight. Setting
        to None will automatically select the precision equal to the input data (e.g. if data is
        np.float32 or np.int32 then dtype=np.float32)

    Returns
    -------
    weights: ndarray of shape (n_features,)
        The learned weights to be applied to columns based on the amount
        of information provided by the column.
    """
    if target is not None and len(target) != data.shape[0]:
        raise ValueError("The length of target must be equal to the number of rows.")
    if column_groups is not None and len(column_groups) != data.shape[1]:
        raise ValueError(
            "The number of columns must match the length of column groups."
        )
    if supervision_weight < 0 or supervision_weight > 1:
        raise ValueError("supervision_weight must be at least 0 and at most 1.")
    if dtype is None and isinstance(data.data, np.float32 | np.int32):
        dtype = np.float32
    else:
        dtype = np.float64
    # Column groups is None == all one column group
    if column_groups is None:
        column_groups = np.broadcast_to(0, data.shape[1])

    data = sp.csc_array(data)

    weights = column_weights(
        data.indptr,
        data.indices,
        data.data,
        data.shape,
        normalize,
        column_groups,
        dtype,
    )

    if target is not None and supervision_weight > 0:
        target_matrix = sum_row_groups(data, target).tocsc()
        target_weights = column_weights(
            target_matrix.indptr,
            target_matrix.indices,
            target_matrix.data,
            target_matrix.shape,
            normalize,
            column_groups,
            dtype,
        )
        weights = combine_unsupervised_and_supervised_weight(
            weights, target_weights, supervision_weight
        )

    if reweight_groups:
        group_weights = compute_group_weights(data, weights, column_groups)
        weights *= group_weights[column_groups]

    return weights
