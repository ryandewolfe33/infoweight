import numba
import numpy as np


@numba.njit(nogil=True, cache=True)
def marginal(
    indptr,
    indices,
    data,
    n_rows,
    target=None,
    column_groups=None,
):
    """
    Compute the marginals to compare each column to. Returns
    an (n column groups) x (n samples) matrix (unsupervised) or an
    (n column groups) x (n targets) matrix (supervised) where each
    row is the marginal of the column group.

    indptr, indices, and data arrays are from csc format.
    """
    n_groups = 1 if column_groups is None else np.max(column_groups) + 1
    n_targets = n_rows if target is None else np.max(target) + 1
    marginal = np.zeros((n_groups, n_targets), dtype=np.float64)
    for col in range(indptr.shape[0] - 1):
        group = 0 if column_groups is None else column_groups[col]
        for i in range(indptr[col], indptr[col + 1]):
            row = indices[i]
            this_target = target[row] if target is not None else row
            if this_target < 0:
                continue
            marginal[group, this_target] += data[i]
    marginal /= np.sum(marginal, axis=1).reshape(-1, 1)
    return marginal


@numba.njit(nogil=True)
def entropy(marginal):
    group_entropy = np.empty(marginal.shape[0], dtype=marginal.dtype)
    for i in range(marginal.shape[0]):
        m = marginal[i, :]
        m = m[m > 0]  # 0s cause -inf in log, but are multiplied by 0
        group_entropy[i] = -np.sum(m * np.log2(m))
    return group_entropy


@numba.njit(nogil=True, cache=True)
def make_marginals(
    indptr,
    indices,
    data,
    n_rows,
    normalize=True,
    target=None,
    column_groups=None,
):
    column_marginal = marginal(
        indptr,
        indices,
        data,
        n_rows,
        target=None,
        column_groups=column_groups,
    )
    column_entropy = (
        entropy(column_marginal)
        if normalize
        else np.full(column_marginal.shape[0], np.nan)
    )
    target_marginal = (
        marginal(
            indptr,
            indices,
            data,
            n_rows,
            target=target,
            column_groups=column_groups,
        )
        if target is not None
        else np.empty((column_marginal.shape[0], 0))
    )
    target_entropy = (
        entropy(target_marginal)
        if normalize and target is not None
        else np.full(target_marginal.shape[0], np.nan)
    )
    return column_marginal, column_entropy, target_marginal, target_entropy