import numba
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray


@numba.njit(nogil=True, cache=True)
def compute_marginal(
    indptr: NDArray[np.integer],
    indices: NDArray[np.integer],
    data: NDArray,
    n_rows: int,
    column_groups: NDArray[np.integer],
    dtype: np.float32 | np.float64 = np.float64,
) -> NDArray[np.floating]:
    """
    Compute the marginals to compare each column to. Returns
    an (n column groups) x (n samples) matrix, where each
    row is the row-marginal of the column group.

    indptr, indices, and data arrays are from csc format.
    """
    n_groups = np.max(column_groups) + 1
    marginal = np.zeros((n_groups, n_rows), dtype=data.dtype)
    for col in range(indptr.shape[0] - 1):
        group = column_groups[col]
        for i in range(indptr[col], indptr[col + 1]):
            row = indices[i]
            marginal[group, row] += data[i]
    return (marginal / np.sum(marginal, axis=1).reshape(-1, 1)).astype(dtype)


def row_group_by(
    data: sp.sparray,
    target: NDArray,
) -> sp.sparray:
    """Sum the rows of the matrix that share a target id."""
    assert data.shape[0] == target.shape[0], (
        "Number of rows in data must match the length of the target array."
    )
    n_rows = data.shape[0]
    n_groups = np.max(target) + 1
    keep_mask = target >= 0
    masked_target = target[keep_mask]
    row_idx = np.arange(n_rows)[keep_mask]
    # Write a matrix that when it multiplies data on the left it projects the rows
    # of data into the new row associated with its target.
    projection = sp.csr_array(
        (np.ones_like(row_idx, dtype=data.dtype), (masked_target, row_idx)),
        shape=(n_groups, n_rows),
    )
    return projection @ data
