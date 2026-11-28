import numba
import numpy as np
from numpy.typing import NDArray

PRIOR_STRENGTH = np.float32(2**-14)  # ~ 1e-4
PRIOR_INFO = np.float32(-14)  # np.log2(PRIOR_STRENGTH)
REL_TOL = np.float64(1e-5)  # Relative tolerance used to compare distribution equality


@numba.njit(nogil=True, inline="always")
def isclose(a: float, b: float) -> bool:
    return abs(a - b) <= REL_TOL * max(abs(a), abs(b))


@numba.njit(
    nogil=True,
    fastmath=True,
    cache=True,
)
def kl_divergence(
    col_indices: NDArray[np.integer],
    col_data: NDArray[np.floating],
    marginal: NDArray[np.floating],
    dtype: np.floating = np.float64,
) -> np.floating:
    """Compute the KL-divergence between the observed column distribution and the
    marginal row distribution. To account for zeros in the column distribution we
    apply a bayesian updating using the marignal distribution with strength 1e-4
    as a prior. The posterior is computed as
    (1-PRIOR_STRENGTH) * observed + PRIOR_STRENGTH * marginal.

    Parameters
    ----------
    col_indices: NDArray
        The indices of non-zero observed probabilities. Same length as col_data.

    col_data: NDArray
        The number of observed counts. Same length as col_indices. Assumed to sum
        to 1 unless all values are zero.

    marginal: NDArray
        The marginal probability distribution. Length equal to the number of rows.
        Assumed to sum to 1.

    Returns
    -------
    The KL divergence between the marginal and sparse column distributions.
    """
    marginal = marginal.astype(dtype)  # Cast to type we work in
    # Zero kl_divergence if all observed values are 0
    sum = np.sum(col_data)
    if sum == 0:
        return dtype(0)

    # Initialize result to 0 with right type of float
    result = dtype(0)
    posterior_equal_prior = True
    # Update result for each non-zero count
    for idx, count in zip(col_indices, col_data):
        prob = dtype(count / sum)
        if not isclose(prob, marginal[idx]):
            posterior_equal_prior = False
        result -= PRIOR_STRENGTH * marginal[idx] * PRIOR_INFO
        posterior_prob = (dtype(1) - PRIOR_STRENGTH) * prob + PRIOR_STRENGTH * marginal[
            idx
        ]
        result += posterior_prob * np.log2(posterior_prob / marginal[idx])

    # Special case if the posterior is equal to the prior
    # Could compute as usual but precision causes issues
    if posterior_equal_prior:
        return dtype(0)

    # Add initial result as if every index were 0 count
    result += PRIOR_STRENGTH * PRIOR_INFO

    return result


@numba.njit(nogil=True, cache=True)
def entropy(
    marginals: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute the entropy of each row in marginal. Assumed each row sums to 1."""
    assert len(marginals.shape) == 2, (
        "marginals must be a 2d array where each row is a distribution."
    )
    group_entropy = np.empty(marginals.shape[0], dtype=marginals.dtype)
    for i in range(marginals.shape[0]):
        m = marginals[i, :]
        m = m[m > 0]  # 0s cause -inf in log, but are multiplied by 0
        group_entropy[i] = -np.sum(m * np.log2(m))
    return group_entropy
