import numba
import numpy as np


DUMMY_FLOAT_ARRAY = np.empty(0, dtype=np.float64)


@numba.njit(nogil=True, cache=True)
def column_kl_divergence(
    count_indices, count_data, marginal, prior_strength, marginal_entropy=np.nan
):
    """Compute the KL-divergence between a prior and a posterior distribution,
    the posterior computed as (1-prior_strength) * observed + prior_strength * prior.

    Parameters
    ----------
    count_indices: NDArray
        The indices of non-zero observed counts. Same length as count_data.

    count_data: NDArray
        The number of observed counts. Same length as count_indices.

    marginal: NDArray
        The marginal probability distribution. Length equal to the number of rows.

    prior_strength: float
        The strength of the prior (marginal) probability distribution in the bayesian update.

    marginal_entropy: float (optional, default=np.nan)
        Normalize the KL divergence by the entropy of the marginal distribution.
        Only applied if entropy is passed.
    """
    # Zero kl_divergence when non count data is observed
    total_count = np.sum(count_data)
    if total_count == 0:
        return 0

    # Special case if the posterior is equal to the prior
    # Could compute as usual but precision causes issues
    posterior_equal_prior = True
    for index, count in zip(count_indices, count_data):
        if count / total_count != marginal[index]:
            posterior_equal_prior = False
            break
    if posterior_equal_prior:
        return 0

    prior_info = np.log2(prior_strength)
    if np.isinf(prior_info):  # Happens if prior_strength == 0
        prior_info = 0

    # Initialize result as if every index were 0 count
    result = prior_strength * prior_info
    for index, count in zip(count_indices, count_data):
        zero_count_contribution = prior_strength * marginal[index] * prior_info
        result -= zero_count_contribution
        posterior_prob = (1 - prior_strength) * (
            count / total_count
        ) + prior_strength * marginal[index]
        posterior_contribution = posterior_prob * np.log2(
            posterior_prob / marginal[index]
        )
        result += posterior_contribution

    if not np.isnan(marginal_entropy):
        result /= marginal_entropy

    return result


@numba.njit(nogil=True, cache=True)
def target_counts(
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
    count_indices,
    count_data,
    marginal,
    prior_strength,
    supervision_weight,
    marginal_entropy=np.nan,
    target=None,
    target_marginal=DUMMY_FLOAT_ARRAY,
    target_marginal_entropy=np.nan,
):
    """ Compute the information weight of a column.

    Parameters
    ----------
    count_indices: NDArray
        An array of the non-zero row indices
    
    count_data: NDArray
        An array of the non-zero row values
    
    marginal: NDArray
        Array of the marginal probability distribution
    
    prior_strength: float
        The strength of the prior (marginal) probability distribution in the bayesian update.
    
    supervision_weight: float
        Amount of weight to put on the supervised part when target is not None.
        Returned weight is unsupervised**(1-supervision_weight) * supervised**supervision_weight.
    
    marginal_entropy: float (default=np.nan)
        Normalize the unsupervised weight by dividing by the marginal entropy if not np.nan.
    
    target: NDArray | None (default=None)
        Array of the target class for each row.
    
    target_marginal: NDArray (optional=DUMMY_FLOAT_ARRAY)
        Array of the marginal distribution over targets. Defaults to an empty dummy array
        to preserve input types for numba.
    
    target_marginal_entropy: float (optional=np.nan)
        Normalize the supervised weight by the entropy of the marginal target distribution
        if not set to np.nan.

    Returns
    -------
    column_weight: float
        The information weight of the column.
    """
    weight = column_kl_divergence(
        count_indices,
        count_data,
        marginal,
        prior_strength,
        marginal_entropy,
    )
    if target is not None:
        target_indices, target_data = target_counts(
            count_indices,
            count_data,
            target,
        )
        supervised_weight = column_kl_divergence(
            target_indices,
            target_data,
            target_marginal,
            prior_strength,
            target_marginal_entropy,
        )
        weight = supervised_weight**supervision_weight * weight ** (
            1 - supervision_weight
        )
    return weight