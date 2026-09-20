from warnings import warn

import numba
import numpy as np
import scipy.sparse
from sklearn.base import BaseEstimator, TransformerMixin


@numba.njit(nogil=True, cache=True)
def column_kl_divergence(
    count_indices,
    count_data,
    prior_probs,
    prior_strength,
    target=None,
):
    """Function to compute the KL-divergence between a prior and a posterior distribution,
    the posterior computed as (1-prior_strength) * observed + prior_strength * prior.

    Parameters
    ----------
    count_indices
        The indices of non-zero observed counts.

    count_data
        The number of observed counts.

    prior_probs
        The prior probability distribution.

    prior_strength
        The strength of the prior probability distribution in the bayesian update.
    """
    # Zero kl_divergence when non count data is observed
    total_count = np.sum(count_data)
    if total_count == 0:
        return 0

    # Special case if the posterior is equal to the prior
    # Could compute as usual but precision causes issues
    posterior_equal_prior = True
    for index, count in zip(count_indices, count_data):
        if count / total_count != prior_probs[index]:
            posterior_equal_prior = False
            break
    if posterior_equal_prior:
        return 0

    prior_info = np.log2(prior_strength)
    if np.isinf(prior_info):  # Happens if prior_strength == -inf
        prior_info = 0

    # Initialize result as if every index were 0 count
    result = prior_strength * prior_info
    for index, count in zip(count_indices, count_data):
        zero_count_contribution = prior_strength * prior_probs[index] * prior_info
        result -= zero_count_contribution
        posterior_prob = (1 - prior_strength) * (
            count / total_count
        ) + prior_strength * prior_probs[index]
        posterior_contribution = posterior_prob * np.log2(
            posterior_prob / prior_probs[index]
        )
        result += posterior_contribution

    return result


@numba.njit(nogil=True, cache=True, parallel=True)
def column_weights(
    indptr,
    indices,
    data,
    baseline_probabilities,
    prior_strength,
    target=None,
    column_groups=None,
):
    n_cols = indptr.shape[0] - 1
    weights = np.ones(n_cols)
    for i in numba.prange(n_cols):
        group = 0
        if column_groups is not None:
            group = column_groups[i]
        count_indices = indices[indptr[i] : indptr[i + 1]]
        count_data = data[indptr[i] : indptr[i + 1]]

        # Make observed target distribution if necessary
        if target is not None:
            target_counts = np.zeros(baseline_probabilities.shape[1], dtype=data.dtype)
            for index, count in zip(count_indices, count_data):
                if target[index] >= 0:
                    target_counts[target[index]] += count
            count_indices = np.nonzero(target_counts)[0].astype(count_indices.dtype)
            count_data = target_counts[count_indices]

        weights[i] = column_kl_divergence(
            count_indices,
            count_data,
            baseline_probabilities[group, :],
            prior_strength=prior_strength,
            target=target,
        )
    return weights


@numba.njit(nogil=True)
def compute_baseline_probabilities(
    indptr,
    indices,
    data,
    target=None,
    column_groups=None,
):
    """
    Compute the marginals to compare each column to. Returns
    an (n column groups) x (n samples) matrix (unsupervised) or an
    (n column groups) x (n targets) matrix (supervised) where each
    row is the marginal of the column group.

    indptr, indices, and data arrays are from csr format.
    """
    n_groups = 1
    if column_groups is not None:
        n_groups = column_groups.max() + 1
    n_targets = indptr.shape[0] - 1
    if target is not None:
        n_targets = target.max() + 1
    counts = np.zeros((n_groups, n_targets), dtype=np.int64)
    for row in range(indptr.shape[0] - 1):
        this_target = row
        if target is not None:
            if target[row] >= 0:
                this_target = target[row]
            else:
                continue
        for i in range(indptr[row], indptr[row + 1]):
            group = 0
            if column_groups is not None:
                group = column_groups[indices[i]]
            counts[group, this_target] += data[i]
    probabilities = counts / np.sum(counts, axis=1).reshape(-1, 1)
    return probabilities


def information_weight(
    data,
    prior_strength=1e-4,
    target=None,
    column_groups=None,
):
    """Compute information based weights for columns. The information weight
    is estimated as the amount of information gained by moving from a baseline
    model to a model derived from the observed counts. In practice this can be
    computed as the KL-divergence between distributions. For the baseline model
    we assume data distributed according to the row sums -- i.e. proportional
    to the frequency of the row. For the observed model we do a bayesian update
    on the prior distribution with weight prior_strength with the observed counts
    distribution with weight (1-prior_strength).

    Parameters
    ----------
    data: scipy sparse matrix (n_samples, n_features)
        A matrix of count data where rows represent observations and
        columns represent features. Column weightings will be learned
        from this data.

    prior_strength: float (optional, default=0.1)
        How strongly to weight the prior when doing a Bayesian update to
        derive a model based on observed counts of a column.

    target: ndarray or None (optional, default=None)
        If supervised target labels are available, these can be used to define distributions
        over the target classes rather than over rows, allowing weights to be
        supervised and target based. If None then unsupervised weighting is used.

    column_groups: ndarray or None (optional, default=None)
        If columns have a natural grouping, i.e. cols 10-15 are a one-hot-encoding of a single
        categorical variable, we should compare the column distribution to the within group
        marginal. If passed None then all columns have the same group.

    Returns
    -------
    weights: ndarray of shape (n_features,)
        The learned weights to be applied to columns based on the amount
        of information provided by the column.
    """
    csr_data = data.tocsr()
    baseline_probabilities = compute_baseline_probabilities(
        csr_data.indptr,
        csr_data.indices,
        csr_data.data,
        target,
        column_groups,
    )

    csc_data = data.tocsc()
    csc_data.sort_indices()
    weights = column_weights(
        csc_data.indptr,
        csc_data.indices,
        csc_data.data,
        baseline_probabilities,
        prior_strength=prior_strength,
        target=target,
        column_groups=column_groups,
    )

    return weights


class InformationWeightTransformer(BaseEstimator, TransformerMixin):
    """A data transformer that re-weights columns of count data. Column weights
    are computed as information based weights for columns. The information weight
    is estimated as the amount of information gained by moving from a baseline
    model to a model derived from the observed counts. In practice this can be
    computed as the KL-divergence between distributions. For the baseline model
    we assume data will be distributed according to the row sums -- i.e.
    proportional to the frequency of the row. For the observed counts we use
    a background prior of pseudo counts equal to ``prior_strength`` times the
    baseline prior distribution. The Bayesian prior can either be computed
    exactly (the default) at some computational expense, or estimated for a much
    fast computation, often suitable for large or very sparse datasets.

    Parameters
    ----------
    prior_strength: float (optional, default=0.1)
        How strongly to weight the prior when doing a Bayesian update to
        derive a model based on observed counts of a column.

    Attributes
    ----------

    information_weights_: ndarray of shape (n_features,)
        The learned weights to be applied to columns based on the amount
        of information provided by the column.
    """

    def __init__(
        self,
        prior_strength=1e-4,
        weight_power=1.0,
        supervision_weight=0.95,
    ):
        self.prior_strength = prior_strength
        self.weight_power = weight_power
        self.supervision_weight = supervision_weight

    def _format_y(self, y):
        # Format y as array of ints if it is not
        if np.issubdtype(y.dtype, np.number) and not np.issubdtype(y.dtype, np.integer):
            cast_y = y.astype(int)
            if np.all(y == cast_y):
                warn(
                    f"Input y was cast from {y.dtype} to {cast_y.dtype} and will be treated"
                    "as array of ints. Consider passing y as an array of ints.",
                    UserWarning,
                    stacklevel=2,
                )
                y = cast_y
            else:
                warn(
                    f"Input y could not be cast from {y.dtype} to {cast_y.dtype} and will be"
                    "treated as array of objects (identical values have the same class).",
                    UserWarning,
                    stacklevel=2,
                )
        if not np.issubdtype(y.dtype, np.integer):
            target_classes = np.unique(y)
            target_dict = {target_classes[i]: i for i in range(target_classes.shape[0])}
            y = np.array([target_dict[label] for label in y], dtype=np.int64)
        return y

    def fit(self, X, y=None, column_groups=None, **fit_kwds):
        """Learn the appropriate column weighting as information weights
        from the observed count data ``X``.

        Parameters
        ----------
        X: ndarray of scipy sparse matrix of shape (n_samples, n_features)
            The count data to be trained on. Note that, as count data all
            entries should be positive or zero.

        Returns
        -------
        self:
            The trained model.
        """
        if not scipy.sparse.isspmatrix(X):
            X = scipy.sparse.csc_matrix(X)

        self.information_weights_ = information_weight(
            X,
            self.prior_strength,
            column_groups=column_groups,
        )

        if y is not None and self.supervision_weight > 0:
            y_ = self._format_y(y)

            print("Supervised")

            supervised_weights = information_weight(
                X,
                self.prior_strength,
                target=y_,
                column_groups=column_groups,
            )

            print(supervised_weights)
            print(self.information_weights_)

            np.power(
                supervised_weights, self.supervision_weight, out=supervised_weights
            )
            np.power(
                self.information_weights_,
                1 - self.supervision_weight,
                out=self.information_weights_,
            )
            self.information_weights_ = supervised_weights * self.information_weights_

        self.information_weights_ = np.power(
            self.information_weights_, self.weight_power
        )

        return self

    def transform(self, X):
        """Reweight data ``X`` based on learned information weights of columns.

        Parameters
        ----------
        X: ndarray of scipy sparse matrix of shape (n_samples, n_features)
            The count data to be transformed. Note that, as count data all
            entries should be positive or zero.

        Returns
        -------
        result: ndarray of scipy sparse matrix of shape (n_samples, n_features)
            The reweighted data.
        """
        result = X @ scipy.sparse.diags(self.information_weights_)
        return result
