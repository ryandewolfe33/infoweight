from warnings import warn

import numba
import numpy as np
import scipy.sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import validate_data


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
    counts = np.zeros((n_groups, n_targets), dtype=data.dtype)
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


@numba.njit(nogil=True, parallel=True)
def compute_baseline_entropies(baseline_probabilities):
    group_entropies = np.empty(baseline_probabilities.shape[0], dtype="float64")
    for i in numba.prange(baseline_probabilities.shape[0]):
        marginal = baseline_probabilities[i, :]
        marginal = marginal[marginal > 0]
        group_entropies[i] = -np.sum(marginal * np.log2(marginal))
    return group_entropies


@numba.njit(nogil=True)
def normalize_by_baseline_entropy(
    weights,
    baseline_probabilities,
    column_groups=None,
):
    baseline_entropies = compute_baseline_entropies(baseline_probabilities)
    if column_groups is None:
        weights /= baseline_entropies[0]
    else:
        for i in numba.prange(weights.shape[0]):
            weights[i] /= baseline_entropies[column_groups[i]]
    return weights


def information_weight(
    data,
    prior_strength=1e-4,
    target=None,
    column_groups=None,
    normalize=True,
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
    if prior_strength < 0 or prior_strength >= 1:
        raise ValueError("prior_strength must be at least 0 and less than 1.")
    if target is not None and len(target) != data.shape[0]:
        raise ValueError("The length of target must be equal to the number of rows.")
    if column_groups is not None and len(column_groups) != data.shape[1]:
        raise ValueError(
            "The number of columns must match the length of column groups."
        )

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

    if normalize:
        normalize_by_baseline_entropy(
            weights,
            baseline_probabilities,
            column_groups=column_groups,
        )

    return weights


class InformationWeightTransformer(TransformerMixin, BaseEstimator):
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
    prior_strength: float (optional, default=0.0001)
        How strongly to weight the prior when doing a Bayesian update to
        derive a model based on observed counts of a column. Must be in
        [0,1).

    weight_power: float (optional, default=1.0)
        Replace each column weight with information_weight**weight_power.
        Weight powers greater than 1 exaggerates the information weight
        transform while values less than 1 dampen. Must be positive.

    supervision_weight: float (optional, default=0.95)
        Parameter for combining supervised and unsupervised weights when
        targets are passed. Final weight is supervised_weight**supervision_weight
        * unsupervised_weight**(1-supervision weight). Must be in (0, 1].

    normalize: bool (optional, default=True)
        Flag to normalize the weights by the entropy of the marginal
        distribution. Essential for combining weights from different
        marginal distribution like is done in supervised mode (when
        supervision weight < 1).

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
        normalize=True,
    ):
        self.prior_strength = prior_strength
        self.weight_power = weight_power
        self.supervision_weight = supervision_weight
        self.normalize = normalize

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "transformer"
        tags.input_tags.sparse = True
        tags.input_tags.positive_only = True
        tags.target_tags.one_d_labels = True
        return tags

    def _validate_data(self, X, y=None, reset=False):
        x_validation = {"accept_sparse": True, "ensure_non_negative": True}
        if y is None:
            X = validate_data(self, X, reset=reset, **x_validation)
        else:
            y_validation = {"ensure_2d": False, "dtype": None}
            X, y = validate_data(
                self,
                X,
                y,
                reset=reset,
                validate_separately=(x_validation, y_validation),
            )
        return X, y

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

    def fit(self, X, y=None, column_groups=None):
        """Learn the appropriate column weighting as information weights
        from the observed count data ``X``.

        Parameters
        ----------
        X: ndarray | scipy.sparse.array | scipy.sparse.matrix
            Input matrix of shape (n_samples, n_features). The count data
            to be trained on. All entries must be positive or zero.

        y: ndarray | None (optional, default=None)
            Array with shape (n_samples, ) of labels for the rows of X that
            are used in supervised mode. If the input array is integers, each
            integer is interpreted as a label and -1 entries are interpreted
            as unlabelled (i.e. semi-supervised mode). Otherwise, each unique
            value is interpreted as a label.

        column_groups: ndarray | None (optional, default=None)
            Array with shape (n_features, ) of labels for known column groups,
            for example if the count data represents several one-hot-encoded
            variables that have been horizontally joined. The information of
            each column is computed with reference to the within column group
            marginal distribution.

        """
        X, y = self._validate_data(X, y, reset=True)
        if not scipy.sparse.isspmatrix(X):
            X = scipy.sparse.csr_matrix(X)

        self.information_weights_ = information_weight(
            X,
            self.prior_strength,
            column_groups=column_groups,
            normalize=self.normalize,
        )

        if y is not None and self.supervision_weight > 0:
            y_ = self._format_y(y)

            supervised_weights = information_weight(
                X,
                self.prior_strength,
                target=y_,
                column_groups=column_groups,
                normalize=self.normalize,
            )

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
        """Reweight data ``X`` based on learned information weights of columns."""
        X, _ = self._validate_data(X, reset=False)
        if isinstance(X, np.ndarray):
            result = X * self.information_weights_
        elif scipy.sparse.issparse(X):
            result = X.multiply(self.information_weights_.reshape(1, -1))
        else:
            raise ValueError("X should be a numpy array or scipy sparse array.")
        return result
