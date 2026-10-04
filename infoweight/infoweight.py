from warnings import warn

import numba
import numpy as np
import scipy.sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import validate_data

DUMMY_FLOAT_ARRAY = np.empty(0, dtype=np.float64)
DUMMY_FLOAT_1ROW_ARRAY = np.empty((1, 0), dtype=np.float64)
NAN_ARRAY = np.array([np.nan])


@numba.njit(nogil=True, cache=True)
def column_kl_divergence(
    count_indices, count_data, marginal, prior_strength, marginal_entropy=np.nan
):
    """Function to compute the KL-divergence between a prior and a posterior distribution,
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
    if np.isinf(prior_info):  # Happens if prior_strength == -inf
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


@numba.njit(nogil=True, cache=True, parallel=True)
def column_weights(
    indptr,
    indices,
    data,
    marginal,
    prior_strength,
    supervision_weight,
    column_groups=None,
    marginal_entropy=NAN_ARRAY,
    target=None,
    target_marginal=DUMMY_FLOAT_1ROW_ARRAY,
    target_marginal_entropy=NAN_ARRAY,
):
    n_cols = indptr.shape[0] - 1
    weights = np.empty(n_cols)
    for i in numba.prange(n_cols):
        count_indices = indices[indptr[i] : indptr[i + 1]]
        count_data = data[indptr[i] : indptr[i + 1]]
        group = column_groups[i] if column_groups is not None else 0
        weights[i] = column_weight(
            count_indices,
            count_data,
            marginal[group],
            prior_strength,
            supervision_weight,
            marginal_entropy=marginal_entropy[group],
            target=target,
            target_marginal=target_marginal[group],
            target_marginal_entropy=target_marginal_entropy[group],
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


@numba.njit
def expected_information_weight(
    weights,
    column_groups,
    column_marginal,
):
    n_groups = np.max(column_groups) + 1
    eiw = np.zeros(n_groups)
    group_total_probs = np.zeros(n_groups)
    for weight, group, prob in zip(weights, column_groups, column_marginal):
        eiw[group] += prob * weight
        group_total_probs[group] += prob
    eiw /= group_total_probs
    return eiw


@numba.njit(nogil=True)
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


@numba.njit
def sample_cdf(
    n_samples,
    rngs,
    information_weights,
    indptr,
    indices,
    data,
    marginal,
    prior_strength,
    supervision_weight,
    marginal_entropy=None,
    target=None,
    target_marginal=None,
    target_marginal_entropy=None,
    column_groups=None,
):
    n_cols = len(indptr) - 1
    cdf = np.zeros(n_cols, dtype=np.float64)

    alias_arrays = np.empty((marginal.shape[0], marginal.shape[1], 2))
    for group in range(marginal.shape[0]):
        alias_arrays[group, :, :] = alias_sampling_setup(marginal[group])

    for col in numba.prange(n_cols):
        group = column_groups[col] if column_groups is not None else 0
        group_marginal = marginal[group]
        group_marginal_entropy = (
            marginal_entropy[group] if marginal_entropy is not None else np.nan
        )
        group_target_marginal = (
            target_marginal[group] if target is not None else DUMMY_FLOAT_ARRAY
        )
        group_target_marginal_entropy = (
            target_marginal_entropy[group] if target is not None else np.nan
        )
        alias_array = alias_arrays[group, :, :]
        count = np.sum(data[indptr[col] : indptr[col + 1]])
        rng = rngs[col]
        for _ in range(n_samples):
            sample_indices, sample_data = sample_column(
                count,
                alias_array,
                rng,
            )
            sample_weight = column_weight(
                sample_indices,
                sample_data,
                group_marginal,
                prior_strength,
                supervision_weight,
                marginal_entropy=group_marginal_entropy,
                target=target,
                target_marginal=group_target_marginal,
                target_marginal_entropy=group_target_marginal_entropy,
            )
            if sample_weight < information_weights[col]:
                cdf[col] += 1
    cdf /= n_samples
    return cdf


def is_stopword(
    p,
    n_samples,
    data,
    information_weights,
    prior_strength,
    supervision_weight,
    normalize=True,
    target=None,
    column_groups=None,
    rng=None,
):
    """Determine which columns (if any) are stopwords. A stopword is
    a column that, with probability p, has less information than if it
    were randomly sampled according to the marginal distribution.

    Parameters
    ----------
    p: float
    n_samples: int
    data: sp.sparse_array | sp.sparse_matrix
    information_weights: NDArray[float]
    prior_strength: float
    supervision_weight: float
    normalize: bool
    target: NDArray[int] | None
    column_groups: NDArray[int] | None
    rng: np.random.Generator | None
    """
    if rng is None:
        rng = np.random.default_rng()

    csr_data = data.tocsr()
    marginal = compute_baseline_probabilities(
        csr_data.indptr,
        csr_data.indices,
        csr_data.data,
        target=None,
        column_groups=column_groups,
    )
    target_marginal = (
        compute_baseline_probabilities(
            csr_data.indptr,
            csr_data.indices,
            csr_data.data,
            target=target,
            column_groups=column_groups,
        )
        if target is not None
        else None
    )

    marginal_entropy = compute_baseline_entropies(marginal) if normalize else np.nan
    target_marginal_entropy = (
        compute_baseline_entropies(target_marginal)
        if normalize and target is not None
        else NAN_ARRAY
    )

    csc_data = data.tocsc()

    # Spawn independent generators for each column so we can process in parallel
    rngs = rng.spawn(data.shape[1])

    cdf = sample_cdf(
        n_samples,
        rngs,
        information_weights,
        csc_data.indptr,
        csc_data.indices,
        csc_data.data,
        marginal,
        prior_strength,
        supervision_weight,
        marginal_entropy,
        target,
        target_marginal,
        target_marginal_entropy,
        column_groups,
    )

    return cdf < (1 - p)


def information_weight(
    data,
    prior_strength=1e-4,
    target=None,
    supervision_weight=0.95,
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

    supervision_weight: float (optional, default=0.95)
        Parameter for combining supervised and unsupervised weights when
        targets are passed. Final weight is supervised_weight**supervision_weight
        * unsupervised_weight**(1-supervision weight). Must be in (0, 1].

    column_groups: ndarray or None (optional, default=None)
        If columns have a natural grouping, i.e. cols 10-15 are a one-hot-encoding of a single
        categorical variable, we should compare the column distribution to the within group
        marginal. If passed None then all columns have the same group.

    normalize: bool (optional, default=True)
        Normalize the information weight by dividing by the entropy of the marginal distribution.
        This normalizes the 'scale' so weights from different distributions can be combined.

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
    if supervision_weight < 0 or supervision_weight > 1:
        raise ValueError("supervision_weight must be at least 0 and at most 1.")

    csr_data = data.tocsr()
    marginal = compute_baseline_probabilities(
        csr_data.indptr,
        csr_data.indices,
        csr_data.data,
        target=None,
        column_groups=column_groups,
    )
    if normalize:
        marginal_entropy = compute_baseline_entropies(marginal)
    else:
        # nan means don't normalize
        marginal_entropy = np.full(marginal.shape[0], np.nan)

    if target is not None:
        target_marginal = compute_baseline_probabilities(
            csr_data.indptr,
            csr_data.indices,
            csr_data.data,
            target=target,
            column_groups=column_groups,
        )
        if normalize:
            target_marginal_entropy = compute_baseline_entropies(target_marginal)
        else:
            # nan means don't normalize
            target_marginal_entropy = np.full(target_marginal.shape[0], np.nan)
    else:
        target_marginal = DUMMY_FLOAT_1ROW_ARRAY
        target_marginal_entropy = NAN_ARRAY

    csc_data = data.tocsc()
    csc_data.sort_indices()

    weights = column_weights(
        csc_data.indptr,
        csc_data.indices,
        csc_data.data,
        marginal,
        prior_strength,
        supervision_weight,
        column_groups=column_groups,
        marginal_entropy=marginal_entropy,
        target=target,
        target_marginal=target_marginal,
        target_marginal_entropy=target_marginal_entropy,
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
        reweight_groups=True,
        p=1,
        n_samples=100,
        rng=None,
    ):
        self.prior_strength = prior_strength
        self.weight_power = weight_power
        self.supervision_weight = supervision_weight
        self.normalize = normalize
        self.reweight_groups = reweight_groups
        self.p = p
        self.n_samples = n_samples
        self.rng = rng

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

        y_ = None
        if y is not None and self.supervision_weight > 0:
            y_ = self._format_y(y)

        self.information_weights_ = information_weight(
            X,
            self.prior_strength,
            target=y_,
            supervision_weight=self.supervision_weight,
            column_groups=column_groups,
            normalize=self.normalize,
        )

        self.stopword_ = None
        if self.p < 1:
            self.stopword_ = is_stopword(
                self.p,
                self.n_samples,
                X,
                self.information_weights_,
                self.prior_strength,
                self.supervision_weight,
                normalize=self.normalize,
                target=y_,
                column_groups=column_groups,
                rng=self.rng,
            )

        if self.reweight_groups and column_groups is not None:
            column_marginal = np.asarray(X.sum(axis=0)).reshape(-1).astype("float64")
            if self.stopword_ is not None:
                column_marginal[~self.stopword_] = 0
            column_marginal /= np.sum(column_marginal)
            self.group_weights_ = expected_information_weight(
                self.information_weights_,
                column_groups,
                column_marginal,
            )
            self.information_weights_ *= self.group_weights_[column_groups]

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
            result = result.tocsc()
        else:
            raise ValueError("X should be a numpy array or scipy sparse array.")
        if self.stopword_ is not None:
            result = result[:, ~self.stopword_]
        return result
