from warnings import warn

import numba
import numpy as np
import scipy.sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import validate_data

from .weight import column_weight
from.marginal import make_marginals
from .stopwords import is_stopword

DUMMY_FLOAT_ARRAY = np.empty(0, dtype=np.float64)
DUMMY_FLOAT_1ROW_ARRAY = np.empty((1, 0), dtype=np.float64)
NAN_ARRAY = np.array([np.nan])


@numba.njit(nogil=True, cache=True, parallel=True)
def column_weights(
    indptr,
    indices,
    data,
    n_rows,
    prior_strength,
    supervision_weight,
    normalize=True,
    target=None,
    column_groups=None,
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

    n_rows: int
        Number of rows in the matrix.

    prior_strength: float
        Prior strength to use in the bayesian update

    normalize: bool (default=True)
        Flag for wether to normalize weights by the entropy of the marginal distribution.

    target: NDArray | None (defualt=None)
        Array of target ids for the rows.

    column_groups: NDArray | None (default=None)
        Array of group ids for the columns.

    Returns
    -------
    weights: NDArray
        Array of the information weight for each column.
    
    """
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


@numba.njit
def expected_group_weight(
    weights,
    column_groups,
    column_marginal,
):
    """Compute the expected (over the row marginal distribution)
    information weight per column group. Used for column group
    reweighting.

    Parameters
    ----------
    weights: NDArray
        Array of the information weight for each column

    column_groups: NDArray
        Array of the group id for each column.
    
    column_marginal: NDArray
        Array of the count (or probability) for each column.
    
    Returns
    -------
    expected_weight: NDArray
        Array with length equal to the number of column groups,
        where each entry is the expected information weight of
        that group.
    """
    n_groups = np.max(column_groups) + 1
    eiw = np.zeros(n_groups)
    group_total_probs = np.zeros(n_groups)
    for weight, group, prob in zip(weights, column_groups, column_marginal):
        eiw[group] += prob * weight
        group_total_probs[group] += prob
    eiw[group_total_probs > 0] /= group_total_probs[group_total_probs > 0]
    return eiw


def information_weight(
    data,
    prior_strength=1e-4,
    supervision_weight=0.95,
    normalize=True,
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

    csc_data = data.tocsc()

    weights = column_weights(
        csc_data.indptr,
        csc_data.indices,
        csc_data.data,
        csc_data.shape[0],
        prior_strength,
        supervision_weight,
        normalize,
        target,
        column_groups,
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

    reweight_group: bool (optional, default=True)
        Flag to reweight each group by the expected information weight
        according to the column marginal. If p<1 and stopwords are
        removed, expected information is computed after dropping
        stopwords.

    p: float (optional, default=1)
        CDF threshold below which a column is a stopword. Setting p=1
        results in no stopwords and the computation is skipped. Sampling
        the CDF is much slower than computing the information weight so
        setting p<1 will increase the runtime.

    offset: float (optional, default=0.01)
        Offset to subtract from the information weight when computing the cdf.

    n_samples: int (optional, default=100)
        Number of samples used in the CDF computation.

    method: str (optional, default="measured")
        Method to determine if a word is a stopword. method='measured' uses
        samples random column distributions according to the observed marginal.
        Setting method='uniform' samples from a uniform marginal, and uses the
        uniform marginal to compute the information weight. Setting
        method='either' give stopwords that satisfy either of the other
        methods.

    rng: np.random.Generator | None (optional, default=None)

    Attributes
    ----------

    information_weights_: ndarray of shape (n_features,)
        The learned weights to be applied to columns based on the amount
        of information provided by the column.

    stopword_: ndarray of shape (n_features,)
        Boolean array that notes if each column is a stopword. If p=1,
        no columns can be stopwords, so stopword_ is set to None.

    group_weights_: ndarray of shape (n_groups,)
        Array holding the expected information of each column group. Only
        set if reweight_groups=True and column_groups is not None.
    """

    def __init__(
        self,
        prior_strength=1e-4,
        weight_power=1.0,
        supervision_weight=0.95,
        normalize=True,
        reweight_groups=True,
        p=1,
        offset=0.01,
        n_samples=100,
        method="measured",
        rng=None,
    ):
        self.prior_strength = prior_strength
        self.weight_power = weight_power
        self.supervision_weight = supervision_weight
        self.normalize = normalize
        self.reweight_groups = reweight_groups
        self.p = p
        self.offset = offset
        self.n_samples = n_samples
        self.method = method
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
            X = scipy.sparse.csc_matrix(X)

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
                self.offset,
                self.n_samples,
                X,
                self.prior_strength,
                self.supervision_weight,
                self.normalize,
                y_,
                column_groups,
                self.method,
                self.rng,
            )

        if (
            self.reweight_groups
            and column_groups is not None
            and not np.all(self.stopword_)
        ):
            column_marginal = np.asarray(X.sum(axis=0)).reshape(-1).astype("float64")
            if self.stopword_ is not None:
                column_marginal[self.stopword_] = 0
            column_marginal /= np.sum(column_marginal)
            self.group_weights_ = expected_group_weight(
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
