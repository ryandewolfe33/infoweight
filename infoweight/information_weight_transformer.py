from warnings import warn

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import validate_data

from .infoweight_ import compute_group_weights, information_weight
from .stopwords import is_stopword


class InformationWeightTransformer(TransformerMixin, BaseEstimator):
    """A data transformer that re-weights columns of count data. Column weights
    are computed as information based weights for columns. The information weight
    is estimated as the amount of information gained by moving from a baseline
    model to a model derived from the observed counts. The baseline model
    assumes data will be distributed according to the row sums -- i.e.
    proportional to the frequency of the row. The information weight is
    computed as the KL-divergence between the observed column distribution and
    the row marginal distribution, making a small adjustment due to 0's in the
    column distribution.

    Additional steps can be taken to normalize, rescale, and drop columns based
    on optional flags that may require additional knowledge.

    Parameters
    ----------
    dtype: np.float32 | np.float64 | None (optional, default=None)
        Precision to compute and return the weights. Setting to None
        will automatically select the precision equal to the input data
        (e.g. if X is np.float32 or np.int32 then dtype=np.float32)

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
        CDF threshold below which a column is a stopword. The default p=1
        results in no stopwords and the computation is skipped. A typical
        value when pruning stopwords is 0.05 (derived from statistical
        significance). Sampling the CDF is much slower than computing the
        information weight so setting p<1 can significant increase the runtime.
        Identified stopwords have their information weight set to 0, before
        column group reweighting.

    offset: float (optional, default=0.01)
        Offset to subtract from the information weight when computing the cdf.

    n_samples: int (optional, default=100)
        Number of samples used in the CDF computation.

    method: str (optional, default="measured")
        Method to determine if a word is a stopword. method='measured' samples
        random column distributions according to the observed marginal. Setting
        method='uniform' samples from a uniform marginal, and uses a uniform marginal
        to compute the information weight. Setting method='either' gives stopwords that
        satisfy either of the other methods.

    drop_stopwords: bool (optional, default=True)
        Flag to drop stopword columns from the returned matrix. If false, all the values
        in a stopword column will be set to 0 and the returned matrix will have the
        same shape as the input matrix.

    rng: np.random.Generator | None (optional, default=None)
        Source of all randomness, only used for stopword identification.

    Attributes
    ----------

    information_weights_: ndarray of shape (n_features,)
        The learned weights to be applied to columns based on the amount
        of information provided by the column.

    stopword_mask_: ndarray of shape (n_features,)
        Boolean array that notes if each column is a stopword. If p=1,
        no columns can be stopwords, so stopword_mask_ is set to None.

    group_weights_: ndarray of shape (n_groups,)
        Array holding the expected information of each column group. Only
        set if reweight_groups=True and column_groups is not None.

    target_dict_: dict
        Dictionary mapping each unique target value to an integer id. Only set
        if y is passed and it is not already an array of integers.
    """

    def __init__(
        self,
        dtype=None,
        weight_power=1.0,
        supervision_weight=0.95,
        normalize=True,
        reweight_groups=True,
        p=1,
        offset=0.01,
        n_samples=100,
        method="measured",
        drop_stopwords=True,
        rng=None,
    ):
        self.dtype = dtype
        self.weight_power = weight_power
        self.supervision_weight = supervision_weight
        self.normalize = normalize
        self.reweight_groups = reweight_groups
        self.p = p
        self.offset = offset
        self.n_samples = n_samples
        self.method = method
        self.drop_stopwords = drop_stopwords
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
        if y is None:
            return None
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
            self.target_dict_ = {
                target_classes[i]: i for i in range(target_classes.shape[0])
            }
            y = np.array([self.target_dict_[label] for label in y], dtype=np.int64)
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
            variables that have been horizontally joined. By default all columns
            are considered as one group. The information of each column is
            computed with reference to the within column group marginal distribution.

        """
        X, y = self._validate_data(X, y, reset=True)
        # Also handled in each of the called functions but doing it here means
        # we only do it once
        X = sp.csc_array(X)
        y_ = self._format_y(y)
        # No column groups cast to all one group.
        if column_groups is None:
            column_groups = np.broadcast_to(0, X.shape[1])

        # Set or infer dtype
        if self.dtype is not None:
            dtype = self.dtype
        elif isinstance(X.data, np.float32 | np.int32):
            dtype = np.float32
        else:
            dtype = np.float64

        self.information_weights_ = information_weight(
            X,
            supervision_weight=self.supervision_weight,
            normalize=self.normalize,
            target=y_,
            column_groups=column_groups,
            reweight_groups=False,  # reweight after stopwords
            dtype=dtype,
        )

        self.stopword_mask_ = None
        if self.p < 1:
            self.stopword_mask_ = is_stopword(
                self.p,
                self.offset,
                self.n_samples,
                X,
                self.method,
                y_,
                column_groups,
                self.normalize,
                self.reweight_groups,
                self.supervision_weight,
                self.dtype,
                self.rng,
                self.information_weights_,
            )
            self.information_weights_[self.stopword_mask_] = 0

        if self.reweight_groups:
            self.group_weights_ = compute_group_weights(
                X, self.information_weights_, column_groups
            )
            self.information_weights_ *= self.group_weights_[column_groups]

        self.information_weights_ **= self.weight_power
        return self

    def transform(self, X):
        """Reweight data ``X`` based on learned information weights of columns."""
        X, _ = self._validate_data(X, reset=False)
        if isinstance(X, np.ndarray):
            result = X * self.information_weights_
        elif sp.issparse(X):
            result = X.multiply(self.information_weights_.reshape(1, -1))
            result = result.tocsc()
        else:
            raise ValueError("X should be a numpy array or scipy sparse array.")
        if self.stopword_mask_ is not None and self.drop_stopwords:
            result = result[:, ~self.stopword_mask_]
        return result
