import numpy as np
import pytest
import scipy.sparse

from infoweight.infoweight import information_weight

test_matrix = scipy.sparse.csr_matrix(
    [[1, 2, 0, 0, 1], [0, 1, 0, 1, 0], [2, 0, 3, 1, 1], [1, 1, 1, 1, 3]]
)


def test_incorrect_prior_strength():
    with pytest.raises(ValueError):
        information_weight(test_matrix, prior_strength=-1)
    with pytest.raises(ValueError):
        information_weight(test_matrix, prior_strength=1)


def test_incorrect_target():
    with pytest.raises(ValueError):
        information_weight(test_matrix, target=np.array([0, 1, 1]))
    with pytest.raises(ValueError):
        information_weight(test_matrix, target=np.array([0, 1, 1, 1, 0]))


def test_incorrect_column_groups():
    with pytest.raises(ValueError):
        information_weight(test_matrix, column_groups=np.array([0, 0, 0, 1]))
    with pytest.raises(ValueError):
        information_weight(test_matrix, column_groups=np.array([0, 0, 1, 1, 1, 1]))
