import numpy as np
import scipy.sparse

from infoweight import InformationWeightTransformer

test_matrix = scipy.sparse.csr_matrix(
    [[1, 2, 0, 0, 1], [0, 1, 0, 1, 0], [2, 0, 3, 1, 1], [1, 1, 1, 1, 3]]
)


def test_iw_transformer_default_result():
    IWT = InformationWeightTransformer()
    IWT.fit(test_matrix)
    answer = np.array(
        [
            0.21625041744889634,
            0.8695094613169168,
            0.7028043022802357,
            0.5317269420080221,
            0.3049147097620183,
        ]
    )
    assert np.allclose(IWT.information_weights_, answer)


def test_iw_transformer_zero_prior_result():
    IWT = InformationWeightTransformer(prior_strength=0)
    IWT.fit(test_matrix)
    answer = np.array(
        [
            0.21641190334415927,
            0.8700893643729614,
            0.7032950483706254,
            0.5320623127944701,
            0.30509356278661054,
        ]
    )
    assert np.allclose(IWT.information_weights_, answer)
