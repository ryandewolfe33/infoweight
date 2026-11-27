import numpy as np
import pytest
import scipy.sparse as sp
from numba import typeof
from numba.core.types import float32, float64, int32, int64

import infoweight


@pytest.mark.parametrize("idx_type", [int32[:], int64[:]])
@pytest.mark.parametrize("count_type", [float32[:], float64[:]])
@pytest.mark.parametrize("marginal_type", [float32[:], float64[:]])
@pytest.mark.parametrize(
    "dtype, return_type",
    [
        (typeof(np.float32), float32),
        (typeof(np.float64), float64),
    ],
)
def test_kl_divergence(idx_type, count_type, marginal_type, dtype, return_type):
    sig = (idx_type, count_type, marginal_type, dtype)
    infoweight.information_theory.kl_divergence.compile(sig)
    assert (
        infoweight.information_theory.kl_divergence.overloads[sig].signature.return_type
        == return_type
    )


test_matrix = sp.csr_array(
    [[1, 2, 0, 0, 1], [0, 1, 0, 1, 0], [2, 0, 3, 1, 1], [1, 1, 1, 1, 3]]
)


@pytest.mark.parametrize("input_type", [np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("output_type", [np.float32, np.float64])
@pytest.mark.parametrize("target", [None, np.array([0, 0, 1, 1])])
@pytest.mark.parametrize("column_groups", [None, np.array([0, 0, 1, 1, 1])])
def test_transformer(input_type, output_type, target, column_groups):
    input = test_matrix.astype(input_type)
    IWT = infoweight.InformationWeightTransformer(dtype=output_type)

    IWT.fit(input, target, column_groups=column_groups)

    assert IWT.dtype == output_type
    assert IWT.information_weights_.dtype == output_type
