import numpy as np
import pytest
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
