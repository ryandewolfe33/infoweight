import numpy as np
import pytest
import scipy.sparse

from infoweight import InformationWeightTransformer

test_matrix = scipy.sparse.csr_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
test_matrix_zero_row = scipy.sparse.csr_matrix([[1, 2, 3], [4, 5, 6], [0, 0, 0]])
test_matrix_zero_row.eliminate_zeros()
test_matrix_zero_column = scipy.sparse.csr_matrix([[1, 2, 0], [4, 5, 0], [7, 8, 0]])
test_matrix_zero_column.eliminate_zeros()


@pytest.mark.parametrize("prior_strength", [0.1, 0.0])
@pytest.mark.parametrize("normalize", [True, False])
def test_iw_transformer(prior_strength, normalize):
    IWT = InformationWeightTransformer(
        prior_strength=prior_strength, normalize=normalize
    )
    result = IWT.fit_transform(test_matrix)
    transform = IWT.transform(test_matrix)
    assert np.allclose(result.toarray(), transform.toarray())


@pytest.mark.parametrize("prior_strength", [0.1, 0.0])
@pytest.mark.parametrize("target", [None, np.array([0, 1, 1])])
@pytest.mark.parametrize("column_groups", [None, np.array([0, 1, 1])])
@pytest.mark.parametrize("normalize", [True, False])
def test_iw_transformer_fit_args(prior_strength, target, column_groups, normalize):
    IWT = InformationWeightTransformer(
        prior_strength=prior_strength, normalize=normalize
    )
    result = IWT.fit_transform(test_matrix, target, column_groups=column_groups)
    transform = IWT.transform(test_matrix)
    assert np.allclose(result.toarray(), transform.toarray())
    assert np.all(IWT.information_weights_ >= 0)


@pytest.mark.parametrize("prior_strength", [0.1, 0.0])
@pytest.mark.parametrize("target", [None, np.array([0, 1, 1])])
@pytest.mark.parametrize("column_groups", [None, np.array([0, 1, 1])])
@pytest.mark.parametrize("normalize", [True, False])
def test_iw_transformer_zero_column(prior_strength, target, column_groups, normalize):
    IWT = InformationWeightTransformer(
        prior_strength=prior_strength, normalize=normalize
    )
    result = IWT.fit_transform(
        test_matrix_zero_column, target, column_groups=column_groups
    )
    transform = IWT.transform(test_matrix_zero_column)
    assert np.allclose(result.toarray(), transform.toarray())
    assert np.all(IWT.information_weights_ >= 0)


@pytest.mark.parametrize("prior_strength", [0.1, 0.0])
@pytest.mark.parametrize("target", [None, np.array([0, 1, 1])])
@pytest.mark.parametrize("column_groups", [None, np.array([0, 1, 1])])
@pytest.mark.parametrize("normalize", [True, False])
def test_iw_transformer_zero_row(prior_strength, target, column_groups, normalize):
    IWT = InformationWeightTransformer(
        prior_strength=prior_strength, normalize=normalize
    )
    result = IWT.fit_transform(
        test_matrix_zero_row, target, column_groups=column_groups
    )
    transform = IWT.transform(test_matrix_zero_row)
    assert np.allclose(result.toarray(), transform.toarray())
    assert np.all(IWT.information_weights_ >= 0)
