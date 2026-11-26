import numpy as np
import pytest
import scipy.sparse

from infoweight import InformationWeightTransformer

test_matrix = scipy.sparse.csr_matrix(
    [[1, 2, 0, 0, 1], [0, 1, 0, 1, 0], [2, 0, 3, 1, 1], [1, 1, 1, 1, 3]]
)
# row sums [4, 3, 7, 7]
test_array = scipy.sparse.csr_array(test_matrix)


@pytest.mark.parametrize("data", [test_matrix, test_array])
def test_iw_transformer_default_result(data):
    IWT = InformationWeightTransformer(normalize=False, reweight_groups=False)
    IWT.fit(data)
    answer = np.array(
        [
            0.216308992554068,
            0.869720202999606,
            0.702982477917093,
            0.531848923510630,
            0.304980051733278,
        ]
    )
    assert np.allclose(IWT.information_weights_, answer)


@pytest.mark.parametrize("data", [test_matrix, test_array])
def test_iw_transformer_supervised_result(data):
    IWT = InformationWeightTransformer(normalize=False, reweight_groups=False)
    IWT.fit(data, np.array([0, 0, 1, 1]))
    answer = np.array(
        [
            0.0104305066205960,
            0.6305931400667026,
            0.5223813580563722,
            0.0047932567219263,
            0.0412408856699958,
        ]
    )
    assert np.allclose(IWT.information_weights_, answer)


@pytest.mark.parametrize("data", [test_matrix, test_array])
def test_iw_transformer_supervised_one_supervised_weight_result(data):
    IWT = InformationWeightTransformer(
        supervision_weight=1, normalize=False, reweight_groups=False
    )
    IWT.fit(data, np.array([0, 0, 1, 1]))
    answer = np.array(
        [
            0.0088920472644947,
            0.6200122609692549,
            0.5142809867639863,
            0.0037410267756482,
            0.0371188162376298,
        ]
    )
    assert np.allclose(IWT.information_weights_, answer)


@pytest.mark.parametrize("data", [test_matrix, test_array])
def test_iw_transformer_semisupervised_result(data):
    IWT = InformationWeightTransformer(normalize=False, reweight_groups=False)
    IWT.fit(data, np.array([0, 0, 1, -1]))
    answer = np.array(
        [
            0.0526881061309598,
            1.1011724096623694,
            0.8820285473808588,
            0.0054476314171496,
            0.0052982430035547,
        ]
    )
    assert np.allclose(IWT.information_weights_, answer)


@pytest.mark.parametrize("data", [test_matrix, test_array])
def test_iw_transformer_column_groups_result(data):
    IWT = InformationWeightTransformer(normalize=False, reweight_groups=False)
    IWT.fit(data, column_groups=np.array([0, 0, 1, 1, 1]))
    answer = np.array(
        [
            0.3536218352117088,
            0.4572723171858078,
            0.4515744415109636,
            0.4519355993488065,
            0.3563620022324862,
        ]
    )
    assert np.allclose(IWT.information_weights_, answer)


@pytest.mark.parametrize("data", [test_matrix, test_array])
def test_iw_transformer_supervised_column_groups_result(data):
    IWT = InformationWeightTransformer(normalize=False, reweight_groups=False)
    IWT.fit(data, np.array([0, 0, 1, 1]), column_groups=np.array([0, 0, 1, 1, 1]))
    answer = np.array(
        [
            0.1947176018587121,
            0.1972363713372797,
            0.2700832944405920,
            0.1269071321524144,
            0.0067652539213802,
        ]
    )
    assert np.allclose(IWT.information_weights_, answer)


@pytest.mark.parametrize("data", [test_matrix, test_array])
def test_iw_transformer_semisupervised_column_groups_result(data):
    IWT = InformationWeightTransformer(normalize=False, reweight_groups=False)
    IWT.fit(data, np.array([0, 0, 1, -1]), column_groups=np.array([0, 0, 1, 1, 1]))
    answer = np.array(
        [
            0.334280772060793,
            0.577497776895318,
            0.483412531858822,
            0.154861479387390,
            0.153032666505991,
        ]
    )
    assert np.allclose(IWT.information_weights_, answer)
