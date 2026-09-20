import numpy as np
import scipy.sparse

from infoweight import InformationWeightTransformer

test_matrix = scipy.sparse.csr_matrix(
    [[1, 2, 0, 0, 1], [0, 1, 0, 1, 0], [2, 0, 3, 1, 1], [1, 1, 1, 1, 3]]
)


def test_iw_transformer_default_result():
    IWT = InformationWeightTransformer(normalize=False)
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
    IWT = InformationWeightTransformer(prior_strength=0, normalize=False)
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


def test_iw_transformer_supervised_result():
    IWT = InformationWeightTransformer(normalize=False)
    IWT.fit(test_matrix, np.array([0, 0, 1, 1]))
    answer = np.array(
        [
            0.010429578294587379,
            0.6305379429705708,
            0.5222153619954761,
            0.00479285019039355,
            0.04123724606682257,
        ]
    )
    assert np.allclose(IWT.information_weights_, answer)


def test_iw_transformer_supervised_one_supervised_weight_result():
    IWT = InformationWeightTransformer(supervision_weight=1, normalize=False)
    IWT.fit(test_matrix, np.array([0, 0, 1, 1]))
    answer = np.array(
        [
            0.008891340950519605,
            0.6199630411917103,
            0.5141158240611795,
            0.0037407379487476647,
            0.037115786584735494,
        ]
    )
    assert np.allclose(IWT.information_weights_, answer)


def test_iw_transformer_semisupervised_column_groups_result():
    IWT = InformationWeightTransformer(normalize=False)
    IWT.fit(
        test_matrix, np.array([0, 0, 1, -1]), column_groups=np.array([0, 0, 1, 1, 1])
    )
    answer = np.array(
        [
            0.3342522938657766,
            0.5773158995466053,
            0.48325836275214235,
            0.15484925944254302,
            0.1530203594658495,
        ]
    )
    assert np.allclose(IWT.information_weights_, answer)


def test_iw_transformer_semisupervised_result():
    IWT = InformationWeightTransformer(normalize=False)
    IWT.fit(test_matrix, np.array([0, 0, 1, -1]))
    answer = np.array(
        [
            0.052683417978751326,
            1.1008690917759285,
            0.8817724537554549,
            0.0054471660185512125,
            0.005297794369109902,
        ]
    )
    assert np.allclose(IWT.information_weights_, answer)


def test_iw_transformer_column_groups_result():
    IWT = InformationWeightTransformer(normalize=False)
    IWT.fit(test_matrix, column_groups=np.array([0, 0, 1, 1, 1]))
    answer = np.array(
        [
            0.35354293404823056,
            0.45713280216144275,
            0.4514702271310239,
            0.45186981953145916,
            0.3562993568566697,
        ]
    )
    assert np.allclose(IWT.information_weights_, answer)


def test_iw_transformer_supervised_column_groups_result():
    IWT = InformationWeightTransformer(normalize=False)
    IWT.fit(
        test_matrix, np.array([0, 0, 1, 1]), column_groups=np.array([0, 0, 1, 1, 1])
    )
    answer = np.array(
        [
            0.19470029553734533,
            0.19721803254086,
            0.26999221258333383,
            0.12689748978402643,
            0.006764705114180968,
        ]
    )
    assert np.allclose(IWT.information_weights_, answer)
