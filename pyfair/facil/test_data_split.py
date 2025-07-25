# coding: utf-8


import numpy as np
# import pdb
from pyfair.facil.utils_const import synthetic_dat

from pyfair.facil.data_split import (
    sklearn_k_fold_cv, sklearn_stratify, manual_repetitive,
    scale_normalize_helper, scale_normalize_data,
    get_splited_set_acdy, sitch_cross_validation,
    situation_split1, situation_split2, situation_split3)


nb_inst, nb_lbl, nb_feat = 21, 3, 5
nb_cv, k = 2, 1  # or 2,3,5
X, y = synthetic_dat(nb_lbl, nb_inst, nb_feat)


def test_sklearn():
    si = sklearn_k_fold_cv(nb_cv, y)
    assert len(si) == nb_cv
    assert all([len(j) + len(k) == nb_inst for j, k in si])
    si = sklearn_stratify(nb_cv, y, X)
    assert len(si) == nb_cv
    assert all([len(j) + len(k) == nb_inst for j, k in si])

    i_trn, i_tst = si[k]
    (_, X_val, _,
     _, y_val, _) = get_splited_set_acdy(
        np.array(X), np.array(y), [i_trn, [], i_tst])
    assert not (X_val or y_val)

    for gen in [False, True]:
        si = manual_repetitive(nb_cv, y, gen)
        assert np.shape(si) == (nb_cv, nb_inst)

    for typ in ['standard', 'min_max', 'min_abs', 'normalize']:
        scaler = scale_normalize_helper(typ, X)
        scaler, X_trn, X_val, X_tst = scale_normalize_data(
            scaler, X, [], X)
        assert np.shape(X_trn) == np.shape(X_tst)
    return


def test_CV():
    si = sitch_cross_validation(nb_cv, y, 'cv2')
    assert all([len(j) + len(k) == nb_inst for j, k in si])
    si = sitch_cross_validation(nb_cv, y, 'cv3')
    assert all([len(i) + len(j) + len(
        k) == nb_inst for i, j, k in si])

    pr_trn, pr_tst = .7, .2
    si = situation_split1(y, pr_trn, None)
    assert len(si[0][0]) + len(si[0][1]) == nb_inst
    si = situation_split1(y, pr_trn, pr_tst)
    assert sum([len(i) for i in si[0]]) == nb_inst

    si = situation_split2(pr_trn, nb_cv, y)
    assert all([len(j) + len(k) == nb_inst for j, k in si])
    si = situation_split3(pr_trn, pr_tst, nb_cv, y)
    assert all([len(i) + len(j) + len(
        k) == nb_inst for i, j, k in si])

    # pdb.set_trace()
    return
