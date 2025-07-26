# coding: utf-8


import numpy as np
# import pdb
# import lightgbm

from pyfair.facil.utils_const import check_equal
from pyfair.marble.metric_fair import (
    marginalised_np_mat, marginalised_np_gen)  # addtl,addl

from pyfair.granite.fair_meas_indiv import (
    GEI_Theil, prop_L_fair, prop_L_loss, DistDirect)

from pyfair.granite.fair_meas_group import (
    UD_grp1_DP, UD_grp1_DisI, UD_grp1_DisT,
    UD_grp2_EO, UD_grp2_EOdd, UD_grp2_PEq,
    UD_grp3_PQP, UD_gammaSubgroup, UD_BoundedGrpLos)

# from pyfair.datasets import PropublicaViolentRecidivism
# from pyfair.preprocessing_hfm import (
#     renewed_prep_and_adversarial, renewed_transform_X_A_and_y,
#     check_marginalised_indices)
# from pyfair.preprocessing_dr import transform_unpriv_tag


n = 110
y = np.random.randint(2, size=n)
y_hat = np.random.randint(2, size=n)

nc, na = 3, 2
A = np.random.randint(nc, size=(n, na)) + 1
idx_Ai_Sjs = [[A[:, i] == j + 1 for j in range(
    nc)] for i in range(na)]
idx_Sjs = [A[:, 0] == j + 1 for j in range(nc)]

A_bin = A.copy()
A_bin[A_bin != 1] = 0
idx_a = [[A_bin[:, i] == 1, A_bin[:, i] != 1] for i in range(na)]
idx_ai = idx_a[0]
pos, priv_val = 1, 1,
A_i, priv_idx = A[:, 1], A[:, 1] == 1
vals_in_Ai = list(set(A_i))
A1_bin, val_A1 = A_bin[:, 1], [1, 0]


"""
ds = PropublicaViolentRecidivism()
df = ds.load_raw_dataset()
(origin_dat, processed_dat, process_mult, perturbed_dat, perturb_mult
 ) = renewed_prep_and_adversarial(ds, df, .97, None)
processed_Xy = process_mult['numerical-multisen']
perturbed_Xy = perturb_mult['numerical-multisen']
X, A, y, _ = renewed_transform_X_A_and_y(ds, processed_Xy, False)
_, Aq, _, _ = renewed_transform_X_A_and_y(ds, perturbed_Xy, False)
# tmp = processed_dat['original'][ds.label_name]
sen_att = ds.get_sensitive_attrs_with_joint()[: 2]
priv_val = ds.get_privileged_group_with_joint('')[: 2]
marginalised_grp = origin_dat['marginalised_groups']
margin_indices = check_marginalised_indices(
    processed_dat['original'], sen_att, priv_val,
    marginalised_grp)
# new_attr = '-'.join(sen_att) if len(sen_att) > 1 else None
# belongs_priv, ptb_with_joint = transform_unpriv_tag(
#     ds, processed_dat['original'], 'both')

X_and_A = np.concatenate([X, A], axis=1)
X_and_Aq = np.concatenate([X, Aq], axis=1)
clf = lightgbm.LGBMClassifier(n_estimators=7)
clf.fit(X_and_A, y)
clf.fit(X_and_A, y)
clf.fit(X_and_A, y)
y_hat = clf.predict(X_and_A)
pos = ds.get_positive_class_val('')
y, priv_val = y.values, 1
A_i = A['race'].values           # A[:, 1]
priv_idx = margin_indices[1][0]  # A_i == priv_val[1]
vals_in_Ai = list(set(A_i))
# pdb.set_trace()
del ds, df, origin_dat, processed_Xy, perturbed_Xy  # , tmp
del processed_dat, process_mult, perturbed_dat, perturb_mult
# del new_attr, belongs_priv, ptb_with_joint
acc = (y == y_hat).mean()
"""


def test_metric_grp1():
    m1 = UD_grp1_DP.bival(y, y_hat, priv_idx, pos)
    m2 = UD_grp1_DP.mu_sp(y, y_hat, A_i, priv_val, pos)
    m3 = UD_grp1_DP.mu_cx(y, y_hat, A_i, priv_val, pos)

    m4 = UD_grp1_DP.yev_sp(y, y_hat, A_i, vals_in_Ai, pos)
    m5 = UD_grp1_DP.yev_cx(y, y_hat, A_i, vals_in_Ai, pos)
    m6 = UD_grp1_DP.yev_cx(y, y_hat, A1_bin, val_A1, pos)
    m7 = UD_grp1_DP.mu_sp(y, y_hat, A1_bin, priv_val, pos)
    m8 = UD_grp1_DP.mu_cx(y, y_hat, A1_bin, priv_val, pos)
    assert check_equal(m1[0][0], [m2[0], m4[0][3][0]])  # m3[0][0],
    assert check_equal(m6[0][2][0], [m7[0], m8[0][1]])  # m8[0][0]])

    qa_1 = UD_grp1_DisI.bival(y, y_hat, priv_idx, pos)
    qa_2 = UD_grp1_DisI.mu_sp(y, y_hat, A_i, priv_val, pos)
    qa_3 = UD_grp1_DisI.mu_cx(y, y_hat, A_i, priv_val, pos)

    qa_4 = UD_grp1_DisI.yev_sp(y, y_hat, A_i, vals_in_Ai, pos)
    qa_5 = UD_grp1_DisI.yev_cx(y, y_hat, A_i, vals_in_Ai, pos)
    qa_6 = UD_grp1_DisI.yev_cx(y, y_hat, A1_bin, val_A1, pos)
    qa_7 = UD_grp1_DisI.mu_sp(y, y_hat, A1_bin, priv_val, pos)
    qa_8 = UD_grp1_DisI.mu_cx(y, y_hat, A1_bin, priv_val, pos)
    assert check_equal(qa_1[0][0], [qa_2[0], qa_4[0][3][0]])
    # assert check_equal(qa_6[0][2][0], [qa_7[0], qa_8[0][1]])
    assert check_equal(qa_7[0], qa_8[0][1])

    qb_1 = UD_grp1_DisT.bival(y, y_hat, priv_idx, pos)
    qb_2 = UD_grp1_DisT.mu_sp(y, y_hat, A_i, priv_val, pos)
    qb_3 = UD_grp1_DisT.mu_cx(y, y_hat, A_i, priv_val, pos)

    qb_5_a = UD_grp1_DisT.mu_sp(y, y_hat, A1_bin, priv_val, pos)
    qb_5_b = UD_grp1_DisT.mu_cx(y, y_hat, A1_bin, priv_val, pos)
    qb_4 = UD_grp1_DisT.yev_sp(y, y_hat, A_i, vals_in_Ai, pos)
    qb_6 = UD_grp1_DisT.yev_sp(y, y_hat, A1_bin, val_A1, pos)
    qb_7 = UD_grp1_DisT.mu_sp(y, y_hat, A1_bin, priv_val, pos)
    qb_8 = UD_grp1_DisT.mu_cx(y, y_hat, A1_bin, priv_val, pos)
    assert check_equal(qb_1[0][0], [qb_2[0], qb_7[0], qb_8[0],
                                    qb_5_a[0], qb_5_b[0]])

    # pdb.set_trace()
    return


def test_metric_grp2():
    m1 = UD_grp2_EO.bival(y, y_hat, priv_idx, pos)
    m2 = UD_grp2_EO.mu_sp(y, y_hat, A_i, priv_val, pos)
    m3 = UD_grp2_EO.mu_cx(y, y_hat, A_i, priv_val, pos)

    m4 = UD_grp2_EO.yev_sp(y, y_hat, A_i, vals_in_Ai, pos)
    m5 = UD_grp2_EO.yev_cx(y, y_hat, A_i, vals_in_Ai, pos)
    m6 = UD_grp2_EO.yev_cx(y, y_hat, A1_bin, val_A1, pos)
    m7 = UD_grp2_EO.mu_sp(y, y_hat, A1_bin, priv_val, pos)
    m8 = UD_grp2_EO.mu_cx(y, y_hat, A1_bin, priv_val, pos)
    assert check_equal(m1[0][0], [m2[0], m4[0][3][0]])
    assert check_equal(m6[0][2][0], [m7[0], m8[0][1]])

    qa_1 = UD_grp2_EOdd.bival(y, y_hat, priv_idx, pos)
    qa_2 = UD_grp2_EOdd.mu_sp(y, y_hat, A_i, priv_val, pos)
    qa_3 = UD_grp2_EOdd.mu_cx(y, y_hat, A_i, priv_val, pos)

    qa_4 = UD_grp2_EOdd.yev_sp(y, y_hat, A_i, vals_in_Ai, pos)
    qa_5 = UD_grp2_EOdd.yev_cx(y, y_hat, A_i, vals_in_Ai, pos)
    qa_6 = UD_grp2_EOdd.yev_cx(y, y_hat, A1_bin, val_A1, pos)
    qa_7 = UD_grp2_EOdd.mu_sp(y, y_hat, A1_bin, priv_val, pos)
    qa_8 = UD_grp2_EOdd.mu_cx(y, y_hat, A1_bin, priv_val, pos)
    # assert check_equal(qa_1[0][0], [qa_2[0][0], qa_4[0][3][0]])
    assert check_equal(qa_1[0][0], [qa_2[0], qa_4[0][3][0]])
    # assert check_equal(qa_6[0][2][0], [qa_7[0], qa_8[0][0]])
    # TODO
    assert check_equal(qa_2[0], [qa_7[0], qa_8[0][1], qa_6[0][1]])

    qb_1 = UD_grp2_PEq.bival(y, y_hat, priv_idx, pos)
    qb_2 = UD_grp2_PEq.mu_sp(y, y_hat, A_i, priv_val, pos)
    qb_3 = UD_grp2_PEq.mu_cx(y, y_hat, A_i, priv_val, pos)

    qb_4 = UD_grp2_PEq.yev_sp(y, y_hat, A_i, vals_in_Ai, pos)
    qb_6 = UD_grp2_PEq.yev_sp(y, y_hat, A1_bin, val_A1, pos)
    qb_7 = UD_grp2_PEq.mu_sp(y, y_hat, A1_bin, priv_val, pos)
    qb_8 = UD_grp2_PEq.mu_cx(y, y_hat, A1_bin, priv_val, pos)
    assert check_equal(qb_1[0][0], qb_2[0])  # [qb_2[0], qb_4[0][3][0]])
    assert check_equal(qb_2[0], [qb_7[0], qb_8[0]])

    # pdb.set_trace()
    return


def test_metric_grp3():
    m1 = UD_grp3_PQP.bival(y, y_hat, priv_idx, pos)
    m2 = UD_grp3_PQP.mu_sp(y, y_hat, A_i, priv_val, pos)
    m3 = UD_grp3_PQP.mu_cx(y, y_hat, A_i, priv_val, pos)

    m4 = UD_grp3_PQP.yev_sp(y, y_hat, A_i, vals_in_Ai, pos)
    # assert check_equal(m1[0], [m2[0], m3[0][0], m4[0][3][0]])
    m5 = UD_grp3_PQP.yev_cx(y, y_hat, A_i, vals_in_Ai, pos)
    m6 = UD_grp3_PQP.yev_cx(y, y_hat, A1_bin, val_A1, pos)
    m7 = UD_grp3_PQP.mu_sp(y, y_hat, A1_bin, priv_val, pos)
    m8 = UD_grp3_PQP.mu_cx(y, y_hat, A1_bin, priv_val, pos)
    assert check_equal(m1[0][0], [m2[0], m4[0][3][0]])
    assert check_equal(m6[0][2][0], [m7[0], m8[0][1]])

    # pdb.set_trace()
    return


def test_metric_indv():
    m1 = UD_gammaSubgroup.bival(y, y_hat, priv_idx, pos)
    m2 = UD_gammaSubgroup.mu_sp(y, y_hat, A_i, priv_val, pos)
    m3 = UD_gammaSubgroup.mu_cx(y, y_hat, A_i, priv_val, pos)
    m4 = UD_gammaSubgroup.yev_sp(y, y_hat, A_i, vals_in_Ai, pos)
    m7 = UD_gammaSubgroup.mu_sp(y, y_hat, A1_bin, priv_val, pos)
    m8 = UD_gammaSubgroup.mu_cx(y, y_hat, A1_bin, priv_val, pos)
    assert check_equal(m1[0][0], [m2[0], m7[0], m8[0][1]])

    m1 = UD_BoundedGrpLos.bival(y, y_hat, priv_idx, pos)
    m2 = UD_BoundedGrpLos.mu_sp(y, y_hat, A_i, priv_val, pos)
    m3 = UD_BoundedGrpLos.mu_cx(y, y_hat, A_i, priv_val, pos)
    m4 = UD_BoundedGrpLos.yev_sp(y, y_hat, A_i, vals_in_Ai, pos)
    m7 = UD_BoundedGrpLos.mu_sp(y, y_hat, A1_bin, priv_val, pos)
    m8 = UD_BoundedGrpLos.mu_cx(y, y_hat, A1_bin, priv_val, pos)
    assert check_equal(m1[0][0], [m2[0], m3[0], m7[0], m8[0]])
    assert check_equal(m2[0], m3[0])

    # pdb.set_trace()
    return
