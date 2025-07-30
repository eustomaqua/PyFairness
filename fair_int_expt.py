# coding: utf-8
# Experiments


import pdb
import time
import numpy as np

from sklearn.ensemble import (
    BaggingClassifier, AdaBoostClassifier)
import lightgbm
# import fairgbm
from pyfair.pkgs_AdaFair_py36 import AdaFair
from pyfair.facil.utils_remark import (
    NAME_INDIVIDUALS, AVAILABLE_ABBR_CLS)
from pyfair.facil.utils_const import DTY_FLT, unique_column


from pyfair.dr_hfm.discriminative_risk import hat_L_fair, hat_L_loss
from pyfair.dr_hfm.dist_drt import (
    DirectDist_bin, DirectDist_nonbin, DirectDist_multiver)
from pyfair.dr_hfm.dist_est_bin import (
    ApproxDist_bin, ApproxDist_bin_alter)
from pyfair.dr_hfm.dist_est_nonbin import (
    ApproxDist_nonbin_mpver, ExtendDist_multiver_mp)
from pyfair.dr_hfm.hfm_df import (
    bias_degree, bias_degree_bin, bias_degree_nonbin)

from pyfair.marble.metric_fair import (
    unpriv_group_one, unpriv_group_two, unpriv_group_thr,
    calc_fair_group, marginalised_np_mat, alterGrps_sing,
    extGrp1_DP_sing, extGrp2_EO_sing, extGrp3_PQP_sing,)
from pyfair.facil.metric_cont import contingency_tab_bi
from pyfair.marble.metric_perf import (
    calc_accuracy, calc_precision, calc_recall, calc_f1_score,
    calc_specificity, imba_geometric_mean,
    imba_discriminant_power, imba_balanced_accuracy)
from pyfair.granite.fair_meas_indiv import GEI_Theil


# =============================
# Exp1: bin-val vs. multi-val


class ComparisonA_setup:
    def get_member_clf(self, X_A_trn, y_trn, nsa_trn=tuple(),
                       constraint='FPR,FNR',
                       sa_idx=None, sa_val=None):
        since = time.time()

        if self._name_cls in AVAILABLE_ABBR_CLS:
            clf = NAME_INDIVIDUALS[self._name_cls]
            clf.fit(X_A_trn, y_trn)

        elif self._name_cls in ['bagging', 'Bagging']:
            clf = BaggingClassifier(n_estimators=self._nb_cls)
            clf.fit(X_A_trn, y_trn)
        elif self._name_cls in ['adaboost', 'AdaBoost']:
            clf = AdaBoostClassifier(n_estimators=self._nb_cls)
            clf.fit(X_A_trn, y_trn)
        elif self._name_cls in ['lightGBM', 'LightGBM']:
            clf = lightgbm.LGBMClassifier(
                n_estimators=self._nb_cls)
            clf.fit(X_A_trn, y_trn)

        elif self._name_cls in ['fairGBM', 'FairGBM']:
            clf = fairgbm.FairGBMClassifier(
                n_estimators=self._nb_cls,
                constraint_type=constraint)
            clf.fit(X_A_trn, y_trn, constraint_group=~nsa_trn)
        elif abbr_cls in ['AdaFair', 'adafair']:
            clf = AdaFair(n_estimators=self._nb_cls,
                          saIndex=sa_idx, saValue=sa_val)
            clf.fit(X_A_trn, y_trn)

        tim_elapsed = time.time() - since
        # self._learner = clf  # self._member
        return clf, tim_elapsed

    def count_scores(
            self,
            y_trn, y_insp, yq_insp, Xb_trn, A_trn, g1m_trn,
            y_tst, y_pred, yq_pred, Xb_tst, A_tst, g1m_tst,
            m1, m2, n_e, pool, pos_label):
        ans_trn = self.count_half_member(
            y_trn, y_insp, yq_insp, Xb_trn, A_trn, g1m_trn,
            pos_label, m1, m2, n_e, pool)
        ans_tst = self.count_half_member(
            y_tst, y_pred, yq_pred, Xb_tst, A_tst, g1m_tst,
            pos_label, m1, m2, n_e, pool)
        return ans_trn + ans_tst

    def count_half_member(self, y, y_hat, y_hat_qtb,
                          X_breve, A, g1m_indices, pos_label,
                          m1, m2, n_e, pool):
        ans, n_a = [], len(g1m_indices)
        tmp = self.subproc_part1_acc(y, y_hat, pos_label)
        tmp_adv = self.subproc_part1_acc(y, y_hat_qtb, pos_label)
        ans.extend(tmp + tmp_adv)  # api/pi
        ans.extend([abs(ti - pi) for ti, pi in zip(tmp, tmp_adv)])

        for i in range(n_a):
            tmp = self.subproc_part2_grp(
                y, y_hat, pos_label, g1m_indices[i][0])
            ans.extend(tmp)
            tmp = self.subproc_part2_ext(
                y, y_hat, pos_label, g1m_indices[i])
            ans.extend(tmp)
        if n_a == 1:
            ans.extend([''] * 10)  # 6+3+1=10
            ans.extend([''] * 19)  # 6+6+6+1=19

        tmp = hat_L_loss(y_hat, y)          # tmp_l(o)s
        ans.extend(tmp)
        tmp = hat_L_fair(y_hat, y_hat_qtb)  # tmp_dr
        ans.extend(tmp)
        # Up to now, ans.shape= (86,) =(8*3+29*2+4,)=28+58

        since = time.time()
        tmp = [GEI_Theil.get_GEI(y, y_hat, float(
            alph))[0] for alph in np.arange(0., 1.01, 0.1)]
        since = time.time() - since
        ans.extend(tmp)
        ans.extend(GEI_Theil.get_Theil(y, y_hat))
        ans.append(since)  # ans.shape= (100,) =(86+11+3,)

        X_nA_y = np.concatenate([y.reshape(-1, 1).astype(
            # 'float'), X_breve.values.astype('float')], axis=1)
            DTY_FLT), X_breve.values.astype(DTY_FLT)], axis=1)
        X_nA_fx = np.concatenate([y_hat.reshape(  # X_nA_y_hat
            -1, 1).astype(DTY_FLT),  # 'float'),
            X_breve.values.astype(DTY_FLT)], axis=1)
        tmp = self.subproc_part4_hfm(X_nA_y, X_nA_fx, g1m_indices)
        ans.extend(tmp)
        tmp = self.subproc_part4_hfm_hat(
            X_nA_y, X_nA_fx, g1m_indices, A.values, m1, m2, n_e, pool)
        ans.extend(tmp)

        # pdb.set_trace()
        return ans  # siz= (158,) =100+29*2

    def subproc_part4_hfm_hat(self, X_nA_y, X_nA_fx,
                              g1m_indices, A, m1, m2, n_e, pool):
        n_a, ans = len(g1m_indices), []
        for i in range(n_a):
            # Ds, t_Ds = ApproxDist_bin(
            #     X_nA_y, A[:, i], g1m_indices[i][0], m1, m2)
            (Ds, _), t_Ds = ApproxDist_bin_alter(
                X_nA_y, g1m_indices[i][0], m1, m2)
            (Df, _), t_Df = ApproxDist_bin_alter(
                X_nA_fx, g1m_indices[i][0], m1, m2)
            df_prev = bias_degree_bin(Ds, Df)
            ans.extend([Ds, Df, df_prev[0], df_prev[1] + t_Ds + t_Df])
        if n_a == 1:
            ans.extend([''] * 4)

        (Ds, Ds_avg, Ds_intermediate), t_Ds = ExtendDist_multiver_mp(
            X_nA_y, A, m1, m2, n_e, pool)   # g1m_indices)
        (Df, Df_avg, Df_intermediate), t_Df = ExtendDist_multiver_mp(
            X_nA_fx, A, m1, m2, n_e, pool)  # g1m_indices)
        ddf = bias_degree_nonbin(Ds, Df)
        ddf_avg = bias_degree_nonbin(Ds_avg, Df_avg)
        ans.extend([Ds, Df, ddf[0], Ds_avg, Df_avg, ddf_avg[0],
                    ddf[1] + ddf_avg[1] + t_Ds + t_Df])

        for i in range(n_a):
            Ds = Ds_intermediate[0][i]
            Ds_avg = Ds_intermediate[1][i]
            t_Ds = Ds_intermediate[2][i]
            Df = Df_intermediate[0][i]
            Df_avg = Df_intermediate[1][i]
            t_Df = Df_intermediate[2][i]
            ddf = bias_degree_nonbin(Ds, Df)
            ddf_avg = bias_degree_nonbin(Ds_avg, Df_avg)
            ans.extend([Ds, Df, ddf[0], Ds_avg, Df_avg, ddf_avg[0],
                        ddf[1] + ddf_avg[1] + t_Ds + t_Df])
        if n_a == 1:
            ans.extend([''] * 7)
        return ans  # siz= (29,) =4*2+7+7*2 =8+7+14

    # def subproc_part4_hfm(self, X_breve, A, y, y_hat,
    #                       g1m_indices, m1, m2, n_e, pool):
    def subproc_part4_hfm(self, X_nA_y, X_nA_fx, g1m_indices):
        pass
        n_a, ans = len(g1m_indices), []
        for i in range(n_a):
            (Ds, _), t_Ds = DirectDist_bin(X_nA_y, g1m_indices[i][0])
            (Df, _), t_Df = DirectDist_bin(X_nA_fx, g1m_indices[i][0])
            df_prev = bias_degree_bin(Ds, Df)
            ans.extend([Ds, Df, df_prev[0], df_prev[1] + t_Ds + t_Df])
        if n_a == 1:
            ans.extend([''] * 4)

        (Ds, Ds_avg, Ds_intermediate), t_Ds = DirectDist_multiver(
            X_nA_y, g1m_indices)
        (Df, Df_avg, Df_intermediate), t_Df = DirectDist_multiver(
            X_nA_fx, g1m_indices)
        ddf = bias_degree_nonbin(Ds, Df)
        ddf_avg = bias_degree_nonbin(Ds_avg, Df_avg)
        ans.extend([Ds, Df, ddf[0], Ds_avg, Df_avg, ddf_avg[0],
                    ddf[1] + ddf_avg[1] + t_Ds + t_Df])

        for i in range(n_a):
            Ds = Ds_intermediate[0][i]
            Ds_avg = Ds_intermediate[1][i]
            t_Ds = Ds_intermediate[2][i]
            Df = Df_intermediate[0][i]
            Df_avg = Df_intermediate[1][i]
            t_Df = Df_intermediate[2][i]
            ddf = bias_degree_nonbin(Ds, Df)
            ddf_avg = bias_degree_nonbin(Ds_avg, Df_avg)
            ans.extend([Ds, Df, ddf[0], Ds_avg, Df_avg, ddf_avg[0],
                        ddf[1] + ddf_avg[1] + t_Ds + t_Df])
        if n_a == 1:
            ans.extend([''] * 7)  # 3 max +3 avg +1 tim =7
        return ans  # siz=(29,) =4*2+7+7*2 =8+7+14=29

    def subproc_part1_acc(self, y, y_hat, pos_label):
        # tmp = tmptp, fp, fn, tn
        intermediate = contingency_tab_bi(y, y_hat, pos_label)
        sen = calc_recall(*intermediate)
        spe = calc_specificity(*intermediate)
        ans = [calc_accuracy(*intermediate),
               calc_precision(*intermediate),
               sen, spe, calc_f1_score(*intermediate),
               imba_geometric_mean(sen, spe),
               imba_balanced_accuracy(sen, spe),
               imba_discriminant_power(sen, spe)]
        return ans  # siz=(8,)

    def subproc_part2_grp(self, y, y_hat, pos_label, non_sa):
        since = time.time()  # non_sa:priv_idx
        g1_Cm, g0_Cm = marginalised_np_mat(y, y_hat, pos_label,
                                           non_sa)
        grp_far_1 = unpriv_group_one(g1_Cm, g0_Cm)
        grp_far_2 = unpriv_group_two(g1_Cm, g0_Cm)
        grp_far_3 = unpriv_group_thr(g1_Cm, g0_Cm)
        ans = list(grp_far_1) + list(grp_far_2) + list(
            grp_far_3) + [calc_fair_group(*grp_far_1),
                          calc_fair_group(*grp_far_2),
                          calc_fair_group(*grp_far_3)]
        since = time.time() - since
        ans.append(since)
        return ans

    def subproc_part2_ext(self, y, y_hat, pos_label, g1m_indices):
        since = time.time()
        grp_far_1, ut_1 = extGrp1_DP_sing(y, y_hat, g1m_indices,
                                          pos_label)
        grp_far_2, ut_2 = extGrp2_EO_sing(y, y_hat, g1m_indices,
                                          pos_label)
        grp_far_3, ut_3 = extGrp3_PQP_sing(y, y_hat, g1m_indices,
                                           pos_label)
        ans = [grp_far_1[0], grp_far_2[0], grp_far_3[0],
               grp_far_1[1], grp_far_2[1], grp_far_3[1]]

        grp_far_1, s1 = alterGrps_sing(grp_far_1[-1], g1m_indices)
        grp_far_2, s2 = alterGrps_sing(grp_far_2[-1], g1m_indices)
        grp_far_3, s3 = alterGrps_sing(grp_far_3[-1], g1m_indices)
        ans.extend([grp_far_1[0], grp_far_2[0], grp_far_3[0],
                    grp_far_1[1], grp_far_2[1], grp_far_3[1],
                    ut_1, ut_2, ut_3, s1, s2, s3])
        since = time.time() - since
        ans.append(since)
        del ut_1, ut_2, ut_3, s1, s2, s3
        return ans


class CompA_sing_learner(ComparisonA_setup):
    def __init__(self, name_cls, nb_cls=1,
                 constraint_type='FPR,FNR',
                 saIndex=tuple(), saValue=tuple()):
        self._name_cls = name_cls
        self._nb_cls = nb_cls
        self._constraint = constraint_type
        self.saIndex, self.saValue = saIndex, saValue
        return

    def schedule_content(
            self, logger, pool,
            XaA_trn, y_trn, XaA_qtb_trn, Xb_trn, A_trn, g1m_trn,
            XaA_tst, y_tst, XaA_qtb_tst, Xb_tst, A_tst, g1m_tst,
            m1=20, m2=8, n_e=2, positive_label=None):
        clf, ut = self.get_member_clf(XaA_trn, y_trn)
        y_insp = clf.predict(XaA_trn)
        y_pred = clf.predict(XaA_tst)
        yq_insp = clf.predict(XaA_qtb_trn)
        yq_pred = clf.predict(XaA_qtb_tst)

        ans_trn = self.count_half_member(
            y_trn, y_insp, yq_insp, Xb_trn, A_trn, g1m_trn,
            positive_label, m1, m2, n_e, pool)
        ans_tst = self.count_half_member(
            y_tst, y_pred, yq_pred, Xb_tst, A_tst, g1m_tst,
            positive_label, m1, m2, n_e, pool)
        # pdb.set_trace()
        return [ut] + ans_trn + ans_tst

    def prepare_trial(self):
        csv_row_1 = unique_column(12 + 158 * 2)
        csv_row_2c = ['Ensem'] + ['Training set'] + [''] * 157 + [
            'Test set'] + [''] * 157

        csv_r3c = ['Performance'] + [''] * (24 - 1) + [
            'sa#1 grp-fairness'] + [''] * (29 - 1) + [
            'sa#2 grp-fairness'] + [''] * (29 - 1) + [
            'DR', '', '', '']
        csv_r4c = ['Normal'] + [''] * 7 + ['Aft perturbation'] + [
            ''] * 7 + ['Delta(performance)'] + [''] * 7 + ([
                'DP', '', 'EO', '', 'PQP', '', 'DP', 'EO', 'PQP',
                '**', 'SP /extGrp*'] + [''] * 17 + ['**']) * 2 + [
            'hat_loss', '', 'hat_bias', '', 'GEI .alph'] + [
            ''] * 10 + ['Theil', 'T(Theil)', 'T(GEIx11)']
        csv_row_3c = [''] + csv_row_3c * 2
        csv_row_4c = ['tim']  # ,csv_row_5c=['tim_elapsed']

        tmp_p1_acc = ['Accuracy', 'Precision', 'Recall /Sensitivity',
                      'Specificity', 'f1_score', 'g_mean',
                      'balanced_acc', 'DiscPower']
        tmp_p2_grp = ['g1', 'g0'] * 3 + ['abs'] * 3 + ['tim'] + [
            'SP /ext* (max)', '-', '-', 'SP /ext* (avg)', '-', '-',
            'meticulous .max', '-', '-', 'meticulous .avg', '-', '-',
            'sep. tim_elapsed', '-', '-', '-', '-', '-', 'tim']
        csv_row_4c
        csv_row_4c.extend(tmp_p1_acc * 3 + tmp_p2_grp * 2 + [
            'loss', 'ut', 'fair', 'ut'])


# =============================


# =============================


# =============================


# =============================
