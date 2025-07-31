# coding: utf-8
# fair_intersectional.py, fair_int_main.py


import argparse
import pdb

import csv
import json
import os
import sys
import time
import warnings

import sklearn
import pandas as pd
import numpy as np
# import numba
from pathos import multiprocessing as pp


from pyfair.utils_empirical import DataSetup
from pyfair.facil.utils_saver import get_elogger, elegant_print
from pyfair.facil.utils_timer import (
    elegant_dated, fantasy_durat, elegant_durat_core)
# from pyfair.facil.utils_const import _get_tmp_name_ens
from pyfair.preprocessing_dr import (
    transform_X_and_y,)  # , transform_unpriv_tag)
from pyfair.preprocessing_hfm import (
    renewed_transform_X_A_and_y, renewed_transform_disturb)
from pyfair.facil.data_split import (
    manual_cross_valid, sklearn_k_fold_cv, sklearn_stratify,
    scale_normalize_helper,)


from fair_int_expt import (
    CompA_sing_learner, CompA_fair_ens, CompA_norm_cls)

warnings.filterwarnings(
    "ignore", module="sklearn",
    category=sklearn.exceptions.ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# =============================
# Experiments


class FairNonbinaryEmpirical(DataSetup):
    def __init__(self, trial_type, data_type, prep=False,
                 nb_cv=5, ratio=.97, m1=20, m2=8, fix_m2=False,
                 n_e=3, mp_cores=2, abbr_cls='DT', nb_cls=1,
                 constraint_type='FPR,FNR',
                 screen=True, logged=False):
        super().__init__(data_type)
        self._fix_m2 = fix_m2  # fixed_m2,appt_m2
        self._mp_cores = mp_cores
        self.preparing_iterator(
            trial_type, prep, nb_cv, ratio, m1, m2, n_e,
            abbr_cls, nb_cls, constraint_type)
        self._screen, self._logged = screen, logged

    def preparing_iterator(self, trial_type, prep, nb_cv,
                           ratio, m1, m2, n_e, abbr_cls, nb_cls,
                           constraint_type):
        self._trial_type = trial_type
        self._prep = prep    # pre-processing data
        self._nb_cv = nb_cv  # default:0 previous `nb_iter`
        self._ratio = ratio
        self._m1, self._m2, self._n_e = m1, m2, n_e

        self._log_document = "_".join([
            trial_type, prep.replace('_', ''),
            "iter{}".format(nb_cv) if nb_cv > 0 else 'sing',
            self._log_document, 'pms'])

        self._abbr_cls = abbr_cls
        self._nb_cls = nb_cls
        self._constraint_type = constraint_type

        if trial_type.endswith('exp1a'):
            self._iterator = CompA_sing_learner(
                abbr_cls, nb_cls,
                constraint_type, self.saIndex, self.saValue)
            self._log_document += '_{}_cls{}'.format(
                abbr_cls, nb_cls)
        elif trial_type.endswith('exp1b'):
            self._iterator = CompA_fair_ens(
                'DT', nb_cls, self.saIndex, self.saValue)
            self._log_document += f'_fair_ens_cls{nb_cls}'
        elif trial_type.endswith('exp1c'):
            self._iterator = CompA_norm_cls(
                'DT', 1, self.saIndex, self.saValue)
            self._log_document += '_regular'

        # self._log_document += ('_gen' * gen + '_rep' * rep)
        return

    def trial_one_process(self, mode="w"):
        since = time.time()
        csv_t = open(self._log_document + ".csv", mode)
        csv_w = csv.writer(csv_t)
        if mode == "a":
            csv_w.writerows([[''], [''], [''], ['']])

        if not (self._screen or self._logged):
            # if (not screen) and (not logged):
            saveout = sys.stdout
            fsock = open(self._log_document + ".log", "w")
            sys.stdout = fsock
        logger = None
        if self._logged:
            if os.path.exists(self._log_document + ".txt"):
                os.remove(self._log_document + ".txt")
            logger, _, _ = get_elogger(
                "intersectional", self._log_document + ".txt")

        elegant_print([
            "[BEGAN AT {}]".format(elegant_dated(since)),
            "EXPERIMENT",
            "\t   trail = {}".format(self._trial_type),
            "\t dataset = {}".format(self._data_type),
            # "\t binary? = {}".format(
            #     not self._trial_type.startswith('mu')),
            "\tdata prep= {}".format(self._prep),
            "PARAMETERS",
            "HYPER-PARAMS", ""], logger)

        # START
        self.coding_per_procedure(csv_w, logger)
        # END

        tim_elapsed = time.time() - since
        elegant_print([
            "",
            "Duration /TimeCost: {}".format(fantasy_durat(
                tim_elapsed, False, abbreviation=True)),
            "[ENDED AT {:s}]".format(elegant_dated(
                time.time())), ""], logger)
        del logger
        if not (self._screen or self._logged):
            fsock.close()
            sys.stdout = saveout
        csv_t.close()
        del csv_t, csv_w, since, tim_elapsed
        return

    # EACH SUBROUTE
    def coding_per_procedure(self, csv_w, logger=None):
        csv_row_2a = [         # 'binary', 'abbr_cls', 'learner'
            'data_name', '#lbl', '#cv', 'ratio', 'm1', 'm2',
            'n_e', '#sen-att',     # 'name_ens,abbr_cls,nb_cls,',
            'name_ens', 'k#', 'tilde(A)']  # 'tilde tim_elapsed']
        # csv_row_2b = ['', self._prep, ] + [''] * 4 + [
        #     'mp#core ={}'.format(self._mp_cores), '', '', '',
        #     'perturbation']
        (csv_row_1, csv_r2c, csv_r3c,
         csv_r4c) = self._iterator.prepare_trial()  # ,csv_r5c
        # csv_w.writerows([csv_row_1, csv_row_2a + csv_r2c,
        #                  csv_row_2b + csv_r3c,
        #                  [''] * 11 + csv_r4c, [''] * 10 + [
        #                      'tim_elapsed'] + csv_r5c, ])

        # START
        res_ans, res_aux = self.coding_per_dataset(logger=logger)
        json_saver = json.dumps({
            "res_aux": res_aux, "res_data": res_ans})
        json_w = open(self._log_document + ".json", "w")
        json_w.write(json_saver)
        json_w.close()
        del json_saver, json_w
        # END

        csv_w.writerows([csv_row_1, csv_row_2a + csv_r2c,
                         res_aux[1] + csv_r3c,
                         # [''] * 8 + [f'nb_cls={self._nb_cls}', '',
                         #             'tim_elapsed'] + csv_r4c])
                         [''] * 10 + ['tim_elapsed'] + csv_r4c])

        csv_w.writerow(res_aux[0])
        # csv_w.writerow(res_aux[1])
        sen_att, priv_val, marginal_grp = res_aux[2:5]
        nk = 1 if self._nb_cv <= 0 else self._nb_cv
        if self._trial_type[-5:] in ('exp1a'):
            for k in range(nk):
                csv_w.writerow([''] * 9 + [k] + [''] + res_ans[k][0])
        elif self._trial_type[-5:] in ('exp1b', 'exp1c'):
            nb_row = np.shape(res_ans)[1]  # (#cv,#alg,#tag_col)
            for r in range(nb_row):
                # for k in range(self._nb_cv):
                #     csv_w.writerow([''] * 7 + res_ans[k][r])
                k = 0
                csv_w.writerow([''] * 7 + res_ans[k][r][:2] + [
                    k, ''] + res_ans[k][r][4:])
                for k in range(1, self._nb_cv):
                    csv_w.writerow([''] * 9 + [k] + res_ans[k][r][3:])

        del nk, sen_att, priv_val, marginal_grp, res_ans, res_aux
        return

    def coding_per_dataset(self, handle='mu', logger=None):
        (handling_info, process_dat, disturb_dat
         ) = self.preparing_curr_dat(
            'dr_intersectional', self._ratio, logger)
        if handle == 'bi':
            processed_Xy = process_dat['numerical-binsensitive']
            disturbed_Xy = disturb_dat['numerical-binsensitive']
        elif handle == 'mu':
            processed_Xy = process_dat['numerical-multival']
            disturbed_Xy = disturb_dat['numerical-multival']
        XA, y = transform_X_and_y(self._dataset, processed_Xy)
        X_breve, A, _, _ = renewed_transform_X_A_and_y(
            self._dataset, processed_Xy, with_joint=False)
        XA_qtb, _ = transform_X_and_y(self._dataset, disturbed_Xy)
        _, A_qtb, _, _ = renewed_transform_X_A_and_y(
            self._dataset, processed_Xy, with_joint=False)
        marginal_grp = handling_info['marginalised_grps']
        # g1m_indices = handling_info['idx_non_sa'] #idx_p(ri)v
        # idx_jt = handling_info['idx_priv_jt']
        g1m_indices = process_dat['g1m_indices']
        X4learn = process_dat['sen-att-2bool']
        X4learn_qtb = disturb_dat['sen-att-2bool']
        if self._dataset.dataset_name == 'german':
            y[y == 2] = 0  # if 2 in y
        pool = None
        if self._mp_cores > 0:  # ==0
            pool = pp.ProcessingPool(nodes=self._mp_cores)

        if not self._fix_m2:
            self._m2 = np.ceil(2 * np.log10(len(y)))
            self._m2 = int(self._m2)
            elegant_print("\tNotice: m2 unfixed", logger)
        elegant_print("Due to #inst = {}".format(len(y)), logger)
        elegant_print("self._m2 = {}".format(self._m2), logger)
        elegant_print("self._m1 = {}".format(self._m1), logger)
        # elegant_print([
        #     "Due to #inst = {}".format(len(y)),
        #     "self._m2 = {}".format(self._m2),
        #     "self._m1 = {}".format(self._m1)], logger)
        tmp = handling_info[
            'processed_dat'][self._dataset.label_name]
        elegant_print([
            "\t BINARY? Y = {}".format(set(y.values)),
            "\t orig.label= {}".format(set(tmp.values)),
            "\t     .shape= {}".format(tmp.shape)], logger)
        del tmp
        sen_att = self._dataset.sensitive_attrs
        tmp_cls = ''
        if self._trial_type.endswith('exp1a'):
            tmp_cls = '{} #{}'.format(self._abbr_cls, self._nb_cls)
        res_aux = [[
            self._dataset.dataset_name, len(set(y.values)),
            self._nb_cv, self._ratio, self._m1, self._m2,
            self._n_e,  # '#sa ={}'.format(len(sen_att)),
            f'#sa={len(sen_att)}: {sen_att}',
            tmp_cls, '', handling_info[
                'perturbation_tim_elapsed']], [
            '', self._prep, '', '', '', '',
             'mp#core ={}'.format(
                 self._mp_cores),  # 'sen-att', '', ''],
            '', f'abbr_cls,nb_cls={self._nb_cls}',
             '', 'perturbation'],
            sen_att, self._dataset.privileged_vals,
            handling_info['marginalised_grps'], ]

        elegant_print("CrossValidation nb_cv={}, {}".format(
            self._nb_cv, self._trial_type[:3]), logger)
        if "mCV" in self._trial_type:
            split_idx = manual_cross_valid(self._nb_cv, y)
        elif "KFS" in self._trial_type:
            split_idx = sklearn_stratify(self._nb_cv, y, X)
        elif "KF" in self._trial_type:
            split_idx = sklearn_k_fold_cv(self._nb_cv, y)
        else:
            raise ValueError("No proper CV (cross-validation).")

        res_ans = []
        for k, (i_trn, i_tst) in enumerate(split_idx):
            (Xb_trn, A_trn, y_trn, _, g1m_trn,
                _) = renewed_transform_disturb(
                X_breve, A, y, A_qtb, i_trn, g1m_indices, [])
            (Xb_tst, A_tst, y_tst, _, g1m_tst,
                _) = renewed_transform_disturb(
                X_breve, A, y, A_qtb, i_tst, g1m_indices, [])
            # idx_jt --> jt_trn, jt_tst; Aq_trn,Aq_tst
            # XaA_trn = XA.iloc[i_trn]
            # XaA_tst = XA.iloc[i_tst]
            # XaA_qtb_trn = XA_qtb.iloc[i_trn]
            # XaA_qtb_tst = XA_qtb.iloc[i_tst]
            y_trn = y_trn.to_numpy()
            y_tst = y_tst.to_numpy()
            XaA_trn = X4learn.iloc[i_trn]  # X4learn_trn
            XaA_tst = X4learn.iloc[i_tst]  # X4learn_tst
            XaA_qtb_trn = X4learn_qtb.iloc[i_trn]  # X4learn_qtb_trn
            XaA_qtb_tst = X4learn_qtb.iloc[i_tst]  # X4learn_qtb_tst

            if self._prep != 'none':
                scaler = scale_normalize_helper(self._prep)
                scaler = scaler.fit(XaA_trn)
                XaA_trn = scaler.transform(XaA_trn)
                XaA_tst = scaler.transform(XaA_tst)
                XaA_qtb_trn = scaler.transform(XaA_qtb_trn)
                XaA_qtb_tst = scaler.transform(XaA_qtb_tst)
                scaler = scale_normalize_helper(self._prep)
                scaler = scaler.fit(Xb_trn)
                Xb_trn = scaler.transform(Xb_trn)
                Xb_tst = scaler.transform(Xb_tst)
            else:
                Xb_trn = Xb_trn.values
                Xb_tst = Xb_tst.values
                XaA_trn = XaA_trn.to_numpy()
                XaA_tst = XaA_tst.to_numpy()
                XaA_qtb_trn = XaA_qtb_trn.values
                XaA_qtb_tst = XaA_qtb_tst.values
            # priv_col = list(range(len(XaA_trn[0])))
            # for i in self.saIndex:
            #     pass
            # pdb.set_trace()

            # i-th K-FOLD
            elegant_print("Iteration {}-th".format(k + 1), logger)
            res_iter = self.trial_one_iteration(
                logger, pool, k,
                XaA_trn, y_trn, XaA_qtb_trn, Xb_trn, A_trn, g1m_trn,
                XaA_tst, y_tst, XaA_qtb_tst, Xb_tst, A_tst, g1m_tst)
            res_ans.append(res_iter)
            del XaA_trn, Xb_trn, A_trn, y_trn, XaA_qtb_trn
            del XaA_tst, Xb_tst, A_tst, y_tst, XaA_qtb_tst
            del g1m_trn, g1m_tst  # , jt_trn, jt_tst
        del split_idx, X4learn, X4learn_qtb, g1m_indices
        del XA, X_breve, A, y, XA_qtb, A_qtb, marginal_grp
        return res_ans, res_aux

    def trial_one_iteration(
            self, logger, pool, k,
            XaA_trn, y_trn, XaA_qtb_trn, Xb_trn, A_trn, g1m_trn,
            XaA_tst, y_tst, XaA_qtb_tst, Xb_tst, A_tst, g1m_tst):
        since = time.time()
        res_iter = []
        positive_label = self._dataset.get_positive_class_val(
            'numerical-binsensitive')
        pms = {'m1': self._m1, 'm2': self._m2, 'n_e': self._n_e,
               'positive_label': positive_label}

        if self._trial_type.endswith('exp1a'):
            res_iter = self._iterator.schedule_content(
                logger, pool,
                XaA_trn, y_trn, XaA_qtb_trn, Xb_trn, A_trn, g1m_trn,
                XaA_tst, y_tst, XaA_qtb_tst, Xb_tst, A_tst, g1m_tst,
                **pms)
            return [res_iter]
        elif self._trial_type[-5:] in ('exp1b', 'exp1c'):
            res_iter = self._iterator.schedule_content(
                logger, pool,
                XaA_trn, y_trn, XaA_qtb_trn, Xb_trn, A_trn, g1m_trn,
                XaA_tst, y_tst, XaA_qtb_tst, Xb_tst, A_tst, g1m_tst,
                **pms)
            # return res_iter

        del positive_label, pms
        tim_elapsed = time.time() - since
        elegant_print("CV iteration {}-th, consumed {}".format(
            k, elegant_durat_core(tim_elapsed, True)), logger)
        return res_iter


# =============================
# Hyper-parameters


def default_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp", "--experiment-id", type=str,
                        required=True)
    parser.add_argument(  # above: type of trial
        "-dat", "--dataset-name", type=str, default="german",
        choices=["ricci", "german", "adult", "ppr", "ppvr"])
    parser.add_argument("-pre", "--data-preprocessing",
                        type=str, default='none', choices=[
                            'none', 'standard', 'min_max',
                            'min_abs', 'normalise', 'normalize'])
    parser.add_argument('-nk', "--nb-cv", type=int, default=5,
                        help="K-fold cross validation")

    parser.add_argument(
        '--ratio', type=float, default=.97,
        help="Disturbing ratio in perturbation")
    parser.add_argument(
        '-m1', '--m1-chosen', type=int, default=20)
    parser.add_argument(
        '-m2', '--m2-chosen', type=int, default=8)
    parser.add_argument('-fix', '--m2-fixed', action='store_true')
    parser.add_argument(
        '-ne', '--n_e_chosen', type=int, default=2)  # or 3
    parser.add_argument('-mpc', '--mp-cores', type=int, default=0)

    parser.add_argument(
        '-clf', "--abbr-cls", type=str, default="DT", choices=[
            "DT", "NB", "SVM", "linSVM",  # "LR",
            "kNNu", "kNNd", "LR1", "LR2", "LM1", "LM2",
            "MLP", "lmSGD", "NN", "LM",
            'bagging', 'AdaBoost', 'LightGBM', 'FairGBM',
            'AdaFair', ], help="Individual classifiers")
    parser.add_argument('-nf', '--nb-cls', type=int, default=7,
                        help='#classifiers')
    parser.add_argument('-constr', '--constraint-type',
                        type=str, default='FPR,FNR')

    parser.add_argument("--screen", action="store_true")
    parser.add_argument("--logged", action="store_false")
    return parser  # help="Where to output"


if __name__ == "__main__":
    # screen = logged = None
    parser = default_parameters()
    args = parser.parse_args()
    kwargs = {}
    kwargs['screen'] = args.screen
    kwargs['logged'] = args.logged

    trial_type = args.experiment_id
    data_type = args.dataset_name
    kwargs['prep'] = args.data_preprocessing
    kwargs['nb_cv'] = args.nb_cv

    kwargs['fix_m2'] = args.m2_fixed
    kwargs['m1'] = args.m1_chosen
    kwargs['m2'] = args.m2_chosen
    kwargs['n_e'] = args.n_e_chosen
    kwargs['mp_cores'] = args.mp_cores
    kwargs['ratio'] = args.ratio

    if trial_type.endswith('exp1a'):
        kwargs['abbr_cls'] = args.abbr_cls
        kwargs['nb_cls'] = args.nb_cls
        kwargs['constraint_type'] = args.constraint_type
    elif trial_type.endswith('exp1b'):
        kwargs['nb_cls'] = args.nb_cls
    elif trial_type.endswith('exp1c'):
        kwargs['nb_cls'] = 1

    case = FairNonbinaryEmpirical(
        trial_type, data_type, **kwargs)
    mode = "a" if data_type == 'adult' else "w"
    case.trial_one_process(mode=mode)


"""
python fair_int_exec.py -exp KF_exp1a -nk 2 -clf AdaBoost -dat ppvr
python fair_int_exec.py -exp KF_exp1a -nk 2 -clf AdaBoost
python fair_int_exec.py -exp KF_exp1b -nk 2 -dat ricci -pre min_max
python fair_int_exec.py -exp KF_exp1c -nk 2 -dat ricci
"""
