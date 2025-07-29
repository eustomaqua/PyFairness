# coding: utf-8
# fair_intersectional.py


import argparse
import pdb

import csv
import json
import os
import sys
import time

import pandas as pd
import numpy as np
import numba


from pyfair.utils_empirical import DataSetup
from pyfair.facil.utils_saver import get_elogger, elegant_print
from pyfair.facil.utils_timer import (
    elegant_dated, fantasy_durat, elegant_durat_core)
from pyfair.preprocessing_dr import (
    transform_X_and_y,)  # , transform_unpriv_tag)
from pyfair.preprocessing_hfm import (
    renewed_transform_X_A_and_y, renewed_transform_disturb)

from pyfair.facil.data_split import (
    manual_cross_valid, sklearn_k_fold_cv, sklearn_stratify,
    scale_normalize_helper,)


class FairNonbinaryEmpirical(DataSetup):
    def __init__(self, trial_type, data_type,
                 ratio=.97,
                 prep=False, screen=True, logged=False):
        super().__init__(data_type)
        self._ratio = ratio
        self.preparing_iterator()
        self._screen, self._logged = screen, logged

    def preparing_iterator(self, trial_type, nb_iter, prep=False):
        self._log_document = "_".join([
            trial_type, prep,
            "iter{}".format(nb_iter) if nb_iter > 0 else 'sing',
            self._log_document, 'pms'])

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
        # END

        tim_elapsed = time.time() - since
        elegant_print([
            "",
            "Duration /TimeCost: {}".format(fantasy_durat(
                tim_elapsed, False)),
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
        csv_row_2a = ['data_name', 'binary', 'abbr_cls']

        # START
        # END

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
        g1m_indices = handling_info['idx_non_sa']  # idx_p(ri)v
        idx_jt = handling_info['idx_priv_jt']
        X4learn = process_dat['sen-att-2bool']
        X4learn_qtb = disturb_dat['sen-att-2bool']

        elegant_print("CrossValidation nb_cv={}, {}".format(
            self._nb_iter, self._trial_type[:3]), logger)
        if "mCV" in self._trial_type:
            split_idx = manual_cross_valid(self._nb_iter, y)
        elif "KFS" in self._trial_type:
            split_idx = sklearn_stratify(self._nb_iter, y, X)
        elif "KF" in self._trial_type:
            split_idx = sklearn_k_fold_cv(self._nb_iter, y)
        else:
            raise ValueError("No proper CV (cross-validation).")

        res_ans = []
        for k, (i_trn, i_tst) in enumerate(split_idx):
            (Xb_trn, A_trn, y_trn, Aq_trn, g1m_trn,
                jt_trn) = renewed_transform_disturb(
                X_breve, A, y, A_qtb, i_trn, g1m_indices, idx_jt)
            (Xb_tst, A_tst, y_tst, Aq_tst, g1m_tst,
                jt_tst) = renewed_transform_disturb(
                X_breve, A, y, A_qtb, i_tst, g1m_indices, idx_jt)
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
            # priv_col = list(range(len(XaA_trn[0])))
            # for i in self.saIndex:
            #     pass

            # i-th K-FOLD
            elegant_print("Iteration {}-th".format(k + 1), logger)

    def trial_one_iteration(
            self, logger, k,
            XaA_trn, y_trn, XaA_qtb_trn, Xb_trn, A_trn,
            XaA_tst, y_tst, XaA_qtb_tst, Xb_tst, A_tst):
        since = time.time()
        res_iter = []

        tim_elapsed = time.time() - since
        elegant_print("CV iteration {}-th, consumed {}".format(
            k, elegant_durat_core(tim_elapsed, True)), logger)
        return res_iter


def default_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp", "--experiment-id", type=str)
    parser.add_argument(  # above: type of trial
        "-dat", "--dataset-name", type=str, default="german",
        choices=["ricci", "german", "adult", "ppr", "ppvr"])
    parser.add_argument("-pre", "--data-preprocessing",
                        type=str, default='none', choices=[
                            'none', 'standard', 'min_max',
                            'min_abs', 'normalise'])
    parser.add_argument('-nk', "--nb-cv", type=int, default=5,
                        help="K-fold cross validation")

    parser.add_argument(
        '--ratio', type=float, default=.97,
        help="Disturbing ratio in perturbation")
    parser.add_argument(
        '-m1', '--m1-chosen', type=int, default=20)
    parser.add_argument(
        '-m2', '--m2-chosen', type=int, default=8)

    parser.add_argument(
        "--abbr-cls", type=str, default="DT", choices=[
            "DT", "NB", "SVM", "linSVM",  # "LR",
            "kNNu", "kNNd", "LR1", "LR2", "LM1", "LM2",
            "MLP", "lmSGD", "NN", "LM",
            'bagging', 'AdaBoost', 'LightGBM', 'FairGBM',
            'AdaFair', ], help="Individual classifiers")
    parser.add_argument(
        '--nb-cls', type=int, default=1, help='#classifiers')
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
