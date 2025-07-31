# coding: utf-8


import argparse
import pdb

import time
import numpy as np
import pandas as pd

from pyfair.utils_empirical import GraphSetup
from pyfair.facil.utils_saver import elegant_print
from pyfair.facil.utils_timer import elegant_durat, elegant_dated

from fair_int_anal import (PlotA_fair_ens, PlotA_norm_cls)


# =============================
# Plotting


class FairNonbinaryPlotting:
    def __init__(self, trial_type, prep=False, nb_cv=5,
                 ratio=.97, m1=20, m2=8, fix_m2=False, n_e=2,
                 mp_cores=2, nb_cls=1, screen=True, logged=False):
        self._dataset = ['ricci', 'german', 'adult', 'ppr', 'ppvr']
        self._ratio = ratio
        self._mp_cores = mp_cores
        # self.preparing_iterator()
        self._screen, self._logged = screen, logged

        # def preparing_iterator(self, trial_type, prep, nb_cv,
        #                     ratio, m1, m2, fix_m2, n_e):
        self._trial_type = trial_type
        self._prep, self._nb_cv = prep, nb_cv
        self._m1, self._m2, self._n_e = m1, m2, n_e
        self._fix_m2 = fix_m2

        self._log_document = "_".join([
            trial_type, prep.replace('_', ''),
            "iter{}".format(nb_cv) if nb_cv > 0 else 'sing',
            'pms'])
        self._nb_cls = nb_cls

    def trial_one_process(self):
        since = time.time()  # logger = None
        elegant_print("[BEGAN {}]".format(elegant_dated(since)))
        # START

        if 'exp1' in self._trial_type:
            self.drawing_exp1()

        # END
        tim_elapsed = time.time() - since
        elegant_print(["Duration /TimeCost: {}".format(
            elegant_durat(tim_elapsed, False)),
            "[ENDED {}]".format(elegant_dated(time.time()))])
        return

    def drawing_exp1(self):
        xlsx_name = f'{self._trial_type}_iter{self._nb_cv}_pms'
        if self._trial_type.endswith('exp1b'):
            xlsx_name += f'_fair_ens_cls{self._nb_cls}'
            self._iterator = PlotA_fair_ens()
        elif self._trial_type.endswith('exp1c'):
            xlsx_name += '_regular'
            self._iterator = PlotA_norm_cls()

        sheet_name = 'exp{}_{}'.format(
            self._trial_type[-2:], self._prep.replace('_', ''))
        raw_df = self._iterator.load_raw_dataset(
            xlsx_name, sheet_name)
        pdb.set_trace()
        return


# =============================
# Hyper-parameters


def default_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp", "--experiment-id", type=str,
                        required=True)
    parser.add_argument("-pre", "--data-preprocessing",
                        type=str, default='min_max', choices=[
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

    # parser.add_argument(
    #     '-clf', "--abbr-cls", type=str, default="DT", choices=[
    #         "DT", "NB", "SVM", "linSVM",  # "LR",
    #         "kNNu", "kNNd", "LR1", "LR2", "LM1", "LM2",
    #         "MLP", "lmSGD", "NN", "LM",
    #         'bagging', 'AdaBoost', 'LightGBM', 'FairGBM',
    #         'AdaFair', ], help="Individual classifiers")
    parser.add_argument('-nf', '--nb-cls', type=int, default=7,
                        help='#classifiers')
    # parser.add_argument('-constr', '--constraint-type',
    #                     type=str, default='FPR,FNR')

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
    # data_type = args.dataset_name
    kwargs['prep'] = args.data_preprocessing
    kwargs['nb_cv'] = args.nb_cv

    kwargs['fix_m2'] = args.m2_fixed
    kwargs['m1'] = args.m1_chosen
    kwargs['m2'] = args.m2_chosen
    kwargs['n_e'] = args.n_e_chosen
    kwargs['mp_cores'] = args.mp_cores
    kwargs['ratio'] = args.ratio

    if trial_type.endswith('exp1b'):
        kwargs['nb_cls'] = args.nb_cls
    elif trial_type.endswith('exp1c'):
        kwargs['nb_cls'] = 1

    case = FairNonbinaryPlotting(trial_type, **kwargs)
    # mode = "a" if data_type == 'adult' else "w"
    case.trial_one_process()  # mode=mode)


"""
python fair_int_draw.py -exp KF_exp1b
"""
