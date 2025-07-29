# coding: utf-8


import numpy as np
import os
import pdb
import time

from pyfair.datasets import (
    DATASETS, DATASET_NAMES, RAW_EXPT_DIR,  # preprocess)
    process_above, process_below)
from pyfair.preprocessing_dr import adverse_perturb  # adversarial
from pyfair.preprocessing_hfm import (
    binarized_data_set, process_addtl, process_addtl_multivalue,
    process_intersectional)


DAT_EXPT_NMS = ['ricci', 'german', 'adult', 'ppr', 'ppvr']
DAT_EXPT_ORG = [
    'Ricci', 'Credit', 'Income', 'COMPAS PPR', 'COMPAS PPVR']


class DataSetup:
    def __init__(self, data_type):
        self._data_type = data_type
        self._log_document = data_type
        # ['ricci', 'german', 'adult', 'ppr', 'ppvr']
        if data_type in ['ppr', 'ppc']:
            self._data_type = DATASET_NAMES[-2]
        elif data_type in ['ppvr', 'ppvc']:
            self._data_type = DATASET_NAMES[-1]
        elif data_type not in ['ricci', 'german', 'adult']:
            raise ValueError("Wrong dataset name `{}`".format(
                data_type))
        idx = DATASET_NAMES.index(self._data_type)

        self._dataset = DATASETS[idx]
        self._data_frame = self._dataset.load_raw_dataset()
        if data_type == 'ricci':
            self.saIndex = [2]     # 'Race' -2
        elif data_type == 'german':
            self.saIndex = [3, 5]  # ['sex','age'] [,12]
        elif data_type == 'adult':
            self.saIndex = [2, 3]  # ['race','sex'] [7,8]
        elif data_type in ['ppr', 'ppc', 'ppvr', 'ppvc']:
            self.saIndex = [0, 2]  # ['sex','race'] [0,3]
        self.saValue = self._dataset.get_privileged_group(
            'numerical-binsensitive')
        # self.saValue = 0  # 1 means the privileged group
        self.saValue = [0 for sa in self.saValue if sa == 1]

        # ricci .columns: [.., 'Race'(2),..]
        # german.columns:[.. 'sex'(3),..'age'(5),.. 'sex-age'(9)]
        #                               adult-1,youth-0
        # adult .columns:['age','education-num','race','sex',
        #               .. 'race-sex'(8), ..]
        # ppr .columns: ['sex','age','race', ..'sex-race'(8)]
        # ppvr.columns: ['sex','age','race', ..'sex-race'(8)]

    # @property
    # def data_type(self):
    #     return self._data_type

    @property
    def log_document(self):
        return self._log_document

    # # ----------- mu -----------
    # def prepare_mu_datasets(self, ratio=.5, logger=None):
    #   pass
    # # ----------- tr -----------
    # # ----------- bi -----------
    # def prepare_bi_datasets(self, ratio=.5, logger=None):
    #   pass

    @property
    def dataset(self):
        return self._dataset

    @property
    def data_frame(self):
        return self._data_frame

    @property
    def trial_type(self):
        return self._trial_type

    def preparing_curr_dat(self, handle, ratio=.97, logger=None):
        # if handle in ['dr']:
        #     processed_dat = process_above(
        #         self._dataset, self._data_frame, logger)
        #     # (proc_numerical, proc_binsensitive,
        #     #  proc_categorical_binsensitive) = process_below(
        #     #     self._dataset, processed_dat)
        #     disturbed_dat = adverse_perturb(
        #         self._dataset, processed_dat, ratio)
        #     # (dist_numerical, dist_binsensitive,
        #     #  dist_categorical_binsensitive) = process_below(
        #     #     self._dataset, disturbed_dat)
        #     # return {
        #     #     "original": processed_dat,
        #     #     "numerical": proc_numerical,
        #     #     "numerical-binsensitive": proc_binsensitive,
        #     #     "categorical-binsensitive": proc_categorical_binsensitive}, {
        #     #     "original": disturbed_dat,
        #     #     "numerical": dist_numerical,
        #     #     "numerical-binsensitive": dist_binsensitive,
        #     #     "categorical-binsensitive": dist_categorical_binsensitive
        #     # }

        # elif handle.startswith('hfm'):
        #     pass

        processed_dat = process_above(
            self._dataset, self._data_frame, logger)
        since = time.time()
        disturbed_dat = adverse_perturb(
            self._dataset, processed_dat, ratio)
        tim_elapsed = time.time() - since
        del since
        handling_info = {
            'processed_dat': processed_dat,
            'disturbed_dat': disturbed_dat,
            'perturbation_tim_elapsed': tim_elapsed}
        belongs_priv = self._dataset.find_where_belongs(
            processed_dat)
        belongs_priv_with_joint = []
        if len(belongs_priv) > 1:
            belongs_priv_with_joint = [
                np.logical_and(belongs_priv[0], belongs_priv[1]),
                np.logical_or(belongs_priv[0], belongs_priv[1]),
            ]
        handling_info['idx_non_sa'] = belongs_priv
        handling_info['idx_priv_jt'] = belongs_priv_with_joint
        if handle not in ['dr', 'hfm_bin', 'hfm_nonbin']:
            # if len(belongs_priv) > 1:
            #     new_attr = self._dataset.get_sensitive_attrs_with_joint()[-1]
            #     processed_Xy.drop(columns=new_attr, inplace=True)
            #     disturbed_Xy.drop(columns=new_attr, inplace=True)
            processed_Xy = process_intersectional(
                self._dataset, processed_dat)
            disturbed_Xy = process_intersectional(
                self._dataset, disturbed_dat)
            marginal_grps = processed_Xy['marginalised_groups']
            handling_info['marginalised_grps'] = marginal_grps
            del processed_Xy['marginalised_groups']
            del disturbed_Xy['marginalised_groups']
            # pdb.set_trace()
            return handling_info, processed_Xy, disturbed_Xy

        # processed_Xy = process_below(self._dataset, processed_dat)
        # processed_Xy = processed_Xy['numerical-binsensitive']
        # disturbed_Xy = process_below(self._dataset, disturbed_dat)
        # disturbed_Xy = disturbed_Xy['numerical-binsensitive']
        _, processed_Xy, _ = process_below(
            self._dataset, processed_dat)
        _, disturbed_Xy, _ = process_below(
            self._dataset, disturbed_dat)
        if handle in ['dr']:
            # _, proc_binsensitive, _ = process_below()
            # return proc_binsensitive, dist_binsensitive
            # return (processed_dat, disturbed_dat
            #         ), processed_Xy, disturbed_Xy
            return handling_info, processed_Xy, disturbed_Xy

        binarized_Xy = binarized_data_set(processed_Xy)
        if handle in ['hfm_bin']:  # in ['hfm']:
            # return (processed_dat, disturbed_dat
            #         ), binarized_Xy, binarized_data_set(
            #     disturbed_Xy)
            return handling_info, binarized_Xy, \
                binarized_data_set(disturbed_Xy)
            # elif handle in ['hfm_nonbin']:
        # elif handle.startswith('hfm'):
        preproc_bin = process_addtl(self._dataset, processed_dat)
        preproc_mu = process_addtl_multivalue(
            self._dataset, preproc_bin['original'])
        perturb_bin = process_addtl(self._dataset, disturbed_dat)
        perturb_mu = process_addtl_multivalue(
            self._dataset, preproc_bin['original'])
        marginalised_grps = preproc_mu['marginalised_groups']
        handling_info['marginalised_grps'] = marginalised_grps
        del preproc_mu['marginalised_groups']
        del perturb_mu['marginalised_groups']
        # return handling_info, (preproc_bin, preproc_mu), (
        #     perturb_bin, perturb_mu)
        preproc_bin = preproc_bin['numerical-binsensitive']
        perturb_bin = perturb_bin['numerical-binsensitive']
        return handling_info, (
            preproc_bin, preproc_mu['numerical-multisen'],
            # preproc_mu['binarized-numerical-sen']), (
            # perturb_bin, perturb_mu['numerical-multisen'],
            # preproc_mu['binarized-numerical-sen'])
        ), (perturb_bin, perturb_mu['numerical-multisen'],)


class GraphSetup:
    def __init__(self, figname=''):
        self._figname = figname
        self._cmap_name = 'muted'  # 'bright'

    @property
    def figname(self):
        return self._figname

    def schedule_mspaint(self, raw_dframe, prep=''):
        raise NotImplementedError

    def load_raw_dataset(self, figname, sheetname):
        filename = os.path.join(
            RAW_EXPT_DIR, filename + '.xlsx')
        dframe = pd.read_excel(filepath, sheetname)
        return dframe
