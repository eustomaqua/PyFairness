# coding: utf-8


import pdb
import pandas as pd
import numpy as np

from pyfair.utils_empirical import GraphSetup
from pyfair.facil.utils_const import unique_column, DTY_FLT

from pyfair.granite.draw_addtl import (
    multi_lin_reg_with_distr, single_line_reg_with_distr,
    multi_lin_reg_without_distr)


# -----------------------------
# Exp1: bin-val vs. multi-val
#
# Plot 1:
# Plot 2:
#


class PlotA_initial(GraphSetup):
    pass

    def obtain_tag_col(self, tag='tst'):
        csv_row_1 = unique_column(12 + 158 * 2)
        tag_trn = csv_row_1[12: 12 + 158]
        tag_tst = csv_row_1[-158:]
        tag_col = tag_trn if tag == 'trn' else tag_tst

        # sub-tags
        st_acc = tag_col[: 24]
        st_grp = [tag_col[24: 24 + 29], tag_col[53: 24 + 58]]
        st_dr = tag_col[24 + 58: 24 + 58 + 18]  # 24+76=100
        st_hfm_drt = tag_col[100: 100 + 29]
        st_hfm_app = tag_col[100 + 29: 100 + 58]

        tag_common = st_acc[:8] + st_acc[-8:] + st_dr[:4] + [
            st_dr[6], st_dr[9], st_dr[12]] + st_dr[-3:] + st_hfm_drt[
            8:15] + st_hfm_app[8:15]  # delta, GEI alph=.2|.5|.8,Theil
        tag_sa1 = st_grp[0][6:22] + st_grp[0][-1:] + st_hfm_drt[
            :4] + st_hfm_drt[15:22] + st_hfm_app[:4] + st_hfm_app[15:22]
        tag_sa2 = st_grp[1][6:22] + st_grp[1][-1:] + st_hfm_drt[
            4:8] + st_hfm_drt[22:29] + st_hfm_app[4:8] + st_hfm_app[22:29]

        # tag_common: perf 8+ delta(perf) 8+ dr(loss,ut,hat_bias,ut) 4+
        #             GEI(alph=.2|.5|.8, Theil,T(Theil),T(GEIx11)) 6+
        #             hfm direct multiver 7+ hfm approx multiver 7
        # tag_sa1/2 : DP,EO,PQP,tim, SP-ext *3, SP-ext-avg *3,
        #             SP-ext-meticulous *3, SP-ext-avg-meticulous *3,tim,
        #             hfm drt (bin 4+ nonbin 7), hfm app (bin 4+ nonbin 7)
        # siz: 16+4+6+14=40, 4+13+11*2=17+22=39
        return tag_common, tag_sa1, tag_sa2

    def obtain_binval_senatt(self, dframe, id_set,  # nb_set,
                             tag='tst'):
        tag_acc, tag_sa1, tag_sa2 = self.obtain_tag_col(tag)
        columns = {t2: t1 for t1, t2 in zip(tag_sa1, tag_sa2)}
        df_raw = dframe.iloc[id_set[1] + 1: id_set[2]][tag_acc + tag_sa1]
        for k in [1, 2]:
            df_tmp = dframe.iloc[id_set[
                k] + 1: id_set[k + 1]][tag_acc + tag_sa2]
            df_tmp = df_tmp.rename(columns=columns)
            df_raw = pd.concat([df_raw, df_tmp], axis=0)

        for k in [3, 4]:
            df_tmp = dframe.iloc[id_set[
                k] + 1: id_set[k + 1]][tag_acc + tag_sa1]
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
        return df_raw

    def obtain_multival_senatt(self, dframe, id_set,  # nb_set,
                               tag='tst'):
        tag_acc, tag_sa1, tag_sa2 = self.obtain_tag_col(tag)
        columns = {t2: t1 for t1, t2 in zip(tag_sa1, tag_sa2)}
        df_raw = dframe.iloc[id_set[2] + 1: id_set[3]][tag_acc + tag_sa1]
        for k in [3, 4]:
            df_tmp = dframe.iloc[id_set[k] + 1: id_set[
                k + 1]][tag_acc + tag_sa2]
            df_tmp = df_tmp.rename(columns=columns)
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
        # df_raw = df_raw.reset_index(drop=True)
        # np.isnan(df_raw.values.astype('float')).any()
        df_tmp = dframe.iloc[id_set[0] + 1: id_set[1]][tag_acc + tag_sa1]
        df_raw = pd.concat([df_raw, df_tmp], axis=0)
        return df_raw

    def draw_extended_grp(self, df, tag_X, tag_Ys, annot_X, annot_Ys):
        X = df[tag_X].values.astype(DTY_FLT)
        Ys = [df[i].values.astype(DTY_FLT) for i in tag_Ys]
        pdb.set_tracet()
        pass


class PlotA_fair_ens(PlotA_initial):
    def __init__(self):
        pass

    def schedule_mspaint(self, raw_dframe, prep=''):
        nb_set, id_set = self.recap_sub_data(
            raw_dframe, sa_ir=3, sa_r=4)
        mk = 'tst'  # flag,mark
        df_bin = self.obtain_binval_senatt(raw_dframe, id_set, mk)
        df_nonbin = self.obtain_multival_senatt(raw_dframe, id_set, mk)
        tag_acc, tag_sa1, _ = self.obtain_tag_col(mk)

        # self.draw_extended_grp(df_nonbin, tag_sa1[3], [tag_sa1[16]], (
        #     # 'Group fairness (bin-val)',
        #     # 'Extended group fairness (multival)'))
        #     'Grp (bin-val)', 'Ext.grp (multival)'))
        self.draw_extended_grp(df_nonbin, tag_sa1[3], [tag_sa1[16]],
                               'Grp (bin-val)', ['Ext.grp (multival)'])
        pdb.set_trace()
        return


class PlotA_norm_cls(PlotA_initial):
    def __init__(self):
        pass

    def schedule_mspaint(self, raw_dframe, prep=''):
        nb_set, id_set = self.recap_sub_data(
            raw_dframe, sa_ir=11, sa_r=0)
        return


# -----------------------------


# -----------------------------


# -----------------------------
