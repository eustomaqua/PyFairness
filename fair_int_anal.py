# coding: utf-8


import pdb
import pandas as pd
import numpy as np

from pyfair.utils_empirical import GraphSetup, GRP_FAIR_COMMON
from pyfair.facil.utils_const import unique_column, DTY_FLT

from pyfair.granite.draw_addtl import (
    multi_lin_reg_with_distr, single_line_reg_with_distr,
    multi_lin_reg_without_distr)
from pyfair.granite.draw_fancy import (
    boxplot_rect, multi_boxplot_rect, radar_chart)
from pyfair.granite.draw_chart import analogous_confusion_extended


# -----------------------------
# Exp1: bin-val vs. multi-val
#
# Plot 1:
# Plot 2:
#


class PlotA_initial(GraphSetup):
    pass

    _perf_metric = [
        'Accuracy', 'Precision', 'Recall',  # 'Sensitivity'
        'Specificity', r'$\mathrm{f}_1$ score', r'$\bar{g}$',
        'bal. acc', 'discr power']  # 'discrim. discrimn.'
    # _dal_metric = [
    #     r'$\Delta(\text{Accuracy})$', r'$\Delta(\text{Precision})$',
    #     r'$\Delta(\text{Recall})$', r'$\Delta(\text{Specificity})$',
    #     r'$\Delta(\mathrm{f}_1 ~\text{score})$', r'$\Delta(\bar{g})$',
    #     r'$\Delta(\text{bal. acc})$', r'$\Delta(\text{discr power})$']
    _dal_metric = [
        r'$\Delta$(Accuracy)', r'$\Delta$(Precision)',
        r'$\Delta$(Recall)', r'$\Delta$(Specificity)',
        r'$\Delta$($\mathrm{f}_1$ score)', r'$\Delta(\bar{g})$',
        r'$\Delta$(bal. acc)', r'$\Delta$(discr power)']

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
            :4] + st_hfm_drt[15:22] + st_hfm_app[:4] + st_hfm_app[
            15:22] + st_grp[0][-7: -1]
        tag_sa2 = st_grp[1][6:22] + st_grp[1][-1:] + st_hfm_drt[
            4:8] + st_hfm_drt[22:29] + st_hfm_app[4:8] + st_hfm_app[
            22:29] + st_grp[1][-7: -1]

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
                               tag='tst', first_incl=False):
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
        if not first_incl:  # if not first: first_incl.
            return df_raw
        df_tmp = dframe.iloc[id_set[0] + 1: id_set[1]][tag_acc + tag_sa1]
        df_raw = pd.concat([df_raw, df_tmp], axis=0)
        return df_raw

    def draw_extended_grp_tim(self, df, tag_X, tag_Ys, figname,
                              verbose=False):  # annot_X, annot_Ys,
        X = df[tag_X].values.astype(DTY_FLT)
        Ys = [df[i].values.astype(DTY_FLT) / X for i in tag_Ys]
        antX = r'T_\text{bin-val}'  # refined
        antYs = [r'T_\text{multival}', r'T_\text{multival alt}']
        antYs.extend([r'T_\text{extGrp}', r'T_\text{alt.extGrp}'])

        annots = [f'${antX}$ (sec)', f'${antYs[0]}$',
                  f'${antYs[0]}={antX}$']
        kws = {'linreg': True, 'snspec': 'sty6'}  # 'sty4'
        annots[1] = r'$\frac{T_\text{multival}}{T_\text{bin-val}}-1$'
        single_line_reg_with_distr(X, Ys[0] - 1., annots,
                                   f'{figname}_tim_st6a', **kws)
        annots[1] = r'$\lg(\frac{T_\text{multival}}{T_\text{bin-val}})$'
        single_line_reg_with_distr(X, np.log10(Ys[0]), annots,
                                   f'{figname}_tim_st6b', **kws)

        # annots[2] = f'${antYs[1]}={antX}$'
        # annots[1] = r'$\frac{T_\text{multival alt}}{T_\text{bin-val}}$'
        if not verbose:
            return
        tmp = [[r'$\frac{ T_\text{extGrp} }{ T_\text{bin-val} }-1$',
                r'$\lg(\frac{ T_\text{extGrp} }{ T_\text{bin-val} })$'], [
               r'$\frac{ T_\text{alt.extGrp} }{ T_\text{bin-val} }-1$',
               r'$\lg(\frac{ T_\text{alt.extGrp} }{ T_\text{bin-val} })$']]
        for i in [-2, -1]:
            annots_sep = [f'${antX}$ (sec)', '', f'${antYs[i]}={antX}$']
            annots_sep[1] = tmp[i][0]
            single_line_reg_with_distr(
                X, Ys[i] - 1, annots_sep,
                f'{figname}_grptim_sep{i}_st6a', **kws)
            annots_sep[1] = tmp[i][1]
            single_line_reg_with_distr(
                X, np.log10(Ys[i]), annots_sep,
                f'{figname}_grp_tim_sep{i}_st6b', **kws)
        return

    def draw_extended_hfm_tim(self, df, tag_X, tag_Ys, figname):
        X = df[tag_X].values.astype(DTY_FLT)  # T(drt bin-val)
        Ys = [df[i].values.astype(DTY_FLT) for i in tag_Ys]
        # T(drt multi-val), T(app bin-val), T(app multi-val)
        # annots = [r'$T_\text{hfm bin-val}$', r'$T_\text{hfm multival}$',
        #           r'$T_\text{hfm multival}=T_\text{hfm bin-val}$']
        annots = [r'$T_{\mathbf{df}_\text{prev} ~\text{(bin-val)}}$',
                  r'$T_{\mathbf{df} ~\text{(multival)}}$',
                  r'$T_{\mathbf{df} ~\text{(multival)}}=T_{\mathbf{df}_\text{prev} ~\text{(bin-val)}}$']
        antZs = [r'$T_{\mathbf{df} ~\text{(multival)}}$', r'$T_{\hat{\mathbf{df}} ~\text{(bin-val)}}$', r'$T_{\hat{\mathbf{df}} ~\text{(multival)}}$']
        # multi_lin_reg_without_distr(X, Ys, antZs, annots,
        #                             f'{figname}_tim_sty4', snspec='sty4')

        n_ell = len(tag_Ys)
        Zs = [i / X - 1. for i in Ys]
        annots[1] = r'$\frac{ T_{\mathbf{df} ~\text{(multival)}} }{ T_{\mathbf{df}_\text{prev} ~\text{(bin-val)}} }-1$'
        multi_lin_reg_without_distr(
            X, Zs, antZs[:n_ell], annots,
            f'{figname}_tim_st6a', snspec='sty6')
        kws = {'linreg': True, 'snspec': 'sty6'}
        single_line_reg_with_distr(
            X, Zs[0], annots, f'{figname}_prtim_6a', **kws)
        Zs = [np.log10(i / X) for i in Ys]
        annots[1] = r'$\lg(\frac{ T_{\mathbf{df} ~\text{(multival)}} }{ T_{\mathbf{df}_\text{prev} ~\text{(bin-val)}} })$'
        multi_lin_reg_without_distr(
            X, Zs, antZs[:n_ell], annots,
            f'{figname}_tim_st6b', snspec='sty6')
        single_line_reg_with_distr(
            X, Zs[0], annots, f'{figname}_prtim_6b', **kws)
        return

    def draw_extended_grp_scat(self, df, tag_grp, tag_ext,
                               tag_ext_alt, figname,
                               verbose=False):
        # labels = ['grp', 'extGrp', 'alt.extGrp']  # 'extAlt'
        labels = ['ori', 'ext', 'alt']
        lbl_hfm = [[r'$\mathbf{df}_\text{prev}$', r'$\mathbf{df}$',
                    r'$\mathbf{df}^{avg}$'], [
            r'$\hat{\mathbf{df}}_\text{prev}$', r'$\hat{\mathbf{df}}$',
            r'$\hat{\mathbf{df}}^{avg}$'], ]
        lbl_dim2 = ['DP', 'EO', 'PQP', r'$\mathbf{df}_\text{prev}$',
                    r'$\hat{\mathbf{df}}_\text{prev}$']
        lbl_dim2[2] = 'PP'
        lbl_dim2[:3] = GRP_FAIR_COMMON
        multi_boxplot_rect(
            df, tag_grp[:3], tag_ext[:3],
            figname=f'{figname}_grpext', annotX=lbl_dim2[:3],
            locate="upper left")
        multi_boxplot_rect(
            df, tag_grp[:3], tag_ext[:3], tag_ext_alt[:3],
            figname=f'{figname}_grpalt', annotX=lbl_dim2[:3],
            locate="upper left")
        if not verbose:
            return

        for i, tg in enumerate(tag_grp):
            data = [df[tg].values.astype(DTY_FLT),
                    df[tag_ext[i]].values.astype(DTY_FLT),
                    df[tag_ext_alt[i]].values.astype(DTY_FLT)]
            fgn = '{}_{}'.format(
                figname, f'grp{i+1}' if i < 3 else f'hfm{i+3}')
            # boxplot_rect(data, labels, fgn + '_prim')
            multi_boxplot_rect(df, [tg, tag_ext[
                i], tag_ext_alt[i]], figname=fgn,
                annotX=labels if i < 3 else lbl_hfm[i - 3])  # not tag_Xs
        multi_boxplot_rect(df, tag_grp, tag_ext,
                           figname=f'{figname}_dim2', annotX=lbl_dim2)
        multi_boxplot_rect(df, tag_grp, tag_ext, tag_ext_alt,
                           figname=f'{figname}_dim3', annotX=lbl_dim2)
        return

    def obtain_sing_dat_cls(self, pick_set, pick_clf,
                            tag_acc, tag_sa1, tag_sa2,
                            dframe, id_set,  # tag='tst',
                            multival=True):  # nonbin=True):
        columns = {t2: t1 for t1, t2 in zip(tag_sa1, tag_sa2)}
        picked_a = id_set[pick_set] + 1 + pick_clf * self._nb_cv
        picked_b = id_set[pick_set] + 1 + (pick_clf + 1) * self._nb_cv
        if multival:
            assert pick_set in [0, 2, 3, 4]
            df_tmp = dframe.iloc[picked_a: picked_b]
            if pick_set in [3, 4]:
                df_tmp = df_tmp[tag_acc + tag_sa2]
                df_tmp = df_tmp.rename(columns=columns)
            else:
                df_tmp = df_tmp[tag_acc + tag_sa1]
        else:
            assert pick_set in [1, 2, 3, 4]
            df_tmp = dframe.iloc[picked_a: picked_b]
            if pick_set in [1, 3, 4]:
                df_tmp = df_tmp[tag_acc + tag_sa1]
            else:
                df_tmp = df_tmp[tag_acc + tag_sa2]
                df_tmp = df_tmp.rename(columns=columns)
            if pick_set == 1:
                df_alt = dframe.iloc[picked_a: picked_b][
                    tag_acc + tag_sa2].rename(columns=columns)
                df_tmp = pd.concat([df_tmp, df_alt], axis=0)
        return df_tmp

    def depict_separately(self, pick_set, pick_clf, df, id_set,
                          tag_mk='tst', fgn='', multival=True,
                          verbose=False):
        tag_acc, tag_sa1, tag_sa2 = self.obtain_tag_col(
            tag_mk)  # mark)
        df_alt = self.obtain_sing_dat_cls(
            pick_set, pick_clf, tag_acc, tag_sa1, tag_sa2,
            df, id_set, multival)
        sub_grp = tag_sa1[:3] + [tag_sa1[16 + 3], tag_sa1[27 + 3]]
        sub_ext = tag_sa1[4:10][:3] + [tag_sa1[16 + 7], tag_sa1[27 + 7]]
        sub_ext_alt = tag_sa1[10:16][:3] + [tag_sa1[
            16 + 10], tag_sa1[27 + 10]]
        sub_idv = tag_acc[16:16 + 4 + 6]  # dr 4+ GEI.alph 3+ Theil+Tx2
        sub_idv = [sub_idv[2], ] + sub_idv[4:-2]
        sub_idv = [sub_idv[0], sub_idv[2], sub_idv[-1]]

        currX = sub_grp[:3] + sub_idv + sub_grp[-2:]
        labels = ['DP', 'EO', 'PP', 'DR', r'GEI ($\alpha=0.5$)',
                  'Theil', r'$\mathbf{df}_\text{prev}$',
                  r'$\hat{\mathbf{df}}_\text{prev}$']
        labels[:3] = GRP_FAIR_COMMON
        df_tmp = df_alt[currX]  # =df_alt[curr_X].mean(axis=0)
        # radar_chart(df_tmp, currX, annotX=labels, figname='test_b', clockwise=True)
        # radar_chart(df_tmp, currX, annotX=labels, figname='test_a')
        for i in currX:
            df_tmp.loc[:, i] = float(df_tmp[i].mean())
        radar_chart(df_tmp, currX, annotX=labels,
                    figname=f'{fgn}_s{pick_set}c{pick_clf}_ori',
                    clockwise=True)

        df_tmp = df_tmp.reset_index(drop=True)[:3]
        df_tmp_tmp = df_alt[sub_ext]
        for i, j in zip(sub_grp, sub_ext):
            df_tmp.loc[1, i] = float(df_tmp_tmp[j].mean())
        df_tmp_tmp = df_alt[sub_ext_alt]
        for i, j in zip(sub_grp, sub_ext_alt):
            df_tmp.loc[2, i] = float(df_tmp_tmp[j].mean())
        annotY = ['ori', 'ext', 'ext.alt']
        if verbose:
            radar_chart(df_tmp[:2], currX, labels, annotY[:2],
                        figname=f'{fgn}_s{pick_set}c{pick_clf}_ext')
        radar_chart(df_tmp, currX, labels, annotY,
                    figname=f'{fgn}_s{pick_set}c{pick_clf}_extalt')
        # pdb.set_trace()
        return

    def draw_trade_off(self, df, pick, tag_X, tag_Ys, figname):
        for pk in pick:
            annotX = self._perf_metric[pk]  # pick]

        key_A = [tag_X[:8][i] for i in pick]
        key_C = [tag_X[8:16][i] for i in pick]
        key_B_bin = tag_Ys[0][:3] + tag_X[-3:] + tag_Ys[0][-2:]
        key_B_nonbin = tag_Ys[1][:3] + tag_X[-3:] + tag_Ys[1][-2:]
        key_B_extalt = tag_Ys[2][:3] + tag_X[-3:] + tag_Ys[2][-2:]
        lbl_A = [self._perf_metric[i] for i in pick]
        lbl_C = [self._dal_metric[i] for i in pick]
        lbl_B_bin = GRP_FAIR_COMMON + [
            'DR', r'GEI ($\alpha=0.5$)', 'Theil',
            r'$\mathbf{df}_\text{prev}$',
            r'$\hat{\mathbf{df}}_\text{prev}$']
        lbl_B_ext = [f'{i} ext.' for i in GRP_FAIR_COMMON] + lbl_B_bin[
            3:6] + [r'$\mathbf{df}$', r'$\hat{\mathbf{df}}$']
        lbl_B_extalt = [
            f'{i} alt.' for i in GRP_FAIR_COMMON] + lbl_B_bin[3:6] + [
            r'$\mathbf{df}^\text{avg}$',
            r'$\hat{\mathbf{df}}^\text{avg}$']
        Mat_B_bin = df[key_B_bin].values.astype(DTY_FLT).T
        Mat_B_ext = df[key_B_nonbin].values.astype(DTY_FLT).T
        Mat_B_extalt = df[key_B_extalt].values.astype(DTY_FLT).T 
        kws = {'cmap_name': 'Blues', 'rotate': 65}
        # analogous_confusion_extended(
        #     df[key_A].values.astype(DTY_FLT).T, Mat_B_bin, lbl_A,
        #     lbl_B_bin, f'{figname}_cont1', **kws)
        analogous_confusion_extended(
            df[key_C].values.astype(DTY_FLT).T, Mat_B_bin, lbl_C,
            lbl_B_bin, f'{figname}_cont1p', **kws)
        kws['cmap_name'] = 'Oranges'
        # analogous_confusion_extended(
        #     df[key_A].values.astype(DTY_FLT).T, Mat_B_ext, lbl_A,
        #     lbl_B_ext, f'{figname}_cont2', **kws)
        analogous_confusion_extended(
            df[key_C].values.astype(DTY_FLT).T, Mat_B_ext, lbl_C,
            lbl_B_ext, f'{figname}_cont2p', **kws)
        kws['cmap_name'] = 'RdPu'
        # analogous_confusion_extended(
        #     df[key_A].values.astype(DTY_FLT).T, Mat_B_extalt,
        #     lbl_A, lbl_B_extalt, f'{figname}_cont3', **kws)
        analogous_confusion_extended(
            df[key_C].values.astype(DTY_FLT).T, Mat_B_extalt,
            lbl_C, lbl_B_extalt, f'{figname}_cont3p', **kws)
        pdb.set_trace()
        return


class PlotA_fair_ens(PlotA_initial):
    def __init__(self):
        pass

    def schedule_mspaint(self, raw_dframe, figname=''):
        nb_set, id_set = self.recap_sub_data(
            raw_dframe, sa_ir=3, sa_r=4)
        mk = 'tst'  # flag,mark
        first_incl = verbose = False
        # df_bin = self.obtain_binval_senatt(raw_dframe, id_set, mk)
        df_nonbin = self.obtain_multival_senatt(raw_dframe, id_set, mk,
                                                first_incl=first_incl)
        tag_acc, tag_sa1, _ = self.obtain_tag_col(mk)

        tmp = tag_sa1[-6: -3]
        df_nonbin['extGrp'] = df_nonbin[
            tmp[0]] + df_nonbin[tmp[1]] + df_nonbin[tmp[2]]
        tmp = tag_sa1[-3:]
        df_nonbin['extAlt'] = df_nonbin[tmp[0]] + df_nonbin[
            tmp[1]] + df_nonbin[tmp[2]] + df_nonbin['extGrp']
        pdb.set_trace()
        pick = [0, 1, 2, 3, 4, 5]  # ,6,7]
        col_grp = tag_sa1[:3] + [tag_sa1[16 + 3], tag_sa1[27 + 3]]
        col_ext = tag_sa1[4:10][:3] + [tag_sa1[16 + 7], tag_sa1[27 + 7]]
        col_ext_alt = tag_sa1[10:16][:3] + [
            tag_sa1[16 + 10], tag_sa1[27 + 10]]
        self.draw_trade_off(df_nonbin, pick, tag_acc[:16] + [
            tag_acc[15 + 3], tag_acc[19 + 2], tag_acc[19 + 4]], [
            col_grp, col_ext, col_ext_alt], f'{figname}_to')
        self.draw_extended_grp_scat(
            df_nonbin, col_grp, col_ext, col_ext_alt,
            f'{figname}_scat', verbose)

        # self.draw_extended_grp(df_nonbin, tag_sa1[3], [tag_sa1[16]], (
        #     # 'Group fairness (bin-val)',
        #     # 'Extended group fairness (multival)'))
        #     'Grp (bin-val)', 'Ext.grp (multival)'))
        # self.draw_extended_grp(df_nonbin, tag_sa1[3], [tag_sa1[16]],
        #                        'Grp (bin-val)', ['Ext.grp (multival)'],
        #                        figname)
        self.draw_extended_grp_tim(df_nonbin, tag_sa1[3], [
            tag_sa1[16], 'extGrp', 'extAlt'], figname + '_grp')
        self.draw_extended_hfm_tim(df_nonbin, tag_sa1[20], [
            tag_sa1[27], tag_sa1[27 + 4], tag_sa1[27 + 4 + 7]],
            figname + '_hfm')
        # # self.draw_extended_hfm_tim(df_nonbin, tag_sa1[20], [
        # #     tag_sa1[27], ], figname + '_hfm_prim')
        # self.draw_extended_grp_scat(df_nonbin, tag_sa1[:3] + [
        #     # tag_sa1[19]],  # tag_sa1[4:4 + 6], tag_sa1[4 + 6: 4 + 12],
        #     # tag_sa1[4:10] + [tag_sa1[19 + 4], tag_sa1[19 + 7]],
        #     # tag_sa1[10:16] + [tag_sa1[26 + 3], tag_sa1[26 + 7]],
        #     tag_sa1[16 + 3], tag_sa1[27 + 3]],
        #     tag_sa1[4:10][:3] + [tag_sa1[16 + 7], tag_sa1[27 + 7]],
        #     tag_sa1[10:16][:3] + [tag_sa1[16 + 10], tag_sa1[27 + 10]],
        #     figname + '_scat', verbose)

        # # self.obtain_sing_dat_cls(0, 2, raw_dframe, id_set, mk)
        # # self.depict_separately(0, 2, mk)
        # self.depict_separately(0, 2, raw_dframe, id_set, mk,
        #                        figname + '_radar')
        # for pks in [2, 3, 4]:
        #     self.depict_separately(pks, 2, raw_dframe, id_set,
        #                            mk, figname + '_radar')
        fgn = f'{figname}_radar'
        for pkc in [0, 1, 2, 6, 10]:  # range(3+4+4):
            for pks in [2, 3, 4]:
                self.depict_separately(
                    pks, pkc, raw_dframe, id_set, mk, fgn)
        if not first_incl:
            return
        for pkc in [0, 1, 2, 6]:     # range(3+4):
            self.depict_separately(  # pks = 0
                0, pkc, raw_dframe, id_set, mk, fgn)
        return


class PlotA_norm_cls(PlotA_initial):
    def __init__(self):
        pass

    def schedule_mspaint(self, raw_dframe, figname=''):
        nb_set, id_set = self.recap_sub_data(
            raw_dframe, sa_ir=11, sa_r=0)
        mk = 'tst'
        first_incl = verbose = False
        df_nonbin = self.obtain_multival_senatt(
            raw_dframe, id_set, mk, first_incl)
        tag_acc, tag_sa1, _ = self.obtain_tag_col(mk)

        tmp = tag_sa1[-6: -3]
        df_nonbin['extGrp'] = df_nonbin[
            tmp[0]] + df_nonbin[tmp[1]] + df_nonbin[tmp[2]]
        tmp = tag_sa1[-3:]
        df_nonbin['extAlt'] = df_nonbin[tmp[0]] + df_nonbin[
            tmp[1]] + df_nonbin[tmp[2]] + df_nonbin['extGrp']
        self.draw_extended_grp_tim(df_nonbin, tag_sa1[
            3], [tag_sa1[16], 'extGrp', 'extAlt'],
            f'{figname}_grp', verbose)
        self.draw_extended_hfm_tim(df_nonbin, tag_sa1[20], [
            tag_sa1[27], tag_sa1[27 + 4], tag_sa1[27 + 4 + 7]],
            f'{figname}_hfm')
        self.draw_extended_grp_scat(df_nonbin, tag_sa1[:3] + [
            tag_sa1[16 + 3], tag_sa1[27 + 3]],
            tag_sa1[4:10][:3] + [tag_sa1[16 + 7], tag_sa1[27 + 7]],
            tag_sa1[10:16][:3] + [tag_sa1[16 + 10], tag_sa1[27 + 10]],
            f'{figname}_scat', verbose)
        fgn = f'{figname}_radar'
        for pks in [2, 3, 4]:
            self.depict_separately(pks, 2, raw_dframe, id_set, mk, fgn)
        if not verbose:
            return
        for pkc in [0, 1, 6, 10]:
            for pks in [2, 3, 4]:
                self.depict_separately(
                    pks, pkc, raw_dframe, id_set, mk, fgn)
        return


# -----------------------------


# -----------------------------


# -----------------------------
