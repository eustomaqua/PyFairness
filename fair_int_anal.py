# coding: utf-8


import os
import pdb
import pandas as pd
import numpy as np

from pyfair.utils_empirical import GraphSetup, GRP_FAIR_COMMON
from pyfair.facil.utils_const import unique_column, DTY_FLT

from pyfair.granite.draw_addtl import (
    multi_lin_reg_with_distr, single_line_reg_with_distr,
    multi_lin_reg_without_distr,
    scatter_with_marginal_distrib, lineplot_with_uncertainty,
    line_reg_with_marginal_distr)
from pyfair.granite.draw_fancy import (
    boxplot_rect, multi_boxplot_rect, radar_chart)
from pyfair.granite.draw_chart import (
    analogous_confusion_extended, multiple_scatter_chart)


# -----------------------------
# Exp1: bin-val vs. multi-val
#
# Plot 1:
# Plot 2:
#


class PlotA_initial(GraphSetup):
    pass
    _dr_ptb = 'K'  # perturb(ation)  # 'K','L'

    _perf_metric = [
        'Accuracy', 'Precision', 'Recall',  # 'Sensitivity'
        'Specificity', r'$\mathrm{f}_1$ score',  # r'$\bar{g}$',
        'G mean', 'bal. acc', 'discr power']  # 'discrim. discrimn.'
    # _dal_metric = [
    #     r'$\Delta(\text{Accuracy})$', r'$\Delta(\text{Precision})$',
    #     r'$\Delta(\text{Recall})$', r'$\Delta(\text{Specificity})$',
    #     r'$\Delta(\mathrm{f}_1 ~\text{score})$', r'$\Delta(\bar{g})$',
    #     r'$\Delta(\text{bal. acc})$', r'$\Delta(\text{discr power})$']
    _dal_metric = [
        r'$\Delta$(Accuracy)', r'$\Delta$(Precision)',
        r'$\Delta$(Recall)', r'$\Delta$(Specificity)',
        r'$\Delta$($\mathrm{f}_1$ score)',  # r'$\Delta(\bar{g})$',
        r'$\Delta$(G mean)',  # r'$\Delta$(g-mean)', 'g_mean'
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
        # return tag_common, tag_sa1, tag_sa2
        return tag_common + csv_row_1[10:12], tag_sa1, tag_sa2

    def obtain_binval_senatt(self, dframe, id_set,  # nb_set,
                             tag='tst'):
        tag_acc, tag_sa1, tag_sa2 = self.obtain_tag_col(tag)
        columns = {t2: t1 for t1, t2 in zip(tag_sa1, tag_sa2)}
        df_raw = dframe.iloc[id_set[1] + 1: id_set[2]][tag_acc + tag_sa1]
        df_raw[self._dr_ptb] = dframe.iloc[id_set[1]][self._dr_ptb]
        for k in [1, 2]:
            df_tmp = dframe.iloc[id_set[
                k] + 1: id_set[k + 1]][tag_acc + tag_sa2]
            df_tmp = df_tmp.rename(columns=columns)
            df_tmp[self._dr_ptb] = dframe.iloc[id_set[k]][self._dr_ptb]
            df_raw = pd.concat([df_raw, df_tmp], axis=0)

        for k in [3, 4]:
            df_tmp = dframe.iloc[id_set[
                k] + 1: id_set[k + 1]][tag_acc + tag_sa1]
            df_tmp[self._dr_ptb] = dframe.iloc[id_set[k]][self._dr_ptb]
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
        return df_raw

    def obtain_multival_senatt(self, dframe, id_set,  # nb_set,
                               tag='tst', first_incl=False):
        tag_acc, tag_sa1, tag_sa2 = self.obtain_tag_col(tag)
        columns = {t2: t1 for t1, t2 in zip(tag_sa1, tag_sa2)}
        df_raw = dframe.iloc[id_set[2] + 1: id_set[3]][tag_acc + tag_sa1]
        df_raw[self._dr_ptb] = dframe.iloc[id_set[2]][self._dr_ptb]
        for k in [3, 4]:
            df_tmp = dframe.iloc[id_set[k] + 1: id_set[
                k + 1]][tag_acc + tag_sa2]
            df_tmp = df_tmp.rename(columns=columns)
            df_tmp[self._dr_ptb] = dframe.iloc[id_set[k]][self._dr_ptb]
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
        # df_raw = df_raw.reset_index(drop=True)
        # np.isnan(df_raw.values.astype('float')).any()
        if not first_incl:  # if not first: first_incl.
            return df_raw
        df_tmp = dframe.iloc[id_set[0] + 1: id_set[1]][tag_acc + tag_sa1]
        df_tmp[self._dr_ptb] = dframe.iloc[id_set[0]][self._dr_ptb]
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
        labels = ['DP', 'EO', 'PP', 'DR',  # r'GEI ($\alpha=0.5$)',
                  r'GEI ($\alpha$=0.5)',
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
        annotY = ['ori', 'ext', 'alt']  # 'ext.alt']
        if verbose:
            radar_chart(df_tmp[:2], currX, labels, annotY[:2],
                        figname=f'{fgn}_s{pick_set}c{pick_clf}_ext')
        radar_chart(df_tmp, currX, labels, annotY,
                    figname=f'{fgn}_s{pick_set}c{pick_clf}_extalt')
        return

    def draw_trade_off(self, df, pick, tag_X, tag_Ys, figname,
                       ver_mark=''):
        annotZs = GRP_FAIR_COMMON + [
            r'GEI ($\alpha$=0.5)', 'Theil', 'DR']
        # tmp_ext = ['{:6s} ext'.format(i) for i in annotZs[:3]]
        # tmp_ext_alt = ['{:6s} alt'.format(i) for i in annotZs[:3]]
        # tmp_ext[1] = f'{annotZs[1]} ext'
        # tmp_ext_alt[1] = f'{annotZs[1]} alt'  # not '{:7s}'
        tmp_ext = [r'$\text{DP}^\text{ext}$',
                   r'$\text{EOpp}^\text{ext}$',
                   r'$\text{PP}^\text{ext}$', ]
        tmp_ext_alt = [r'$\text{DP}^\text{alt}$',
                       r'$\text{EOpp}^\text{alt}$',
                       r'$\text{PP}^\text{alt}$', ]
        for pk in pick:
            annotX = self._perf_metric[pk]  # pick]
            # annots = (annotX, "Fairness")  # 'Fairness measure'
            # X = df[tag_X[pk]].values.astype(DTY_FLT)
            # Ys = df[tag_Ys[0][:3] + tag_X[-2:]].values.astype(DTY_FLT).T
            # multiple_scatter_chart(
            #     X, Ys, annots, annotZs, f'{figname}_to{pk}v',
            #     ind_hv='v', identity=False)
            # scatter_with_marginal_distrib(
            #     df, tag_X[pk], 'Fairness', tag_Ys[0][:3],
            #     GRP_FAIR_COMMON, annotX=annotX, annotY='Fairness',
            #     figname=f'{figname}_to{pk}_s4')
            # line_reg_with_marginal_distr(
            #     df, tag_X[pk], 'Fairness', tag_Ys[0][:3] + tag_X[
            #         -2:] + tag_X[-3:-2], annotZs, annotX=annotX,
            #     annotY='Fairness', snspec='sty4b',
            #     figname=f'{figname}_to{pk}_s4')
            line_reg_with_marginal_distr(  # tag_X[-2:]+tag_X[-3:-2]
                df, tag_X[pk], 'Fairness', tag_X[-3:], annotZs[-3:],
                annotX=annotX, annotY='Individual fairness',
                snspec='sty4b', figname=f'{figname}_to{pk}_s4')
            line_reg_with_marginal_distr(
                df, tag_X[pk], 'Fairness', tag_Ys[0][:3], annotZs[:3],
                annotX=annotX, annotY='Group fairness (bin-val)',
                snspec='sty4b', figname=f'{figname}_to{pk}_s1')
            line_reg_with_marginal_distr(
                df, tag_X[pk], 'Fairness', tag_Ys[1][:3],
                # [f'{i} ext.' for i in annotZs[:3]], annotX=annotX,
                tmp_ext, annotX=annotX,
                annotY='Extended group fairness (multival)',
                snspec='sty4b', figname=f'{figname}_to{pk}_s2')
            line_reg_with_marginal_distr(
                # df, tag_X[pk], 'Fairness', tag_Ys[0][:3],
                # [f'{i} ext. alt' for i in annotZs[:3]], annotX=annotX,
                df, tag_X[pk], 'Fairness', tag_Ys[2][:3],
                tmp_ext_alt, annotX=annotX,
                annotY='Alternative extended group fairness (multival)',
                snspec='sty4b', figname=f'{figname}_to{pk}_s3')

        key_A = [tag_X[:8][i] for i in pick]
        key_C = [tag_X[8:16][i] for i in pick]
        key_B_bin = tag_Ys[0][:3] + tag_X[-3:] + tag_Ys[0][-2:]
        key_B_nonbin = tag_Ys[1][:3] + tag_X[-3:] + tag_Ys[1][-2:]
        key_B_extalt = tag_Ys[2][:3] + tag_X[-3:] + tag_Ys[2][-2:]
        lbl_A = [self._perf_metric[i] for i in pick]
        lbl_C = [self._dal_metric[i] for i in pick]
        # lbl_B_bin = GRP_FAIR_COMMON + [
        #     'DR', r'GEI ($\alpha=0.5$)', 'Theil',
        #     r'$\mathbf{df}_\text{prev}$',
        #     r'$\hat{\mathbf{df}}_\text{prev}$']
        lbl_B_bin = annotZs + [r'$\mathbf{df}_\text{prev}$',
                               r'$\hat{\mathbf{df}}_\text{prev}$']
        # lbl_B_ext = [f'{i} ext.' for i in GRP_FAIR_COMMON] + lbl_B_bin[
        #     3:6] + [r'$\mathbf{df}$', r'$\hat{\mathbf{df}}$']
        # lbl_B_extalt = [
        #     f'{i} alt.' for i in GRP_FAIR_COMMON] + lbl_B_bin[3:6] + [
        #     r'$\mathbf{df}^\text{avg}$',
        #     r'$\hat{\mathbf{df}}^\text{avg}$']
        lbl_B_ext = tmp_ext + lbl_B_bin[3:6] + [
            r'$\mathbf{df}$', r'$\hat{\mathbf{df}}$']
        lbl_B_extalt = tmp_ext_alt + lbl_B_bin[3:6] + [
            r'$\mathbf{df}^\text{avg}$',
            r'$\hat{\mathbf{df}}^\text{avg}$']
        Mat_B_bin = df[key_B_bin].values.astype(DTY_FLT).T
        Mat_B_ext = df[key_B_nonbin].values.astype(DTY_FLT).T
        Mat_B_extalt = df[key_B_extalt].values.astype(DTY_FLT).T 
        kws = {'cmap_name': 'Blues', 'rotate': 65}
        # analogous_confusion_extended(
        #     df[key_A].values.astype(DTY_FLT).T, Mat_B_bin, lbl_A,
        #     lbl_B_bin, f'{figname}_cont1', **kws)
        # analogous_confusion_extended(
        #     df[key_C].values.astype(DTY_FLT).T, Mat_B_bin, lbl_C,
        #     lbl_B_bin, f'{figname}_cont1p', **kws)
        analogous_confusion_extended(
            df[key_C + key_A].values.astype(DTY_FLT).T, Mat_B_bin,
            lbl_C + lbl_A, lbl_B_bin, f'{figname}_cont1p', **kws)
        kws['cmap_name'] = 'Oranges'
        # analogous_confusion_extended(
        #     df[key_A].values.astype(DTY_FLT).T, Mat_B_ext, lbl_A,
        #     lbl_B_ext, f'{figname}_cont2', **kws)
        # analogous_confusion_extended(
        #     df[key_C].values.astype(DTY_FLT).T, Mat_B_ext, lbl_C,
        #     lbl_B_ext, f'{figname}_cont2p', **kws)
        analogous_confusion_extended(
            df[key_C + key_A].values.astype(DTY_FLT).T, Mat_B_ext,
            lbl_C + lbl_A, lbl_B_ext, f'{figname}_cont2p', **kws)
        kws['cmap_name'] = 'RdPu'
        # analogous_confusion_extended(
        #     df[key_A].values.astype(DTY_FLT).T, Mat_B_extalt,
        #     lbl_A, lbl_B_extalt, f'{figname}_cont3', **kws)
        # analogous_confusion_extended(
        #     df[key_C].values.astype(DTY_FLT).T, Mat_B_extalt,
        #     lbl_C, lbl_B_extalt, f'{figname}_cont3p', **kws)
        analogous_confusion_extended(
            df[key_C + key_A].values.astype(DTY_FLT).T, Mat_B_extalt,
            lbl_C + lbl_A, lbl_B_extalt, f'{figname}_cont3p', **kws)
        return

    def draw_extended_idv_tim(self, df, tag_X, tag_Ys, figname):
        tim_grp, tim_grp_nonbin = tag_X
        tim_idv, tim_df, tim_df_pl = tag_Ys
        X = df[tim_grp].values.astype(DTY_FLT)
        Ys = [df[i].values.astype(DTY_FLT) / X for i in [
            tim_grp_nonbin, ] + tim_idv]
        antX = r'T_\text{gf (bin-val)}'      # GF,gf
        antY = r'T_\text{if (multival)}'
        annots = [f'${antX}$ (sec)', f'${antY}$',
                  f'${antY}={antX}$']  # antYs[1]
        antZs = [r'$T_\text{gf (multival)}$',
                 # r'$T_{\text{GEI (} \alpha\text{=0.5)}}$',
                 r'$T_\text{GEI}$',    # alpha 0-1 in list
                 r'$T_\text{Theil}$', r'$T_\text{DR}$']
        # Z = df['idv_ptb'].values.astype(DTY_FLT) / X
        # Z = Ys[:-1] + [Z, ]
        annots[2] = r'$T_\text{if}= T_\text{gf (bin-val)}$'

        kws = {'snspec': 'sty7'}  # 'sty6' # {'linreg': True,
        annots[1] = r'$\frac{ T_\text{if (multival)} }{ T_\text{gf (bin-val)} }-1$'
        # multi_lin_reg_without_distr(
        #     X, [i - 1. for i in Ys], antZs, annots,
        #     f'{figname}_tim_st6a', **kws)
        # multi_lin_reg_without_distr(
        #     X, [i - 1. for i in Z], antZs, annots,
        #     f'{figname}_tim_st7a', **kws)
        multi_lin_reg_without_distr(
            X, [i - 1. for i in Ys[:-1]], antZs[:-1], annots,
            f'{figname}_tim_st8a', **kws)
        # multi_lin_reg_without_distr(
        #     X, [i - 1. for i in Ys[1:]], antZs[1:], annots,
        #     f'{figname}_tim_st9a', **kws)
        # annots[1] = r'$\lg(\frac{ T_\text{IF (multival)} }{ T_\text{GF (bin-val)} })$'
        annots[1] = r'$\lg(\frac{ T_\text{if (multival)} }{ T_\text{gf (bin-val)} })$'
        # multi_lin_reg_without_distr(
        #     X, [np.log10(i) for i in Ys], antZs, annots,
        #     f'{figname}_tim_st6b', **kws)
        # multi_lin_reg_without_distr(
        #     X, [np.log10(i) for i in Z], antZs, annots,
        #     f'{figname}_tim_st7b', **kws)
        multi_lin_reg_without_distr(
            X, [np.log10(i) for i in Ys[:-1]], antZs[:-1], annots,
            f'{figname}_tim_st8b', **kws)
        # multi_lin_reg_without_distr(
        #     X, [np.log10(i) for i in Ys[1:]], antZs[1:], annots,
        #     f'{figname}_tim_st9b', **kws)

        # fgn = f'{figname}_df_tim'
        # self.sub_draw_idv_df(df, tim_df, tim_df_pl, tag_X, X, fgn, kws)
        # pdb.set_trace()
        return

    # def sub_draw_idv_bin(self, df, tag_X, tag_Ys, figname):
    #     return
    # def sub_draw_idv_nonbin(self, df, tag_X, tag_Ys, figname):
    #     return

    def sub_draw_idv_df(self, df, tim_df, tim_df_pl, tag_X, X, fgn, kws):
        antZs_drt = [r'$T_\text{gf (multival)}$',
                     r'$T_{\mathbf{df}_\text{prev} \text{ (bin-val)}}$',
                     r'$T_{\mathbf{df} \text{ (multival)}}$',
                     r'$T_{\mathbf{df} \text{ intersectional}}$']
        annots_drt = [r'$T_\text{gf (bin-val)}$ (sec)', '',
                      r'$T_{\mathbf{df}} =T_\text{gf (bin-val)}$']
        Ys = [df[i].values.astype(DTY_FLT) / X for i in tag_X[
            1:] + tim_df[:2] + tim_df_pl[:1]]
        annots_drt[1] = r'$\frac{ T_{\mathbf{df} \text{ (multival)}} }{T_\text{gf (bin-val)}}-1$'
        multi_lin_reg_without_distr(
            X, [i - 1. for i in Ys], antZs_drt, annots_drt,
            f'{fgn}_da', **kws)
        # multi_lin_reg_without_distr(
        #     X, [i - 1. for i in Ys[1:]], antZs_drt[1:],
        #     annots_drt, f'{fgn}_d1p', **kws)
        annots_drt[1] = r'$\lg(\frac{ T_{\mathbf{df} \text{ (multival)}} }{T_\text{gf (bin-val)}})$'
        multi_lin_reg_without_distr(
            X, [np.log10(i) for i in Ys], antZs_drt, annots_drt,
            f'{fgn}_db', **kws)
        # multi_lin_reg_without_distr(
        #     X, [np.log10(i) for i in Ys[1:]], antZs_drt[1:],
        #     annots_drt, f'{fgn}_d2p', **kws)

        antZs_app = [
            r'$T_\text{gf (multival)}$',
            r'$T_{\hat{\mathbf{df}}_\text{prev} \text{ (bin-val)}}$',
            r'$T_{\hat{\mathbf{df}} \text{ (multival)}}$',
            r'$T_{\hat{\mathbf{df}} \text{ intersectional}}$']
        annots_app = [r'$T_\text{gf (bin-val)}$ (sec)', '',
                      r'$T_{\hat{\mathbf{df}}} =T_\text{gf (bin-val)}$']
        Ys = [df[i].values.astype(DTY_FLT) / X for i in tag_X[
            1:] + tim_df[2:] + tim_df_pl[1:]]
        annots_app[1] = r'$\frac{ T_{\hat{\mathbf{df}} \text{ (multival)}} }{T_\text{gf (bin-val)}}-1$'
        multi_lin_reg_without_distr(
            X, [i - 1. for i in Ys], antZs_app, annots_app,
            f'{fgn}_aa', **kws)
        # multi_lin_reg_without_distr(
        #     X, [i - 1. for i in Ys[1:]], antZs_app[1:],
        #     annots_app, f'{fgn}_a1p', **kws)
        annots_app[1] = r'$\lg(\frac{ T_{\hat{\mathbf{df}} \text{ (multival)}} }{T_\text{gf (bin-val)}})$'
        multi_lin_reg_without_distr(
            X, [np.log10(i) for i in Ys], antZs_app, annots_app,
            f'{fgn}_ab', **kws)
        # multi_lin_reg_without_distr(
        #     X, [np.log10(i) for i in Ys[1:]], antZs_app[1:],
        #     annots_app, f'{fgn}_a2p', **kws)
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
        # pdb.set_trace()
        pick = [0, 4, 5]  # 1,2,3,] # ,6,7]
        col_grp = tag_sa1[:3] + [tag_sa1[16 + 3], tag_sa1[27 + 3]]
        col_ext = tag_sa1[4:10][:3] + [tag_sa1[16 + 7], tag_sa1[27 + 7]]
        col_ext_alt = tag_sa1[10:16][:3] + [
            tag_sa1[16 + 10], tag_sa1[27 + 10]]
        self.draw_trade_off(df_nonbin, pick, tag_acc[:16] + [
            tag_acc[19 + 2], tag_acc[19 + 4], tag_acc[15 + 3], ], [
            col_grp, col_ext, col_ext_alt], f'{figname}_to')
        self.draw_extended_grp_scat(
            df_nonbin, col_grp, col_ext, col_ext_alt,
            f'{figname}_scat', verbose)

        tim_idv = [tag_acc[16 + 3], ] + tag_acc[16 + 4 + 4:][:2]
        tim_idv = tim_idv[:: -1]   # DR,Theil,GEI: then reverse
        tim_df_pl = [tag_acc[26:][6], tag_acc[26:][6 + 7],
                     ]  # df/hat_df multiver (df intersectional)
        tim_grp = [tag_sa1[3], tag_sa1[16], 'extGrp', 'extAlt']  # three
        tim_df = [tag_sa1[20], tag_sa1[27], tag_sa1[27 + 4], tag_sa1[
            27 + 4 + 7]]  # df4one sen-att: bin-val, multival, hat_df x2
        df_nonbin['idv_ptb'] = df_nonbin[tag_acc[-2]] + df_nonbin[
            tim_idv[-1]]  # 'idvDR_perturb','idv_dr_ptb', 'idvDR_'
        self.draw_extended_idv_tim(df_nonbin, tim_grp[:2], [
            tim_idv, tim_df, tim_df_pl], figname + '_idv')

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

    def schedule_mspaint_avg(self, raw_dframe, figname=''):
        nb_set, id_set = self.recap_sub_data(raw_dframe, sa_ir=3, sa_r=4)
        mk, first_incl, verbose = 'tst', False, False
        df_nonbin = self.obtain_multival_senatt(
            raw_dframe, id_set, mk, first_incl=first_incl)
        tag_acc, tag_sa1, _ = self.obtain_tag_col(mk)
        tmp = tag_sa1[-6: -3]
        df_nonbin['extGrp'] = df_nonbin[tmp[
            0]] + df_nonbin[tmp[1]] + df_nonbin[tmp[2]]    # TimeCost
        tmp = tag_sa1[-3:]
        df_nonbin['extAlt'] = df_nonbin[tmp[0]] + df_nonbin[tmp[
            1]] + df_nonbin[tmp[2]] + df_nonbin['extGrp']  # TimeCost
        pick = [0, 4, 5]

        # extension in average forms, above is maximal forms
        col_grp = tag_sa1[:3] + [tag_sa1[16 + 3], tag_sa1[27 + 3]]
        col_ext = tag_sa1[4:10][-3:] + [tag_sa1[23], tag_sa1[34]]
        col_ext_alt = tag_sa1[10:16][-3:] + [tag_sa1[26], tag_sa1[37]]
        self.avg_draw_trade_off(df_nonbin, pick, tag_acc[
            :16] + [tag_acc[21], tag_acc[23], tag_acc[18], ], [
            col_grp, col_ext, col_ext_alt],
            f'{figname}_to_avg')  # f'{figname}_avg_to')
        self.avg_draw_extended_grp_scat(
            df_nonbin, col_grp, col_ext + tag_sa1[4:7],
            col_ext_alt + tag_sa1[10:13],
            # f'{figname}_avg_scat', verbose)
            f'{figname}_scat_avg', verbose)
        # pdb.set_trace()
        fgn = f'{figname}_radar_avg'  # f'{figname}_avg_radar'
        for pkc in [0, 1, 2, 6, 10]:
            for pks in [2, 3, 4]:
                self.avg_depict_separately(
                    pks, pkc, raw_dframe, id_set, mk, fgn)
                if pkc == 2:
                    continue
                os.remove(f'{fgn[:-4]}_s{pks}c{pkc}_ori.pdf')
        # if not first_incl:
        #     return
        # for pkc in [0, 1, 2, 6]:
        #     self.avg_depict_separately(
        #         0, pkc, raw_dframe, id_set, mk, fgn)

        col_ext_max = tag_sa1[4:10][:3] + [tag_sa1[23], tag_sa1[34]]
        col_ext_alt_max = tag_sa1[10:16][:3] + [
            tag_sa1[26], tag_sa1[37]]
        self.avg_draw_trade_off_alt(
            df_nonbin, pick, tag_acc[:16] + [
                tag_acc[21], tag_acc[23], tag_acc[18], ], [
                col_grp, col_ext_max, col_ext_alt_max,
                col_ext, col_ext_alt], f'{figname}_to_alt')
        self.avg_draw_incompatible_alt(
            df_nonbin, tag_acc[:16] + [
                tag_acc[21], tag_acc[23], tag_acc[18], ], [
                col_grp, col_ext_max, col_ext_alt_max,
                col_ext, col_ext_alt], f'{figname}_nc')
        return

    def avg_depict_separately(self, pick_set, pick_clf, df, id_set,
                              tag_mk='tst', fgn='', multival=True):
        #                       verbose=True):
        # if not verbose:
        #     os.remove(f'{fgn[:-4]}_s{pick_set}c{pick_clf}_ori.pdf')

        tag_acc, tag_sa1, tag_sa2 = self.obtain_tag_col(tag_mk)
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

        sub_ext_avg = tag_sa1[4:10][-3:] + [tag_sa1[23], tag_sa1[34]]
        sub_ext_alt_avg = tag_sa1[10:16][-3:] + [
            tag_sa1[26], tag_sa1[37]]
        currX = sub_grp[:3] + sub_idv + sub_grp[-2:]
        labels = GRP_FAIR_COMMON + [
            'DR', r'GEI ($\alpha$=0.5)', 'Theil',
            r'$\mathbf{df}_\text{prev}$',
            r'$\hat{\mathbf{df}}_\text{prev}$']
        df_tmp = df_alt[currX]
        for i in currX:
            df_tmp.loc[:, i] = float(df_tmp[i].mean())
        # radar_chart(df_tmp, currX, annotX=labels,
        #             figname=f'{fgn}_s{pick_set}c{pick_clf}_ori',
        #             clockwise=True)

        df_tmp = df_tmp.reset_index(drop=True)
        df_tmp_tmp = df_alt[sub_ext]
        for i, j in zip(sub_grp, sub_ext):
            df_tmp.loc[1, i] = float(df_tmp_tmp[j].mean())
        df_tmp_tmp = df_alt[sub_ext_alt]
        for i, j in zip(sub_grp, sub_ext_alt):
            df_tmp.loc[2, i] = float(df_tmp_tmp[j].mean())

        df_tmp_tmp = df_alt[sub_ext_avg]
        for i, j in zip(sub_grp, sub_ext_avg):
            df_tmp.loc[3, i] = float(df_tmp_tmp[j].mean())
        df_tmp_tmp = df_alt[sub_ext_alt_avg]
        for i, j in zip(sub_grp, sub_ext_alt_avg):
            df_tmp.loc[4, i] = float(df_tmp_tmp[j].mean())
        annotY = ['ori', 'ext', 'alt', 'ext (avg)', 'alt (avg)']
        radar_chart(df_tmp, currX, labels, annotY,
                    figname=f'{fgn}_s{pick_set}c{pick_clf}')
        # pdb.set_trace()
        return

    def avg_draw_extended_grp_scat(self, df, tag_grp, tag_ext,
                                   tag_ext_alt, figname,
                                   verbose=False, ver_mark=' (avg)'):
        labels = ['ori', 'ext', 'alt',
                  f'ext{ver_mark}', f'alt{ver_mark}']
        lbl_dim2 = GRP_FAIR_COMMON + [
            r'$\mathbf{df}_\text{prev}$',
            r'$\hat{\mathbf{df}}_\text{prev}$']
        fgn = figname.replace('_avg', '')
        multi_boxplot_rect(df, tag_grp[:3], tag_ext[:3],
                           labels =['ori'] + labels[-2:],
                           figname=f'{fgn}_grpext_avg',
                           annotX=lbl_dim2[:3], locate="upper left")
        multi_boxplot_rect(
            df, tag_grp[:3], tag_ext[:3], tag_ext_alt[:3],
            labels =['ori'] + labels[-2:],
            figname=f'{fgn}_grpalt_avg',
            annotX=lbl_dim2[:3], locate="upper left")
        multi_boxplot_rect(
            df, tag_grp[:3], tag_ext[-3:], tag_ext_alt[-3:],
            tag_ext[:3], tag_ext_alt[:3],
            figname=f'{fgn}_group_max_avg',
            annotX=lbl_dim2[:3], locate="upper left",
            figsize='M-NT')

        multi_boxplot_rect(
            df, tag_grp[:3], tag_ext[-3:], labels =labels[:3],
            # figname='{}_grpext'.format(figname.replace('avg', 'max')),
            figname=f'{fgn}_grpext',  # f'{fgn}_grpext_max',
            annotX=lbl_dim2[:3], locate="upper left")
        multi_boxplot_rect(
            df, tag_grp[:3], tag_ext[-3:], tag_ext_alt[-3:],
            labels =labels[:3],
            # figname=f'{figname.replace('avg', 'max')}_grpalt',
            # figname='{}_grpext'.format(figname.replace('avg', 'max')),
            figname=f'{fgn}_grpalt',  # f'{fgn}_grpalt_max',
            annotX=lbl_dim2[:3], locate="upper left")
        # os.remove(f'{figname[:-9]}_scat_grpalt.pdf')
        # os.remove(f'{figname[:-9]}_scat_grpext.pdf')

        # labels = labels[:3] if not ver_mark else ['ori'] + labels[-2:]
        # lbl_hfm = [[r'$\mathbf{df}_\text{prev}$', r'$\mathbf{df}$',
        #             r'$\mathbf{df}^{avg}$'], [
        #     r'$\hat{\mathbf{df}}_\text{prev}$', r'$\hat{\mathbf{df}}$',
        #     r'$\hat{\mathbf{df}}^{avg}$'], ]
        # for i, tg in enumerate(tag_grp):
        #     data = [df[tg].values.astype(DTY_FLT),
        #             df[tag_ext[i]].values.astype(DTY_FLT),
        #             df[tag_ext_alt[i]].values.astype(DTY_FLT)]
        #     fgn = '{}_{}'.format(
        #         figname, f'grp{i+1}' if i < 3 else f'hfm{i+3}')
        #     multi_boxplot_rect(df, [tg, tag_ext[
        #         i], tag_ext_alt[i]], figname=fgn,
        #         annotX=labels if i < 3 else lbl_hfm[i - 3])  # not tag_Xs
        # multi_boxplot_rect(df, tag_grp, tag_ext[:5],
        #                    figname=f'{figname}_dim2', annotX=lbl_dim2)
        # multi_boxplot_rect(df, tag_grp, tag_ext[:5], tag_ext_alt[:5],
        #                    figname=f'{figname}_dim3', annotX=lbl_dim2)
        # pdb.set_trace()
        return

    def avg_draw_trade_off(self, df, pick, tag_X, tag_Ys, figname,
                           ver_mark=' (avg)'):
        annotZs = GRP_FAIR_COMMON + [
            r'GEI ($\alpha$=0.5)', 'Theil', 'DR']
        annotY = 'Extended group fairness (multival)'
        # tmp_ext = ['{:6s} ext{}'.format(
        #     i, ver_mark) for i in annotZs[:3]]
        # tmp_ext_alt = ['{:6s} alt{}'.format(  # 'ext. alt{}'
        #     i, ver_mark) for i in annotZs[:3]]
        # tmp_ext[1] = f'{annotZs[1]} ext{ver_mark}'
        # tmp_ext_alt[1] = f'{annotZs[1]} alt{ver_mark}'
        tmp_ext = [r'$\text{DP}^\text{ext(avg)}$',
                   r'$\text{EOpp}^\text{ext(avg)}$',
                   r'$\text{PP}^\text{ext(avg)}$', ]
        tmp_ext_alt = [r'$\text{DP}^\text{alt(avg)}$',
                       r'$\text{EOpp}^\text{alt(avg)}$',
                       r'$\text{PP}^\text{alt(avg)}$', ]

        for pk in pick:
            annotX = self._perf_metric[pk]
            line_reg_with_marginal_distr(
                df, tag_X[pk], 'Fairness', tag_Ys[1][:3],
                # [f'{i} ext.{ver_mark}' for i in annotZs[:3]],
                tmp_ext, annotX=annotX, annotY=annotY,
                snspec='sty4b', figname=f'{figname}_to{pk}_s2')
            line_reg_with_marginal_distr(
                # df, tag_X[pk], 'Fairness', tag_Ys[0][:3],
                # [f'{i} ext. alt{ver_mark}' for i in annotZs[:3]],
                df, tag_X[pk], 'Fairness', tag_Ys[2][:3],
                tmp_ext_alt, annotX=annotX, annotY=annotY.replace(
                    'Extended', 'Alternative extended'),
                snspec='sty4b', figname=f'{figname}_to{pk}_s3')

        key_A = [tag_X[:8][i] for i in pick]
        key_C = [tag_X[8:16][i] for i in pick]
        key_B_bin = tag_Ys[0][:3] + tag_X[-3:] + tag_Ys[0][-2:]
        key_B_nonbin = tag_Ys[1][:3] + tag_X[-3:] + tag_Ys[1][-2:]
        key_B_extalt = tag_Ys[2][:3] + tag_X[-3:] + tag_Ys[2][-2:]
        lbl_A = [self._perf_metric[i] for i in pick]
        lbl_C = [self._dal_metric[i] for i in pick]
        lbl_B_bin = annotZs + [r'$\mathbf{df}_\text{prev}$',
                               r'$\hat{\mathbf{df}}_\text{prev}$']
        Mat_B_bin = df[key_B_bin].values.astype(DTY_FLT).T
        Mat_B_ext = df[key_B_nonbin].values.astype(DTY_FLT).T
        Mat_B_extalt = df[key_B_extalt].values.astype(DTY_FLT).T 
        kws = {'cmap_name': 'Blues', 'rotate': 65}

        # lbl_B_ext = [f'{i} ext{ver_mark}' for i in GRP_FAIR_COMMON
        #              ] + lbl_B_bin[3:6] + [
        #     r'$\mathbf{df}$', r'$\hat{\mathbf{df}}$']
        # lbl_B_extalt = [f'{i} alt{ver_mark}' for i in GRP_FAIR_COMMON
        #                 ] + lbl_B_bin[3:6] + [
        #     r'$\mathbf{df}^\text{avg}$',
        #     r'$\hat{\mathbf{df}}^\text{avg}$']
        lbl_B_ext = tmp_ext + lbl_B_bin[3:6] + [
            r'$\mathbf{df}$', r'$\hat{\mathbf{df}}$']
        lbl_B_extalt = tmp_ext_alt + lbl_B_bin[3:6] + [
            r'$\mathbf{df}^\text{avg}$',
            r'$\hat{\mathbf{df}}^\text{avg}$']
        kws['cmap_name'] = 'Oranges'
        analogous_confusion_extended(
            df[key_C + key_A].values.astype(DTY_FLT).T, Mat_B_ext,
            lbl_C + lbl_A, lbl_B_ext, f'{figname}_cont2p', **kws)
        kws['cmap_name'] = 'RdPu'
        analogous_confusion_extended(
            df[key_C + key_A].values.astype(DTY_FLT).T, Mat_B_extalt,
            lbl_C + lbl_A, lbl_B_extalt, f'{figname}_cont3p', **kws)
        return

    def avg_draw_trade_off_alt(self, df, pick, tag_X, tag_Ys,
                               figname):
        annotZs = GRP_FAIR_COMMON + [
            r'GEI ($\alpha$=0.5)', 'Theil', 'DR']
        key_A = [tag_X[:8][i] for i in pick]
        key_C = [tag_X[8:16][i] for i in pick]
        key_B_bin = tag_Ys[0][:3] + tag_X[-3:] + tag_Ys[0][-2:]
        lbl_A = [self._perf_metric[i] for i in pick]
        lbl_C = [self._dal_metric[i] for i in pick]
        lbl_B_bin = annotZs + [r'$\mathbf{df}_\text{prev}$',
                               r'$\hat{\mathbf{df}}_\text{prev}$']
        Mat_B_bin = df[key_B_bin].values.astype(DTY_FLT).T
        kws = {'cmap_name': 'Blues', 'rotate': 65}

        key_B_nonbin = tag_Ys[1][:3] + tag_Ys[3][:3] + tag_Ys[1][-2:]
        key_B_extalt = tag_Ys[2][:3] + tag_Ys[4][:3] + tag_Ys[2][-2:]
        # lbl_B_ext = [f'{i} ext' for i in GRP_FAIR_COMMON] + [
        #     f'{i} ext(avg)' for i in GRP_FAIR_COMMON] + [
        #     r'$\mathbf{df}$', r'$\hat{\mathbf{df}}$']
        # lbl_B_extalt = [f'{i} alt' for i in GRP_FAIR_COMMON] + [
        #     f'{i} alt(avg)' for i in GRP_FAIR_COMMON] + [
        #     r'$\mathbf{df}^\text{avg}$',
        #     r'$\hat{\mathbf{df}}^\text{avg}$']
        Mat_B_ext = df[key_B_nonbin].values.astype(DTY_FLT).T
        Mat_B_extalt = df[key_B_extalt].values.astype(DTY_FLT).T

        lbl_B_ext = [r'$\text{DP}^\text{ext}$',
                     r'$\text{EOpp}^\text{ext}$',
                     r'$\text{PP}^\text{ext}$',
                     r'$\text{DP}^\text{ext(avg)}$',
                     r'$\text{EOpp}^\text{ext(avg)}$',
                     r'$\text{PP}^\text{ext(avg)}$', ] + [
            r'$\mathbf{df}$', r'$\hat{\mathbf{df}}$']
        lbl_B_extalt = [r'$\text{DP}^\text{alt}$',
                        r'$\text{EOpp}^\text{alt}$',
                        r'$\text{PP}^\text{alt}$',
                        r'$\text{DP}^\text{alt(avg)}$',
                        r'$\text{EOpp}^\text{alt(avg)}$',
                        r'$\text{PP}^\text{alt(avg)}$', ] + [
            r'$\mathbf{df}^\text{avg}$',
            r'$\hat{\mathbf{df}}^\text{avg}$']

        fgn = figname[:-4]  # figname.replace('_alt', '')
        os.remove(f'{fgn}_cont1p.pdf')
        os.remove(f'{fgn}_cont2p.pdf')
        os.remove(f'{fgn}_cont3p.pdf')
        os.remove(f'{fgn}_avg_cont2p.pdf')
        os.remove(f'{fgn}_avg_cont3p.pdf')
        Mat_C_A = df[key_C + key_A].values.astype(DTY_FLT).T
        analogous_confusion_extended(
            Mat_C_A, Mat_B_bin, lbl_C + lbl_A, lbl_B_bin,
            f'{fgn}_cont1p', **kws)

        kws['cmap_name'] = 'Oranges'
        analogous_confusion_extended(
            Mat_C_A, Mat_B_ext, lbl_C + lbl_A, lbl_B_ext,
            f'{figname}_cont2p', **kws)
        kws['cmap_name'] = 'RdPu'
        analogous_confusion_extended(
            Mat_C_A, Mat_B_extalt, lbl_C + lbl_A,
            lbl_B_extalt, f'{figname}_cont3p', **kws)
        return

    def avg_draw_incompatible_alt(self, df, tag_X, tag_Ys,
                                  figname):
        annotZs = GRP_FAIR_COMMON + [
            r'GEI ($\alpha$=0.5)', 'Theil', 'DR']
        # ta, ra = f"{'':<5}", f"{'':>5}"
        # tmp_ext = [r'$\text{DP}^\text{ext}$' + ta,
        #            r'$\text{EOpp}^\text{ext}$',
        #            r'$\text{PP}^\text{ext}$' + ta, ]
        # tmp_ext_alt = [r'$\text{DP}^\text{alt}$' + ta,
        #                r'$\text{EOpp}^\text{alt}$',
        #                r'$\text{PP}^\text{alt}$' + ta, ]
        # tmp_ext_avg = [r'$\text{DP}^\text{ext(avg)}$' + ra,
        #                r'$\text{EOpp}^\text{ext(avg)}$',
        #                r'$\text{PP}^\text{ext(avg)}$' + ra]
        # tmp_ext_alt_avg = [r'$\text{DP}^\text{alt(avg)}$' + ra,
        #                    r'$\text{EOpp}^\text{alt(avg)}$',
        #                    r'$\text{PP}^\text{alt(avg)}$' + ra]
        # annotZs[0] += f"{'':>4}"  # ta
        # annotZs[2] += f"{'':<4}"  # ra
        #
        # kws = {'annotY': 'Group fairness', 'snspec': 'sty5b'}  # invt_a=True
        # for i, pk in enumerate(tag_X[-3:]):
        #     annotX = annotZs[i + 3]
        #     annotX = f'Individual fairness: {annotX}'
        #     # annotX = f'Individual fairness ({annotX})'
        #     # f'{fgn}_bin' _ext,_alt,_ext_avg,_alt_avg
        #     fgn = f'{figname}_corr{i+3}'
        #     line_reg_with_marginal_distr(
        #         df, pk, 'Fairness', tag_Ys[0][:3], annotZs[:3],
        #         annotX=annotX, figname=f'{fgn}_g1', **kws)
        #     line_reg_with_marginal_distr(
        #         df, pk, 'Fairness', tag_Ys[1][:3], tmp_ext,
        #         annotX=annotX, figname=f'{fgn}_g2', **kws)
        #     line_reg_with_marginal_distr(
        #         df, pk, 'Fairness', tag_Ys[2][:3], tmp_ext_alt,
        #         annotX=annotX, figname=f'{fgn}_g3', **kws)
        #     line_reg_with_marginal_distr(
        #         df, pk, 'Fairness', tag_Ys[3][:3], tmp_ext_avg,
        #         annotX=annotX, figname=f'{fgn}_g4', **kws)
        #     line_reg_with_marginal_distr(
        #         df, pk, 'Fairness', tag_Ys[4][:3], tmp_ext_alt_avg,
        #         annotX=annotX, figname=f'{fgn}_g5', **kws)

        ta, ra = f"{'':<7}", f"{'':>8}"
        tmp_ext = [r'$\text{DP}^\text{ext}$' + ta,
                   r'$\text{EOpp}^\text{ext}$' + ta,
                   r'$\text{PP}^\text{ext}$' + ta, ]
        tmp_ext_alt = [r'$\text{DP}^\text{alt}$' + ra,
                       r'$\text{EOpp}^\text{alt}$' + ra,
                       r'$\text{PP}^\text{alt}$' + ra, ]
        tmp_ext_avg = [r'$\text{DP}^\text{ext(avg)}$',
                       r'$\text{EOpp}^\text{ext(avg)}$',
                       r'$\text{PP}^\text{ext(avg)}$']
        tmp_ext_alt_avg = [r'$\text{DP}^\text{alt(avg)}$' + ' ',
                           r'$\text{EOpp}^\text{alt(avg)}$' + ' ',
                           r'$\text{PP}^\text{alt(avg)}$' + ' ']
        annotZs[0] += ra + f"{'':>3}"
        annotZs[2] += ra + f"{'':>2}"
        annotZs[1] += ra + f"{'':<3}"
        grp1 = [y[0] for y in tag_Ys]
        grp2 = [y[1] for y in tag_Ys]
        grp3 = [y[2] for y in tag_Ys]
        lbl_g1 = [annotZs[0], tmp_ext[0], tmp_ext_alt[0],
                  tmp_ext_avg[0], tmp_ext_alt_avg[0]]
        lbl_g2 = [annotZs[1], tmp_ext[1], tmp_ext_alt[1],
                  tmp_ext_avg[1], tmp_ext_alt_avg[1]]
        lbl_g3 = [annotZs[2], tmp_ext[2], tmp_ext_alt[2],
                  tmp_ext_avg[2], tmp_ext_alt_avg[2]]
        kws = {'snspec': 'sty5b'}  # kws.pop('annotY')
        for i, pk in enumerate(tag_X[-3:]):
            annotX = annotZs[i + 3]
            fgn = f'{figname}_corr{i+4}'
            line_reg_with_marginal_distr(
                df, pk, 'Fairness', grp1, lbl_g1, annotX=annotX,
                annotY=lbl_g1[0], figname=f'{fgn}_grp1', **kws)
            line_reg_with_marginal_distr(
                df, pk, 'Fairness', grp2, lbl_g2, annotX=annotX,
                annotY=lbl_g2[0], figname=f'{fgn}_grp2', **kws)
            line_reg_with_marginal_distr(
                df, pk, 'Fairness', grp3, lbl_g3, annotX=annotX,
                annotY=lbl_g3[0], figname=f'{fgn}_grp3', **kws)

        hfm_drt = [tag_Ys[0][-2], tag_Ys[1][-2], tag_Ys[2][-2]]
        hfm_app = [tag_Ys[0][-1], tag_Ys[1][-1], tag_Ys[2][-1]]
        lbl_drt = [r'$\mathbf{df}_\text{prev}$', r'$\mathbf{df}$',
                   r'$\mathbf{df}^\text{avg}$']
        lbl_app = [r'$\hat{\mathbf{df}}_\text{prev}$',
                   r'$\hat{\mathbf{df}}$',
                   r'$\hat{\mathbf{df}}^\text{avg}$']
        for k, (pi, pj) in enumerate(zip(hfm_drt, hfm_app)):
            fgn = f'{figname}_df{k}'
            # _drt_g1,_drt_g2,_drt_g3,_app_g1,_app_g2,_app_g3
            line_reg_with_marginal_distr(
                df, pi, 'Fairness', grp1, lbl_g1, annotX=lbl_drt[k],
                annotY=lbl_g1[0], figname=f'{fgn}_g1d', **kws)
            line_reg_with_marginal_distr(
                df, pi, 'Fairness', grp2, lbl_g2, annotX=lbl_drt[k],
                annotY=lbl_g2[0], figname=f'{fgn}_g2d', **kws)
            line_reg_with_marginal_distr(
                df, pi, 'Fairness', grp3, lbl_g3, annotX=lbl_drt[k],
                annotY=lbl_g3[0], figname=f'{fgn}_g3d', **kws)
            line_reg_with_marginal_distr(
                df, pj, 'Fairness', grp1, lbl_g1, annotX=lbl_app[k],
                annotY=lbl_g1[0], figname=f'{fgn}_g1a', **kws)
            line_reg_with_marginal_distr(
                df, pj, 'Fairness', grp2, lbl_g2, annotX=lbl_app[k],
                annotY=lbl_g2[0], figname=f'{fgn}_g2a', **kws)
            line_reg_with_marginal_distr(
                df, pj, 'Fairness', grp3, lbl_g3, annotX=lbl_app[k],
                annotY=lbl_g3[0], figname=f'{fgn}_g3a', **kws)
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
