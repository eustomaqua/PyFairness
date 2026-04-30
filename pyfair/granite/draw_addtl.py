# coding: utf-8
#
# TARGET:
#   Measuring fairness via data manifolds
#


import matplotlib.pyplot as plt
import seaborn as sns
# import itertools
import numpy as np
import pandas as pd
import pdb


# from fairml.facilc.draw_graph import (
# from pyfair.senior.draw_graph import (
from pyfair.facil.draw_prelim import (
    PLT_LOCATION, PLT_FRAMEBOX, _setup_config,
    _style_set_axis, _setup_figsize, _setup_figshow,
    _setup_rgb_color)
from pyfair.granite.draw_graph import _sns_line_err_bars
# from pyfair.senior.draw_graph import (
#     _sns_line_err_bars, _setup_rgb_color)
# _setup_locater,_set_quantile, cnames, cname_keys, cmap_names,
# _backslash_distributed, _barh_patterns, _sns_line_fit_regs,
from pyfair.facil.utils_const import DTY_FLT, subfig_ind
from pyfair.marble.draw_hypos import Pearson_correlation


# ===============================
# Preliminaries
# Matlab plot
# -------------------------------


# ===============================
# Python Matlablib plot


# -------------------------------
# multiple line_chart.m
#

_line_styler = [
    '-', '--', '-.', ':']  # solid, dashed, dash-dot, dotted
_line_marker = [
    '.', ',', 'o', 'v', '^', '<', '>',  # point, pixel, circle,
    '1', '2', '3', '4',  # triangle_.., tri_down/up/left/right,
    's', 'p', '*', 'h', 'H',  # square, pentagon, star, hexagon,
    '+', 'x', 'D', 'd', '|', '_']  # /thin_diamond, vline/hline

_colr_nms = [
    '#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30',
    '#4DBEEE', '#A2142F']  # Matlab 2014, parula


_pl_myclr = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
_navy = 'navy'  # _dflt_navy

_navy = '#0b599e'
_pl_myclr = [_navy, '#9a0019', '#f5001c', '#fe8462', '#98deeb', '#1390c6',
             '#2a347a', '#8f96bd', '#9fc3d5', '#3ac3de']  # '#fed5bc',
# # _pl_myclr = ['#9a0019', '#f5001c', '#fe8462', '#fed5bc', '#98deeb',
# #              '#3ac3de', '#1390c6', '#0b599e'][::-1]


# '''
# def multiple_line_chart(X, Ys, annots=(
#     r"$\lambda$", r"Test Accuracy (%)"),
#         annotY=('',), mkrs=None,
#         figname='lam', figsize='S-NT'):
#     # X .shape (#num,)
#     # Ys.shape (#num, #baseline_for_comparison, #iter)
#
#     if mkrs is None:
#         mkrs = []
#         # mkrs += [for i in _line_marker]
#
#     num_bs = Ys.shape[1]
# '''


# def multiple_lines_with_errorbar(Ys, picked_keys, annotY='Acc',
# TODO!
def multiple_lines_with_errorbar(X, Ys, picked_keys=('Baseline #1',),
                                 annotX=r'$\lambda$', annotY='Acc',
                                 cmap_name='GnBu_r',
                                 figname='lam_sns', figsize='M-WS'):
    # similar usage: box_plot(Ys[:, i, :])
    #                only works for one algorithm
    # X or picked_keys: (#num,)
    # Ys.shape (#num, #baseline_for_comparison, nb_iter)

    # num, pick_baseline, _ = Ys.shape  # pick_baseline,nb_iter
    _, pick_baseline, _ = Ys.shape
    # fig, ax = plt.subplots(figsize=_setup_config['M-NT'])
    fig = plt.figure(figsize=_setup_config['M-NT'])
    ax = fig.gca()

    cs, _ = _setup_rgb_color(pick_baseline, cmap_name)  # ,cl
    kws = {'color': _navy, 'lw': 1}  # plt.plot(.5, 0.5)
    for j in range(pick_baseline):
        kws['color'] = cs[j]
        kws['label'] = picked_keys[j]
        # TODO 好像有点不对
        _sns_line_err_bars(ax, kws, X, Ys[:, j, :].mean(axis=1))

    ax.autoscale_view()
    fig = _setup_figsize(fig, figsize)
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


# -------------------------------
# 箱线图 带误差线
#


def box_plot(Ys, picked_keys, annotY='Acc',
             annotX='', patch_artist=False,
             figname='box_lam', figsize='M-WS', rotate=60):
    # Ys.shape (#baseline_for_comparison, #iter)

    pick_baseline = Ys.shape[0]  # ,nb_iter #picked_ways/method,
    fig, ax = plt.subplots(figsize=_setup_config['M-NT'])
    ax.boxplot(Ys.T, patch_artist=patch_artist)  # bp=

    ind = np.arange(pick_baseline) + 1
    ax.set_xticks(ind)
    ax.set_xticklabels(picked_keys, rotation=rotate)
    ax.set_ylabel(annotY)
    ax.set_xlabel(annotX)

    ax.autoscale_view()
    fig = _setup_figsize(fig, figsize)
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


def styled_box_plot():
    pass


# -------------------------------

# -------------------------------


# ===============================
# Python Matlablib plot


# -------------------------------
# fairmanf
#   for 单独一个数据集，类似上图，results of 5 iterations
#   (real approx values) + analysis (like mean+-std)
#


def scatter_k_cv_with_real(X, Ys, z,  # y/z: real values
                           # picked_keys=('Baseline #1',),
                           annotX=r'hyper-pm', annotY='value',
                           tidy_cv=False, ddof=0,  # 1,tidy_cv=True,
                           figsize='M-WS',  # cmap_name='GnBu_r',
                           figname='hyperpm_effect'):
    # This is for results from k-cross validation
    # X : possible values of some certain hyper-parameter
    # Ys: results of 算法的估计值
    # z : real value of 真实值

    # X .shape= (#num,)
    # Ys.shape= (nb_iter, #num)  # Ys.shape= (#num, nb_iter)
    # z .shape= (nb_iter,)

    nb_iter = Ys.shape[0]  # nb_iter, num = Ys.shape
    fig, ax = plt.subplots(figsize=_setup_config['M-NT'])

    kws = {'color': '#F65F47', 'lw': 1}
    for i in range(nb_iter):
        plt.scatter(X, Ys[i], s=2.5, **kws)

    kws['color'] = '#465386'
    tz_avg, tz_std = np.mean(z), np.std(z, ddof=ddof)
    tx_min, tx_max = ax.get_xlim()
    line, = ax.plot([tx_min, tx_max], [tz_avg, tz_avg],
                    label='Real value', **kws)
    line.sticky_edges.x[:] = (tx_min, tx_max)
    ax.fill_between([tx_min, tx_max],  # ax.get_xlim(),
                    [tz_avg - tz_std] * 2,
                    [tz_avg + tz_std] * 2, alpha=.15,
                    facecolor='#465386')  # **kws)
    # kws.pop('edgecolor')
    # kws.pop('facecolor')

    kws['color'] = '#F65F47'
    if not tidy_cv:
        tX = np.array([X] * nb_iter)
        tYs = Ys.reshape(-1)
        # tz = np.array([z] * num).T
    else:
        tX = np.array([X] * nb_iter).T
        tYs = Ys.T.reshape(-1)
        # tz = np.array([z] * num)
    kws['linestyle'] = '--'
    _sns_line_err_bars(ax, kws, tX.reshape(-1), tYs)
    kws.pop('linestyle')

    ax.ticklabel_format(
        style='sci', scilimits=(-1, 2), axis='y')
    plt.legend(loc='best', frameon=False)  # PLT_LOCATION,PLT_FRAMEBOX)
    ax.set_xlabel(annotX)
    ax.set_ylabel(annotY)

    ax.autoscale_view()
    fig = _setup_figsize(fig, figsize)
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


# -------------------------------
# fairmanf
#   时间代价


def boxplot_k_cv_with_real(X, Ys, z,
                           annotX=r'hyperpm', annotY='value',
                           patch_artist=False, ddof=0,
                           figsize='M-WS',
                           figname='hyperpm_boxmu'):
    # X .shape= (#num,)
    # Ys.shape= (nb_iter, #num)
    # z .shape= (nb_iter,)

    nb_iter = Ys.shape[0]  # nb_iter, num = Ys.shape
    fig, ax = plt.subplots(figsize=_setup_config['M-NT'])

    ax.boxplot(Ys, positions=X, patch_artist=patch_artist)  # bp=
    # ax.set_xticks(X)
    ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')

    kws = {'color': '#F65F47', 'lw': 1}
    for i in range(nb_iter):
        plt.scatter(X, Ys[i], s=2.5, **kws)
    kws['color'] = '#465386'
    tz_avg, tz_std = np.mean(z), np.std(z, ddof=ddof)
    tx_min, tx_max = ax.get_xlim()
    ax.plot([tx_min, tx_max], [tz_avg, tz_avg],
            label='Real value', **kws)
    ax.fill_between(
        [tx_min, tx_max],
        [tz_avg - tz_std] * 2, [tz_avg + tz_std] * 2,
        alpha=.15, facecolor='#465386')

    kws['color'] = '#F65F47'
    tX = np.array([X] * nb_iter).reshape(-1)
    # tz = np.array([z] * num).T.reshape(-1)
    kws['linestyle'] = '--'
    _sns_line_err_bars(ax, kws, tX, Ys.reshape(-1))
    kws.pop('linestyle')
    plt.legend(loc=PLT_LOCATION, frameon=PLT_FRAMEBOX)

    ax.set_xlabel(annotX)
    ax.set_ylabel(annotY)
    ax.autoscale_view()
    fig = _setup_figsize(fig, figsize)
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


# -------------------------------
# fairmanf
#   for 若干个数据集放在一起


# '''
# def _diff_between_approx_and_direct(Yss, zs):
#     # difference: abs(approx - direct) / direct
#     # Yss.shape= (#att_sens, nb_iter, #num)
#     # zs .shape= (#att_sens, nb_iter)
#
#     nb_att, nb_iter, num = Yss.shape
#     diff = np.zeros_like(Yss) - 1.
#     for j in range(nb_att):
#         for i in range(nb_iter):
#             diff[j][i] = np.abs(Yss[j][i] - zs[j][i])
#             diff[j][i] /= check_zero(zs[j][i])
#     return diff
#
#
# def approximated_dist_comparison(X, Yss, zs, picked_keys,
#                                  figsize='M-WS',
#                                  figname='hyperpm_multi'):
#     # nb_att, nb_iter, num = Yss.shape
#     diff = _diff_between_approx_and_direct(Yss, zs)
# '''


def approximated_dist_comparison(
        X, Ys, picked_keys, annotX='pm',
        annotY=r'$\frac{abs(\hat{\mathbf{D}}-\mathbf{D})}{\mathbf{D}}$',
        figsize='M-WS', cmap_name='Dark2_r',  # 'Accent_r',
        figname='hyperpm_multi'):
    # X  : possible values of some certain hyper-parameter
    # Yss: results of 算法的估计值
    # zs : real value of 真实值

    # X  .shape= (#num,)  # 21 for m2, 24 for m1
    # Yss.shape= (#att_sen, #iter, #num)
    # zs .shape= (#att_sen, #iter)

    # Ys = abs(Yss - zs) / zs
    # Ys .shape= (#att_sen, #iter, #num)

    nb_att, nb_iter = Ys.shape[:2]  # nb_att,nb_iter,num=Ys.shape
    fig, ax = plt.subplots(figsize=_setup_config['M-NT'])

    # cs, cl = _setup_rgb_color(nb_iter, cmap_name)
    cs = sns.color_palette(cmap_name)  # cl = len(cs)
    kws = {'color': _navy, 'lw': 1}
    if isinstance(X, list):
        tX = X * nb_iter
    else:
        # tX = np.repeat(X, nb_iter).reshape(num, -1).T.reshape(-1)
        tX = np.array([X] * nb_iter).reshape(-1)
    for i in range(nb_att):
        kws['color'] = cs[i]
        tYs = Ys[i].reshape(-1)  # .shape= (#num*nb_iter,)
        plt.scatter(tX, tYs, s=2.5, label=picked_keys[i], **kws)
        _sns_line_err_bars(ax, kws, tX, tYs)

    # bp = ax.boxplot()
    plt.legend(loc='best', frameon=True)
    ax.set_xlabel(annotX)
    ax.set_ylabel(annotY)
    ax.autoscale_view()
    fig = _setup_figsize(fig, figsize)
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


def multiple_scatter_comparison(X, Yss, zs, picked_keys,
                                annotX=r'hyper-pm',
                                annotY='Approximated value',
                                patch_artist=False, ddof=0,
                                cmap_name='Accent',
                                figsize='M-WS',  # scat
                                figname='hyperpm_bboxs'):
    # X  .shape= (#num,)
    # Yss.shape= (#att_sen, #iter, #num)
    # zs .shape= (#att_sen, #iter)
    # picked_keys.shape= (#att_sen,) list of names

    # nb_att, nb_iter, num = Yss.shape
    nb_att, nb_iter, _ = Yss.shape
    fig, ax = plt.subplots(figsize=_setup_config['M-NT'])
    cs = sns.color_palette(cmap_name)
    cl = len(cs)

    tX = np.array([X] * nb_iter).reshape(-1)
    for i in range(nb_att):
        kws = {'color': cs[i % cl], 'lw': 1}
        tYs = Yss[i].reshape(-1)
        plt.scatter(tX, tYs, s=2.5, **kws)
        kws['label'] = picked_keys[i]
        kws['linestyle'] = '--'
        _sns_line_err_bars(ax, kws, tX, tYs)

    tx_min, tx_max = ax.get_xlim()
    tz_avg = np.mean(zs, axis=1)            # (#att_sen,)
    # tz_std = np.std(zs, axis=1, ddof=ddof)  # (#att_sen,)
    for i in range(nb_att):
        kws = {'color': cs[i % cl], 'lw': 1}
        ax.plot([tx_min, tx_max], [tz_avg[i], tz_avg[i]], **kws)
        # '''
        # ax.fill_between([tx_min, tx_max],
        #             [tz_avg[i] - tz_std[i]] * 2,
        #             [tz_avg[i] + tz_std[i]] * 2,
        #             alpha=.15, facecolor=cs[i % cl])
        # '''

    ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    plt.legend(loc=PLT_LOCATION, frameon=PLT_FRAMEBOX)
    ax.set_xlabel(annotX)
    ax.set_ylabel(annotY)
    ax.autoscale_view()
    fig = _setup_figsize(fig, figsize)
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


# ===============================
# Python Matlablib plot


# -------------------------------
# fairmanf


# -------------------------------
# Linear regression with marginal distributions


def _internal_marg_dist_s1(ax1, df_all, col_X, current_palette):
    ax1.spines[:].set_linewidth(.4)  # 设置坐标轴线宽
    ax1.tick_params(width=.6, length=2.5, labelsize=8
                    )  # 设置坐标轴刻度的宽度与长度、数值刻度的字体
    sns.kdeplot(data=df_all, x=col_X, hue='learning',
                fill=True, common_norm=False, legend=False,
                palette=current_palette, alpha=.5,
                linewidth=.5, ax=ax1)  # 边缘分布图
    # ax1.set_xlim(-75, 1575)
    ax1.set_xticks([])
    ax1.set_xlabel("")
    ax1.set_yticks([])
    ax1.set_ylabel("")

    # ax1.spines['top'].set_visible(False)
    # ax1.spines['right'].set_visible(False)
    sns.despine(ax=ax1, bottom=True,
                top=True, right=True, left=True)
    return ax1


def _internal_marg_dist_s2(ax2, df_all, col_Y, current_palette):
    ax2.spines[:].set_linewidth(.4)
    ax2.tick_params(width=.6, length=2.5, labelsize=8)
    sns.kdeplot(data=df_all, y=col_Y, hue='learning',
                fill=True, common_norm=False, legend=False,
                palette=current_palette, alpha=.5,
                linewidth=.5, ax=ax2)
    # ax2.set_ylim(-10, 210)
    ax2.set_xticks([])
    ax2.set_xlabel("")
    ax2.set_yticks([])
    ax2.set_ylabel("")

    sns.despine(ax=ax2, left=True,
                right=True, top=True, bottom=True)
    return ax2


def _marginal_distrib_step1(grid, df_all, col_X, current_palette):

    # 4.1 绘制长度的边缘分布图
    ax1 = plt.subplot(grid[0, 0: 5])
    return _internal_marg_dist_s1(ax1, df_all, col_X, current_palette)


def _marginal_distrib_step2(grid, df_all, col_Y, current_palette):

    # 4.2 绘制宽度的边缘分布图
    ax2 = plt.subplot(grid[1: 6, 5])
    return _internal_marg_dist_s2(ax2, df_all, col_Y, current_palette)


def _marginal_distrib_step3(grid, dfs_pl, columns, col_X, col_Y,
                            # mycolor=None, annotX=r'X', annotY=r'Y'):
                            annotX=r'X', annotY=r'Y',  # mycolor=None,
                            # cmap_name='muted'):  # deep
                            mycolor=None, loc=None):
    _curr_sz = [30, 15, 15, 10, 10, 15]  # 20
    _curr_mark = ['*', '^', 'v', 'x', 'D', 'd']
    if len(columns) > 6:
        _curr_sz = [30, 15, 15, 10, 10, 10, 15, 15, 15]
        _curr_mark = ['*', '^', 'v', 'x', 'o', 'D', 'd', '<', '>']

    # 4.3 绘制二元分布图（散点图）
    ax3 = plt.subplot(grid[1: 6, 0: 5])
    ax3.spines[:].set_linewidth(.4)
    ax3.tick_params(width=.6, length=2.5, labelsize=8)
    ax3.grid(linewidth=.6, ls='-.', alpha=.4)

    for i, df in enumerate(dfs_pl):
        ax3.scatter(x=df[col_X], y=df[col_Y], s=_curr_sz[i], alpha=1,
                    marker=_curr_mark[i], color=mycolor[i],
                    edgecolors='w', linewidths=.5, label=columns[i])
    # _curr_font = 'Times New Roman'  # 'SimSun'
    _curr_font = plt.rcParams['font.family']
    legend_font = {'family': _curr_font, 'size': 8}
    _curr_nc = 1 if len(columns) <= 4 else 2

    loc_kws = {'loc': (.98, 1.01), 'frameon': False, 'ncol': _curr_nc
               # } if loc == None else {'loc': loc, 'frameon': True}
               } if loc is None else {'loc': loc, 'frameon': True}
    ax3.legend(
        prop=legend_font, labelspacing=.35,
        handleheight=1.2, handletextpad=0,
        columnspacing=.3, **loc_kws)
    del _curr_nc

    ax3.set_xlabel(annotX, fontsize=9, family=_curr_font, x=.55)
    ax3.set_ylabel(annotY, fontsize=9, family=_curr_font, y=.55)
    return ax3


def _marginal_distr_read_in(raw_df, col_X, col_Y, tag_Ys,
                            picked_keys):
    # 1. 读取数据
    dfs_pl = []

    # 2. 重组表格数据
    for i, tag_Y in enumerate(tag_Ys):
        tmp = raw_df[[col_X, tag_Y]].rename(columns={tag_Y: col_Y})
        tmp = tmp.apply(pd.to_numeric, errors='coerce')
        tmp['learning'] = picked_keys[i]
        dfs_pl.append(tmp)

    df_all = pd.concat(dfs_pl, axis=0).reset_index(drop=True)  # 合并表格
    return dfs_pl, df_all


def scatter_with_marginal_distrib(df, col_X, col_Y, tag_Ys,
                                  picked_keys,
                                  annotX='acc', annotY='fair',
                                  cmap_name='muted',
                                  figsize='M-WS', figname='smd'):
    # X .shape= (#num,)
    # Ys.shape= (#num, #baseline_for_comparison)
    # columns, i.e., picked_keys.shape= (#baseline,)
    # num, nb_way = Ys.shape

    dfs_pl, df_all = _marginal_distr_read_in(
        df, col_X, col_Y, tag_Ys, picked_keys)

    # 3. 设置seaborn颜色格式
    current_palette = sns.color_palette(cmap_name, len(picked_keys))
    sns.palplot(current_palette)

    # 4. 开始绘图
    fig = plt.figure(figsize=_setup_config[figsize], dpi=300)
    plt.subplots_adjust(left=.11, bottom=.11, right=.98, top=.995)
    grid = plt.GridSpec(6, 6, wspace=.05, hspace=.05)
    # columns = picked_keys

    # ax1, ax2, ax3 =
    _marginal_distrib_step1(grid, df_all, col_X, current_palette)
    _marginal_distrib_step2(grid, df_all, col_Y, current_palette)
    _marginal_distrib_step3(grid, dfs_pl, picked_keys, col_X, col_Y,
                            annotX, annotY, current_palette)

    # 5. 保存图片
    _setup_figshow(fig, figname=figname)
    plt.close()
    return


# -------------------------------
# Linear regression with marginal distributions


def _marginal_distr_step4(grid, dfs_pl, columns, col_X, col_Y,
                          annotX, annotY, mycolor, snspec='sty1',
                          identity=None, distrib=True,
                          curr_legend_nb_split=4):  # or 6
    _curr_sz = [30, 15, 15, 10, 10, 15]  # 20
    _curr_mk = ['*', '^', 'v', 'x', 'D', 'd']  # mark
    if len(columns) > 6:
        # _curr_sz = [30, 15, 15, 10, 10, 10, 15, 15, 15]
        # _curr_mk = ['*', '^', 'v', 'x', 'o', 'D', 'd', '<', '>']
        _curr_mk = ['D', '^', 'v', 'o', '*', 's', 'd', '<', '>']  # 'p'
        _curr_sz = [10, 15, 15, 12, 29, 11, 15, 15, 15]
    # _curr_mc = ['w'] * 3 + [None] + [''] * 5

    ax4 = plt.subplot(grid[1: 6, 0: 5])
    ax4.spines[:].set_linewidth(.4)
    ax4.tick_params(width=.6, length=2.5, labelsize=8)
    if distrib:
        ax4.grid(linewidth=.6, ls='-.', alpha=.4)

    # if snspec:
    if snspec == 'sty1':  # 's1', 'sns1'
        for i, df in enumerate(dfs_pl):
            tX = df[col_X].values.astype(DTY_FLT)
            tY = df[col_Y].values.astype(DTY_FLT)
            R = np.corrcoef(tX, tY)[1, 0]
            key = 'Correlation = %.4f' % R
            # regr = np.polyfit(tX, tY, deg=1)
            # estimated = np.polyval(regr, tX)

            ax4.scatter(x=df[col_X], y=df[col_Y], s=_curr_sz[i],
                        alpha=1,  # edgecolors=_curr_mc[i],
                        marker=_curr_mk[i], color=mycolor[i],
                        edgecolors='w', linewidths=.5,
                        label='{:4s} {}'.format(columns[i], key))
            # kws = {'color': mycolor[i], 'lw': .87, 'alpha': 1}
            # _sns_line_err_bars(ax4, kws, tX, tY)
            if identity is None:
                _sns_line_err_bars(ax4, {'color': mycolor[
                    i], 'lw': .87, 'alpha': 1}, tX, tY)

    elif snspec.startswith('sty5'):
        tmp_Xs, tmp_Ys = [], []
        for i, df in enumerate(dfs_pl):
            tX = df[col_X].values.astype(DTY_FLT)
            tY = df[col_Y].values.astype(DTY_FLT)
            R = np.corrcoef(tX, tY)[1, 0]
            key = 'Correlation = %.4f' % R
            ax4.scatter(x=df[col_X], y=df[col_Y], s=_curr_sz[i],
                        alpha=1,
                        marker=_curr_mk[i], color=mycolor[i],
                        edgecolors='w', linewidths=.5,
                        label='{:4s} {}'.format(columns[i], key))
            tmp_Xs.append(tX)
            tmp_Ys.append(tY)

        for i, (tX, tY) in enumerate(zip(tmp_Xs, tmp_Ys)):
            kws = {'color': mycolor[i], 'lw': .87, 'alpha': 1}
            _sns_line_err_bars(ax4, kws, tX, tY)
        del tX, tY, tmp_Xs, tmp_Ys
        if identity is not None:
            tx_min, tx_max = ax4.get_xlim()
            ax4.plot([tx_min, tx_max], [tx_min, tx_max], 'k--', lw=.5,
                     label=identity)  # label=r'identity')
            del tx_min, tx_max

    elif snspec == 'sty4a':
        tmp_Xs, tmp_Ys = [], []
        for i, df in enumerate(dfs_pl):
            tX = df[col_X].values.astype(DTY_FLT)
            tY = df[col_Y].values.astype(DTY_FLT)
            ax4.scatter(
                x=df[col_X], y=df[col_Y], s=_curr_sz[i], alpha=1,
                marker=_curr_mk[i], color=mycolor[i],
                edgecolors='w', linewidths=.5, label=columns[i])
            tmp_Xs.append(tX)
            tmp_Ys.append(tY)
        for i, (tX, tY) in enumerate(zip(tmp_Xs, tmp_Ys)):
            kws = {'color': mycolor[i], 'lw': .87, 'alpha': .78}
            _sns_line_err_bars(ax4, kws, tX, tY)
        del tX, tY, tmp_Xs, tmp_Ys
        if identity is not None:
            tx_min, tx_max = ax4.get_xlim()
            ax4.plot([tx_min, tx_max], [tx_min, tx_max], 'k--', lw=.5,
                     label=identity)  # label=r'identity')
            del tx_min, tx_max
    elif snspec == 'sty4b':
        tmp_Xs, tmp_Ys = [], []
        for i, df in enumerate(dfs_pl):
            tX = df[col_X].values.astype(DTY_FLT)
            tY = df[col_Y].values.astype(DTY_FLT)
            ax4.scatter(
                x=df[col_X], y=df[col_Y], s=_curr_sz[i], alpha=1,
                marker=_curr_mk[i], color=mycolor[i],
                edgecolors='w', linewidths=.5, label=columns[i])
            tmp_Xs.append(tX)
            tmp_Ys.append(tY)
        del tX, tY, tmp_Xs, tmp_Ys
        if identity is not None:
            tx_min, tx_max = ax4.get_xlim()
            ax4.plot([tx_min, tx_max], [tx_min, tx_max], 'k--', lw=.5,
                     label=identity)  # label=r'identity')
            del tx_min, tx_max

    elif snspec == 'sty3':
        for i, df in enumerate(dfs_pl):
            sns.regplot(x=col_X, y=col_Y, data=df, label=columns[i],
                        marker=_curr_mk[i], color=mycolor[i],
                        line_kws={'lw': .87},
                        scatter_kws={'s': _curr_sz[i] / 4})

    elif snspec == 'sty2':
        for i, df in enumerate(dfs_pl):
            tX = df[col_X].values.astype(DTY_FLT)
            tY = df[col_Y].values.astype(DTY_FLT)
            R = np.corrcoef(tX, tY)[1, 0]
            key = 'Correlation = %.4f' % R
            # regr = np.polyfit(tX, tY, deg=1)
            # estimated = np.polyval(regr, tX)
            ax4.scatter(tX, tY,
                        label='{:4s} {}'.format(columns[i], key),
                        s=_curr_sz[i] / 4, marker=_curr_mk[i],
                        color=mycolor[i])
            kws = {'color': mycolor[i], 'lw': .87, 'alpha': 1}
            _sns_line_err_bars(ax4, kws, tX, tY)

    _curr_ft = plt.rcParams['font.family']  # 'Times New Roman'
    legend_font = {'family': _curr_ft, 'size': 7}
    _curr_lc = (.98, 1.01) if distrib else PLT_LOCATION
    _curr_nc = 1 if len(columns) <= curr_legend_nb_split else 2
    # _curr_nc = 1 if len(columns) <= 4 else 2  # <=4,6
    if snspec == 'sty5b':
        _curr_nc = 1
    _curr_kw = {'frameon': False}
    if not distrib:
        _curr_kw['frameon'] = True
        _curr_kw['framealpha'] = .7
    if _curr_nc >= 2:
        _curr_kw['columnspacing'] = .4
    ax4.legend(
        prop=legend_font, labelspacing=.35, handleheight=1.2,
        handletextpad=0, loc=_curr_lc, ncol=_curr_nc,
        **_curr_kw)  # mode='expand',
    del _curr_lc, _curr_nc, _curr_kw
    if not distrib:
        _style_set_axis(ax4)

    ax4.set_xlabel(annotX, fontsize=9, family=_curr_ft, x=.55)
    ax4.set_ylabel(annotY, fontsize=9, family=_curr_ft, y=.55)
    return ax4


def _marginal_distr_step5(grid, dfs_pl, col_X, col_Y, mycolor):
    if col_X is not None:

        # 4.1 绘制长度的边缘分布图
        ax1 = plt.subplot(grid[0, 0: 5])
        ax1.spines[:].set_linewidth(.4)  # 设置坐标轴线宽
        ax1.tick_params(width=.6, length=2.5, labelsize=8
                        )  # 设置坐标轴刻度的宽度与长度、数值刻度的字体
        for i, df in enumerate(dfs_pl):
            sns.kdeplot(data=df, x=col_X, fill=True,
                        common_norm=False, legend=False, color=mycolor[i],
                        alpha=.5, linewidth=.5, ax=ax1)  # 边缘分布图
        ax1.set_xticks([])
        ax1.set_xlabel("")
        ax1.set_yticks([])
        ax1.set_ylabel("")
        sns.despine(ax=ax1, left=True, bottom=True)

    else:
        ax1 = None
    if col_Y is not None:

        # 4.2 绘制宽度的边缘分布图
        ax2 = plt.subplot(grid[1: 6, 5])
        ax2.spines[:].set_linewidth(.4)
        ax2.tick_params(width=.6, length=2.5, labelsize=8)
        for i, df in enumerate(dfs_pl):
            sns.kdeplot(
                data=df, y=col_Y, fill=True,
                common_norm=False, legend=False, color=mycolor[i],
                alpha=.5, linewidth=.5, ax=ax2)
        ax2.set_xticks([])
        ax2.set_xlabel("")
        ax2.set_yticks([])
        ax2.set_ylabel("")
        sns.despine(ax=ax2, left=True, bottom=True)

    else:
        ax2 = None
    return ax1, ax2


def line_reg_with_marginal_distr(df, col_X, col_Y, tag_Ys,
                                 picked_keys,
                                 annotX='acc', annotY='fair',
                                 invt_a=False, snspec='sty0',
                                 distrib=True,
                                 cmap_name='muted',
                                 figsize='M-WS', figname='smd',
                                 identity=None,
                                 curr_legend_nb_split=4):
    dfs_pl, df_all = _marginal_distr_read_in(
        df, col_X, col_Y, tag_Ys, picked_keys)
    mycolor = sns.color_palette(cmap_name, len(picked_keys))

    fig = plt.figure(figsize=_setup_config[figsize], dpi=300)
    plt.subplots_adjust(left=.11, bottom=.11, right=.98, top=.995)
    grid = plt.GridSpec(6, 6, wspace=.05, hspace=.05)

    if invt_a:
        col_X, col_Y = col_Y, col_X
        annotX, annotY = annotY, annotX
    if distrib:
        # ax1 = _marginal_distrib_step1(grid, df_all, col_X, mycolor)
        # ax2 = _marginal_distrib_step2(grid, df_all, col_Y, mycolor)
        _marginal_distrib_step1(grid, df_all, col_X, mycolor)
        _marginal_distrib_step2(grid, df_all, col_Y, mycolor)
    if snspec == 'sty0':
        # ax3 =
        _marginal_distrib_step3(grid, dfs_pl, picked_keys,
                                col_X, col_Y, annotX, annotY,
                                mycolor)  # , distrib=distrib)
    elif snspec in ['sty1', 'sty2', 'sty3',  # 'sty4', 'sty5',
                    'sty6', 'sty4a', 'sty4b', 'sty5a', 'sty5b']:
        # ax4 =
        _marginal_distr_step4(
            grid, dfs_pl, picked_keys, col_X, col_Y,
            annotX, annotY, mycolor, snspec,
            identity=identity, distrib=distrib,
            curr_legend_nb_split=curr_legend_nb_split)

    _setup_figshow(fig, figname=figname)
    plt.close(fig)
    return


def _marginal_distr_step7a(ax4, dfs_pl, columns, col_X, col_Y,
                           mycolor, snspec, curr_key=True, distrib=False
                           ):  # subfig=''):
    #                     , curr_legend_nb_split=5):
    # ax4 = plt.subplot(grid[1:6, 0:5])
    ax4.spines[:].set_linewidth(.4)
    ax4.tick_params(width=.6, length=2.5, labelsize=8)
    if distrib:
        ax4.grid(linewidth=.6, ls='-.', alpha=.4)
    _curr_sz = [22, 19, 25, 14, 19]  # [21, 16, 19, 20, 23]
    _curr_mk = ['o', 's', 'd', 's', 'd']

    tmp_Xs, tmp_Ys = [], []
    for i, df in enumerate(dfs_pl):
        tX = df[col_X].values.astype(DTY_FLT)
        tY = df[col_Y].values.astype(DTY_FLT)
        tw = {'color': mycolor[i], 'edgecolors': 'w'}
        if snspec.startswith('sty5'):
            if not (0 <= i < 3):  # i > 2:  # 0 < i < 3:
                tw = {'color': 'w', 'edgecolors': mycolor[i]}
            # R =  np.corrcoef(tX,tY)[1,0]  # % R
            key = 'Correlation=%.4f' % Pearson_correlation(tX, tY)[0]
            key = f'{key}{"":3s}{columns[i] if curr_key else ""}'
            ax4.scatter(x=df[col_X], y=df[col_Y], linewidths=.5,
                        alpha=1 if i > 0 else .7, s=_curr_sz[i],
                        marker=_curr_mk[i], label=key, **tw)
        elif snspec.startswith('sty4'):
            if (i == 1 and snspec == 'sty4b') or (
                    i == 0 and snspec == 'sty4a'):
                tw = {'color': 'w', 'edgecolors': mycolor[i]}
            ax4.scatter(x=df[col_X], y=df[col_Y], linewidths=.5,
                        alpha=1 if i > 0 else .7, s=_curr_sz[i],
                        marker=_curr_mk[i], label=(
                            columns[i] if curr_key else ""), **tw)
            # if subfig:
            #     ax4.set_xlabel(subfig)
        tmp_Xs.append(tX)
        tmp_Ys.append(tY)
    if snspec.startswith('sty5'):
        for i, (tX, tY) in enumerate(zip(tmp_Xs, tmp_Ys)):
            kws = {'color': mycolor[i], 'lw': .87, 'alpha': 1}
            _sns_line_err_bars(ax4, kws, tX, tY)
        # del tX, tY, tmp_Xs, tmp_Ys, tw
    del tmp_Xs, tmp_Ys, tX, tY, tw
    return ax4


def _marginal_distr_step7b(ax4, annotX, annotY, _curr_lc=(1.01, .78),
                           identity=None, distrib=False, handles=None,
                           pad=3):  # default:4
    if identity is not None:
        tx_min, tx_max = ax4.get_xlim()
        ax4.plot([tx_min, tx_max], [tx_min, tx_max], 'k--',
                 lw=.5, label=identity)
        del tx_min, tx_max

    _curr_ft = plt.rcParams['font.family']  # 'Times New Roman'
    legend_font = {'family': _curr_ft, 'size': 7}
    _curr_nc = 1  # if len(columns) <= curr_legend_nb_split else 2
    _curr_kw = {'frameon': (not distrib), 'framealpha': .7}
    if _curr_nc >= 2:
        _curr_kw['columnspacing'] = .4
    # _curr_lc = (1.01, .78) if distrib else PLT_LOCATION
    ax4.legend(prop=legend_font, labelspacing=.35, handleheight=1.2,
               handletextpad=0, loc=_curr_lc, ncol=_curr_nc,
               handles=handles, **_curr_kw)  # handles=[],
    del _curr_nc, _curr_kw, _curr_lc
    if not distrib:
        _style_set_axis(ax4)
    ax4.set_xlabel(annotX, fontsize=9, family=_curr_ft, x=.55, labelpad=pad)
    ax4.set_ylabel(annotY, fontsize=9, family=_curr_ft, y=.55)
    return ax4


def linreg_w_marg_dist_revised(
        df, col_X, col_Y, tag_Ys, picked_keys, figname='smd',
        annotX='acc', annotY='fair', snspec='sty5b',
        distrib=True, curr_key=False, invt_a=False,
        palette_X = ('#00a087',) * 5, palette_Y=(
            'black',
            '#066190', '#C42238', '#024163', '#8E0F31',
            '#77AECD', '#D98380', '#066190', '#C42238')):
    dfs_pl, df_all = _marginal_distr_read_in(
        df, col_X, col_Y, tag_Ys, picked_keys)
    fig = plt.figure(figsize=_setup_config['M-WS'], dpi=300)
    plt.subplots_adjust(left=.11, bottom=.11, right=.98, top=.995)
    grid = plt.GridSpec(6, 6, wspace=.05, hspace=.05)
    if invt_a:
        col_X, col_Y = col_Y, col_X
        annotX, annotY = annotY, annotX
        # palette_X, palette_Y = palette_Y, palette_X

    # palette_Y = palette_Y[: len(picked_keys)]
    # palette_Y = [palette_Y[i] for i in [0, 3, 4, 7, 8]]
    palette_Y = [palette_Y[i] for i in [0, 1, 2, 5, 6]]
    if distrib and invt_a:
        _marginal_distrib_step1(grid, df_all, col_X, palette_Y)
        if curr_key:
            _marginal_distrib_step2(grid, df_all, col_Y, palette_X)
    elif distrib:
        _marginal_distrib_step1(grid, df_all, col_X, palette_X)
        _marginal_distrib_step2(grid, df_all, col_Y, palette_Y)
    # _marginal_distr_step4(grid, dfs_pl, picked_keys, col_X, col_Y,
    #                       annotX, annotY, palette, 'sty5b',
    #                       identity=None, distrib=True,
    #                       curr_legend_nb_split=4)
    # _marginal_distr_step7(ax4, dfs_pl, picked_keys, col_X, col_Y,
    #                       annotX, annotY, palette_Y,
    #                       # if not invt_a else palette_X,
    #                       snspec, distrib=distrib, curr_key=curr_key)
    ax4 = plt.subplot(grid[1:6, 0:5])
    ax4 = _marginal_distr_step7a(ax4, dfs_pl, picked_keys, col_X, col_Y,
                                 palette_Y, snspec, curr_key=curr_key,
                                 distrib=distrib)
    ax4 = _marginal_distr_step7b(ax4, annotX, annotY, distrib=distrib)
    _setup_figshow(fig, figname=figname)
    return


def _rev_sup_pv1_wo_subtit(fig, df, existing, square_pm, *, share, lgth=3):
    pick, col_Xs, col_Y, tag_Yss, picked_keys = existing
    palette_X, palette_Y, antX, antYs, snspec, distrib = square_pm
    nk, ny = len(pick), len(antYs)  # wspace=.315, hspace=.25)

    grid = plt.GridSpec(ny * 5 + 1, nk * 5 + 1, wspace=.235, hspace=.25)
    row_refs, col_refs = [None] * ny, [None] * nk
    for ir in range(ny):
        for ic in range(nk):
            pk = pick[ic]
            dfs_pl, df_all = _marginal_distr_read_in(
                df, col_Xs[pk], col_Y, tag_Yss[ir], picked_keys)
            if distrib and ir == 0:
                ax00 = fig.add_subplot(grid[0, ic * 5: ic * 5 + 5])       # plt.subplot
                _internal_marg_dist_s1(ax00, df_all, col_Xs[pk], palette_X)
            if distrib and ic == nk - 1:
                ax11 = fig.add_subplot(grid[ir * 5 + 1: ir * 5 + 6, -1])  # plt.subplot
                _internal_marg_dist_s2(ax11, df_all, col_Y, palette_Y)
            # ax55 = plt.subplot(grid[ir * 5 + 1: ir * 5 + 6, ic * 5: ic * 5 + 5])

            sharey_ax, sharex_ax = row_refs[ir], col_refs[ic]
            kws = dict(sharex=sharex_ax, sharey=sharey_ax) if share else {}
            ax55 = fig.add_subplot(grid[ir * 5 + 1:ir * 5 + 6, ic * 5:ic * 5 + 5], **kws)
            if row_refs[ir] is None:
                row_refs[ir] = ax55
            if col_refs[ic] is None:
                col_refs[ic] = ax55
            if ic > 0:
                ax55.tick_params(labelleft=False)
            if ir < ny - 1:
                ax55.tick_params(labelbottom=False)

            ax55 = _marginal_distr_step7a(
                ax55, dfs_pl,  # picked_keys if gap else (..),
                # picked_keys[:3] if ir == 0 else picked_keys[-3:],
                picked_keys[:lgth] if ir == 0 else picked_keys[-lgth:],
                col_Xs[pk], col_Y, palette_Y,
                # ax55, dfs_pl, picked_keys, col_Xs[pk], col_Y, palette_Y,
                snspec, curr_key=ic == nk - 1, distrib=distrib)
            ax55 = _marginal_distr_step7b(
                ax55, antX[pk] if ir == ny - 1 else '',
                # subfig_ind(ir * nk + ic) + f"\n{antX[pk]}" * (ir == ny - 1),
                antYs[ir] if ic == 0 else '', distrib=distrib,
                _curr_lc=(1.11, .76))  # =(1.11, .86))

    del pick, col_Xs, col_Y, tag_Yss, picked_keys
    del palette_X, palette_Y, antX, antYs, snspec, distrib
    return fig


def _rev_sup_pv1_w_subtit(fig, df, existing, square_pm, *, share, lgth=3,
                          tit='bottom',  # ('top','bottom',False,'right'):#True,
                          start_pt_ind=0):
    pick, col_Xs, col_Y, tag_Yss, picked_keys = existing
    palette_X, palette_Y, antX, antYs, snspec, distrib = square_pm
    nk, ny = len(pick), len(antYs)  # wspace=.335, hspace=.35)

    ttp = 2 if tit in ['bottom', 'right'] else 3  # 2 + int(tit not in['b','r'])
    tt = 9  # 10
    grid = plt.GridSpec(ny * (tt + 1) + ttp, nk * tt + ttp,  # nk * tt + 2
                        wspace=.54 - .021 * (tit == 'bottom'), hspace=.31)
    # grid = plt.GridSpec(ny * 6 + 1, nk * 5 + 1, wspace=.135, hspace=.15)
    row_refs, col_refs = [None] * ny, [None] * nk
    for ir in range(ny):
        for ic in range(nk):
            pk = pick[ic]
            dfs_pl, df_all = _marginal_distr_read_in(
                df, col_Xs[pk], col_Y, tag_Yss[ir], picked_keys)
            tt_cc = ic * tt              # ic * tt + tt
            tt_rr = ir * (tt + 1) + ttp  # (ir + 1) * (tt + 1)

            if distrib and ir == 0 and start_pt_ind == 0:
                ax00 = fig.add_subplot(grid[0:2, tt_cc: tt_cc + tt])
                # ax00 = fig.add_subplot(grid[0, ic * 5: ic * 5 + 5])
                _internal_marg_dist_s1(ax00, df_all, col_Xs[pk], palette_X)
            if distrib and ic == nk - 1:
                ax11 = fig.add_subplot(grid[tt_rr:tt_rr + tt, -ttp:])  # -2:])
                # ax11 = fig.add_subplot(grid[ir * 6 + 1:ir * 6 + 6, -1])
                _internal_marg_dist_s2(ax11, df_all, col_Y, palette_Y)

            sharey_ax, sharex_ax = row_refs[ir], col_refs[ic]
            kws = dict(sharex=sharex_ax, sharey=sharey_ax) if share else {}
            ax55 = fig.add_subplot(grid[tt_rr:tt_rr + tt, tt_cc:tt_cc + tt], **kws)
            # ax55 = fig.add_subplot(grid[ir * 6 + 1: ir * 6 + 6, ic * 5:ic * 5 + 5], **kws)
            if not tit:
                ax55.tick_params(axis='x', which='major', pad=0)
                ax55.tick_params(axis='y', which='major', pad=1)
                ax55.xaxis.set_ticks_position('top')

            if row_refs[ir] is None:
                row_refs[ir] = ax55
            if col_refs[ic] is None:
                col_refs[ic] = ax55
            if ic > 0:
                ax55.tick_params(labelleft=False)
            if tit and ir < ny - 1:
                ax55.tick_params(labelbottom=False)
            if not tit and ir > 0:
                ax55.tick_params(labeltop=False)

            tt_curr = f"{antX[pk]}" * (ir == ny - 1)
            tt_curr = f"{tt_curr:8s}\n"  # :20s :8s
            tt_ind = subfig_ind(ir * nk + ic + start_pt_ind) + "\n"
            if tit == 'right':
                tt_curr = tt_ind.strip() + " " + tt_curr  # =tt_ind*(tit!='top')+tt_curr
            else:
                tt_curr = tt_ind * (not tit) + tt_curr + tt_ind.strip() * (tit == 'bottom')
            # tt_curr = (subfig_ind(ir * nk + ic) + "\n") * (tit not in [
            #     'top', 'bottom']) + tt_curr + "\n" + (
            #     subfig_ind(ir * nk + ic)) * (tit == 'bottom')  # tit!='top'
            ax55 = _marginal_distr_step7a(
                ax55, dfs_pl,
                picked_keys[:lgth] if ir == 0 else picked_keys[-lgth:],
                col_Xs[pk], col_Y, palette_Y,
                snspec, curr_key=ic == nk - 1, distrib=distrib)
            ax55 = _marginal_distr_step7b(
                ax55, tt_curr.strip(), antYs[ir] * (ic == 0),
                distrib=distrib, _curr_lc=(1.11, .76), pad=2 * (
                    not tit) + 3 * (tit == 'top') + 1 * (tit == 'bottom'),
                handles=None if ic == nk - 1 else [])
            if tit == 'top':
                ax55.set_title(subfig_ind(ir * nk + ic), pad=3, fontsize=9)
            ax55.tick_params(axis='both', labelsize=7.8, pad=.6)  # labelsize=7)

    del tt_cc, tt_rr, tt, ttp, tt_curr, tt_ind
    del pick, col_Xs, col_Y, tag_Yss, picked_keys
    del palette_X, palette_Y, antX, antYs, snspec, distrib
    return fig


def linreg_w_marg_dist_rev_sup_pv1(
        df, col_Y, pick, col_Xs, tag_Yss, picked_keys, figname='smd',
        snspec='sty4b', antYs=('fair',), antX=('acc',), distrib=True,
        palette_X=('#F0AE97',) * 3,  # gap=False, #curr_key=False,invt_a=False,
        palette_Y=('#168E6A', ) + ('#1E827B', '#586395') * 4,
        subfig=False, start_pt_ind=0):
    # from mpl_toolkits.axes_grid1 import ImageGrid
    # fig = plt.figure(figsize=(15, 10), dpi=300)
    # grid = ImageGrid(fig, 111, nrows_ncols=(len(antYs), nk),
    #                  axes_pad=.15, cbar_location='right',
    #                  cbar_mode='single', cbar_size='7%', cbar_pad=.15)
    # for ik, ax in enumerate(grid):
    #     ir, ic = ik // nk, ik % nk

    fig = plt.figure(figsize=(7.8, 4.27 if subfig else 4.01), dpi=300)
    plt.subplots_adjust(left=.11, bottom=.11, right=.98, top=.995)

    # '' '
    # nk, ny = len(pick), len(antYs)
    # from matplotlib.gridspec import GridSpecFromSubplotSpec
    # fig, grid = plt.subplots(ny, nk, figsize=(11, 6.8), constrained_layout=True)
    # # fig = plt.figure(figsize=(10, 8), dpi=300)
    # # outer = fig.add_gridspec(ny, nk)
    # # 用来存储每行/每列的参考轴 (主轴)
    # row_refs = [None] * ny  # 每一行一个 y参考
    # col_refs = [None] * nk  # 每一列一个 x参考
    # for ir in range(ny):
    #     for ic in range(nk):
    #         dfs_pl, df_all = _marginal_distr_read_in(
    #             df, col_Xs[pick[ic]], col_Y, tag_Yss[ir], picked_keys)
    #
    #         grid[ir, ic].remove()  # 1. 先删除原来的占位子图
    #         curr_r, curr_c = 6 - int(ir != 0), 6 - int(ic != nk - 1)
    #         # 2. 在原位置创建一个新的 GridSpec
    #         gs = GridSpecFromSubplotSpec(curr_r, curr_c,
    #                                      subplot_spec=fig.add_gridspec(ny, nk)[
    #                                          ir, ic], wspace=0.05, hspace=0.05)
    #         # inner = outer[ir, ic].subgridspec(6, 6, wspace=.05, hspace=.05)
    #         # 3. 在这个 6x6 网格中创建子图
    #         # ax00 = fig.add_subplot(inner[0, 0:5])  # gs[0, 0])
    #         # ax11 = fig.add_subplot(inner[1:6, 5])  # gs[1:3, 1:4])
    #         # ax55 = fig.add_subplot()  # gs[5, :])  # ax55.axis('off')
    #         # 你可以继续在 gs[...] 中添加更多子图
    #
    #         # 这里我们约定：每个cell里有一个主轴，用来参与sharex/sharey，比如就用右下角那个
    #         main_r, main_c = curr_r - 1, curr_c - 1
    #         # 找到这一行/列的参考轴（如果已有的话）
    #         sharey_ax, sharex_ax = row_refs[ir], col_refs[ic]
    #
    #         if distrib and ir == 0:
    #             ax00 = fig.add_subplot(gs[0, 0:5])  # gs[0, 0:5])
    #             _internal_marg_dist_s1(ax00, df_all, col_Xs[pick[ic]], palette_Y)
    #         if distrib and ic == nk - 1:
    #             ax11 = fig.add_subplot(gs[int(ir == 0):, 5])  # gs[1:6, 5])
    #             _internal_marg_dist_s2(ax11, df_all, col_Y, palette_Y)
    #         ax55 = fig.add_subplot(gs[int(ir == 0):, 0:5],  # gs[1:6, 0:5])
    #                                sharex=sharex_ax, sharey=sharey_ax)
    #
    #         # 如果这一行还没有y参考轴，就把当前主轴记为参考
    #         # 如果这一列还没有x参考轴，就把当前主轴记为参考
    #         if row_refs[ir] is None:
    #             row_refs[ir] = ax55  # ax_main
    #         if col_refs[ic] is None:
    #             col_refs[ic] = ax55  # ax_main
    #         # 关键：隐藏重复 tick labels
    #         if ic > 0:
    #             ax55.tick_params(labelleft=False)
    #         if ir < ny - 1:
    #             ax55.tick_params(labelbottom=False)
    #
    #         ax55 = _marginal_distr_step7a(
    #             ax55, dfs_pl, picked_keys, col_Xs[pick[ic]], col_Y,
    #             palette_Y, snspec, curr_key=ic == nk - 1, distrib=distrib)
    #         ax55 = _marginal_distr_step7b(
    #             ax55, antX[pick[ic]] if ir == ny - 1 else '',
    #             antYs[ir] if ic == 0 else '', distrib=distrib,
    #             _curr_lc=(.51, .81))
    #     # if ir == 0:
    #     #     _marginal_distrib_step1(grid, df_all=, colx)
    # '' '

    existing = [pick, col_Xs, col_Y, tag_Yss, picked_keys]
    sqr_pm = [palette_X, palette_Y[:3], antX, antYs, snspec, distrib]
    if not subfig:
        fig = _rev_sup_pv1_wo_subtit(fig, df, existing, sqr_pm, share=True)
    else:
        fig = _rev_sup_pv1_w_subtit(fig, df, existing, sqr_pm, share=True,
                                    start_pt_ind=start_pt_ind)
    # pdb.set_trace()

    _setup_figshow(fig, figname=figname)
    return


def linreg_w_marg_dist_rev_sup_pv2(
        df, col_Y, pick, col_Xs, tag_Yss, picked_keys, figname='smd',
        snspec='sty4b', antYs=('fair',), antX=('acc',), distrib=True,
        palette_X=('#00a087',) * 5, palette_Y=(  # gap=True,
            'black', '#066190', '#C42238', '#024163', '#8E0F31',
            '#77AECD', '#D98380', '#066190', '#C42238'),
        subfig=True, start_pt_ind=0):  # starting_point_index
    nk, ny = len(pick), len(antYs)
    fig = plt.figure(figsize=(9.9, 6.7 if subfig else 5.76), dpi=300)
    plt.subplots_adjust(left=.11, bottom=.11, right=.98, top=.995)
    ttd, tt, ttp = (5, 3, 1) if not subfig else (10, 6, 3)  # (7, 4, 2)
    grid = plt.GridSpec(ny * (ttd + int(subfig)) + 1 + 1, nk * (ttd + tt),
                        wspace=.364 - .1 * subfig, hspace=.3 - .1 * subfig
                        )  # wspace=.364,hspace=.32)
    row_refs, col_refs = [None] * ny, [None] * nk
    for ir in range(ny):
        for ic in range(nk):
            pk = pick[ic]
            dfs_pl, df_all = _marginal_distr_read_in(
                df, col_Xs[pk], col_Y, tag_Yss[ir], picked_keys)
            tt_cc = ic * (ttd + tt)                 # ic * 8: ic * 8 + 5
            tt_rr = ir * (ttd + int(subfig)) + ttp  # ir * 5 + 1: ir * 5 + 6

            if distrib and ir == 0 and start_pt_ind == 0:
                ax00 = plt.subplot(grid[:ttp, tt_cc: tt_cc + ttd])
                # ax00 = plt.subplot(grid[0, ic * 8: ic * 8 + 5])
                _internal_marg_dist_s1(ax00, df_all, col_Xs[pk], palette_X)
            if distrib and ic == nk - 1:
                ax11 = plt.subplot(grid[tt_rr: tt_rr + ttd, -tt:-tt + ttp])
                # ax11 = plt.subplot(grid[ir * 5 + 1: ir * 5 + 6, -3])
                _internal_marg_dist_s2(ax11, df_all, col_Y, palette_Y)
            # ax55 = plt.subplot(grid[ir * 5 + 1:ir * 5 + 6, ic * 8:ic * 8 + 5])
            sharey_ax, sharex_ax = row_refs[ir], col_refs[ic]
            ax55 = fig.add_subplot(
                grid[tt_rr:tt_rr + ttd, tt_cc:tt_cc + ttd],
                # grid[ir * 5 + 1:ir * 5 + 6, ic * 8:ic * 8 + 5],
                sharex=sharex_ax, sharey=sharey_ax)
            if row_refs[ir] is None:
                row_refs[ir] = ax55
            if col_refs[ic] is None:
                col_refs[ic] = ax55
            if ic > 0:
                ax55.tick_params(labelleft=False)
            if ir < ny - 1:
                ax55.tick_params(labelbottom=False)

            tt_curr = f"\n{antX[pk]}" * (ir == ny - 1)
            tt_curr = tt_curr + ("\n" + subfig_ind(
                ir * nk + ic + start_pt_ind)) * subfig
            ax55 = _marginal_distr_step7a(
                ax55, dfs_pl, picked_keys, col_Xs[pk], col_Y, palette_Y,
                snspec, curr_key=ic == nk - 1, distrib=distrib)
            ax55 = _marginal_distr_step7b(
                ax55, tt_curr.strip(),  # antX[pk] if ir == ny - 1 else '',
                # subfig_ind(ir * nk + ic) + f"\n{antX[pk]}" * (ir == ny - 1),
                antYs[ir] if ic == 0 else '', distrib=distrib,
                # _curr_lc=(1.04,.61), (1.11,.76))
                _curr_lc =((1.09 if ic == nk - 1 else .98) - 0.012 * subfig,
                           0.56 + 0.02 * subfig), pad=4 * (not subfig))
            if subfig:
                ax55.tick_params(axis='both', pad=.6)
    del tt_curr, tt, ttd, tt_cc, tt_rr
    _setup_figshow(fig, figname=figname)
    return


def single_line_reg_with_distr(X, Y, annots=('X', 'Y', 'Z'),
                               figname='sing_linreg', figsize='M-WS',
                               linreg=False, distrib=False,
                               snspec='sty2', cmap_name='coolwarm',
                               sci_format_y=False):
    # mycolor = sns.color_palette(cmap_name)

    fig = plt.figure(figsize=_setup_config[figsize], dpi=300)
    plt.subplots_adjust(left=.11, bottom=.11, right=.98, top=.995)
    grid = plt.GridSpec(6, 6, wspace=.05, hspace=.05)

    if distrib:
        df = pd.DataFrame({'x': X, 'y': Y})
        kwd = {'fill': True, 'common_norm': False, 'legend': False,
               'alpha': .5, 'linewidth': .5}  # 'palette': mycolor,

        # def _marginal_distrib_step1:
        ax1 = plt.subplot(grid[0, 0: 5])
        ax1.spines[:].set_linewidth(.4)
        ax1.tick_params(width=.6, length=2.5, labelsize=8)
        sns.kdeplot(data=df, x='x', ax=ax1, **kwd)
        ax1.set_xticks([])
        ax1.set_xlabel("")
        ax1.set_yticks([])
        ax1.set_ylabel("")
        sns.despine(ax=ax1, left=True, bottom=True)

        # def _marginal_distrib_step2:
        ax2 = plt.subplot(grid[1: 6, 5])
        ax2.spines[:].set_linewidth(.4)
        ax2.tick_params(width=.6, length=2.5, labelsize=8)
        sns.kdeplot(data=df, y='y', ax=ax2, **kwd)
        ax2.set_xticks([])
        ax2.set_xlabel("")
        ax2.set_yticks([])
        ax2.set_ylabel("")
        sns.despine(ax=ax2, left=True, bottom=True)

    # def _marginal_distr_step4:
    ax4 = plt.subplot(grid[1: 6, 0: 5])
    ax4.spines[:].set_linewidth(.4)
    ax4.tick_params(width=.6, length=2.5, labelsize=8)
    # ax4.grid(linewidth=.6, ls='-.', alpha=.4)
    _curr_ft = plt.rcParams['font.family']  # 'Times New Roman'
    legend_font = {'family': _curr_ft, 'size': 8}

    R = np.corrcoef(X, Y)[1, 0]
    key = 'Correlation = %.4f' % R
    regr = np.polyfit(X, Y, deg=1)
    estimated = np.polyval(regr, X)
    Z = sorted(X)
    annotZ = annots[2] if len(annots) > 2 else r'$f(x)=x$'

    if distrib:
        sns.regplot(
            x='x', y='y', data=df, label=key,
            # line_kws={'lw': .87}, scatter_kws={'s': 10})
            scatter_kws={'s': 27, 'edgecolors': 'w',
                         'lw': .1},  # , 'color': 'blue'},
            line_kws={'lw': 1})
        if snspec == 'sty5':
            plt.plot(Z, Z, 'k--', lw=1,
                     label='{:4s}{}'.format('', annotZ))
            plt.plot(X, estimated, '-', lw=1, color=_navy)
        ax4.legend(
            prop=legend_font, labelspacing=.35, handleheight=1.2,
            # handletextpad=0, loc=(.98, 1.01), frameon=False)
            handletextpad=0, loc='best', frameon=False)
    elif linreg:  # if snspec == 'sty1':

        if snspec == 'sty2':
            # ax4.scatter(X, Y, label=key)
            ax4.scatter(x=X, y=Y, alpha=1, edgecolors='w',
                        linewidths=.5, label=key)  # s='.',
            plt.plot(X, estimated, 'k-', lw=1)
        elif snspec == 'sty3a':
            ax4.scatter(x=X, y=Y, alpha=1, edgecolors='w',
                        linewidths=.4, facecolor=_navy)
            plt.plot(X, estimated, '-', lw=1, label=key, color=_navy)
            plt.plot(Z, Z, 'k--', lw=1, label=annotZ)
            # plt.plot(X, estimated, '-', lw=1, label=key, color='navy')
        elif snspec == 'sty3b':
            ax4.scatter(x=X, y=Y, alpha=1, edgecolors='w',
                        linewidths=.4, label=key, facecolor=_navy)
            plt.plot(X, estimated, '-', lw=1, color=_navy)
            plt.plot(Z, Z, 'k--', lw=1, label=annotZ)
        elif snspec == 'sty4':
            ax4.scatter(x=X, y=Y, alpha=1, edgecolors='w',
                        # linewidths=.2, s=27, facecolor=_navy)
                        linewidths=.4, facecolor=_navy)  # s=42,
            plt.plot(Z, Z, 'k--', lw=1, label=annotZ)
        elif snspec == 'sty6':
            tx = _pl_myclr[1] if 'hfm' in figname else _navy
            ax4.scatter(x=X, y=Y, alpha=1, edgecolors='w',
                        linewidths=.4, facecolor=tx)
            tx_min, tx_max = ax4.get_xlim()
            annotZ = r'$\hat{\mathbf{D}}_{\cdot}=\mathbf{D}_{\cdot}$'
            if len(annots) > 2:
                annotZ = annots[2]
            ax4.plot([tx_min, tx_max], [0, 0], 'k--', lw=1, label=annotZ)
            del tx_min, tx_max, tx

        # if snspec != 'sty4':
        if snspec not in ['sty4', 'sty6']:
            kws = {'color': _navy, 'lw': 1}
            _sns_line_err_bars(ax4, kws, X, Y)
        ax4.legend(prop=legend_font, loc='best', frameon=False)
        _style_set_axis(ax4)
    del annotZ, Z

    ax4.set_xlabel(annots[0], fontsize=9, family=_curr_ft, x=.55)
    ax4.set_ylabel(annots[1], fontsize=9, family=_curr_ft, y=.55)
    # def end.
    if sci_format_y:  # if sci_y_format:
        # ax4.ticklabel_format(style='sci', scilimits=(-2, 3), axis='y')
        ax4.ticklabel_format(style='sci', scilimits=(-3, 4), axis='y')
        ax4.yaxis.get_offset_text().set_fontsize(8)  # 7)

    _setup_figshow(fig, figname=figname)
    plt.close(fig)
    return


# -------------------------------
# shaded uncertainty region in line plot


def _uncertainty_read_in(dfs_pl, col_X, col_Y, num_gap=1000,
                         alpha_loc='b4|af', alpha_rev=False):
    # dfs_pl: list of pd.DataFrame,
    #         len()= #baseline_for_comparison
    # each element is pd.DataFrame,
    #         columns= ['col_X', 'col_Y', 'learning']
    #         shape  = (#num, 3)

    X = np.linspace(0, 1, num_gap)
    baseline_Ys = []
    for df in dfs_pl:  # for i, df in enumerate(dfs_pl):
        Ys = []
        for alpha in X:
            # '''
            # if (alpha_loc == 'b4') and (not alpha_rev):
            #     tmp = df[col_X] * alpha + (1. - alpha) * df[col_Y]
            # elif (alpha_loc == 'af') and (not alpha_rev):
            #     tmp = df[col_X] * (1. - alpha) + alpha * df[col_Y]
            # elif (alpha_loc == 'b4') and alpha_rev:  # reverse
            #     tmp = (1 - df[col_X]) * alpha + (1. - alpha) * df[col_Y]
            # elif (alpha_loc == 'af') and alpha_rev:  # reverse
            #     tmp = (1 - df[col_X]) * (1. - alpha) + alpha * df[col_Y]
            # '''

            # assert alpha_loc in ('b4', 'af')
            if (not alpha_rev) and (alpha_loc == 'b4'):
                tmp = df[col_X] * alpha + (1. - alpha) * df[col_Y]
            elif not alpha_rev:           # (alpha_loc == 'af') and
                tmp = df[col_X] * (1. - alpha) + alpha * df[col_Y]
            elif alpha_loc == 'b4':     # and alpha_rev:  # reverse
                tmp = (1 - df[col_X]) * alpha + (1. - alpha) * df[col_Y]
            else:  # if (alpha_loc=='af') and alpha_rev:  # reverse
                tmp = (1 - df[col_X]) * (1. - alpha) + alpha * df[col_Y]
            Ys.append(tmp.values)  # (#num,)
        baseline_Ys.append(Ys)   # (num_gap, #num)
    # baseline_Ys.shape= (#baseline, #gap, #num)

    # baseline_Ys = np.array(baseline_Ys, dtype='float')
    # return X, baseline_Ys, np.mean(
    #     baseline_Ys, axis=2), np.std(baseline_Ys, axis=2, ddof=ddof)
    return X, np.array(baseline_Ys, dtype=DTY_FLT)


def _sub_unc_text(annotY, alpha_loc):
    # if annotY is None:
    #     if alpha_loc == 'b4':
    #         annotY = r'$\alpha·$ performance $+(1-\alpha)·$ fairness'
    #     elif alpha_loc == 'af':
    #         annotY = r'$(1-\alpha)·$ performance $+\alpha·$ fairness'
    # else:
    #     if alpha_loc == 'b4':
    #         annotY = r'$\alpha·${} $+($1$-\alpha)·$ fairness'.format(annotY)
    #     elif alpha_loc == 'af':
    #         annotY = r'$($1$-\alpha)·${} $+\alpha·$ fairness'.format(annotY)

    assert alpha_loc in ['b4', 'af']
    if annotY is None:
        if alpha_loc == 'b4':
            annotY = r'$\alpha·$ performance $+(1-\alpha)·$ fairness'
        elif alpha_loc == 'af':
            annotY = r'$(1-\alpha)·$ performance $+\alpha·$ fairness'
        return annotY
    if alpha_loc == 'b4':
        annotY = r'$\alpha·${} $+($1$-\alpha)·$ fairness'.format(annotY)
    elif alpha_loc == 'af':
        annotY = r'$($1$-\alpha)·${} $+\alpha·$ fairness'.format(annotY)
    return annotY    # CC 7->6


def _uncertainty_plotting(X, Ys, picked_keys, annotY=None, ddof=0,
                          alpha_loc='b4|af', cmap_name='husl',
                          figsize='M-WS', figname='lwu',
                          alpha_clarity=.3):
    '''
    _curr_sty = ['-.', '-.', '-.', '--', '-', '-']
    if len(picked_keys) > 6:
        # _curr_sty = ['-.', '-.', '-.', '--', '-', '-', ':', '-', ':']
        _curr_sty.extend([':', '-', ':'])
    '''
    _curr_sty = ['-.', '-.', '-.', '--', '-', '-'] + [':', '-', ':']

    # X                                     # (#gap,)
    Ys_avg = np.mean(Ys, axis=2)            # (#baseline, #gap)
    Ys_std = np.std(Ys, axis=2, ddof=ddof)  # (#baseline, #gap)
    # picked_keys                           # (#baseline,)

    fig, ax = plt.subplots(figsize=_setup_config['M-NT'])
    # clrs = sns.color_palette(cmap_name, len(picked_keys))
    # with sns.axes_style('darkgrid'):
    #   # epochs = list(range(101))
    clrs = sns.color_palette(cmap_name, len(picked_keys))  # palette=)

    for i, _ in enumerate(picked_keys):  # _:key
        ax.plot(X, Ys_avg[i],
                _curr_sty[i], label=picked_keys[i], c=clrs[i], lw=1)
        # ax.plot(X, Ys_avg[i], label=picked_keys[i], c=clrs[i])
        ax.fill_between(
            X, Ys_avg[i] - Ys_std[i], Ys_avg[i] + Ys_std[i],
            # alpha=.3, facecolor=clrs[i])
            alpha=alpha_clarity, facecolor=clrs[i])  # .1
    # ax.legend()
    # ax.set_yscale('log')

    ax.legend(labelspacing=.1, prop={
        'size': 9 if len(picked_keys) <= 6 else 8})
    ax.set_xlim(X[0], X[-1])
    ax.set_xlabel(r'$\alpha$')
    annotY = _sub_unc_text_beta(annotY, alpha_loc)

    # '''
    # assert alpha_loc in ('b4', 'af')
    # if (annotY is None) and (alpha_loc == 'b4'):
    #     annotY = r'$\alpha·$ performance $+(1-\alpha)·$ fairness'
    # elif annotY is None:  # and (alpha_loc == 'af'):
    #     annotY = r'$(1-\alpha)·$ performance $+\alpha·$ fairness'
    # elif alpha_loc == 'b4':
    #     annotY = r'$\alpha·${} $+($1$-\alpha)·$ fairness'.format(annotY)
    # else:  # alpha_loc == 'af':
    #     annotY = r'$($1$-\alpha)·${} $+\alpha·$ fairness'.format(annotY)
    # '''
    ax.set_ylabel(annotY, fontsize=9)
    ax.autoscale_view()
    fig = _setup_figsize(fig, figsize)
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


def _sub_unc_text_beta(annotY, alpha_loc):
    assert alpha_loc in ['b4', 'af']
    if annotY is None:
        if alpha_loc == 'b4':
            annotY = r'$\beta·$ performance $+(1-\beta)·$ fairness'
        elif alpha_loc == 'af':
            annotY = r'$(1-\beta)·$ performance $+\beta·$ fairness'
        return annotY
    if alpha_loc == 'b4':
        annotY = r'$\beta·${}$+($1$-\beta)·$ fairness'.format(annotY)
    elif alpha_loc == 'af':
        annotY = r'$($1$-\beta)·${} $+\beta·$ fairness'.format(annotY)
    return annotY


def _uncertainty_plot_beta(X, Ys, picked_keys, annotY=None, ddof=0,
                           alpha_loc='b4|af', cmap_name='hus1',
                           figsize='M-WS', figname='lwu',
                           alpha_clarity=.3):
    _curr_sty = ['-.', '-.', '-.', '--', '-', '-']
    # _curr_sty = ['--^', '--<', '-->', '-.p', '-s', '-h']
    # _curr_sty = ['--', '--', '--', '-.', '-', '-']
    # _curr_mks = ['o', 'o', 'o', 'd', 'p', 's']
    # X                                     # (#gap,)
    Ys_avg = np.mean(Ys, axis=2)            # (#baseline, #gap)
    Ys_std = np.std(Ys, axis=2, ddof=ddof)  # (#baseline, #gap)
    # picked_keys                           # (#baseline,)
    fig, ax = plt.subplots(figsize=_setup_config['M-NT'])
    clrs = sns.color_palette(cmap_name, len(picked_keys))
    # clrs = clrs[::-1]

    for i, _ in enumerate(picked_keys):  # _:key
        ax.plot(X, Ys_avg[i], _curr_sty[i], label=picked_keys[i],
                c=clrs[i], lw=1.5)  # marker=_curr_mks[i])  # ,
        #       # markerfacecolor=_curr_sty[i])
        ax.fill_between(
            X, Ys_avg[i] - Ys_std[i], Ys_avg[i] + Ys_std[i],
            alpha=alpha_clarity, facecolor=clrs[i])

    ax.legend(labelspacing=.1, prop={
        'size': 9 if len(picked_keys) <= 6 else 8})
    ax.set_xlim(X[0], X[-1])
    ax.set_xlabel(r'$\beta$')
    annotY = _sub_unc_text_beta(annotY, alpha_loc)

    ax.set_ylabel(annotY, fontsize=9)
    ax.autoscale_view()
    fig = _setup_figsize(fig, figsize)
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


def lineplot_with_uncertainty(df, col_X, col_Y, tag_Ys, picked_keys,
                              # annotX='acc', annotY='fair',
                              annotY=None, ddof=0, num_gap=100,
                              alpha_loc='b4', cmap_name='husl',
                              alpha_rev=True,  # middle of annotY
                              figsize='M-WS', figname='lwu',
                              alpha_clarity=.3, whether_beta=False):
    dfs_pl, _ = _marginal_distr_read_in(  # ,df_all
        df, col_X, col_Y, tag_Ys, picked_keys)  # alp_loc/rev
    X, Ys = _uncertainty_read_in(dfs_pl, col_X, col_Y, num_gap=num_gap,
                                 alpha_loc=alpha_loc, alpha_rev=alpha_rev)
    kwargs = {'figsize': figsize, 'figname': figname}
    if whether_beta:
        _uncertainty_plot_beta(X, Ys, picked_keys, annotY, ddof,
                               alpha_loc=alpha_loc,
                               alpha_clarity=alpha_clarity,
                               cmap_name=cmap_name, **kwargs)
        return
    _uncertainty_plotting(X, Ys, picked_keys, annotY, ddof,
                          alpha_loc=alpha_loc,
                          alpha_clarity=alpha_clarity,
                          cmap_name=cmap_name, **kwargs)


# -------------------------------


# ===============================
# Python Matlablib plot


# -------------------------------
# fairmanf ext.(extension)


# -------------------------------
# refers to:
#   def single_line_reg_with_distr():
#


def _subproc_pl_lin_reg(ax4, X, Y, Z, annotZ, snspec, clr=_navy,
                        reverse=False, corr=True, sz=None, mk=None):
    if Y is None:
        return

    R = Pearson_correlation(X, Y)[0]  # or pearsonccs(X,Y)[1,0]
    # R = np.corrcoef(X, Y)[1, 0]
    key = 'Correlation = %.4f' % R
    regr = np.polyfit(X, Y, deg=1)
    estimated = np.polyval(regr, X)
    # # Z = sorted(X)
    # '''
    # if not reverse:
    #     key = '{} {}'.format(key, Z)  # Z, key)
    # elif reverse:
    #     key = '{:9s} {}'.format(Z, key)
    # '''
    key = '{} {}'.format(key, Z) if (
        not reverse) else '{:9s} {}'.format(Z, key)
    if snspec == 'sty3bf':
        key = Z

    if snspec == 'sty3a':
        ax4.scatter(x=X, y=Y, alpha=1, edgecolors='w',
                    linewidths=.4, color=clr)
        plt.plot(X, estimated, '-', lw=1, label=key, color=clr)
        # plt.plot(Z, Z, 'k--', lw=1, label=annotZ)
    elif snspec in ('sty3b', 'sty3c', 'sty3d', 'sty3e',
                    'sty3bf',):  # =='sty3b':
        ax4.scatter(x=X, y=Y, alpha=1, edgecolors='w',
                    linewidths=.4, label=key, color=clr)
        plt.plot(X, estimated, '-', lw=1, color=clr)
    elif snspec in ('sty4', 'sty4c', 'sty4e',):
        # snspec.startswith('sty4'): #=='sty4':#in['sty4','sty8']:
        ax4.scatter(x=X, y=Y, alpha=1, edgecolors='w',
                    linewidths=.2, s=27, label=Z, color=clr)
    elif snspec == 'sty8a':
        ax4.scatter(x=X, y=Y, alpha=1, edgecolors='w',
                    linewidths=.4, s=42, label=Z,
                    color=clr)  # 'navy')
        ax4.xaxis.get_offset_text().set_size(7)  # 8)
        ax4.yaxis.get_offset_text().set_size(7)  # 8)
    elif snspec == 'sty8b':
        plt.plot(X, estimated, '-', lw=1, color=clr)
        ax4.scatter(x=X, y=Y, alpha=1, edgecolors='w',
                    linewidths=.4, label=key, color=clr)
    elif snspec in ('sty6', 'sty6c', 'sty6d', 'sty6e',):  # =='sty6':
        ax4.scatter(x=X, y=Y, alpha=1, edgecolors='w',
                    linewidths=.4, label=Z, color=clr)

    elif snspec == 'sty5a':
        ax4.scatter(x=X, y=Y, s=sz, marker=mk, alpha=1,
                    edgecolors='w', linewidths=.5, color=clr)
        ax4.plot(X, estimated, '-', lw=1,
                 label=key if corr else Z, color=clr)
    elif snspec == 'sty5b':
        ax4.scatter(x=X, y=Y, s=sz, marker=mk, alpha=1,
                    edgecolors='w', linewidths=.5, color=clr,
                    label=key if corr else Z)
        ax4.plot(X, estimated, '-', lw=1, color=clr)

    return


def _subproc_pl_lin_reg_alt(ax4, X, Y, snspec, clr=_navy):
    if Y is None:
        return
    if snspec.startswith('sty8'):
        kws = {'color': _navy, 'lw': 1}
        _sns_line_err_bars(ax4, kws, X, Y)
    elif snspec not in ['sty4', 'sty6', 'sty6c', 'sty6d',
                        'sty4d', 'sty4c', 'sty4e', 'sty6e', ]:
        kws = {'color': clr, 'lw': 1}
        _sns_line_err_bars(ax4, kws, X, Y)
    return


def _subproc_pl_identity(ax4, Xs, annotZ, snspec):
    # Zs = [sorted(X) for X in Xs]  # , clr='navy'):
    tX = np.concatenate(Xs, axis=0)
    tX = tX.tolist()  # fmpar env.
    Z = sorted(tX)  # tZ = sorted(tX)
    if snspec in ['sty3a', 'sty3b', 'sty3bf', 'sty4', 'sty5a', 'sty5b',
                  'sty8a', 'sty8b', 'sty3c', 'sty3d',
                  'sty4d', 'sty4c', 'sty3e', 'sty4e', ]:
        plt.plot(Z, Z, 'k--', lw=1, label=annotZ)
    elif snspec in ('sty6', 'sty6c', 'sty6d', 'sty6e',):  # =='sty6':
        tx_min, tx_max = ax4.get_xlim()
        ax4.plot([tx_min, tx_max], [0, 0], 'k--', lw=1, label=annotZ)
        del tx_min, tx_max
    del Z, tX
    return


# _pl_myclr = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
#              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def multi_lin_reg_with_distr(Xs, Ys, Zs, annots=('X', 'Y', 'Z'),
                             figname='pl_linreg', figsize='M-WS',
                             # linreg=False,  # distrib=False,
                             snspec='sty2', cmap_name='coolwarm',
                             sci_format_y=True):  # default:False
    # mycolor = sns.color_palette(cmap_name)
    fig = plt.figure(figsize=_setup_config[figsize], dpi=300)
    plt.subplots_adjust(left=.11, bottom=.11, right=.98, top=.995)
    grid = plt.GridSpec(6, 6, wspace=.05, hspace=.05)

    ax4 = plt.subplot(grid[1: 6, 0: 5])
    ax4.spines[:].set_linewidth(.4)
    ax4.tick_params(width=.6, length=2.5, labelsize=8)
    _curr_ft = plt.rcParams['font.family']  # 'Times New Roman'
    legend_font = {'family': _curr_ft, 'size': 8}

    annotZ = r'$\hat{\mathbf{D}}_\cdot=\mathbf{D}_\cdot$'
    annotZ = annots[2] if len(annots) > 2 else r'$f(x)=x$'
    # myclr = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    #          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    myclr = _pl_myclr

    _subproc_pl_lin_reg(
        ax4, Xs[0], Ys[0][0], Zs[0][0], annotZ, snspec, myclr[0])
    _subproc_pl_lin_reg(
        ax4, Xs[0], Ys[0][1], Zs[0][1], annotZ, snspec, myclr[1])
    _subproc_pl_lin_reg(
        ax4, Xs[1], Ys[1], Zs[1], annotZ, snspec, myclr[2])
    _subproc_pl_lin_reg_alt(ax4, Xs[0], Ys[0][0], snspec, clr=myclr[0])
    _subproc_pl_lin_reg_alt(ax4, Xs[0], Ys[0][1], snspec, clr=myclr[1])
    _subproc_pl_lin_reg_alt(ax4, Xs[1], Ys[1], snspec, clr=myclr[2])
    _subproc_pl_identity(ax4, Xs, annotZ, snspec)

    if snspec in ['sty3a', 'sty3b', ]:
        _curr_fram = {'frameon': False, 'loc': 'upper left'}
    elif snspec in ['sty6', ]:
        _curr_fram = {'frameon': True, 'framealpha': .5,
                      'loc': 'upper right'}  # 'loc': 'best'}
    elif snspec in ['sty4', ]:
        _curr_fram = {'loc': 'best', 'frameon': True, 'framealpha': .5}
    ax4.legend(prop=legend_font, labelspacing=.35, **_curr_fram)
    if sci_format_y:
        ax4.ticklabel_format(style='sci', scilimits=(-3, 4), axis='y')
        ax4.yaxis.get_offset_text().set_fontsize(8)  # 7)
    _style_set_axis(ax4)
    ax4.set_xlabel(annots[0], fontsize=9, family=_curr_ft, x=.55)
    ax4.set_ylabel(annots[1], fontsize=9, family=_curr_ft, y=.55)
    _setup_figshow(fig, figname=figname)
    plt.close(fig)
    return


# -------------------------------
# refers to
#   def multi_lin_reg_with_distr
#


def _add_on_DR_plot(ax4, X, Ys, Zs, annotZ, snspec, myclr):
    n_k = len(Ys)  # aka. len(Zs)
    start_i = 2 if n_k == 2 else 1
    if snspec.startswith('sty8'):  # 'navy','royalblue'
        # myclr = ['#1f77b4'] * (len(Ys) + start_i)
        myclr = _pl_myclr[0] * (len(Ys) + start_i)

    if len(Ys) == 1 and snspec in ('sty3e',):  # 'sty3b'):
        # myclr = ['navy', ] * (1 + start_i)
        # elif snspec == 'sty3e':
        # ax4.scatter(x=X, y=Ys[0], alpha=1, edgecolors='w',
        #             linewidths=.4, color=myclr[0 + start_i])
        _subproc_pl_identity(ax4, [X, X], annotZ, snspec)
        ttt = np.mean(Ys[0] <= X).tolist() * 100.
        tx_min, tx_max = ax4.get_xlim()
        ax4.plot([tx_min], [tx_min], 'w-', label=(
            r'$f(x)\leq x$ coverage {:.2f}%'.format(ttt)))
        del ttt, tx_min, tx_max
        # plt.plot(X, estimated, '-', lw=1, color='navy')

        myclr = [_navy, ] + myclr  # myclr[:1]+
        # R = np.corrcoef(X, Y)[1, 0]  # or pearsonccs(X,Y)[1,0]
        R = Pearson_correlation(X, Ys[0])[0]
        key = 'Correlation = %.4f' % R
        regr = np.polyfit(X, Ys[0], deg=1)
        estimated = np.polyval(regr, X)
        # key = '{} {}'.format(key, Z) if (
        #     not reverse) else '{:9s} {}'.format(Z, key)
        # plt.plot(X, estimated, '-', lw=1, label=key, color='navy')
        ax4.scatter(x=X, y=Ys[0], alpha=1, edgecolors='w',
                    linewidths=.4, label=key, color=myclr[start_i])
        plt.plot(X, estimated, '-', lw=1, color=_navy)
        del R, key, regr, estimated
        _sns_line_err_bars(ax4, {'color': _navy, 'lw': 1}, X, Ys[0])
        return

    # if len(Ys) == 1 and snspec == 'sty3b':
    #     myclr = ['navy', ] + myclr
    for i in range(n_k):
        _subproc_pl_lin_reg(ax4, X, Ys[i], Zs[i], annotZ,
                            snspec, myclr[i + start_i])
    # if len(Ys) == 1 and snspec == 'sty3b':
    #     myclr = myclr[:1] * (len(Ys) + start_i)
    for i in range(n_k):
        _subproc_pl_lin_reg_alt(
            ax4, X, Ys[i], snspec, myclr[i + start_i])
    # pdb.set_trace()
    # if len(Ys) == 1 and snspec.startswith('sty8'):
    #     ttt = np.mean(Ys[0] <= X).tolist()
    #     # annotZ += '\nCoverage = {:2f}%'.format(ttt * 100.)
    #     annotZ += ' (Coverage {:.2f}%)'.format(ttt * 100.)
    #     del ttt
    _subproc_pl_identity(ax4, [X, X], annotZ, snspec)
    if len(Ys) == 1 and snspec in ('sty8a', 'sty8b',):  # 'sty3b'):
        ttt = np.mean(Ys[0] <= X).tolist() * 100.
        tx_min, tx_max = ax4.get_xlim()
        ax4.plot([tx_min], [tx_min], 'w-',
                 label=r'$f(x)\leq x$ coverage {:.2f}%'.format(ttt))
        del ttt, tx_min, tx_max
    return


def multi_lin_reg_without_distr(X, Ys, Zs, annots=('X', 'Y', 'Z'),
                                figname='pl_linreg', figsize='M-WS',
                                snspec='sty4',  # cmap_names='coolwarm',
                                sci_format_y=False):
    fig = plt.figure(figsize=_setup_config[figsize], dpi=300)
    plt.subplots_adjust(left=.11, bottom=.11, right=.98, top=.995)
    grid = plt.GridSpec(6, 6, wspace=.05, hspace=.05)
    ax4 = plt.subplot(grid[1: 6, 0: 5])
    ax4.spines[:].set_linewidth(.4)
    ax4.tick_params(width=.6, length=2.5, labelsize=8)
    _curr_ft = plt.rcParams['font.family']  # 'Times New Roman'
    legend_font = {'family': _curr_ft, 'size': 8}

    myclr = _pl_myclr
    if snspec.startswith('sty7'):
        myclr = [_navy, ] + myclr[:6]  # +_pl_myclr[3:]
        del myclr[4]   # del myclr[-2]  # 1+5=6
        # myclr = ['navy', _pl_myclr[0], '#DED031', '#96DE31', _pl_myclr[
        #     4]]  # '#B4A81F', '#1FB4A8' # '#27F5B0','#27D3F5','#D3F527',
        # # myclr[1], myclr[4] = '#3196DE', '#7931DE'
        # myclr[2:] = ['#E58E50', '#8E50E5', '#A7E550', '#50A8E5']
        del myclr[2:4]  # myclr[3] = '#81C718'; del myclr[2]
        # myclr[3] = _pl_myclr[7]  # default:_pl_myclr[5]
        snspec = snspec.replace('y7', 'y6')
    elif snspec in ('sty3c', 'sty6c', 'sty4c',):  # =='sty3c'
        myclr = _pl_myclr[1:]  # HFM_ext
    elif snspec in ('sty3d', 'sty6d', 'sty4d',):
        myclr = _pl_myclr[2:]  # HFM_ext
    elif snspec in ('sty3e', 'sty4e', 'sty6e',):
        # myclr = _pl_myclr[3:]  # HFM_ext
        myclr = _pl_myclr[1:3] + _pl_myclr[5:]
        if len(Ys) == 1:
            # myclr = ['#1f77b4', ] + myclr  # default:|'#17becf'
            myclr = _pl_myclr[:1] + myclr
    annotZ = annots[2] if len(annots) > 2 else r'$f(x)=x$'

    _add_on_DR_plot(ax4, X, Ys, Zs, annotZ, snspec, myclr)

    if snspec in ['sty3a', 'sty3b', 'sty3c', 'sty3d', 'sty3e']:
        # _curr_fram = {'frameon': False, 'loc': 'upper left',
        #               'fontsize': 6,  # 'loc': 'lower right',
        #               'framealpha': .5}  # 'loc':'best',
        _curr_fram = {  # 'frameon': snspec == 'sty3b',
            'frameon': snspec in ('sty3b', 'sty3e', 'sty3c',),
            'loc': 'upper left' if snspec != 'sty3e' else 'best',
            'fontsize': 6, 'framealpha': .5}
    elif snspec == 'sty3bf':
        _curr_fram = {'frameon': 'lower right', 'fontsize': 6, 'framealpha': .5}
    elif snspec in ['sty6', 'sty6c', 'sty6d', 'sty6e', ]:
        _curr_fram = {'frameon': True, 'framealpha': .5,
                      'loc': 'best'}  # 'loc': 'upper right'}
    elif snspec in ['sty4', 'sty8a', 'sty8b', 'sty4d', 'sty4c',
                    'sty4e', ]:
        _curr_fram = {'loc': 'best', 'frameon': True,
                      'framealpha': .5}
        if snspec.startswith('sty8'):
            _curr_fram['frameon'] = False
            # _curr_fram['loc'] = 'upper left'
        elif snspec == 'sty4e':
            _curr_fram['loc'] = 'upper left'
    ax4.legend(prop=legend_font, labelspacing=.14,  # =.35
               handletextpad=.21, **_curr_fram)
    # ax4.legend(prop=legend_font, labelspacing=.35, **_curr_fram)
    if sci_format_y:
        ax4.ticklabel_format(style='sci', scilimits=(-3, 4),
                             axis='y')
        ax4.yaxis.get_offset_text().set_fontsize(8)
    _style_set_axis(ax4)
    ax4.set_xlabel(annots[0], fontsize=9, family=_curr_ft, x=.55)
    ax4.set_ylabel(annots[1], fontsize=9, family=_curr_ft, y=.55)
    _setup_figshow(fig, figname=figname)
    plt.close(fig)
    return


# -------------------------------


def scatter_parl_chart_renew(centralised, distributed, mp_cores=3,
                             figname='parl', figsize='M-WS',
                             identity=True):
    speed_up = np.divide(centralised, distributed)
    efficiency = speed_up / mp_cores
    annotY = [r'Speedup = $\frac{ T }{ T_{par} }$',
              r'Efficiency = $\frac{T}{m T_{par}}$']
    annotX = r'Sequential running time $T$ (sec)'
    picked_key = r'$m$ = {}'.format(mp_cores)

    fig = plt.figure(figsize=_setup_config['L-NT'])
    plt.scatter(centralised, speed_up, s=19, c='royalblue',
                label=picked_key, edgecolors='w', linewidths=.2)
    if identity:
        annotZ = 'Speedup =1'  # 'speedup =1'
        tx_min, tx_max = fig.gca().get_xlim()
        plt.plot([tx_min, tx_max], [1, 1], 'k--', lw=.8, label=annotZ)
    # plt.xlabel('No parallel computing')
    # plt.ylabel('Computing in parallel')
    plt.xlabel(annotX)
    plt.ylabel(annotY[0])  # plt.ylabel('Speedup')
    plt.legend(loc=PLT_LOCATION, labelspacing=.1, frameon=True)
    fig = _setup_figsize(fig, figsize)
    _setup_figshow(fig, figname + '_parl_sp')
    plt.clf()  # fig)

    plt.scatter(centralised, efficiency, s=19, c='slateblue',
                label=picked_key, edgecolors='w', linewidths=.2)
    if identity:
        annotZ = 'Efficiency =1'  # 'efficiency =1'
        plt.plot([tx_min, tx_max], [1, 1], 'k--', lw=.8, label=annotZ)
        del tx_min, tx_max
    plt.xlabel(annotX)
    plt.ylabel(annotY[1])  # plt.ylabel('Efficiency')
    plt.legend(loc=PLT_LOCATION, labelspacing=.1, frameon=True)
    fig = _setup_figsize(fig, figsize)
    _setup_figshow(fig, figname + '_parl_ep')
    plt.close(fig)
    return


# def _hyper_pm_step4(ax4, snspec):
#   pass
def hyper_params_lin_reg(X, Ys, tag_Ys, picked_keys,
                         annots=('acc', 'fair', 'Z'),
                         figname='smd', figsize='M-WS',
                         distrib=False, identity=True,
                         sci_format_y=False, corr=False,
                         curr_legend_nb_split=4,
                         cmap_name='muted', snspec='sty5b'):
    mycolor = sns.color_palette(cmap_name, len(picked_keys))
    fig = plt.figure(figsize=_setup_config[figsize], dpi=300)
    plt.subplots_adjust(left=.11, bottom=.11, right=.98, top=.995)
    grid = plt.GridSpec(6, 6, wspace=.05, hspace=.05)

    _curr_mk = ['D', '^', 'v', 'o', '*', 's', 'd', '<', '>']
    _curr_sz = [10, 15, 15, 12, 29, 11, 15, 15, 15]
    ax4 = plt.subplot(grid[1: 6, 0: 5])
    ax4.spines[:].set_linewidth(.4)
    ax4.tick_params(width=.6, length=2.5, labelsize=8)
    if distrib:
        ax4.grid(linewidth=.6, ls='-.', alpha=.4)

    annotZ = annots[2] if len(annots) > 2 else r'$f(x)=x$'
    # identity
    # nb_choice = len(picked_keys)
    # for i in range(nb_choice):
    for i, key in enumerate(picked_keys):
        _subproc_pl_lin_reg(
            ax4, X, Ys[key], tag_Ys[key], annotZ, snspec,
            mycolor[i], True, corr, _curr_sz[i], _curr_mk[i])
    for i, key in enumerate(picked_keys):
        _subproc_pl_lin_reg_alt(ax4, X, Ys[key], snspec, mycolor[i])
    if identity:
        _subproc_pl_identity(ax4, [X, X], annotZ, snspec)

    if snspec in ['sty3a', 'sty3b']:
        _curr_fram = {'frameon': False, 'loc': 'upper left'}
    elif snspec in ['sty5a', 'sty5b']:
        _curr_nc = 1 if len(picked_keys) <= curr_legend_nb_split else 2
        _curr_fram = {
            'frameon': True, 'loc': 'upper left', 'framealpha': .5,
            'handleheight': 1.2, 'handletextpad': 0, 'ncol': _curr_nc}
        del _curr_nc
    elif snspec in ['sty6', ]:
        _curr_fram = {'frameon': True, 'framealpha': .5,
                      'loc': 'upper right'}  # 'loc': 'best'}
    elif snspec in ['sty4']:
        _curr_fram = {'loc': 'best', 'frameon': True, 'framealpha': .5}
    _curr_ft = plt.rcParams['font.family']  # 'Times New Roman'
    legend_font = {'family': _curr_ft, 'size': 8}
    ax4.legend(prop=legend_font, labelspacing=.35, **_curr_fram)
    if sci_format_y:
        ax4.ticklabel_format(style='sci', scilimits=(-3, 4), axis='y')
        ax4.yaxis.get_offset_text().set_fontsize(8)  # 7)
    _style_set_axis(ax4)
    ax4.set_xlabel(annots[0], fontsize=9, family=_curr_ft, x=.55)
    ax4.set_ylabel(annots[1], fontsize=9, family=_curr_ft, y=.55)
    del _curr_ft, legend_font, _curr_fram, _curr_sz, _curr_mk

    _setup_figshow(fig, figname=figname)
    plt.close(fig)
    return


# -------------------------------

# -------------------------------


# ===============================
# RR plot


# -------------------------------

# -------------------------------

# -------------------------------


# ===============================
# FairGBM
# https://arxiv.org/pdf/2209.07850


# -------------------------------
# Figure 2(a)


def FairGBM_scatter(Xs, Ys, annot, label=('X', 'Y'),
                    cmap_name='deep',  # 'colorblind',
                    figname='FairGBM', figsize='M-WS'):
    # Xs.shape  (#model, #experiment /iteration)
    # Ys.shape  (#model, #experiment /iteration)
    curr_palette = sns.color_palette(cmap_name, len(annot))
    sns.palplot(curr_palette)  # current color theme

    dfs_pl = []
    for i, ant in enumerate(annot):
        tmp = pd.DataFrame({'x': Xs[i], 'y': Ys[i]})
        # tmp = tmp.apply(pd.to_numeric, errors='coerce')
        tmp['learning'] = ant
        dfs_pl.append(tmp)
    df_all = pd.concat(dfs_pl, axis=0).reset_index(drop=True)

    fig = plt.figure(figsize=_setup_config[figsize], dpi=300)
    plt.subplots_adjust(left=.11, bottom=.11, right=.98, top=.995)
    grid = plt.GridSpec(6, 6, wspace=.05, hspace=.05)
    # ax1, ax2, ax3 =
    _marginal_distrib_step1(grid, df_all, 'x', curr_palette)
    _marginal_distrib_step2(grid, df_all, 'y', curr_palette)
    _marginal_distrib_step3(grid, dfs_pl, annot, 'x', 'y',
                            label[0], label[1], curr_palette,
                            loc='best')

    _setup_figshow(fig, figname=figname)
    plt.close()
    return


# -------------------------------
# Figure 2(b)


def FairGBM_tradeoff_v1(X, Y, annot, label=('X', 'Y'),
                        num_gap=1000,
                        alpha_loc='b4|af', alpha_rev=False,
                        alpha_clarity=.15, cmap_name='colorblind',
                        figname='FairGBM', figsize='M-WS'):
    # X.shape  (#model,)
    # Y.shape  (#model,)

    Z = np.linspace(0, 1, num_gap)
    baseline_Ys = []
    # for i, ant in enumerate(annot):
    #   tmp = []
    for alpha in Z:
        if (alpha_loc == 'b4') and (not alpha_rev):
            tmp = X * alpha + (1. - alpha) * Y
            label_y = r'$\alpha·$ performance $+(1-\alpha)·$ fairness'
        elif (alpha_loc == 'af') and (not alpha_rev):
            tmp = X * (1. - alpha) + alpha * Y
            label_y = r'$(1-\alpha)·$ performance $+\alpha·$ fairness'
        elif (alpha_loc == 'b4') and alpha_rev:
            tmp = (1. - X) * alpha + (1. - alpha) * Y
            label_y = r'$\alpha·${} $+($1$-\alpha)·$ {}'.format(*label)
        elif (alpha_loc == 'af') and alpha_rev:
            tmp = (1. - X) * (1. - alpha) + alpha * Y
            label_y = r'$($1$-\alpha)·${} $+\alpha·$ {}'.format(*label)
        # tmp.shape  (#model,)
        baseline_Ys.append(tmp)
    baseline_Ys = np.array(baseline_Ys).transpose()
    # baseline_Ys.shape  (#model, #gap)

    curr_palette = sns.color_palette(cmap_name, len(annot))
    sns.palplot(curr_palette)
    _curr_sty = ['-.', '-.', '-.', '--', '-', '-', ':', '-', ':']
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    kws = {'color': _navy, 'lw': 1}
    for i, ant in enumerate(annot):
        kws['color'] = curr_palette[i]
        kws['label'] = ant  # annot[i]
        kws['linestyle'] = _curr_sty[i]
        _sns_line_err_bars(ax, kws, Z, baseline_Ys[i])

    ax.set_xlabel(r'$\alpha$')  # label[0])
    ax.set_ylabel(label_y)      # label[1])
    plt.legend(loc='best', frameon=True,
               labelspacing=.07, prop={'size': 9})
    fig = _setup_figsize(fig, figsize)
    _setup_figshow(fig, figname=figname)
    plt.close()
    return


def FairGBM_tradeoff_v3(Xs, Ys, annot, label=('X', 'Y'),
                        num_gap=1000, cmap_name='colorblind',
                        alpha_loc='b4|af', alpha_rev=False,
                        alpha_clarity=.15,
                        figname='FairGBM', figsize='M-WS'):
    # Xs.shape  (#model, #experiment /iteration)
    # Ys.shape  (#model, #experiment /iteration)
    Z = np.linspace(0, 1, num_gap)  # (#gap,)
    baseline_Ys = []
    for alpha in Z:
        if (alpha_loc == 'b4') and (not alpha_rev):
            tmp = Xs * alpha + (1. - alpha) * Ys
        elif (alpha_loc == 'af') and (not alpha_rev):
            # tmp = Xs * (1. - alpha) + alpha * Ys[i]
            tmp = Xs * (1. - alpha) + alpha * Ys
        elif (alpha_loc == 'b4') and alpha_rev:
            tmp = (1. - Xs) * alpha + (1. - alpha) * Ys
        elif (alpha_loc == 'af') and alpha_rev:
            tmp = (1. - Xs) * (1. - alpha) + alpha * Ys
        # tmp.shape  (#model, #experiment /iteration)
        baseline_Ys.append(tmp)
    baseline_Ys = np.array(baseline_Ys).transpose(1, 0, 2)
    # baseline_Ys.shape  (#model, #gap, #iteration)
    # baseline_Ys = np.array(baseline_Ys)  # (#model, #gap,#..)
    n = baseline_Ys.shape[2]
    ZZ = np.repeat(Z, n).reshape(-1, n).T.reshape(-1)
    base_Ys = baseline_Ys.reshape(-1, num_gap * n)
    # ZZ = np.array([Z for _ in range(n)]).reshape(-1)
    del n

    kwargs = {'figsize': figsize, 'figname': figname}
    kwargs['alpha_loc'] = alpha_loc
    kwargs['alpha_clarity'] = alpha_clarity
    curr_palette = sns.color_palette(cmap_name, len(annot))
    sns.palplot(curr_palette)
    _curr_sty = ['-.', '-.', '-.', '--', '-', '-', ':', '-', ':']
    if alpha_loc == 'b4':
        label_y = r'$\alpha·${} $+($1$-\alpha)·$ {}'.format(*label)
    elif alpha_loc == 'af':
        label_y = r'$($1$-\alpha)·${} $+\alpha·$ {}'.format(*label)
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    kws = {'color': _navy, 'lw': 1}
    for i, ant in enumerate(annot):
        kws['color'] = curr_palette[i]
        kws['label'] = ant  # annot[i]
        kws['linestyle'] = _curr_sty[i]
        _sns_line_err_bars(ax, kws, ZZ, base_Ys[i])

    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(label_y)
    plt.legend(loc='best', frameon=True,
               labelspacing=.07, prop={'size': 9})
    fig = _setup_figsize(fig, figsize)
    _setup_figshow(fig, figname=figname)
    plt.close()
    return


def FairGBM_tradeoff_v2(Xs, Ys, annot, label=('X', 'Y'),
                        num_gap=1000, cmap_name='colorblind',
                        alpha_loc='b4|af', alpha_rev=False,
                        alpha_clarity=.15,
                        figname='FairGBM', figsize='M-WS'):
    # Xs.shape  (#model, #experiment /iteration)
    # Ys.shape  (#model, #experiment /iteration)
    Z = np.linspace(0, 1, num_gap)  # (#gap,)
    baseline_Ys = []
    for alpha in Z:
        if (alpha_loc == 'b4') and (not alpha_rev):
            tmp = Xs * alpha + (1. - alpha) * Ys
        elif (alpha_loc == 'af') and (not alpha_rev):
            # tmp = Xs * (1. - alpha) + alpha * Ys[i]
            tmp = Xs * (1. - alpha) + alpha * Ys
        elif (alpha_loc == 'b4') and alpha_rev:
            tmp = (1. - Xs) * alpha + (1. - alpha) * Ys
        elif (alpha_loc == 'af') and alpha_rev:
            tmp = (1. - Xs) * (1. - alpha) + alpha * Ys
        # tmp.shape  (#model, #experiment /iteration)
        baseline_Ys.append(tmp)
    baseline_Ys = np.array(baseline_Ys).transpose(1, 0, 2)
    # baseline_Ys.shape  (#model, #gap, #iteration)
    # baseline_Ys = np.array(baseline_Ys)  # (#model, #gap,#..)

    kwargs = {'figsize': figsize, 'figname': figname}
    kwargs['alpha_loc'] = alpha_loc
    kwargs['alpha_clarity'] = alpha_clarity
    _uncertainty_plotting(Z, baseline_Ys, annot, label[1], **kwargs)

    # _setup_figshow(fig, figname=figname)
    # plt.close()
    return


# -------------------------------

# -------------------------------
#
# matplotlib(legend)
# https://blog.csdn.net/henkekao/article/details/75282446

# -------------------------------
# Revision

def _subtim_afterbody(ax, annots, sci_format_y, _curr_ft):
    ax.set_xlabel(annots[0], fontsize=9, family=_curr_ft, x=.55)
    ax.set_ylabel(annots[1], fontsize=9, family=_curr_ft, y=.55)
    if sci_format_y:
        ax.ticklabel_format(style='sci', scilimits=(-3, 4), axis='y')
        ax.yaxis.get_offset_text().set_fontsize(8)
    return ax  # tail


def _subtim_sing_lin(ax, X, Y, snspec='sty2',
                     # lbl_X='x', lbl_Y='y', lbl_Z=r'$f(x)=x$'):
                     annots=('X', 'Y', 'Z'), sci_format_y=False):
    R = Pearson_correlation(X, Y)[0]
    # R = np.corrcoef(X, Y)[1, 0]
    key = 'Correlation = %.4f' % R
    regr = np.polyfit(X, Y, deg=1)
    estimated = np.polyval(regr, X)
    Z = sorted(X)
    annotZ = annots[2] if len(annots) > 2 else r'$f(x)=x$'

    kw = dict(alpha=1, linewidths=.4,)
    if snspec in ['sty6', 'sty9']:
        tx = _navy if snspec == 'sty6' else _pl_myclr[1]
        ax.scatter(x=X, y=Y, edgecolors='w', facecolor=tx, linewidths=.4)
        tx_min, tx_max = ax.get_xlim()
        ax.plot([tx_min, tx_max], [0, 0], 'k--', lw=1, label=annotZ)
        del tx, tx_max, tx_min
    # if snspec == 'sty2':
    #     ax.scatter(edgecolors='w', linewidths=.5, label=key, **kw)
    #     ax.plot(X, estimated, 'k-', lw=1)
    # elif snspec == 'sty3a':
    #     ax.scatter(edgecolors='w', linewidths=.4, facecolor=_navy, **kw)
    #     ax.plot(X, estimated, '-', lw=1, label=key, color=_navy)
    #     ax.plot(Z, Z, 'k--', lw=1, label=lbl_Z)
    _style_set_axis(ax)

    _curr_ft = plt.rcParams['font.family']
    legend_font = {'size': 8.7, 'family': _curr_ft}
    if snspec not in ['sty6', 'sty9']:
        kw = {'color': _navy, 'lw': 1}
        _sns_line_err_bars(ax, kw, X, Y)
    ax.legend(prop=legend_font, loc='best', frameon=False)
    ax = _subtim_afterbody(ax, annots, sci_format_y, _curr_ft)
    ax.tick_params(width=.6, length=2.5, labelsize=8.7)
    return ax


def _subtim_multi_lin(ax, X, Ys, snspec, annots, Zs,
                      sci_format_y=False):
    ax.tick_params(width=.6, length=2.5, labelsize=8.7)
    _curr_ft = plt.rcParams['font.family']
    legend_font = {'family': _curr_ft, 'size': 8.7}

    myclr = _pl_myclr
    if snspec.startswith('sty7'):
        myclr = [_navy, ] + myclr[:6]
        del myclr[4]
        del myclr[2:4]
        snspec = snspec.replace('y7', 'y6')
    annotZ = annots[2] if len(annots) > 2 else r'$f(x)=x$'

    n_k = len(Ys)
    start_i = 2 if n_k == 2 else 1
    for i in range(n_k):
        _subproc_pl_lin_reg(
            ax, X, Ys[i], Zs[i], annotZ, snspec, myclr[i + start_i])
    for i in range(n_k):
        _subproc_pl_lin_reg_alt(ax, X, Ys[i], snspec, myclr[i + start_i])
    _subproc_pl_identity(ax, [X, X], annotZ, snspec)

    if snspec.startswith('sty6'):
        _curr_fram = {'frameon': True, 'framealpha': .5, 'loc': 'best'}
    ax.legend(prop=legend_font, labelspacing=.14, handletextpad=.21, **_curr_fram)
    ax = _subtim_afterbody(ax, annots, sci_format_y, _curr_ft)
    _style_set_axis(ax)
    return ax


def gathering_lin_reg_sup_tim(df, X, Ys, figname, ant_X, ant_Ys, lbl_Zs):
    num = len(X)     # figsize=(8.7-10.7, 4.27-4.37)
    # fig, ax = plt.subplots(nrows=2, ncols=num, sharex='col',  # True,
    #                        figsize=(9.86, 4.17), dpi=300,
    #                        constrained_layout=True)
    # # plt.subplots_adjust(left=.06, bottom=.06, right=.98, top=.995)

    import matplotlib.gridspec as gridspec  # 设置较大的画布   (11.6, 4.37)
    fig = plt.figure(figsize=(12.1, 4.42), dpi=300)  # ,constrained_layout=True)
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1.15, 1.09, 1.12],
                           figure=fig, wspace=.3, hspace=.2)  # gs[0, i]
    # 3. 定义 GridSpec：1行3列，中间列比两边宽 (宽比例为 1:2:1)
    # gs = gridspec.GridSpec(1, 3, width_ratios=[1, 2, 1])
    # axes = []
    # for r in range(2):
    #     tmp = []
    #     for c in range(4):
    #         ax = fig.add_subplot(gs[r, c])
    #         tmp.append(ax)
    #     axes.append(tmp)  # ax)
    # ax = axes
    axes = [fig.add_subplot(gs[0, c]) for c in range(4)]
    for ax in axes:   # 第一行
        ax.tick_params(labelbottom=False)
    tmp = [fig.add_subplot(gs[1, c], sharex=axes[c]) for c in range(4)]
    ax = [axes, tmp]
    del tmp, axes  # pdb.set_trace()

    for i in [0, 2]:   # if i % 2 == 0:
        tag_X, tag_Y = X[i], Ys[i]
        num_X = df[tag_X].values.astype(DTY_FLT)
        num_Y = df[tag_Y].values.astype(DTY_FLT)
        num_Y = num_Y / num_X
        snspec = 'sty6' if i == 0 else 'sty9'
        curr_antYs = ant_Ys[i] + r'$-1$'
        annots = [subfig_ind(i), curr_antYs, lbl_Zs[i]]
        ax[0][i] = _subtim_sing_lin(ax[0][i], num_X, num_Y - 1, snspec, annots)
        annots[0] = ant_X[i] + '\n' + subfig_ind(num + i)
        annots[1] = r'$\lg($' + ant_Ys[i] + r'$)$'
        ax[1][i] = _subtim_sing_lin(ax[1][i], num_X, np.log10(num_Y), snspec, annots)
        del tag_X, tag_Y, num_X, num_Y, snspec, curr_antYs, annots

    for i in [1, 3]:   # for i, tag_X in enumerate(X):
        tag_Ys = Ys[i]  # cur_antX,cur_antYs = ant_X[i],ant_Ys[i]
        num_X = df[X[i]].values.astype(DTY_FLT)
        num_Ys = [df[k].values.astype(DTY_FLT) / num_X - 1 for k in tag_Ys]
        snspec = 'sty7' if i == 1 else 'sty6'
        j = i if i <= 2 else i - 1
        curr_antYs = ant_Ys[j] + r'$-1$'
        annots = [subfig_ind(i), curr_antYs, lbl_Zs[i][0]]
        ax[0][i] = _subtim_multi_lin(ax[
            0][i], num_X, num_Ys, snspec, annots, lbl_Zs[i][1:])
        annots[0] = ant_X[j] + '\n' + subfig_ind(num + i)
        annots[1] = r'$\lg($' + ant_Ys[j] + r'$)$'
        ax[1][i] = _subtim_multi_lin(ax[1][i], num_X, [np.log10(
            k + 1) for k in num_Ys], snspec, annots, lbl_Zs[i][1:])
        # if i == 1:
        #     break
        del tag_Ys, num_X, num_Ys, snspec, j, curr_antYs, annots

    # fig.set_constrained_layout_pads(
    #     w_pad=4.0,   # 子图之间的水平间距
    #     h_pad=2.0,   # 子图之间的垂直间距
    #     wspace=0.2, hspace=0.2)  # 轴之间的额外空间
    # plt.tight_layout()  # pdb.set_trace()
    # fig.subplots_adjust(wspace=16, hspace=14)
    _setup_figshow(fig, figname=figname)
    return


# -------------------------------
