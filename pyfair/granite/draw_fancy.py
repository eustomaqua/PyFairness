# coding: utf-8


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
# import pdb
from pyfair.facil.draw_prelim import (
    _setup_figsize, _setup_figshow, _setup_config)
from pyfair.facil.utils_const import DTY_FLT


# ------------------------------


def _bp_rect_dat(Ys, annotX):
    dfs = [pd.DataFrame({'fair': i}) for i in Ys]
    for i, df in enumerate(dfs):
        # df['bel'] = annotX[i]
        df.loc[:, 'bel'] = annotX[i]
    df_tmp = pd.concat(dfs, axis=0)
    return df_tmp


def boxplot_rect(Ys, annotX, figname,  # notch=False,
                 figsize='M-WS'):
    df = _bp_rect_dat(Ys, annotX)
    fig, ax = plt.subplots(figsize=_setup_config[
        figsize])  # 111)  # bplot_rect =
    # ax.boxplot(x=Ys, notch=notch, vert=True, widths=.3,
    #            labels=annotX, patch_artist=True,
    #            medianprops={'linewidth': 1.5},
    #            showmeans=True, meanline=True,
    #            showfliers=True)
    sns.boxplot(ax=ax, data=df, x="bel", y="fair")
    # ax.yaxis.grid(True)
    ax.set_xlabel('')
    ax.set_ylabel('')
    fig = _setup_figsize(fig, figsize, invt=False)
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


def _bp_dat_X(df, tag_Xs):
    df_tmp = _bp_rect_dat([df[i].values.astype(
        DTY_FLT) for i in tag_Xs], tag_Xs)
    return df_tmp


def _bp_dat_XY(df, tag_Xs, tag_Ys):
    df_tX = df[tag_Xs]
    # df_tX['hue_dim'] = "ori"  # "ori."  # OG,orig
    df_tX.loc[:, ('hue_dim',)] = "ori"
    columns = {t2: t1 for t1, t2 in zip(tag_Xs, tag_Ys)}
    df_tY = df[tag_Ys].rename(columns=columns)
    # df_tY['hue_dim'] = "ext"  # "ext."
    df_tY.loc[:, ('hue_dim',)] = "ext"
    df_alt = pd.concat([df_tX, df_tY], axis=0)

    dfs = [df_alt[[i, 'hue_dim']].rename(columns={
        i: 'fair'}) for i in tag_Xs]
    for i, df in enumerate(dfs):
        df['bel'] = tag_Xs[i]
    return pd.concat(dfs, axis=0)


def _bp_dat_XYZ(df, tag_Xs, tag_Ys, tag_Zs):
    df_tX = df[tag_Xs]
    df_tX.loc[:, ('hue_dim',)] = 'ori'
    columns = {t2: t1 for t1, t2 in zip(tag_Xs, tag_Ys)}
    df_tY = df[tag_Ys].rename(columns=columns)
    df_tY.loc[:, ('hue_dim',)] = 'ext'
    columns = {t3: t1 for t1, t3 in zip(tag_Xs, tag_Zs)}
    df_tZ = df[tag_Zs].rename(columns=columns)
    df_tZ.loc[:, ('hue_dim',)] = 'ext.alt'
    df_alt = pd.concat([df_tX, df_tY, df_tZ], axis=0)

    dfs = [df_alt[[i, 'hue_dim']].rename(columns={
        i: 'fair'}) for i in tag_Xs]
    for i, df in enumerate(dfs):
        df['bel'] = tag_Xs[i]  # belong
    return pd.concat(dfs, axis=0)


def multi_boxplot_rect(df, tag_Xs, tag_Ys=None, tag_Zs=None,
                       annotX=tuple(), figname='',
                       locate="best", figsize='M-WS'):
    fig, ax = plt.subplots(figsize=_setup_config[figsize])
    if tag_Ys is None:  # and (tag_Z is None):
        df_alt = _bp_dat_X(df, tag_Xs)
        sns.boxplot(ax = ax, data = df_alt, x = "bel", y = "fair")
    elif tag_Zs is None:
        df_alt = _bp_dat_XY(df, tag_Xs, tag_Ys)
        sns.boxplot(ax=ax, data=df_alt, x="bel", y="fair",
                    hue="hue_dim")
        sns.move_legend(ax, locate, title='')
    else:
        df_alt = _bp_dat_XYZ(df, tag_Xs, tag_Ys, tag_Zs)
        sns.boxplot(ax=ax, data=df_alt, x="bel", y="fair",
                    hue="hue_dim")
        sns.move_legend(ax, locate, title='')
    if annotX:
        tmp = ax.get_xticks()
        ax.set_xticks(tmp)
        ax.set_xticklabels(annotX)  # ,rotation=rotate|0)
    ax.set_xlabel('')
    ax.set_ylabel('')
    fig = _setup_figsize(fig, figsize, invt=False)
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


def _radar_X(ax, df, tag_Xs, annotX, clockwise=False,
             stylish=False):
    # scores = [df[i].values.astype(DTY_FLT) for i in tag_Xs]
    # scores = [np.concatenate([i, [i[0]]]) for i in scores]
    angles = np.linspace(
        0, 2 * np.pi, len(tag_Xs), endpoint=False)
    labels = annotX + [annotX[0]]
    angles = np.concatenate([angles, [angles[0], ]])
    if clockwise:
        angles = angles[::-1]
    # if stylish:
    #     plt.style.use('ggplot')  # 使用ggplot的绘图风格
    scores = df[tag_Xs].values.astype(DTY_FLT)
    scores = np.concatenate([scores, scores[:, 0].reshape(
        -1, 1)], axis=1)
    for i, sc in enumerate(scores):
        ax.plot(angles, sc)
    # ax.set_thetagrids(angles * 180 / np.pi, labels)  # 标签显示
    # ax.set_theta_zero_location('N')  # 设置雷达图的0度起始位置
    # ax.set_rlim(0, 100)  # 设置雷达图的坐标刻度范围
    # ax.set_rlabel_position(270)  # 设置坐标显示角度，相对于起始角度的偏移量

    kws = {}  # {'fontsize': 14, 'style': 'italic'}
    if stylish:
        for i, sc in enumerate(scores):
            ax.fill(angles, sc, alpha=.25)
        kws['style'] = 'italic'
    ax.set_thetagrids(angles * 180 / np.pi, labels, **kws)
    ax.set_theta_zero_location('N')  # 'E'
    ax.set_rlabel_position(225)
    return ax


def radar_chart(df, tag_Xs,  # tag_Ys=None, tag_Zs=None,
                annotX=tuple(), annotY=tuple(), clockwise=True,
                stylish=False, figname='', figsize='M-WS'):
    fig = plt.figure(figsize=_setup_config[figsize])
    ax = fig.add_subplot(111, polar=True)  # 设置极坐标格式
    ax = _radar_X(ax, df, tag_Xs, annotX, clockwise,
                  stylish=stylish)
    if annotY:
        plt.legend(annotY, loc="best",
                   labelspacing=.07, prop={'size': 9})

    # if tag_Ys is None:
    #     ax = _radar_X(ax, df, tag_Xs, annotX, clockwise)
    # elif tag_Zs is None:
    #     pass
    # else:
    #     pass
    # fig = _setup_figsize(fig, figsize, invt = False)
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


# ------------------------------
# refs:
#
# https://zhuanlan.zhihu.com/p/375866522
# https://www.cnblogs.com/shijingwen/p/15011142.html
# https://www.cnblogs.com/metafullstack/p/17651922.html
# https://blog.csdn.net/weixin_42699538/article/details/134362019
# https://blog.csdn.net/weixin_39675038/article/details/111843998
# https://blog.csdn.net/zyh960/article/details/118278429
# https://zhuanlan.zhihu.com/p/686319124
# https://developer.baidu.com/article/details/2795826
# https://matplotlib.org.cn/stable/gallery/color/color_sequences.html
# https://matplotlib.org.cn/stable/gallery/color/colormap_reference.html
#
