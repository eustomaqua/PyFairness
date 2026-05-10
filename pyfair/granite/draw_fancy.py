# coding: utf-8


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from pyfair.facil.draw_prelim import (
    _setup_figsize, _setup_figshow, _setup_config, _style_set_axis)
from pyfair.facil.utils_const import DTY_FLT  # ,subfig_ind
from matplotlib.patches import Rectangle


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


def _bp_dat_XY(df, tag_Xs, tag_Ys,
               labels=('ori', 'ext',)):
    df_tX = df[tag_Xs]
    # df_tX['hue_dim'] = "ori"  # "ori."  # OG,orig
    df_tX.loc[:, ('hue_dim',)] = labels[0]  # "ori"
    columns = {t2: t1 for t1, t2 in zip(tag_Xs, tag_Ys)}
    df_tY = df[tag_Ys].rename(columns=columns)
    # df_tY['hue_dim'] = "ext"  # "ext."
    df_tY.loc[:, ('hue_dim',)] = labels[1]  # "ext"
    df_alt = pd.concat([df_tX, df_tY], axis=0)

    dfs = [df_alt[[i, 'hue_dim']].rename(columns={
        i: 'fair'}) for i in tag_Xs]
    for i, df in enumerate(dfs):
        df['bel'] = tag_Xs[i]
    return pd.concat(dfs, axis=0)


def _bp_dat_XYZ(df, tag_Xs, tag_Ys, tag_Zs,
                labels=('ori', 'ext', 'alt',)):
    df_tX = df[tag_Xs]
    df_tX.loc[:, ('hue_dim',)] = labels[0]  # 'ori'
    columns = {t2: t1 for t1, t2 in zip(tag_Xs, tag_Ys)}
    df_tY = df[tag_Ys].rename(columns=columns)
    df_tY.loc[:, ('hue_dim',)] = labels[1]  # 'ext'
    columns = {t3: t1 for t1, t3 in zip(tag_Xs, tag_Zs)}
    df_tZ = df[tag_Zs].rename(columns=columns)
    df_tZ.loc[:, ('hue_dim',)] = labels[2]  # 'ext.alt'
    df_alt = pd.concat([df_tX, df_tY, df_tZ], axis=0)

    dfs = [df_alt[[i, 'hue_dim']].rename(columns={
        i: 'fair'}) for i in tag_Xs]
    for i, df in enumerate(dfs):
        df['bel'] = tag_Xs[i]  # belong
    return pd.concat(dfs, axis=0)


def _bp_dat_XYZT_extra(df, tag_Xs, tag_Ys, tag_Zs,
                       tag_Ts, tag_Es, labels):
    df_tX = df[tag_Xs]
    df_tX.loc[:, ('hue_dim',)] = labels[0]  # 'ori'
    columns = {t2: t1 for t1, t2 in zip(tag_Xs, tag_Ys)}
    df_tY = df[tag_Ys].rename(columns=columns)
    df_tY.loc[:, ('hue_dim',)] = labels[1]  # 'ext'
    columns = {t3: t1 for t1, t3 in zip(tag_Xs, tag_Zs)}
    df_tZ = df[tag_Zs].rename(columns=columns)
    df_tZ.loc[:, ('hue_dim',)] = labels[2]  # 'alt', 'ext.alt'

    columns = {t4: t1 for t1, t4 in zip(tag_Xs, tag_Ts)}
    df_tT = df[tag_Ts].rename(columns=columns)
    df_tT.loc[:, ('hue_dim',)] = labels[3]  # 'ext (avg)'
    columns = {t5: t1 for t1, t5 in zip(tag_Xs, tag_Es)}
    df_tE = df[tag_Es].rename(columns=columns)
    df_tE.loc[:, ('hue_dim',)] = labels[4]  # 'alt (avg)'

    df_alt = pd.concat([df_tX, df_tY, df_tZ,
                        df_tT, df_tE], axis=0)
    dfs = [df_alt[[i, 'hue_dim']].rename(columns={
        i: 'fair'}) for i in tag_Xs]
    for i, df in enumerate(dfs):
        df['bel'] = tag_Xs[i]  # belong
    return pd.concat(dfs, axis=0)


def multi_boxplot_rect(df, tag_Xs, tag_Ys=None, tag_Zs=None,
                       tag_Ts=None, tag_Extra=None,
                       labels=('ori', 'ext', 'alt', 'ext (avg)',
                               'alt (avg)',),  # tag_Es=None,
                       annotX=tuple(), figname='',
                       locate="best", figsize='M-WS'):
    fig, ax = plt.subplots(figsize=_setup_config[figsize])
    if tag_Ys is None:  # and (tag_Z is None):
        df_alt = _bp_dat_X(df, tag_Xs)
        sns.boxplot(ax = ax, data = df_alt, x = "bel", y = "fair")
        # showfliers=True, flierprops={'marker': 's', 'markerfacecolor': 'red', 'markersize': 5, 'markeredgecolor': 'black'})
    elif tag_Zs is None:
        df_alt = _bp_dat_XY(df, tag_Xs, tag_Ys,
                            labels=labels[: 2])
        sns.boxplot(ax=ax, data=df_alt, x="bel", y="fair",
                    hue="hue_dim")
        sns.move_legend(ax, locate, title='')
    elif tag_Ts is None:
        df_alt = _bp_dat_XYZ(df, tag_Xs, tag_Ys, tag_Zs,
                             labels=labels[: 3])
        sns.boxplot(ax=ax, data=df_alt, x="bel", y="fair",
                    hue="hue_dim")
        sns.move_legend(ax, locate, title='')
    else:
        df_alt = _bp_dat_XYZT_extra(
            df, tag_Xs, tag_Ys, tag_Zs, tag_Ts, tag_Extra,
            labels)
        sns.boxplot(ax=ax, data=df_alt, x="bel", y="fair",
                    hue="hue_dim")
        sns.move_legend(ax, locate, title='')
        ax.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0,
                  loc='upper left', labelspacing=.07,
                  prop={'size': 9})
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


def _internal_with_swarm(ax, df, tag_Xs, clrs, x, bar_w, lw=2):  # i,g,xi,
    def _swarm_positions(y, x_centre=0.0, dx=0.04, dy=None):
        # 给一组纵坐标生成swarmplot风格的横坐标
        y, n = np.asarray(y, dtype=float), len(y)
        if n == 0:
            return np.array([])
        # 自动设一个纵向碰撞阈值
        if dy is None:
            y_range = np.ptp(y)  # max - min
            dy = max(y_range * 0.03, 1e-8)
        # 按y从小到大排序，逐个放置
        order = np.argsort(y)
        x_sorted = np.zeros(n, dtype=float)
        y_sorted = y[order]
        placed = []
        for i, yi in enumerate(y_sorted):
            # 候选位置：中间、左、右、左2、右2……
            candidates = [0.0]
            for k in range(1, 200):
                candidates.extend([-k * dx, k * dx])
            chosen_x = None
            for cand in candidates:
                collision = False
                for (xp, yp) in placed:
                    # 只检查纵向接近的点
                    if abs(yi - yp) < dy:
                        # 横向也太近，就算撞上
                        if abs(cand - xp) < dx * 0.95:
                            collision = True
                            break
                if not collision:
                    chosen_x = cand
                    break
            if chosen_x is None:
                chosen_x = 0.0
            x_sorted[i] = x_centre + chosen_x
            placed.append((chosen_x, yi))
        # 还原回原始顺序
        x_positions = np.empty(n, dtype=float)
        x_positions[order] = x_sorted
        return x_positions

    bp = [df[g].values for g in tag_Xs]
    bp = ax.boxplot(bp, positions=x, widths=bar_w, patch_artist=True,
                    showfliers=False)  # 异常点也可以自己叠加，先关掉
    for i, box in enumerate(bp['boxes']):
        box.set(facecolor='white', edgecolor=clrs[i], linewidth=lw)
    for i, med in enumerate(bp['medians']):
        med.set(color=clrs[i], linewidth=lw)
    for item in bp['whiskers']:
        item.set(color='black', linewidth=lw * 0.8)
    for item in bp['caps']:
        item.set(color='black', linewidth=lw * 0.8)
    # 再叠加 swarm点
    for i, g in enumerate(tag_Xs):
        y = df[g].values
        xs = _swarm_positions(y, x_centre=x[i], dx=0.045)
        ax.scatter(xs, y, s=28, color=clrs[i], zorder=4)
    return ax


def _sub_internal_sem(arr):
    s = np.std(arr, ddof=1) / np.sqrt(len(arr))
    return np.mean(arr).tolist(), s.tolist()


def _sub_internal_med(arr):
    m = np.median(arr).tolist()
    q1 = np.percentile(arr, 25).tolist()
    q3 = np.percentile(arr, 75).tolist()
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    # return m, q1, q3, iqr, lower, upper

    # # Tukey whiskers: 落在范围内的最小/最大值
    # whis_low = np.min(arr[arr >= lower])
    # whis_high = np.max(arr[arr <= upper])
    m = np.percentile(arr, 50).tolist()

    inliers = arr[(arr >= lower) & (arr <= upper)]
    outliers = arr[(arr < lower) | (arr > upper)]
    # whiskers用 Tukey boxplot的定义
    whis_low = np.min(inliers).tolist()
    whis_high = np.max(inliers).tolist()
    return m, q1, q3, iqr, whis_low, whis_high, inliers, outliers


def _nat_grped_bar1(ax, df, tag_Xs, clrs):  # ,p_texts=None):
    x = np.arange(len(tag_Xs))  # groups
    bar_w = .62
    # means, sems = [], []
    # for g in tag_Xs:
    #     s = np.std(df[g], ddof=1) / np.sqrt(df.shape[0])
    #     sems.append(s.tolist())
    #     means.append(np.mean(df[g]).tolist())
    med, q1, q3 = [], [], []
    for g in tag_Xs:
        # med.append(np.median(df[g]).tolist())
        q1.append(np.percentile(df[g], 25).tolist())
        q3.append(np.percentile(df[g], 75).tolist())
        med.append(np.percentile(df[g], 50).tolist())

    def _internal_outliers(i, g, xi, lw=2):
        iqr = q3[i] - q1[i]
        lower = q1[i] - 1.5 * iqr
        upper = q3[i] + 1.5 * iqr
        # Tukey whiskers: 落在范围内的最小/最大值
        arr = df[g].values.astype(DTY_FLT)
        # whis_low = np.min(arr[arr >= lower])
        # whis_high = np.max(arr[arr <= upper])
        inliers = arr[(arr >= lower) & (arr <= upper)]
        outliers = arr[(arr < lower) | (arr > upper)]
        # # whiskers用 Tukey boxplot的定义
        whis_low = np.min(inliers).tolist()
        whis_high = np.max(inliers).tolist()

        rect = Rectangle((
            xi - bar_w / 2, q1[i]), bar_w, iqr, facecolor='white',
            edgecolor=clrs[i], linewidth=lw, zorder=2)
        ax.add_patch(rect)
        ax.hlines(med[i], xi - bar_w / 2, xi + bar_w / 2, colors=clrs[i],
                  linewidth=lw, zorder=3)  # 中位数横线
        kw = {'linewidth': lw * 0.8, 'zorder': 2}  # whiskers竖线
        ax.vlines(xi, whis_low, q1[i], colors=clrs[i], **kw)
        ax.vlines(xi, q3[i], whis_high, colors=clrs[i], **kw)
        capw = bar_w * .76  # bar_w * 0.35  # whisker caps
        kw['colors'] = clrs[i]
        ax.hlines(whis_low, xi - capw / 2, xi + capw / 2, **kw)
        ax.hlines(whis_high, xi - capw / 2, xi + capw / 2, **kw)
        # # 盒子内/正常点：加横向抖动
        if len(inliers) > 0:
            jitter_in = np.linspace(-0.13, 0.13, len(inliers))
            ax.scatter(np.full(len(inliers), xi) + jitter_in,
                       inliers, s=28, color=clrs[i], zorder=4)
        # # 异常点：单独画，通常也略加一点横向抖动
        if len(outliers) > 0:
            jitter_out = np.linspace(-0.05, 0.05, len(outliers))
            ax.scatter(np.full(len(outliers), xi) + jitter_out,
                       # outliers, s=32, color=clrs[i], zorder=5)
                       outliers, s=34, facecolor='white', edgecolor=clrs[i],
                       linewidth=lw * 0.6, zorder=5)
        return ax

    for i, g in enumerate(tag_Xs):  # 柱子+误差线
        # ax.bar(x[i], means[i], width=bar_w, facecolor='white',
        #        edgecolor=clrs[i], linewidth=2.2, zorder=1)
        # ax.errorbar(x[i], means[i], yerr=sems[i], fmt='none',
        #             ecolor=clrs[i], elinewidth=1.8,
        #             capsize=6, capthick=1.8, zorder=3)
        # # 散点：给一点横向抖动
        # yvals = df[g]
        # jitter = np.linspace(-.13, .13, len(yvals))
        # ax.scatter(np.full(len(yvals), x[i]) + jitter, yvals,
        #            s=28, color=clrs[i], zorder=4)

        # lower = med[i] - q1[i]
        # upper = q3[i] - med[i]
        # ax.bar(x[i], med[i], width=bar_w, facecolor='white',
        #        edgecolor=clrs[i], linewidth=2.2, zorder=1)
        # ax.errorbar(x[i],  # med[i], yerr=np.array([[lower], [upper]]),
        #             med[i], yerr=np.array([[lower, upper], ]).T, fmt='_',
        #             color=clrs[i], ecolor=clrs[i], elinewidth=1.8,
        #             capsize=6, capthick=1.8, markersize=18, zorder=4)
        ax = _internal_outliers(i, g, x[i], lw=2)
    # _internal_with_swarm(ax, df, tag_Xs, clrs, x, bar_w, lw=2)
    return ax


def _sub_inter_jitter(ax, bar_w, xi, arr, c, cp, lw=1.26, jit=True,
                      lb=None):  # lb_box=None,lb_pts=None,lb_out=None):
    # med, q1, q3, iqr, lower, upper = _sub_internal_med(arr)
    # inliers = arr[(arr >= lower) & (arr <= upper)]
    # outliers = arr[(arr < lower) | (arr > upper)]
    # whis_low = np.min(inliers).tolist()
    # whis_high = np.max(inliers).tolist()
    (med, q1, q3, iqr, whis_low, whis_high,
     inliers, outliers) = _sub_internal_med(arr)
    rect = Rectangle((xi - bar_w / 2, q1), bar_w, iqr, facecolor=c,
                     edgecolor='black', linewidth=lw, zorder=1,
                     label=lb)  # label=None if jit else lb) #label=lb_box)
    ax.add_patch(rect)
    ax.hlines(med, xi - bar_w / 2, xi + bar_w / 2, colors='black',
              linewidth=lw, zorder=2)    # 中位数横线 colors=c,
    kw = {'linewidth': lw, 'zorder': 1}  # whiskers竖线
    ax.vlines(xi, whis_low, q1, colors=c, **kw)   # =c
    ax.vlines(xi, q3, whis_high, colors=c, **kw)  # =c
    capw = bar_w * 0.78  # whisker caps
    kw['colors'] = 'black'  # c
    ax.hlines(whis_low, xi - capw / 2, xi + capw / 2, **kw)
    ax.hlines(whis_high, xi - capw / 2, xi + capw / 2, **kw)

    tin, tout = bar_w * 0.31, bar_w * 0.24
    # 异常点：单独画，通常也略加一点横向抖动
    if len(outliers) > 0:
        jitter_out = np.linspace(-tout, tout, len(outliers))
        # jitter_out = np.linspace(-0.072, 0.072, len(outliers))
        ax.scatter(np.full(len(outliers), xi) + jitter_out,
                   outliers, s=24, facecolor='white', edgecolor=cp,
                   linewidth=lw * 0.6, zorder=4)  # ,label=lb_out)
    if not jit:
        return ax
    # 盒子内/正常点：加横向抖动
    if len(inliers) > 0:
        jitter_in = np.linspace(-tin, tin, len(inliers))
        # jitter_in = np.linspace(-0.11, 0.11, len(inliers))  # 0.045
        ax.scatter(np.full(len(inliers), xi) + jitter_in, inliers,
                   s=21, color=cp, edgecolor='none', zorder=3
                   )  # ,label=lb)  # ,label=lb_pts)
    return ax


def _nat_grped_bar2(ax, df, tag_Xs, tag_Ys, clrs, lb, lw=1.6):
    x = np.arange(len(tag_Xs))  # groups
    bar_w = 0.42  # 0.32
    # def _internal(i, prenatal, postnatal):
    #     x1 = float(x[i] - bar_w / 2)
    #     x2 = float(x[i] + bar_w / 2)
    #     return ax

    for i, (g_pre, g_post) in enumerate(zip(tag_Xs, tag_Ys)):
        prenatal = df[g_pre].values.astype(DTY_FLT)
        postnatal = df[g_post].values.astype(DTY_FLT)
        # m1, s1 = _sub_internal_sem(prenatal)
        # m2, s2 = _sub_internal_sem(postnatal)
        x1 = float(x[i] - bar_w / 2)
        x2 = float(x[i] + bar_w / 2)
        # light_c, dark_c = clrs[2 * i + 1], clrs[2 * i + 2]
        # light_c, dark_c = clrs[1], clrs[2]

        # # Prenatal 柱子
        # ax.bar(x1, m1, width=bar_w, facecolor=light_c, edgecolor='black',
        #        linewidth=lw, zorder=1)
        # pdb.set_trace()
        # ax.errorbar(x1, m1, yerr=s1, fmt='none', ecolor='black',
        #             elinewidth=lw - 0.3, capsize=4, capthink=1.3, zorder=3)
        # # Postnatal 柱子
        # ax.bar(x2, m2, width=bar_w, facecolor=dark_c, edgecolor='black',
        #        linewidth=lw, zorder=1)
        # ax.errorbar(x2, m2, yerr=s2, fmt='none', ecolor='black',
        #             elinewidth=lw - 0.3, capsize=4, capthink=1.3, zorder=3)
        # # 散点
        # jitter1 = np.linspace(-0.045, 0.045, len(prenatal))
        # jitter2 = np.linspace(-0.045, 0.045, len(postnatal))
        # ax.scatter(np.full(len(prenatal), x1) + jitter1, prenatal,
        #            s=18, color=light_c, edgecolor='none', zorder=4)
        # ax.scatter(np.full(len(postnatal), x2) + jitter2, postnatal,
        #            s=18, color=dark_c, edgecolor='none', zorder=4)
        # # 显著性横线

        # ax = _sub_inter_jitter(ax, bar_w, x1, prenatal, light_c, clrs[3])
        # ax = _sub_inter_jitter(ax, bar_w, x2, postnatal, dark_c, clrs[4])
        ax = _sub_inter_jitter(ax, bar_w, x1, prenatal, 'darkgrey', clrs[0],
                               # lb_box=lb[i] if i == 0 else None,
                               # lb_pts=lb[i] if i == 0 else None,
                               # lb_out=lb[i] if i == 0 else None)
                               lb=lb[i] if i == 0 else None)
        ax = _sub_inter_jitter(ax, bar_w, x2, postnatal, clrs[1], clrs[3],
                               lb=lb[i] if i == 0 else None)
    return ax


def _nat_grped_bar3(ax, df, tag_Xs, tag_Ys, tag_Zs, clrs, lb, lw=1.6):
    x = np.arange(len(tag_Xs)).tolist()  # groups
    bar_w = 0.305  # 0.36  # bar_w = 0.61
    for i, (g_pre, g_post, g_alt) in enumerate(zip(tag_Xs, tag_Ys, tag_Zs)):
        prenatal = df[g_pre].values.astype(DTY_FLT)
        postnatal = df[g_post].values.astype(DTY_FLT)
        alter = df[g_alt].values.astype(DTY_FLT)
        x1 = float(x[i] - bar_w)  # / 2)
        x2 = float(x[i] + bar_w)  # / 2)
        dflt, light_c, dark_c = clrs[0], clrs[1], clrs[2]
        ax = _sub_inter_jitter(ax, bar_w, x1, prenatal, 'darkgrey', dflt,
                               lb=lb[i] if i == 0 else None)
        ax = _sub_inter_jitter(ax, bar_w, x[i], postnatal, light_c, clrs[3],
                               lb=lb[i] if i == 0 else None)
        ax = _sub_inter_jitter(ax, bar_w, x2, alter, dark_c, clrs[4],
                               lb=lb[i] if i == 0 else None)
    return ax


def _nat_grped_barc(ax, df, clrs, lb, *tags, lw=1.6):
    tag_Xs, bar_w = tags[0], 0.19
    x = np.arange(len(tag_Xs)).tolist()
    for i, gs in enumerate(zip(*tags)):
        prenatal = df[gs[0]].values.astype(DTY_FLT)
        postnatal = df[gs[1]].values.astype(DTY_FLT)
        alter = df[gs[2]].values.astype(DTY_FLT)
        alter_max = df[gs[3]].values.astype(DTY_FLT)
        alter_avg = df[gs[4]].values.astype(DTY_FLT)
        x1 = float(x[i] - bar_w * 2)
        x2 = float(x[i] - bar_w)
        x4 = float(x[i] + bar_w)
        x5 = float(x[i] + bar_w * 2)
        dflt, light_c, dark_c = clrs[0], clrs[1], clrs[2],
        # pdb.set_trace()
        ax = _sub_inter_jitter(ax, bar_w, x1, prenatal, 'darkgrey', dflt,
                               lb=lb[0] if i == 0 else None)
        ax = _sub_inter_jitter(ax, bar_w, x2, postnatal, light_c, clrs[3],
                               lb=lb[1] if i == 0 else None)
        ax = _sub_inter_jitter(ax, bar_w, x[i], alter, dark_c, clrs[4],
                               lb=lb[2] if i == 0 else None)
        ax = _sub_inter_jitter(ax, bar_w, x4, alter_max, clrs[5], clrs[7],
                               lb=lb[3] if i == 0 else None)
        ax = _sub_inter_jitter(ax, bar_w, x5, alter_avg, clrs[6], clrs[8],
                               lb=lb[4] if i == 0 else None)
    return ax


def multi_boxplot_rect_revised(df, tag_Xs, tag_Ys=None, tag_Zs=None,
                               tag_Ts=None, tag_Extra=None,
                               labels=('ori', 'ext', 'alt', 'ext (avg)',
                                       'alt (avg)'), annotX=tuple(),
                               # palette=['black', '#1b8e3e', '#1b8e3e'],
                               # palette=[
                               #     'black',
                               #     # '#295F85', '#8D3322', '#116DA9', '#B03C2B',
                               #     '#184879', '#762E29', '#387EB8', '#C24E44',
                               #     # '#024163', '#8E0f31',
                               #     '#066190', '#C42238', '#77AECD', '#D98380'],
                               palette=('black', '#387EB8', '#C24E44', '#184879', '#762E29', '#066190', '#C42238', '#024163', '#8E0F31'),
                               locate='best', figname='', figsize='M-WS'):
    fig, ax = plt.subplots(figsize=_setup_config[figsize])
    if tag_Ys is None:
        df_alt = df[tag_Xs]  # _bp_dat_X(df, tag_Xs)
        ax = _nat_grped_bar1(ax, df_alt, tag_Xs, palette)
        # ax.set_xticks(np.arange(len(tag_Xs)))
        # ax.set_xticklabels(labels[:len(tag_Xs)])  # tag_Xs)
        # # ax.tick_params(axis='both', direction='out')
    elif tag_Zs is None:
        df_alt = df[tag_Xs + tag_Ys]
        ax = _nat_grped_bar2(ax, df_alt, tag_Xs, tag_Ys, palette, labels)
        # ax.set_xticks(np.arange(len(tag_Xs)))
        # ax.set_xticklabels(labels[:len(tag_Xs)])  # tag_Xs)
        # ax.legend()
    elif tag_Ts is None:
        # x = np.arange(len(tag_Xs)) * 2
        fig, ax = plt.subplots(figsize=(3.45, 2.52))
        # fig.set_size_inches(3.95, 2.52)  # 动态修改为 8x4英寸
        df_alt = df[tag_Xs + tag_Ys + tag_Zs]
        ax = _nat_grped_bar3(ax, df_alt, tag_Xs, tag_Ys, tag_Zs, palette, labels)
        # ax.set_xticks(np.arange(len(tag_Xs)))     # x)
        # ax.set_xticklabels(labels[:len(tag_Xs)])
    else:
        fig, ax = plt.subplots(figsize=(5.95, 2.52))
        df_alt = df[tag_Xs + tag_Ys + tag_Zs + tag_Ts + tag_Extra]
        ax = _nat_grped_barc(ax, df_alt, palette, labels, *(
            tag_Xs, tag_Ys, tag_Zs, tag_Ts, tag_Extra,))
        # sns.move_legend(ax, locate, title='')
        ax.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0,
                  loc=locate, labelspacing=.07, prop={'size': 9},
                  frameon=False)  # ax.legend()

    if annotX:
        ax.set_xticks(np.arange(len(tag_Xs)))
        # tmp = ax.get_xticks()
        # ax.set_xticks(tmp)
        ax.set_xticklabels(annotX)  # ,rotation=rotate|0)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax = _style_set_axis(ax, invt=False)  # fig.gca()
    ax.tick_params(direction='out')  # axis='x|y','both',
    # ax.tick_params(axis='x', direction='out')
    # ax.tick_params(axis='y', direction='out')
    # fig = _setup_figsize(fig, figsize, invt=False)
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
    for sc in scores:  # for i, sc in enumerate(scores):
        ax.plot(angles, sc)
    # ax.set_thetagrids(angles * 180 / np.pi, labels)  # 标签显示
    # ax.set_theta_zero_location('N')  # 设置雷达图的0度起始位置
    # ax.set_rlim(0, 100)  # 设置雷达图的坐标刻度范围
    # ax.set_rlabel_position(270)  # 设置坐标显示角度，相对于起始角度的偏移量

    kws = {}  # {'fontsize': 14, 'style': 'italic'}
    if stylish:
        for sc in scores:  # for i, sc in enumerate(scores):
            ax.fill(angles, sc, alpha=.25)
        kws['style'] = 'italic'
    ax.set_thetagrids(angles * 180 / np.pi, labels, **kws)
    ax.set_theta_zero_location('N')  # 'E'
    ax.set_rlabel_position(225)
    return ax


def _radar_X_revised(ax, df, tag_Xs, annotX, clockwise=False,
                     palette = (
        # # 'darkgrey', 'black', '#066190', '#C42238', '#024163', '#8E0F31',
        # # '#77AECD', '#D98380', '#066190', '#C42238'], stylish=False):
        # # 'darkgrey', 'black', '#066190', '#024163', '#C42238', '#8E0F31',
        # # '#77AECD', '#066190', '#D98380', '#C42238'], stylish=False):
        # 'black', 'darkgrey', '#024163', '#77AECD', '#8E0F31', '#D98380',
        # '#066190', '#77AECD', '#C42238', '#D98380'], stylish=False):
        'black', 'darkgrey', '#184879', '#6299CA', '#762E29', '#C87271',
        '#387EB8', '#6299CA', '#C24E44', '#C87271'), stylish=False):

    angles = np.linspace(0, 2 * np.pi, len(tag_Xs), endpoint=False)
    labels = annotX + [annotX[0]]
    angles = np.concatenate([angles, [angles[0], ]])
    if clockwise:
        angles = angles[::-1]
    scores = df[tag_Xs].values.astype(DTY_FLT)
    scores = np.concatenate([scores, scores[:, 0].reshape(-1, 1)], axis=1)

    for i, sc in enumerate(scores):
        k = {'linestyle': '-.'} if i > 2 else {}  # 'dashed'}
        ax.plot(angles, sc, color=palette[i * 2], **k)
    ax.grid(alpha=.21)  # False)
    kws = {}
    if stylish:
        for i, sc in enumerate(scores):
            ax.fill(angles, sc, alpha=.25, facecolor=palette[i * 2 + 1])
        kws['style'] = 'italic'
    ax.set_thetagrids(angles * 180 / np.pi, labels, **kws)
    ax.set_theta_zero_location('N')
    ax.set_rlabel_position(225)
    return ax


def radar_chart(df, tag_Xs,  # tag_Ys=None, tag_Zs=None,
                annotX=tuple(), annotY=tuple(), clockwise=True,
                stylish=False, figname='', figsize='M-WS'):
    fig = plt.figure(figsize=_setup_config[figsize])
    ax = fig.add_subplot(111, polar=True)  # 设置极坐标格式
    # ax = _radar_X(ax, df, tag_Xs, annotX, clockwise,
    #               stylish=stylish)
    ax = _radar_X_revised(
        ax, df, tag_Xs, annotX, clockwise, stylish=stylish)

    if annotY:  # if len(annotY) > 3:
        # plt.legend(annotY, loc="best",
        #            labelspacing=.07, prop={'size': 9})
        plt.legend(annotY,  # bbox_to_anchor=(1.05, 1),
                   bbox_to_anchor=(1.25, 1), frameon=False,
                   borderaxespad=0, loc='upper left',
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


def grped_radar_cht(df_pl, tag_Xs, annotX=tuple(), annotY=tuple(),
                    clockwise=True, figname='', stylish=True):
    if len(df_pl) >= 5:
        fsz, fb, ft = (17, 3.4), (1.65, 1), {'fontsize': 'large'}
    else:
        fsz, fb, ft = (13, 2.47), (1.76, 1), {}
    # fsz = (17, 3.4) if len(df_pl) >= 5 else (13, 2.47)  # fb
    fig, ax = plt.subplots(ncols=len(df_pl), figsize=fsz,
                           subplot_kw={'projection': 'polar'})
    for i, df in enumerate(df_pl):
        ax[i] = _radar_X(ax[i], df, tag_Xs, annotX, clockwise,
                         stylish=stylish)
    if annotY:
        # lines, labels = fig.axes[-1].get_legend_handles_labels()
        # fig.legend(lines, labels,
        plt.legend(annotY,  # loc='upper right', prop={'size': 9})
                   # bbox_to_anchor=(.74, .96), ncol=4, framealpha=1)
                   # , fontsize='large') (1.65,1)
                   bbox_to_anchor=fb, borderaxespad=.05,
                   labelspacing=.09, frameon=False, **ft)
    del fsz, fb, ft  # del fsz  # del fsz, fb
    # fig.tight_layout()  # 调整整体空白，调整子图间距
    kw = {'left': None, 'bottom': None, 'right': None, 'top': None}
    plt.subplots_adjust(wspace=.34, hspace=0, **kw)
    fig.savefig("{}{}".format(
        figname, '.pdf'), dpi=300, bbox_inches='tight')
    # _setup_figshow(fig, figname)
    plt.close(fig)
    return


# ------------------------------
# 南丁格尔玫瑰图
# https://zhuanlan.zhihu.com/p/367450085
# https://blog.csdn.net/gb4215287/article/details/108744093
# https://stackoverflow.com/questions/76047803/typeerror-figurebase-gca-got-an-unexpected-keyword-argument-projection

# https://comate.baidu.com/zh/page/to0m6xpc4v0
# https://blog.csdn.net/Drajor/article/details/116572069


def _petal_X(ax, df, tag_Xs, annotX):
    # 准备好角度和半径
    angles = np.arange(0, 2 * np.pi, 2 * np.pi / len(tag_Xs))
    radius = np.array(df[tag_Xs].mean())  # or .mean().values
    # plt.bar(angles, radius, color=[
    #     'blue', 'red', 'yellow', 'green'], width=.3)

    plt.bar(angles, radius, alpha=.74)  # 绘制南丁格尔玫瑰图
    plt.xticks(angles, tag_Xs)          # 添加X轴的标签
    # plt.ylim(-25, 125)  # 设置Y的取值范围，以让中间出现圆圈/显示空心
    plt.yticks([])      # 不显示Y轴的数字标签, 添加数值和标题
    # for a, b in zip(angles, radius):
    #     plt.text(a + 0.03, b + 1, b, va='center', ha='center')
    # plt.title('coxcomb chart', loc='center')

    plt.ylim(0, .817)
    for a, b in zip(angles, radius):
        ia = round(a + .03, 4)  # float('{:.4f}'.format(a+.03))
        ib = round(b + .14, 4)   # float('{:.4f}'.format(b+1.))
        plt.text(ia, ib, round(b - 1, 4), va='center', ha='center')
    return ax


def petal_chart(df, tag_Xs, annotX=tuple(), annotY=tuple(),
                clockwise=True, figname='', figsize='M-WS'):
    fig = plt.figure(figsize=_setup_config[figsize])
    ax = fig.add_subplot(111, polar=True)
    # ax = fig.gca(polar=True)
    ax.set_theta_offset(np.pi / 2)
    if clockwise:
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(0)
    ax = _petal_X(ax, df, tag_Xs, annotX)  # ,clockwise)
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


def tabular_chart(df, tag_Xs, annotX=tuple(), annotY=tuple(),
                  data='', algo='', figname='', cumulate=False):
    columns, rows = tag_Xs, annotY[::-1]
    # Get some pastel shades for the colors
    colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
    n_rows = df.shape[0]  # len(df)
    index = np.arange(len(columns)) + 0.3
    bar_width = 0.4
    # Initialize the vectical-offset for the stacked bar chart.
    y_offset = np.zeros(len(columns))

    # tmp = [0, 3, 4, 1, 2]
    # df_tmp = df.iloc[tmp]  # 0, 2, 4, 1, 3]]

    fig = plt.figure(figsize=(4.95, 1.95))  # 2.65))
    # Plot bars and create text labels for the table
    cell_text = []
    for row in range(n_rows):  # for ri, row in enumerate(tmp):
        plt.bar(index, df.loc[row].values.astype(DTY_FLT), bar_width,
                bottom=y_offset, color=colors[row])  # ri])
        if cumulate:
            y_offset = y_offset + df.loc[row]  # df.loc[row]
        # cell_text.append(['%1.4f' % x for x in y_offset])
        cell_text.append(['%1.4f' % x for x in df.loc[row]])
    # Reserve colors and text labels to display the last value at the top.
    colors = colors[::-1]
    cell_text.reverse()  # cell_text = [cell_text[i] for i in tmp[::-1]]

    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text, rowLabels=rows, rowColours=colors,
                          colLabels=annotX, loc='bottom')  # colLabels=columns,
    # Adjust layerout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.ylabel(data)  # algo)  # algo)  # "")
    # plt.yticks()
    plt.xticks([])
    plt.title(algo)   # data)  # data.upper())  # "L")
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


# ------------------------------
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
# https://github.com/fonttools/fonttools/issues/3538
#
# https://blog.csdn.net/weixin_44637060/article/details/126127700
# https://www.zhihu.com/question/507342145
# https://blog.csdn.net/GAN_player/article/details/78543643
# https://blog.csdn.net/qq_40994260/article/details/114478555
# https://blog.csdn.net/CSDN_LYY/article/details/114014856
#
_nat_sci_cs = ['#073068', '#206FB6', '#6BADD7', '#C5DAEE',
               '#FDDFD0', '#FC9171', '#EE3B2A', '#A60E16']
