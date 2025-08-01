# coding: utf-8


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import pdb

from pyfair.granite.draw_fancy import (
    boxplot_rect, multi_boxplot_rect, radar_chart)


n, n_d = 10, 5
far_grp = np.random.rand(n, n_d)
far_ext = np.random.rand(n, n_d)
far_ext_alt = np.random.rand(n, n_d)
labels = ['DP', 'EO', 'PQP', r'$\mathbf{df}_\text{prev}$',
          r'$\hat{\mathbf{df}}_\text{prev}$']
annots = ['grp', 'ext', 'ext_alt']

df = sns.load_dataset("titanic")
lb_grp = ['DP', 'EO', 'PP', 'df_prev', 'df_hat_prev']
lb_ext = ['ext1', 'ext2', 'ext3', 'df', 'hat_df']  # nonbin
lb_ext_alt = ['alt1', 'alt2', 'alt3', 'df_avg', 'hat_df_avg']
df_grp = pd.DataFrame(far_grp, columns=lb_grp)
df_ext = pd.DataFrame(far_ext, columns=lb_ext)
df_ext_alt = pd.DataFrame(far_ext, columns=lb_ext_alt)
df_alt = pd.concat([df_grp, df_ext, df_ext_alt], axis=1)


def test_bplot():
    i = 0
    boxplot_rect([far_grp[:, i], far_ext[:, i], far_ext_alt[:, i]
                  ], annots, f'chart_far_grp{i+1}')
    df_tmp = pd.DataFrame({annots[0]: far_grp[:, i], annots[
        1]: far_ext[:, i], annots[2]: far_ext_alt[:, i]})
    multi_boxplot_rect(df_tmp, annots, figname=f'chart_far_gr{i+1}p')

    df_lbl = ['First', 'Second', 'Third']  # {'Third', ..}
    # boxplot_rect([np.nan_to_num(df[df['class'] == i][
    #     'age'].values) for i in df_lbl], df_lbl, 'chart_titanic')
    boxplot_rect([df[df['class'] == i][
        'age'].values for i in df_lbl], df_lbl, 'chart_titanic')
    multi_boxplot_rect(df_alt, lb_grp, figname=f'chart_far_gr{i+1}_dim1')
    multi_boxplot_rect(df_alt, lb_grp, lb_ext, figname=f'chart_far_gr{i+1}_dim2')
    radar_chart(df_alt, lb_grp, annotX=lb_grp,
                figname=f'chart_radar_dim1')
    # pdb.set_trace()

    fig, axs = plt.subplots(2, 3, figsize=(8.1, 4.7))  # =(6,5))
    sns.boxplot(ax=axs[0, 0], data=df, x="class", y="age")
    axs[0, 0].yaxis.grid(True)
    sns.boxplot(ax=axs[0, 1], data=df, x="class", y="age",
                hue="alive")
    axs[0, 1].yaxis.grid(True)
    sns.boxplot(ax=axs[0, 2], data=df[["fare", "age"]],
                orient="v")  # orient="h"
    sns.boxplot(ax=axs[1, 0], data=df, x="deck", y="fare",
                hue="deck", dodge=False)
    axs[1, 0].yaxis.grid(True)
    sns.boxplot(ax=axs[1, 1], data=df, x="deck", y="fare",
                order=["G", "F", "E", "D", "C", "B", "A"],
                hue="deck", dodge=True)
    axs[1, 1].yaxis.grid(True)
    sns.boxplot(ax=axs[1, 2], data=df, x="class", y="age",
                notch=True, showcaps=False, flierprops={
                "marker": "^"}, boxprops={"facecolor": (
                    .4, .6, .8, .5)}, medianprops={"color": "r"})
    axs[1, 2].yaxis.grid(True)
    plt.savefig("chart_bps_sns.pdf", dpi=300)  # plt.show()
    # pdb.set_trace()
    return
