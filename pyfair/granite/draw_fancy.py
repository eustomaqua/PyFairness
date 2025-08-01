# coding: utf-8


import matplotlib.pyplot as plt

from pyfair.facil.draw_prelim import (
    _setup_figsize, _setup_figshow, _setup_config)


def boxplot_rect(Ys, annotX, figname, notch=False, figsize='M-WS'):
    fig, ax = plt.subplots(figsize=_setup_config[
        figsize])  # 111)  # bplot_rect =
    ax.boxplot(x=Ys, notch=notch, vert=True, widths=.3,
               labels=annotX, patch_artist=False,
               medianprops={'linewidth': 1.5},
               showmeans=True, meanline=True,
               showfliers=True)
    fig = _setup_figsize(fig, figsize, invt=False)
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


def multi_boxplot_rect():
    pass


def radar_chart():
    pass


# ------------------------------
# refs:
#
# https://zhuanlan.zhihu.com/p/375866522
# https://www.cnblogs.com/shijingwen/p/15011142.html
# https://www.cnblogs.com/metafullstack/p/17651922.html
#
