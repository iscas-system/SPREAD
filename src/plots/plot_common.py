from os import path
from basic import *
from cycler import cycler
from scipy import stats
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import os
import threading

line_cycler   = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
                 cycler(linestyle=["-", "--", "-.", ":", "-", "--", "-."]))
marker_cycler = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
                 cycler(linestyle=["none", "none", "none", "none", "none", "none", "none"]) +
                 cycler(marker=["4", "2", "3", "1", "+", "x", "."]))
color_cycler   = cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"])
colors = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442", "gray"]
# plt.rc("text", usetex=True)
# plt.rc("text.latex", preamble=r"\usepackage{newpxtext}\usepackage{newpxmath}\usepackage{commath}\usepackage{mathtools}")
# plt.rc("font", family="serif", size=18.)
plt.rc("legend", loc="best", fontsize="medium", fancybox=True, framealpha=0.5)
plt.rc("lines", linewidth=2.5, markersize=10, markeredgewidth=2.5)

def output_path(filename):
    figure_path_env = os.getenv("FIGURE_PATH")
    if figure_path_env is not None:
        return path.join(figure_path_env, filename)
    return path.join('outputs', filename)


mkdir_lock = threading.Lock()

def must_exist_dir(dir_path: str):
    if os.path.exists(dir_path):
        assert os.path.isdir(dir_path)
        return
    try:
        mkdir_lock.acquire()
        if os.path.exists(dir_path):
            assert os.path.isdir(dir_path)
            return
        else:
            os.mkdir(dir_path)
            return
    finally:
        mkdir_lock.release()

def inside_ticks(ax, x=True, y=True):
    if y:
        ax.tick_params(axis="y", which='major', direction="in", length=4)
        ax.tick_params(axis="y", which='minor', direction="in", length=2)
    if x:
        ax.tick_params(axis="x", which='major', direction="in", length=4)
        ax.tick_params(axis="x", which='minor', direction="in", length=2)


def save_fig(fig, fig_path):
    print(f"### saving figure to {fig_path} ###")
    fig.savefig(fig_path, dpi=400, format='pdf', bbox_inches='tight')


def plot_single(plot_func):
    fig, ax = plt.subplots()
    plot_func(ax)
    return fig

def most(iterable):
    return stats.mode(iterable)[0][0]