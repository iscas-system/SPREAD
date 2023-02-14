from os import path
from basic import *
from cycler import cycler
from scipy import stats
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

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
    return path.join('outputs', filename)


def inside_ticks(ax, x=True, y=True):
    if y:
        ax.tick_params(axis="y", which='major', direction="in", length=4)
        ax.tick_params(axis="y", which='minor', direction="in", length=2)
    if x:
        ax.tick_params(axis="x", which='major', direction="in", length=4)
        ax.tick_params(axis="x", which='minor', direction="in", length=2)


def save_fig(fig, fig_path):
    fig.savefig(fig_path, dpi=400, format='pdf', bbox_inches='tight')


def plot_single(plot_func):
    fig, ax = plt.subplots()
    plot_func(ax)
    return fig

def most(iterable):
    return stats.mode(iterable)[0][0]