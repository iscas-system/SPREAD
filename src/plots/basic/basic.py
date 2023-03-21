import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as plt_ticker
import matplotlib.lines as mlines
from matplotlib.patches import Patch
# from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator
import numpy as np


def init_global_params():
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.size'] = 15
    mpl.rcParams['font.family'] = 'Arial'


if __name__ == "__main__":
    init_global_params()
