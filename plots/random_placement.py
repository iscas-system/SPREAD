import numpy as np

from record_preprocess import *
from itertools import count
from common import *
from typing import Callable, Any
from matplotlib.ticker import PercentFormatter, FuncFormatter


def plot_random_placement_box(ax, data_source: DataSourceName, schedulers: List[SchedulerName]):

    box_data = list()
    cluster_name = ClusterName.Cluster10GPUs
    total_profit = cluster_name_to_spec(cluster_name)["total_profit"]
    for scheduler in schedulers:
        play_record = extract_play_record(SessionMode.RandomPlacement,
                                          cluster_name=cluster_name,
                                          data_source_name=data_source,
                                          scheduler_name=scheduler)
        assert len(play_record) == 1
        play_record = play_record[0]
        items = list()
        for stat in play_record.assignment_statistics:
            items.append(1. - stat.profit / total_profit)
        print(f"scheduler {scheduler}, avg profits: {np.mean(items)}")
        box_data.append(items)

    handles = list()

    bp = ax.boxplot(box_data, patch_artist=True, notch="True")
    for i, patch in enumerate(bp["boxes"]):
        scheduler = schedulers[i]
        spec = scheduler_to_spec(scheduler_name=scheduler)
        color = spec["color"]
        patch.set_facecolor(color)
        hatch = "/"
        patch.set_hatch(hatch)
        handle = Patch(
            facecolor=color,
            edgecolor="black",
            label=spec["label"],
            hatch=hatch
        )
        handles.append(handle)

    for whisker in bp['whiskers']:
        whisker.set(color='black',
                    linewidth=1.5,
                    linestyle=":")

    # changing color and linewidth of
    # caps
    for cap in bp['caps']:
        cap.set(color='black',
                linewidth=2)

    # changing color and linewidth of
    # medians
    for median in bp['medians']:
        median.set(color='black',
                   linewidth=3)

    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker='D',
                  color="black",
                  alpha=0.5)

    # xtick_labels = [scheduler_to_spec(scheduler)["label"] for scheduler in schedulers]
    # ax.set_xticklabels(xtick_labels, rotation=45)
    ax.yaxis.set_major_formatter(plt_ticker.FuncFormatter('{0:.0%}'.format))
    ax.yaxis.grid(True)
    ax.set_xticks([])
    ax.set_ylabel("RFD")
    ax.set_xlabel("Schedulers")
    ax.set_xlabel(data_source_to_spec(data_source_name=data_source)["label"])
    return handles


def plot_random_placement_boxes():
    original_fontsize = mpl.rcParams["font.size"]
    mpl.rcParams.update({'font.size': 28})
    schedulers = [SchedulerName.MMKP_strict,
                  SchedulerName.MMKP_strict_no_split,
                  SchedulerName.KubeShare,
                  SchedulerName.Gavel,
                  SchedulerName.Tiresias,
                  SchedulerName.Kubernetes]
    data_source_names = [
        DataSourceName.DataSourceAli,
        DataSourceName.DataSourceAliFixNew,
        DataSourceName.DataSourceAliUni,
        DataSourceName.DataSourcePhi,
        DataSourceName.DataSourcePhiFixNew,
        DataSourceName.DataSourcePhiUni
    ]
    col = 2
    fig, axes = plt.subplots(3, col, figsize=(20, 14))
    handles = None
    for i, data_source_name in enumerate(data_source_names):
        handles = plot_random_placement_box(axes[i // col, i % col], data_source_name, schedulers)

    fig.tight_layout()
    lgd = fig.legend(handles=handles, loc=(0.005, 0.92), ncol=len(handles))
    lgd.get_frame().set_alpha(None)
    fig.subplots_adjust(top=0.87)
    save_fig(fig, output_path("random_placement_10GPUs_boxes.pdf"))
    mpl.rcParams.update({'font.size': original_fontsize})


def plot_spreading_distribution_bar():
    original_fontsize = mpl.rcParams["font.size"]
    mpl.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize=(12, 4))
    width = 0.3
    cluster_name = ClusterName.Cluster10GPUs
    items = list()
    data_source_names = [
        DataSourceName.DataSourceAli,
        DataSourceName.DataSourceAliFixNew,
        DataSourceName.DataSourceAliUni,
        DataSourceName.DataSourcePhi,
        DataSourceName.DataSourcePhiFixNew,
        DataSourceName.DataSourcePhiUni
    ]
    rfd_improvements = list()
    total_profit = cluster_name_to_spec(cluster_name)["total_profit"]
    for i, data_source_name in enumerate(data_source_names):
        total_deployed_jobs = 0
        total_spread_jobs = 0
        play_record = extract_play_record(SessionMode.RandomPlacement,
                                          cluster_name=cluster_name,
                                          data_source_name=data_source_name,
                                          scheduler_name=SchedulerName.MMKP_strict)
        assert len(play_record) == 1
        play_record = play_record[0]
        for stat in play_record.assignment_statistics:
            total_deployed_jobs += stat.deployed_job_size
            total_spread_jobs += stat.deployed_dist_job_size
        items.append(total_spread_jobs/total_deployed_jobs)

        no_split_play_record = extract_play_record(SessionMode.RandomPlacement,
                                          cluster_name=cluster_name,
                                          data_source_name=data_source_name,
                                          scheduler_name=SchedulerName.MMKP_strict_no_split)
        assert len(no_split_play_record) == 1
        no_split_play_record = no_split_play_record[0]

        def get_avg_rfd(rc):
            profits = list()
            for st in rc.assignment_statistics:
                profits.append(1. - st.profit / total_profit)
            return np.mean(profits)

        rfd_sp = get_avg_rfd(play_record)
        rfd_sp_no_split = get_avg_rfd(no_split_play_record)

        rfd_improvements.append(rfd_sp_no_split - rfd_sp)

    X = np.arange(len(data_source_names))
    spec = scheduler_to_spec(scheduler_name=SchedulerName.MMKP_strict)
    ax.bar(X, items,
           width=width,
           color=spec["color"],
           label="Ratio of Spread Jobs",
           edgecolor="black",
           hatch="/")

    xticks = [data_source_to_spec(data_source_name)["label"] for data_source_name in data_source_names]
    ax.tick_params(axis='y', labelcolor=spec["color"])
    ax.set_xticks(X, xticks)
    ax.yaxis.set_major_formatter(plt_ticker.FuncFormatter('{0:.0%}'.format))
    ax.set_ylabel("Ratio of Spread Jobs", color=spec["color"])
    ax.set_xlabel("Workloads")
    # ax.yaxis.grid(True)

    ax_ratio = ax.twinx()
    # ax_ratio.set_ylim([0., 1.])
    ratio_color = colors[2]
    ax_ratio.set_ylabel("RFD Improvement", color=ratio_color)
    ax_ratio.plot(
        X, rfd_improvements,
        linestyle="solid", marker="o",
        label="RFD Improvement",
        color=ratio_color
    )
    ax_ratio.yaxis.set_major_formatter(plt_ticker.FuncFormatter('{0:.0%}'.format))
    ax_ratio.tick_params(axis='y', labelcolor=ratio_color)
    fig.legend(loc=(0.2, 0.6))
    # ax.xaxis.grid(True)
    fig.tight_layout()
    save_fig(fig, output_path("random_placement_spreading_distribution.pdf"))
    mpl.rcParams.update({'font.size': original_fontsize})

def main():
    plot_random_placement_boxes()
    plot_spreading_distribution_bar()


if __name__ == '__main__':
    main()
