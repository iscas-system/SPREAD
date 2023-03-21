import numpy as np

from record_preprocess import *
from itertools import count
from plot_common import *
from typing import Callable, Any
from matplotlib.ticker import PercentFormatter, FuncFormatter


def plot_selectors_box(ax, data_source: DataSourceName, schedulers: List[SchedulerName], session=SessionMode.Selectors):

    box_data = list()
    cluster_name = ClusterName.Cluster10GPUs
    total_profit = cluster_name_to_spec(cluster_name)["total_profit"]
    scheduler_to_session = {
        SchedulerName.MMKP_strict: SessionMode.RandomPlacement,
        SchedulerName.MMKP_strict_no_split: SessionMode.RandomPlacement,
    }

    for scheduler in schedulers:
        sess = scheduler_to_session.get(scheduler, session)
        play_record = extract_play_record(sess,
                                          cluster_name=cluster_name,
                                          data_source_name=data_source,
                                          scheduler_name=scheduler)
        if len(play_record) != 1:
            print(cluster_name, data_source, scheduler)
        assert len(play_record) == 1
        play_record = play_record[0]
        items = list()
        for stat in play_record.assignment_statistics:
            items.append(1. - stat.profit / total_profit)
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
    ax.set_xlabel(data_source_to_spec(data_source_name=data_source)["label"])
    return handles


def plot_selectors_boxes():
    original_fontsize = mpl.rcParams["font.size"]
    mpl.rcParams.update({'font.size': 28})
    schedulers = [SchedulerName.MMKP_strict,
                  SchedulerName.MMKP_strict_random_select,
                  SchedulerName.MMKP_strict_no_split,
                  SchedulerName.MMKP_strict_no_split_random_select,
                  ]
    data_source_names = [
        DataSourceName.DataSourceAli,
        DataSourceName.DataSourceAliFixNew,
        DataSourceName.DataSourceAliUni,
        DataSourceName.DataSourcePhi,
        DataSourceName.DataSourcePhiFixNew,
        DataSourceName.DataSourcePhiUni
    ]
    col = 3
    fig, axes = plt.subplots(2, col, figsize=(18, 10))
    handles = None
    for i, data_source_name in enumerate(data_source_names):
        handles = plot_selectors_box(axes[i // col, i % col], data_source_name, schedulers)

    fig.tight_layout()
    lgd = fig.legend(handles=handles, loc=(0.05, 0.92), ncol=len(handles))
    lgd.get_frame().set_alpha(None)
    fig.subplots_adjust(top=0.88)
    save_fig(fig, output_path("selectors_10GPUs_boxes.pdf"))
    mpl.rcParams.update({'font.size': original_fontsize})


def plot_spreading_distribution_bar():
    original_fontsize = mpl.rcParams["font.size"]
    mpl.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize=(12, 4))
    width = 0.3
    cluster_name = ClusterName.Cluster10GPUs
    data_source_names = [
        DataSourceName.DataSourceAli,
        DataSourceName.DataSourceAliFixNew,
        DataSourceName.DataSourceAliUni,
        DataSourceName.DataSourcePhi,
        DataSourceName.DataSourcePhiFixNew,
        DataSourceName.DataSourcePhiUni
    ]

    def get_spread_ratios(scheduler_name: SchedulerName):
        items = list()
        for i, data_source_name in enumerate(data_source_names):
            total_deployed_jobs = 0
            total_spread_jobs = 0
            play_record = extract_play_record(SessionMode.Selectors,
                                              cluster_name=cluster_name,
                                              data_source_name=data_source_name,
                                              scheduler_name=scheduler_name)
            assert len(play_record) == 1
            play_record = play_record[0]
            for stat in play_record.assignment_statistics:
                total_deployed_jobs += stat.deployed_job_size
                total_spread_jobs += stat.deployed_dist_job_size
            items.append(total_spread_jobs/total_deployed_jobs)
        return items

    MMKP_ratios = get_spread_ratios(SchedulerName.MMKP_strict)
    MMKP_random_ratios = get_spread_ratios(SchedulerName.MMKP_strict_random_select)
    X = np.arange(len(data_source_names))
    for i, group in enumerate(zip((SchedulerName.MMKP_strict, SchedulerName.MMKP_strict_random_select), [MMKP_ratios, MMKP_random_ratios])):
        scheduler, data = group
        spec = scheduler_to_spec(scheduler)
        ax.bar(X + i*width,
               data,
               width=width,
               color=spec["color"],
               label=spec["label"],
               hatch="/",
               edgecolor="black"
               )

    xticks = [data_source_to_spec(data_source_name)["label"] for data_source_name in data_source_names]
    ax.set_xticks(X + width/2, xticks)
    ax.yaxis.set_major_formatter(plt_ticker.FuncFormatter('{0:.0%}'.format))
    lgd = ax.legend(loc=(0.01, 0.3))
    ax.set_ylabel("Ratio of Spread Jobs")
    ax.set_xlabel("Workloads")
    ax.yaxis.grid(True)
    # ax.xaxis.grid(True)
    fig.tight_layout()
    save_fig(fig, output_path("selectors_spreading_distribution.pdf"))
    mpl.rcParams.update({'font.size': original_fontsize})


def plot_spreading_distribution_bar_variants():
    original_fontsize = mpl.rcParams["font.size"]
    mpl.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.3
    cluster_name = ClusterName.Cluster10GPUs
    data_source_names = [
        DataSourceName.DataSourceAli,
        DataSourceName.DataSourceAliFixNew,
        DataSourceName.DataSourceAliUni,
        DataSourceName.DataSourcePhi,
        DataSourceName.DataSourcePhiFixNew,
        DataSourceName.DataSourcePhiUni
    ]

    def get_spread_ratios(scheduler_name: SchedulerName):
        items = list()
        for i, data_source_name in enumerate(data_source_names):
            total_deployed_jobs = 0
            total_spread_jobs = 0
            if scheduler_name == SchedulerName.MMKP_strict_rand_3:
                play_record = extract_play_record(SessionMode.RandomPlacement,
                                              cluster_name=cluster_name,
                                              data_source_name=data_source_name,
                                              scheduler_name=SchedulerName.MMKP_strict_rand_3)
                if len(play_record) != 1:
                    play_record = extract_play_record(SessionMode.RandomPlacement,
                                                      cluster_name=cluster_name,
                                                      data_source_name=data_source_name,
                                                      scheduler_name=SchedulerName.MMKP_strict)
            else:
                play_record = extract_play_record(SessionMode.RandomPlacement,
                                                  cluster_name=cluster_name,
                                                  data_source_name=data_source_name,
                                                  scheduler_name=scheduler_name)
            assert len(play_record) == 1
            play_record = play_record[0]
            for stat in play_record.assignment_statistics:
                total_deployed_jobs += stat.deployed_job_size
                total_spread_jobs += stat.deployed_dist_job_size
            items.append(total_spread_jobs/total_deployed_jobs)
        return items

    MMKP_ratios = get_spread_ratios(SchedulerName.MMKP_strict_rand_3)
    MMKP_random_ratios = get_spread_ratios(SchedulerName.MMKP_strict_rand_variants)
    X = np.arange(len(data_source_names))
    for i, group in enumerate(zip((SchedulerName.MMKP_strict_rand_3, SchedulerName.MMKP_strict_rand_variants), [MMKP_ratios, MMKP_random_ratios])):
        scheduler, data = group
        spec = scheduler_to_spec(scheduler)
        ax.bar(X + i*width,
               data,
               width=width,
               color=spec["color"],
               label=spec["label"],
               hatch="/",
               edgecolor="black"
               )

    xticks = [data_source_to_spec(data_source_name)["label"] for data_source_name in data_source_names]
    ax.set_xticks(X + width/2, xticks, rotation=35)
    ax.yaxis.set_major_formatter(plt_ticker.FuncFormatter('{0:.0%}'.format))
    lgd = ax.legend(loc="upper right")
    ax.set_ylabel("Ratio of Job Variants")
    ax.set_xlabel("Workloads")
    # ax.yaxis.grid(True)
    # ax.xaxis.grid(True)
    fig.tight_layout()
    save_fig(fig, output_path("selectors_variants_spreading_distribution.pdf"))
    mpl.rcParams.update({'font.size': original_fontsize})


def plot_selectors_variants_boxes():
    def grouped_boxplots(data_groups, ax=None, max_width=0.6, pad=0.05, **kwargs):
        if ax is None:
            ax = plt.gca()

        max_group_size = max(len(item) for item in data_groups)
        total_padding = pad * (max_group_size - 1)
        width = (max_width - total_padding) / max_group_size
        kwargs['widths'] = width

        def positions(group, i):
            span = width * len(group) + pad * (len(group) - 1)
            ends = (span - width) / 2
            x = np.linspace(-ends, ends, len(group))
            return x + i

        artists = []
        done_schedulers = set()
        handles = []
        for i, group in enumerate(data_groups, start=1):
            bp = ax.boxplot(group, positions=positions(group, i), widths=width, patch_artist=True, notch="True")
            artists.append(bp)

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
                if scheduler in done_schedulers:
                    continue
                done_schedulers.add(scheduler)
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

        # ax.margins(0.05)
        ax.set(xticks=np.arange(len(data_groups)) + 1)
        ax.autoscale()
        return handles

    original_fontsize = mpl.rcParams["font.size"]
    mpl.rcParams.update({'font.size': 24})
    schedulers = [SchedulerName.MMKP_strict_rand_3,
                  SchedulerName.MMKP_strict_rand_variants]
    data_source_names = [
        DataSourceName.DataSourceAli,
        DataSourceName.DataSourceAliFixNew,
        DataSourceName.DataSourceAliUni,
        DataSourceName.DataSourcePhi,
        DataSourceName.DataSourcePhiFixNew,
        DataSourceName.DataSourcePhiUni
    ]
    fig, ax = plt.subplots(figsize=(7, 4))

    data_groups = list()
    total_profit = cluster_name_to_spec(ClusterName.Cluster10GPUs)["total_profit"]
    for data_source in data_source_names:
        data_group = list()
        for scheduler in schedulers:
            if scheduler == SchedulerName.MMKP_strict_rand_3:
                play_record = extract_play_record(SessionMode.RandomPlacement,
                                              cluster_name=ClusterName.Cluster10GPUs,
                                              data_source_name=data_source,
                                              scheduler_name=SchedulerName.MMKP_strict_rand_3)
                if len(play_record) != 1:
                    play_record = extract_play_record(SessionMode.RandomPlacement,
                                                      cluster_name=ClusterName.Cluster10GPUs,
                                                      data_source_name=data_source,
                                                      scheduler_name=SchedulerName.MMKP_strict)
            else:
                play_record = extract_play_record(SessionMode.RandomPlacement,
                                                  cluster_name=ClusterName.Cluster10GPUs,
                                                  data_source_name=data_source,
                                                  scheduler_name=scheduler)
            assert len(play_record) == 1
            play_record = play_record[0]
            l = list()
            for stat in play_record.assignment_statistics:
                l.append(1. - stat.profit / total_profit)
            data_group.append(l)
        data_groups.append(data_group)

    handles = grouped_boxplots(data_groups, ax,
                              patch_artist=True, notch=True)

    fig.tight_layout()
    ax.yaxis.set_major_formatter(plt_ticker.FuncFormatter('{0:.0%}'.format))
    ax.yaxis.grid(True)
    lgd = fig.legend(handles=handles, loc=(0.27, 0.71))
    # lgd.get_frame().set_alpha(None)
    # fig.subplots_adjust(top=0.88)
    ax.set(xlabel='Workloads', ylabel='RFD', axisbelow=True)
    ax.set_xticklabels([data_source_to_spec(data_source)["label"] for data_source in data_source_names], rotation=35)
    save_fig(fig, output_path("selectors_variants_10GPUs_boxes.pdf"))
    mpl.rcParams.update({'font.size': original_fontsize})


def main():
    plot_selectors_boxes()
    plot_spreading_distribution_bar()
    plot_selectors_variants_boxes()
    plot_spreading_distribution_bar_variants()


if __name__ == '__main__':
    main()
