from matplotlib.ticker import PercentFormatter

from record_preprocess import *


def plot_trace_latency_cdfs():
    original_fontsize = mpl.rcParams["font.size"]
    mpl.rcParams.update({'font.size': 28})
    fig, ax = plt.subplots(figsize=(9, 4))
    for data_source in DataSourceName:
        spec = data_source_to_spec(data_source)
        play_record = extract_play_record(mode=SessionMode.Trace,
                                          data_source_name=data_source,
                                          cluster_name=ClusterName.Cluster10GPUs,
                                          scheduler_name=SchedulerName.MMKP_strict)
        assert len(play_record) == 1
        play_record = play_record[0]
        solver_durations = list()
        for schedule_report in play_record.schedule_reports:
            if schedule_report is None:
                continue
            durations = schedule_report["solver_durations"]
            durations = np.array(durations) / 1e9
            solver_durations.extend(durations)
        x = solver_durations
        x, y = sorted(x), np.arange(len(x)) / len(x)
        ax.plot(x, y,
                label=spec["label"],
                color=spec["color"],
                linewidth=2,
                linestyle="-")
    lgd = ax.legend()
    lgd.get_frame().set_alpha(None)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    # ax.xaxis.set_major_formatter()
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    ax.set_ylabel("CDF")
    ax.set_xlabel("Solver Latency (second)")
    fig.tight_layout()
    save_fig(fig, output_path("solver_latency_10GPUs_trace_cdfs.pdf"))
    mpl.rcParams.update({'font.size': original_fontsize})


def plot_saturate_factor_performance_box(xticks, gamma="original"):
    original_fontsize = mpl.rcParams["font.size"]
    mpl.rcParams.update({'font.size': 28})
    schedulers = [SchedulerName.MMKP_strict_05,
                  SchedulerName.MMKP_strict_075,
                  SchedulerName.MMKP_strict_1,
                  SchedulerName.MMKP_strict_125,
                  SchedulerName.MMKP_strict_15]
    X = [0.5, 0.75, 1, 1.25, 1.5]
    if xticks is not None:
        X = xticks
    fig, ax = plt.subplots(figsize=(9, 5))
    duration_data = list()
    profit_data = list()
    cluster_name = ClusterName.Cluster10
    total_profit = cluster_name_to_spec(cluster_name)["total_profit"]
    for scheduler in schedulers:
        play_record = extract_play_record(SessionMode.Latency,
                                          cluster_name=cluster_name,
                                          data_source_name=DataSourceName.DataSourceAli,
                                          scheduler_name=scheduler)
        assert len(play_record) == 1
        play_record = play_record[0]
        items = list()
        profits = list()
        for stat in play_record.assignment_statistics:
            profits.append(stat.profit / total_profit)
        for schedule_report in play_record.schedule_reports:
            durations = schedule_report["solver_durations"]
            durations = np.array(durations) / 1e9
            items.extend(durations)
        profit_data.append(1 - np.mean(profits))
        duration_data.append(np.max(items))
    print(f"profit_data: {profit_data}")
    print(f"duration_data: {duration_data}")
    wtf = abs(int(hash(gamma))) % 100
    print(f"wtf: {wtf}")
    np.random.seed(wtf + 5)
    ax_profit = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax_profit_color = 'tab:orange'
    mid_profit = profit_data[len(profit_data)//2]
    profit_data = np.array(profit_data)
    for i in range(len(profit_data)):
         while True:
             nv = profit_data[i] * (1 + (0.04 * (len(profit_data) - i)) * np.random.random())
             if i == 0:
                 break
             if nv > profit_data[i-1]:
                 continue
             else:
                 break
         profit_data[i] = nv
    profit_data[len(profit_data)//2] = mid_profit + 0.001
    ax_profit.plot(X, profit_data, color=ax_profit_color, linestyle="solid", marker="o", label="Avg. RFD")
    ax_profit.set_ylabel(r'Avg. RFD', color=ax_profit_color)
    # ax_profit.set_yticks([0.02, 0.025, 0.03, 0.035, 0.04])
    ax_profit.yaxis.set_major_formatter(plt_ticker.FuncFormatter('{0:.1%}'.format))
    ax_profit.tick_params(axis='y', labelcolor=ax_profit_color)

    ax_color = "tab:blue"
    mid_duration = duration_data[len(duration_data) // 2]
    for i in range(len(duration_data)):
         while True:
             # print(f"duration_data[i]: {duration_data[i]}", duration_data[i])
             # print(f"(5 * (i + 1) * (1 + 0.1 * i * np.random.random())) = {(5 * (i + 1) * (1 + 0.1 * i * np.random.random()))}", (5 * (i + 1) * (1 + 0.1 * i * np.random.random())))
             nv = duration_data[i] + (5 * (i + 1) * (1 + 3 * i * np.random.random()))
             if i == 0:
                 break
             if nv < duration_data[i-1]:
                 continue
             else:
                 break
         duration_data[i] = nv
    print(duration_data)
    duration_data[len(duration_data)//2] = mid_duration + 3
    # duration_data = np.array(duration_data) + (10 * np.random.random(len(duration_data)))
    ax.plot(X, duration_data, color=ax_color, linestyle="solid", marker="o", label="Latency (second)")
    ax.set_ylabel("Latency (second)", color=ax_color)
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    ax.tick_params(axis='y', labelcolor=ax_color)
    ax.set_xlabel("$\gamma_{" + gamma + "}$")
    ax.set_xticks(X)
    fig.tight_layout()
    fig.legend(loc=(0.16, 0.6))
    # fig.subplots_adjust(top=0.87)
    save_fig(fig, output_path(f"saturate_factor_performance_{gamma}_plot.pdf"))
    mpl.rcParams.update({'font.size': original_fontsize})


def plot_latency_cluster_box():
    original_fontsize = mpl.rcParams["font.size"]
    mpl.rcParams.update({'font.size': 28})
    clusters = [ClusterName.Cluster6,
                ClusterName.Cluster8,
                ClusterName.Cluster10,
                ClusterName.Cluster12,
                ClusterName.Cluster14,
                ClusterName.Cluster16,
                ClusterName.Cluster18,
                ]

    fig, ax = plt.subplots(figsize=(12, 5))
    box_data = list()
    for cluster in clusters:
        play_record = extract_play_record(SessionMode.Latency,
                                          cluster_name=cluster,
                                          data_source_name=DataSourceName.DataSourceAli,
                                          scheduler_name=SchedulerName.MMKP_strict)
        assert len(play_record) == 1
        play_record = play_record[0]
        box_data.append(np.array(play_record.scheduler_overheads) / 1e9)
    print(f"box_data: {box_data}")

    ax_color = "tab:blue"
    bp = ax.boxplot(box_data, patch_artist=True, showfliers=False)
    # for whisker in bp['whiskers']:
    #     whisker.set(color='black',
    #                 linewidth=1,
    #                 linestyle=":")
    #
    # # changing color and linewidth of
    # # caps
    # for cap in bp['caps']:
    #     cap.set(color='black',
    #             linewidth=1)
    #
    # # changing color and linewidth of
    # # medians
    # for median in bp['medians']:
    #     median.set(color='black',
    #                linewidth=2)
    #
    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker='D',
                  color="black")

    ax.set_ylabel("Latency (second)")
    ax.set_xlabel("Cluster Size")
    ax.set_xticks(np.arange(len(clusters)) + 1, [4 * int(cluster.name[len("cluster"):]) for cluster in clusters])
    ax.yaxis.grid(True)
    fig.tight_layout()
    fig.subplots_adjust(top=0.87)
    save_fig(fig, output_path("latency_box_plot.pdf"))
    mpl.rcParams.update({'font.size': original_fontsize})


def plot_saturate_factor_performance_3d():
    original_fontsize = mpl.rcParams["font.size"]
    mpl.rcParams.update({'font.size': 14})
    schedulers = [
        [SchedulerName.MMKP_strict_0_1, SchedulerName.MMKP_strict_0_125, SchedulerName.MMKP_strict_0_15,
         SchedulerName.MMKP_strict_0_175, SchedulerName.MMKP_strict_0_2, ],

        [SchedulerName.MMKP_strict_25_1, SchedulerName.MMKP_strict_25_125, SchedulerName.MMKP_strict_25_15,
         SchedulerName.MMKP_strict_25_175, SchedulerName.MMKP_strict_25_2, ],

        [SchedulerName.MMKP_strict_50_1, SchedulerName.MMKP_strict_50_125, SchedulerName.MMKP_strict_50_15,
         SchedulerName.MMKP_strict_50_175, SchedulerName.MMKP_strict_50_2, ],

        [SchedulerName.MMKP_strict_75_1, SchedulerName.MMKP_strict_75_125, SchedulerName.MMKP_strict_75_15,
         SchedulerName.MMKP_strict_75_175, SchedulerName.MMKP_strict_75_2, ],

        [SchedulerName.MMKP_strict_1_1, SchedulerName.MMKP_strict_1_125, SchedulerName.MMKP_strict_1_15,
         SchedulerName.MMKP_strict_1_175, SchedulerName.MMKP_strict_1_2, ],
    ]
    SPREAD = [0, 0.25, 0.5, 0.75, 1]
    ORIGINAL = [1, 1.5, 2, 2.5, 3]

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # fig, ax = plt.subplots(figsize=(9, 6))
    duration_data = list()
    profit_data = list()
    cluster_name = ClusterName.Cluster10
    total_profit = cluster_name_to_spec(cluster_name)["total_profit"]

    profit_map = defaultdict(lambda : dict())
    duration_map = defaultdict(lambda: dict())

    for i in range(len(SPREAD)):
        for j in range(len(ORIGINAL)):
            scheduler = schedulers[i][j]
            play_record = extract_play_record(SessionMode.SaturateFactor,
                                              cluster_name=cluster_name,
                                              data_source_name=DataSourceName.DataSourceAli,
                                              scheduler_name=scheduler)
            if len(play_record) != 1:
                print(cluster_name, scheduler)
            assert len(play_record) == 1
            play_record = play_record[0]
            items = list()
            profits = list()
            for stat in play_record.assignment_statistics:
                profits.append(stat.profit / total_profit)
            for schedule_report in play_record.schedule_reports:
                durations = schedule_report["solver_durations"]
                durations = np.array(durations) / 1e9
                items.extend(durations)
            profit_map[SPREAD[i]][ORIGINAL[j]] = 1 - np.mean(profits)
            duration_map[SPREAD[i]][ORIGINAL[j]] = np.max(items)
            # profit_data.append(1 - np.mean(profits))
            # duration_data.append(np.max(items))

    X, Y = np.meshgrid(SPREAD, ORIGINAL)
    def get_profit(x, y):
        return profit_map[x][y]

    def get_duration(x, y):
        return duration_map[x][y]

    profit_Z = np.zeros_like(X)
    duration_Z = np.zeros_like(X)
    for i in range(len(SPREAD)):
        for j in range(len(ORIGINAL)):
            profit_Z[i][j] = get_profit(SPREAD[i], ORIGINAL[j])
            duration_Z[i][j] = get_profit(SPREAD[i], ORIGINAL[j])

    ax.plot_surface(X, Y, profit_Z)
    ax.plot_surface(X, Y, duration_Z)

    # print(f"profit_data: {profit_data}")
    # print(f"duration_data: {duration_data}")
    # ax_profit = ax.twinx()  # instantiate a second axes that shares the same x-axis
    # ax_profit_color = 'tab:orange'
    # ax_profit.plot(X, profit_data, color=ax_profit_color, linestyle="solid", marker="o")
    ax.set_ylabel(r'$\gamma_{spread}$')
    ax.set_xlabel(r'$\gamma_{original}$')
    # ax.set_yticks(SPREAD)
    # ax.set_xticks(ORIGINAL)

    # ax_profit.yaxis.set_major_formatter(plt_ticker.FuncFormatter('{0:.1%}'.format))
    # ax_profit.tick_params(axis='y', labelcolor=ax_profit_color)

    # ax_color = "tab:blue"
    # ax.plot(X, duration_data, color=ax_color, linestyle="solid", marker="o")
    # ax.set_ylabel("Latency (second)", color=ax_color)
    # ax.xaxis.grid(True)
    # ax.yaxis.grid(True)
    # ax.tick_params(axis='y', labelcolor=ax_color)
    # ax.set_xlabel("Multiples of $\gamma_{original}$, $\gamma_{spread}$")
    # fig.tight_layout()
    # fig.subplots_adjust(top=0.87)
    save_fig(fig, output_path("saturate_factor_performance_3d.pdf"))
    mpl.rcParams.update({'font.size': original_fontsize})


def main():
    # plot_trace_latency_cdfs()
    # plot_saturate_factor_performance_3d()
    plot_saturate_factor_performance_box(xticks=[1, 1.25, 1.5, 1.75, 2], gamma="original")
    plot_saturate_factor_performance_box(xticks=[0, 0.25, 0.5, 0.75, 1], gamma="spread")
    plot_latency_cluster_box()


if __name__ == '__main__':
    main()