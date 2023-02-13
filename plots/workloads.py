from matplotlib.ticker import PercentFormatter, FuncFormatter

from record_preprocess import *


def plot_data_source_plan_GPU_distribution():
    original_fontsize = mpl.rcParams["font.size"]
    mpl.rcParams.update({'font.size': 24})
    col = 2
    fig, axes = plt.subplots(3, col, figsize=(12, 8))
    plan_GPU_options = [
        *{*np.arange(5, 105, 5)}, 200
    ]
    width = 0.15
    data_sources = [
        DataSourceName.DataSourceAli,
        DataSourceName.DataSourcePhi,
        DataSourceName.DataSourceAliFixNew,
        DataSourceName.DataSourcePhiFixNew,
        DataSourceName.DataSourceAliUni,
        DataSourceName.DataSourcePhiUni
    ]
    for i, data_source_name in enumerate(data_sources):
        record = extract_play_record(
            mode=SessionMode.Trace,
            data_source_name=data_source_name,
            cluster_name=ClusterName.Cluster10GPUs,
            scheduler_name=SchedulerName.MMKP_strict)
        if len(record) != 1:
            print(data_source_name)
        assert len(record) == 1
        record = record[0]
        plan_GPU_to_size = {opt: 0 for opt in plan_GPU_options}
        for job_spec in record.job_specs.values():
            plan_GPU_to_size[int(job_spec.plan_GPU)] += 1
        X = np.arange(len(plan_GPU_options))
        sizes = [plan_GPU_to_size[opt] for opt in plan_GPU_options]
        ax = axes[i // col, i % col]
        sizes = np.array(sizes) * 4
        ax.bar(X, sizes, width=width)
        xticks = ["", "10", "", "", "25", "", "", "", "", "50", "", "", "65", "", "", "", "85", "", "", "", ">100"]
        ax.set_xticks(X, xticks)
        ax.set_ylabel("# of Jobs")
        ax.set_xlabel("Computation Demands (%)")
        ax.set_title(f"{data_source_to_spec(data_source_name)['label']}")
    # fig.suptitle("Computation Quota Distribution for Each Workload")
    fig.tight_layout()
    save_fig(fig, output_path("workloads_comp_quota_distribution.pdf"))
    mpl.rcParams.update({'font.size': original_fontsize})


def plot_data_source_runtime_distribution():
    original_fontsize = mpl.rcParams["font.size"]
    mpl.rcParams.update({'font.size': 32})
    fig, ax = plt.subplots(figsize=(16, 7))
    data_sources = [DataSourceName.DataSourceAli, DataSourceName.DataSourcePhi]
    for data_source in data_sources:
        record = extract_play_record(mode=SessionMode.Trace,
                                     data_source_name=data_source,
                                     cluster_name=ClusterName.Cluster10GPUs,
                                     scheduler_name=SchedulerName.MMKP_strict)
        assert len(record) == 1
        record = record[0]
        runtimes = []
        for job_spec in record.job_specs.values():
            runtimes.append(job_spec.run_time)

        x = runtimes
        x = list(filter(lambda v: v < 300000, x))

        # N is the count in each bin, bins is the lower-limit of the bin
        # ax.hist(runtimes, bins=100)
        #
        x, y = sorted(x), np.arange(len(x)) / len(x)
        data_source_to_label = {
            DataSourceName.DataSourceAli: "Alibaba",
            DataSourceName.DataSourcePhi: "Philly"
        }
        ax.plot(x, y, label=data_source_to_label[data_source], linewidth=4)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    ax.legend()
    ax.set_xticks([(i * 2 * 60 * 60) for i in range(1, 28, 4)])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(v / (60 * 60))}"))
    ax.set_ylabel("CDF")
    ax.set_xlabel("Runtime (hour)")
    fig.tight_layout()
    save_fig(fig, output_path("workloads_CDF_runtime.pdf"))
    mpl.rcParams.update({'font.size': original_fontsize})


def main():
    plot_data_source_plan_GPU_distribution()
    plot_data_source_runtime_distribution()


if __name__ == '__main__':
    main()
