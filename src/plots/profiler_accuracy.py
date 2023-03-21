from mono_job_data_preprocess import *


def plot_pre_profile_accuracy(mono_job_exec_infos: Dict[ModelName, List[JobExecInfo]]):
    model_names = [model_name for model_name in ModelName]
    batch_sizes_len = 4
    profile_length = 5
    Data = namedtuple(typename="data", field_names=["mode", "batch_size_idx", "avg_duration", "yerr"])
    model_to_data_list: Dict[ModelName, List[Data]] = defaultdict(list)
    for model_name in model_names:
        infos = MonoJobExecInfoLoader.extract(mono_job_exec_infos[model_name],
                                              train_or_inference=TrainOrInference.train, worker_count=1,
                                              acc_type=AccType.RTX_2080Ti,
                                              computation_proportion_predicate=lambda comp: comp == 100)
        batch_sizes = MonoJobExecInfoLoader.batch_sizes(infos)
        batch_sizes = sorted(list(batch_sizes))[1:]
        batch_sizes = batch_sizes[:batch_sizes_len]
        for batch_size_idx, batch_size in enumerate(batch_sizes):
            info = MonoJobExecInfoLoader.extract(infos, batch_size=batch_size)
            assert len(info) == 1
            info = info[0]
            iteration_intervals = info.stabled_iteration_intervals
            i = 0
            avg_interval_for_each_slice = list()
            while True:
                total_interval = 0
                intervals = list()
                if i >= len(iteration_intervals):
                    break
                sub = iteration_intervals[i:]
                for iteration_interval in sub:
                    total_interval += iteration_interval
                    intervals.append(iteration_interval)
                    i += 1
                    if total_interval > profile_length * 1e9:
                        break
                avg_interval = np.mean(intervals)
                avg_interval_for_each_slice.append(avg_interval)
            avg_interval_for_each_slice = np.array(avg_interval_for_each_slice)
            all_avg_duration = np.mean(iteration_intervals)
            avg_interval_for_each_slice /= all_avg_duration

            avg_duration = np.mean(avg_interval_for_each_slice)
            yerr = np.std(avg_interval_for_each_slice)
            model_to_data_list[model_name].append(Data(mode="long_run",
                                                       batch_size_idx=batch_size_idx,
                                                       avg_duration=1,
                                                       yerr=0))
            model_to_data_list[model_name].append(Data(mode="profile",
                                                       batch_size_idx=batch_size_idx,
                                                       avg_duration=avg_duration,
                                                       yerr=yerr))

    batch_size_idx_to_data_list = defaultdict(list)
    for batch_size_idx in range(batch_sizes_len):
        for model_name in model_names:
            data_list = model_to_data_list[model_name]
            for data in data_list:
                if data.batch_size_idx != batch_size_idx:
                    continue
                batch_size_idx_to_data_list[batch_size_idx].append(data)

    X = np.arange(len(model_names))
    width = 0.1
    fig, ax = plt.subplots(figsize=(20, 8))

    profile_hatch = "/"
    long_run_hatch = "\\"
    edge_color = "black"
    bottom = 0.9

    for batch_size_idx in range(batch_sizes_len):
        data_list = batch_size_idx_to_data_list[batch_size_idx]
        profile_data_list = list(filter(lambda d: d.mode == "profile", data_list))
        long_run_data_list = list(filter(lambda d: d.mode == "long_run", data_list))
        profile_avg_duration = [data.avg_duration for data in profile_data_list]
        profile_yerr = [data.yerr for data in profile_data_list]
        long_run_avg_duration = [data.avg_duration for data in long_run_data_list]
        ax.bar(
            X + (2 * batch_size_idx) * width,
            np.array(profile_avg_duration) - bottom,
            edgecolor=edge_color,
            width=width,
            color=batch_size_idx_color(batch_size_idx),
            label=batch_size_idx_label(batch_size_idx),
            hatch=profile_hatch,
            yerr=profile_yerr,
            error_kw=dict(lw=1, capsize=3, capthick=1),
            bottom=bottom
        )

        ax.bar(
            X + (2 * batch_size_idx + 1) * width,
            np.array(long_run_avg_duration) - bottom,
            edgecolor=edge_color,
            width=width,
            color=batch_size_idx_color(batch_size_idx),
            label=batch_size_idx_label(batch_size_idx),
            hatch=long_run_hatch,
            bottom=bottom
        )
    ax.spines['bottom'].set_position(('data', bottom))
    handles = list()
    for batch_size_idx in range(batch_sizes_len):
        handle = Patch(
            facecolor=batch_size_idx_color(batch_size_idx),
            edgecolor=edge_color,
            label=batch_size_idx_label(batch_size_idx),
        )
        handles.append(handle)
    handles.append(Patch(
        facecolor="white",
        edgecolor=edge_color,
        label="Profiled",
        hatch=profile_hatch,
    ))
    handles.append(Patch(
        facecolor="white",
        edgecolor=edge_color,
        label="Ground Truth",
        hatch=long_run_hatch,
    ))

    lgd = fig.legend(handles=handles, loc=(0.01, 0.86), ncol=len(handles))
    lgd.get_frame().set_alpha(None)
    # fig.suptitle(f"Memory Quotas of Workers With Various Spreading Configurations", fontsize="x-large")
    # fig.subplots_adjust(top=0.95)

    ax.set_xticks(X + (2*batch_sizes_len - 1) * width/2,
                  [model_name_spec(model_name)["label"] for model_name in model_names])
    ax.yaxis.grid(True)
    ax.yaxis.set_major_formatter(plt_ticker.FuncFormatter('{0:.0%}'.format))
    ax.set_xlabel("Models")
    ax.set_ylabel("Profiled Iteration Duration")
    fig.tight_layout()
    save_fig(fig, output_path("profiler_pre_accuracy.pdf"))


def main():
    mono_job_exec_infos = MonoJobExecInfoLoader.load_infos("./datas/mono_data")

    plot_pre_profile_accuracy(mono_job_exec_infos)


if __name__ == '__main__':
    main()
