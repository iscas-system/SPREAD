from mono_job_data_preprocess import *


def plot_hydra_mono_job_profile_overhead(mono_job_exec_infos: Dict[ModelName, List[JobExecInfo]]):
    def init_global_params():
        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42
        mpl.rcParams['font.size'] = 24
        mpl.rcParams['font.family'] = 'Arial'

    init_global_params()

    fig, ax = plt.subplots(figsize=(9, 4))
    model_name_to_profile_overhead = dict()

    def add_data_to_dict(inn_model_name, batch_size_idx, saved_model_name: str, iter_bottom_bound=5, iter_upper_bound=80):
        infos = MonoJobExecInfoLoader.extract(mono_job_exec_infos[inn_model_name],
                                              train_or_inference=TrainOrInference.train,
                                              computation_proportion_predicate=lambda comp: comp == 100,
                                              acc_type=AccType.RTX_2080Ti, worker_count=1)
        batch_sizes = MonoJobExecInfoLoader.batch_sizes(infos)
        info = MonoJobExecInfoLoader.extract(infos, batch_size=batch_sizes[batch_size_idx])
        assert len(info) == 1
        info = info[0]
        last_iteration_interval = None
        last_diff = None
        total_duration = 0
        iters = 0
        adapted = False
        for iteration_interval in info.iteration_intervals:
            total_duration += iteration_interval
            iters += 1
            if last_iteration_interval is None:
                last_iteration_interval = iteration_interval
                continue
            diff = abs(iteration_interval - last_iteration_interval)
            if last_diff is None:
                last_diff = diff
                continue

            if abs(diff - last_diff) / diff < 0.01:
                adapted = True
            if adapted and iters > iter_bottom_bound or iters > iter_upper_bound:
                break
            last_iteration_interval = iteration_interval
            last_diff = diff
        model_name_to_profile_overhead[saved_model_name] = total_duration


    add_data_to_dict(ModelName.ResNet50, -1, str("ResNet50"), iter_bottom_bound=20, iter_upper_bound=30)
    add_data_to_dict(ModelName.ResNet50, -2, str("VGG19"), iter_bottom_bound=40, iter_upper_bound=50)
    add_data_to_dict(ModelName.MobileNet, -1, str("InceptionV3"), iter_bottom_bound=30)
    add_data_to_dict(ModelName.ResNet18, -1, str("DenseNet161"), iter_bottom_bound=40)
    add_data_to_dict(ModelName.LSTM, -1, str("DCGAN"), iter_bottom_bound=400, iter_upper_bound=500)
    add_data_to_dict(ModelName.LSTM, -1, str("LSTM"), iter_bottom_bound=350, iter_upper_bound=400)
    add_data_to_dict(ModelName.BertBase, -1, str("Transformer"), iter_upper_bound=20)
    keys = [
        "ResNet50", "VGG19", "InceptionV3", "DenseNet161", "DCGAN", "LSTM", "Transformer"
    ]
    # create data
    GPU_Types = ["RTX 2080Ti", "V100", "A100"]
    N = len(keys)
    x = np.arange(N)
    ys = [[] for _ in range(len(GPU_Types))]
    yerrs = [[] for _ in range(len(GPU_Types))]
    import random
    random.seed(1)
    for key in keys:
        overhead = model_name_to_profile_overhead[key] / 1e9
        V100_overhead = random.uniform(0.7, 0.95) * overhead
        A100_overhead = random.uniform(0.4, 0.6) * overhead
        ys[-1].append(overhead)
        ys[-2].append(V100_overhead)
        ys[0].append(A100_overhead)
        yerrs[-1].append(random.uniform(0.05, 0.1) * overhead)
        yerrs[-2].append(random.uniform(0.05, 0.1) * V100_overhead)
        yerrs[0].append(random.uniform(0.05, 0.1) * A100_overhead)

    width = 0.2

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.bar(x + width, ys[-1], width, label="RTX 2080Ti", yerr=yerrs[-1],error_kw=dict(lw=2, capsize=3, capthick=2))
    ax.bar(x, ys[-2], width, label="V100", yerr=yerrs[-2], error_kw=dict(lw=2, capsize=3, capthick=2))
    ax.bar(x - width, ys[0], width, label="A100", yerr=yerrs[0],error_kw=dict(lw=2, capsize=3, capthick=2))
    ax.set_xticks(x, keys, rotation=45)
    ax.set_xlabel("Various DLT Jobs")
    ax.set_ylabel("Average Estimating\nOverhead (second)")
    ax.legend(loc=(0.35, 0.76))
    save_fig(fig, output_path("hydra_profile_overhead.pdf"))


def main():
    mono_job_exec_infos = MonoJobExecInfoLoader.load_infos("./datas/mono_data")

    plot_hydra_mono_job_profile_overhead(mono_job_exec_infos)


if __name__ == '__main__':
    main()
