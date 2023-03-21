import argparse
import os
import pathlib
import runpy
import sys

results_path = pathlib.Path(__file__).parent / "results"
os.environ["FIGURE_PATH"] = str((results_path / "figures").absolute())
os.environ["DATA_PATH"] = str((results_path / "data").absolute())

src_path = (pathlib.Path(__file__).parent / "src").absolute()
plots_script_dir = str((src_path / "plots").absolute())

src_dir = str(src_path)


def add_plot_root():
    sys.path.append(plots_script_dir)
    sys.path.append(src_dir)


def add_run_root():
    sys.path.append(src_dir)


def config_filepath(filename: str):
    return str((src_path / "configs" / filename).absolute())


def refresh_experiment_data():
    from distutils.dir_util import copy_tree, remove_tree
    from_path = str(pathlib.Path(plots_script_dir) / "datas" / "reports")
    to_path = os.environ["DATA_PATH"]
    if os.path.exists(to_path):
        remove_tree(to_path)
    copy_tree(from_path, to_path)


def run_placement_experiment():
    add_run_root()
    from src.run import run
    print("Running solver & partitioning & distributing experiment. This may take hours...")
    run(config_filepath("Solver_config.json"))


def run_trace_experiment():
    add_run_root()
    from src.run import run
    print("Running trace experiment. This may take hours...")
    run(config_filepath("Trace_config.json"))


def plot_job_profiling_data():
    print("Plotting job profiling data figures")
    os.chdir(plots_script_dir)
    add_plot_root()
    runpy.run_path(str(pathlib.Path(plots_script_dir) / "mono_job_performance.py"))


def plot_solver_eval():
    print("Plotting solver & part & dist evaluation figures")
    os.chdir(plots_script_dir)
    add_plot_root()
    runpy.run_path(str(pathlib.Path(plots_script_dir) / "random_placement.py"))


def plot_trace_eval():
    print("Plotting trace evaluation figures")
    os.chdir(plots_script_dir)
    add_plot_root()
    runpy.run_path(str(pathlib.Path(plots_script_dir) / "time_series_profit.py"))


def plot_preemption_eval():
    print("Plotting preemption evaluation figures")
    os.chdir(plots_script_dir)
    add_plot_root()
    runpy.run_path(str(pathlib.Path(plots_script_dir) / "checkpoint.py"))


def plot_scalability_eval():
    print("Plotting scalability evaluation figures")
    os.chdir(plots_script_dir)
    add_plot_root()
    runpy.run_path(str(pathlib.Path(plots_script_dir) / "solver_latency.py"))


def plot_simulator_eval():
    print("Plotting simulator evaluation figures")
    os.chdir(plots_script_dir)
    add_plot_root()
    runpy.run_path(str(pathlib.Path(plots_script_dir) / "time_series_profit_small_scale.py"))


def plot_workloads():
    print("Plotting workloads figures")
    os.chdir(plots_script_dir)
    add_plot_root()
    runpy.run_path(str(pathlib.Path(plots_script_dir) / "workloads.py"))


with open(str(pathlib.Path(__file__).parent / "usage.md"), "r") as f:
    usage = f.read()

parser = argparse.ArgumentParser(
    description="SPREAD: Towards Optimal GPU sharing by Job Spreading for Lightweight Deep Learning Training Jobs. "
                "This is a helper CLI program to run SPREAD simulator and draw all plots in the paper. Please refer to the flag descriptions to run experiment or ",
    usage=usage)

group = parser.add_mutually_exclusive_group()

group.add_argument(
    "--run-placement-experiment",
    help="Run solver & partitioning & distributing experiment and collect data",
    action="store_true"
)

group.add_argument(
    "--run-trace-experiment",
    help="Run trace experiment and collect data",
    action="store_true"
)

group.add_argument(
    "--plot-job-profiling-data",
    help="Plot job profiling data figures",
    action="store_true"
)

group.add_argument(
    "--plot-solver-part-dist-eval",
    help="Plot solver & partitioning & distributing evaluation figures",
    action="store_true"
)

group.add_argument(
    "--plot-trace-eval",
    help="Plot trace evaluation figures",
    action="store_true"
)

group.add_argument(
    "--plot-preemption-eval",
    help="Plot preemption evaluation figures",
    action="store_true"
)

group.add_argument(
    "--plot-scalability-eval",
    help="Plot scalability evaluation figures",
    action="store_true"
)

group.add_argument(
    "--plot-simulator-eval",
    help="Plot simulator evaluation figures",
    action="store_true"
)

group.add_argument(
    "--plot-workloads",
    help="Plot workloads figures",
    action="store_true"
)

group.add_argument(
    "--refresh-data",
    help="Replace all experiment data by the initial data",
    action="store_true"
)

args = parser.parse_args()

if not any(vars(args).values()):
    parser.print_help()
    exit(0)

if args.run_placement_experiment:
    run_placement_experiment()

if args.run_trace_experiment:
    run_trace_experiment()

if args.plot_job_profiling_data:
    plot_job_profiling_data()

if args.plot_solver_part_dist_eval:
    plot_solver_eval()

if args.plot_trace_eval:
    plot_trace_eval()

if args.plot_preemption_eval:
    plot_preemption_eval()

if args.plot_scalability_eval:
    plot_scalability_eval()

if args.plot_simulator_eval:
    plot_simulator_eval()

if args.refresh_data:
    refresh_experiment_data()
