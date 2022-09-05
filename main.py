from log import *
from schedulers.solver import AssignmentSolver
from config import get_config, Config
from itertools import product
from simulator import Simulator
import argparse
import pathlib
import os


init_logging()


def do_test():
    # test data
    dist_job_to_tasks = {
        "job_1": ("task_1_job_1", "task_2_job_1")
    }
    GPU_to_comp_mem_capacity = {
        "T4_1": (10, 15),
        "T4_2": (10, 15)
    }
    task_comp_mem_requirements_and_profits = {
        "task_1_job_1": (5, 5, 8),
        "task_2_job_1": (5, 5, 8),
        "task_1_job_2": (3, 10, 13),
        "task_1_job_3": (2, 5, 7),
    }

    assignment, duration, profit = AssignmentSolver.MMKP(dist_job_to_tasks=dist_job_to_tasks,
                                       GPU_comp_mem_capacity=GPU_to_comp_mem_capacity,
                                       task_comp_mem_requirements_and_profits=task_comp_mem_requirements_and_profits)
    logging.info(assignment)


def main():
    parser = argparse.ArgumentParser(description='MMKP')
    parser.add_argument('--config-path',
                        type=str,
                        required=False,
                        default=str(pathlib.Path(__file__).parent / "configs" / "test_config.json"),
                        help="config path")
    args = parser.parse_args()
    assert os.path.exists(args.config_path), "config path not exists"
    c = get_config(args.config_path)
    run(c)


def run(c: Config):
    for data_source_config_name, cluster_config_name in product(c.enabled_data_source_configs, c.enabled_cluster_configs):
        data_source_config = c.data_source_configs[data_source_config_name]
        cluster_config = c.cluster_configs[cluster_config_name]
        sim = Simulator(data_source_config=data_source_config, cluster_config=cluster_config)
        sim.play()


if __name__ == '__main__':
    main()
    # do_test()
