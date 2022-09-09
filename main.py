import argparse
import os
import pathlib
from itertools import product

from config import get_config, Config
from log import info
from schedulers.solver import AssignmentSolver
from simulator import Simulator


def do_test():
    # test data
    dist_job_to_tasks = {
        "job_ID_148": ("job_ID_148|task_0", "job_ID_148|task_1")
    }
    GPU_to_comp_mem_capacity = {
        "RTX_2080Ti_0": (20, 22),
        "RTX_2080Ti_1": (20, 22),
        "RTX_2080Ti_2": (20, 22),
        "RTX_2080Ti_3": (20, 22)
    }
    task_comp_mem_requirements_and_profits = {
        'job_ID_148|task_0': (10, 11, 2.28181818181818183),
        'job_ID_148|task_1': (10, 11, 2.28181818181818183),
        'job_ID_129|task_0': (4, 5, 0.42727272727272725),
        'job_ID_126|task_0': (4, 4, 0.38181818181818183),
        'job_ID_102|task_0': (4, 5, 0.42727272727272725),
        'job_ID_137|task_0': (12, 17, 1.3727272727272726),
        'job_ID_101|task_0': (4, 5, 0.42727272727272725),
        'job_ID_142|task_0': (6, 7, 0.6181818181818182),
        'job_ID_118|task_0': (6, 6, 0.5727272727272728),
        'job_ID_100|task_0': (16, 13, 1.390909090909091),
        'job_ID_110|task_0': (20, 20, 1.9090909090909092),
        'job_ID_112|task_0': (20, 20, 1.9090909090909092),
        'job_ID_139|task_0': (20, 22, 2.0),
        'job_ID_135|task_0': (14, 20, 1.609090909090909),
        'job_ID_127|task_0': (12, 12, 1.1454545454545455),
        'job_ID_104|task_0': (16, 22, 1.8),
    }

    assignment, duration, profit = AssignmentSolver.MMKP_1(strict=False, dist_job_to_tasks=dist_job_to_tasks,
                                                           GPU_comp_mem_capacity=GPU_to_comp_mem_capacity,
                                                           task_comp_mem_requirements_and_profits=task_comp_mem_requirements_and_profits)
    info(assignment)


def do_test_2():
    # test data
    dist_job_to_tasks = {
        "job_ID_148": ("splitting_true|job_ID_148|task_0", "splitting_true|job_ID_148|task_1")
    }
    GPU_to_comp_mem_capacity = {
        "RTX_2080Ti_0": (20, 22),
        "RTX_2080Ti_1": (20, 22),
    }
    task_comp_mem_requirements_and_profits = {
        'splitting_true|job_ID_148|task_0': (10, 11, 2.28181818181818183),
        'splitting_true|job_ID_148|task_1': (10, 11, 2.28181818181818183),
        'splitting_false|job_ID_148|task_0': (20, 22, 4.28181818181818183),
        # 'job_ID_129|task_0': (4, 5, 0.42727272727272725),
        # 'job_ID_126|task_0': (4, 4, 0.38181818181818183),
    }
    splitting_task_IDs_list_list = [
        ["splitting_true|job_ID_148|task_0", "splitting_true|job_ID_148|task_1", "splitting_false|job_ID_148|task_0"]
    ]
    assignment, profit = AssignmentSolver.MMKP_2(
        splitting_task_IDs_list_list=splitting_task_IDs_list_list,
        dist_job_to_tasks=dist_job_to_tasks,
        GPU_comp_mem_capacity=GPU_to_comp_mem_capacity,
        task_comp_mem_requirements_and_profits=task_comp_mem_requirements_and_profits)
    info(assignment)


def do_test_3():
    # test data
    dist_tasks = [
        ("splitting_2|job_ID_148|task_0", "splitting_2|job_ID_148|task_1"),
        ("splitting_4|job_ID_148|task_0", "splitting_4|job_ID_148|task_1", "splitting_4|job_ID_148|task_2", "splitting_4|job_ID_148|task_3"),
    ]

    GPU_to_comp_mem_capacity = {
        "RTX_2080Ti_0": (20, 22),
        "RTX_2080Ti_1": (20, 22),
        "RTX_2080Ti_2": (20, 22),
        "RTX_2080Ti_3": (20, 22),
    }
    task_comp_mem_requirements_and_profits = {
        'splitting_4|job_ID_148|task_0': (5, 5, 1.11181818181818183),
        'splitting_4|job_ID_148|task_1': (5, 5, 1.11181818181818183),
        'splitting_4|job_ID_148|task_2': (5, 5, 1.11181818181818183),
        'splitting_4|job_ID_148|task_3': (5, 5, 1.11181818181818183),
        'splitting_2|job_ID_148|task_0': (10, 11, 2.28181818181818183),
        'splitting_2|job_ID_148|task_1': (10, 11, 2.28181818181818183),
        'splitting_false|job_ID_148|task_0': (20, 22, 4.57181818181818183),
        # 'job_ID_129|task_0': (4, 5, 0.42727272727272725),
        # 'job_ID_126|task_0': (4, 4, 0.38181818181818183),
    }
    splitting_job_ID_task_sets = {
        "job_ID_148": [
            ["splitting_false|job_ID_148|task_0"],
            ["splitting_2|job_ID_148|task_1", "splitting_2|job_ID_148|task_0"],
            [
                "splitting_4|job_ID_148|task_0",
                "splitting_4|job_ID_148|task_1",
                "splitting_4|job_ID_148|task_2",
                "splitting_4|job_ID_148|task_3"
            ],
        ]
    }

    assignment, profit = AssignmentSolver.MMKP_3(
        timeout=30,
        splitting_job_ID_task_sets=splitting_job_ID_task_sets,
        dist_tasks=dist_tasks,
        GPU_comp_mem_capacity=GPU_to_comp_mem_capacity,
        task_comp_mem_requirements_and_profits=task_comp_mem_requirements_and_profits)
    info(assignment)


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
    for data_source_config_name, cluster_config_name in product(c.enabled_data_source_configs,
                                                                c.enabled_cluster_configs):
        data_source_config = c.data_source_configs[data_source_config_name]
        cluster_config = c.cluster_configs[cluster_config_name]
        sim = Simulator(data_source_config=data_source_config, cluster_config=cluster_config)
        sim.play()


if __name__ == '__main__':
    main()
    # do_test_3()
