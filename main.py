from log import *
from schedulers.solver import AssignmentSolver

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
    ...
    # run_server()


if __name__ == '__main__':
    # main()
    do_test()
