import datetime
import json
import logging
import os.path
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pydantic

from common import get_fig_dir, get_session_dir
from model import *


def init_global_params():
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.size'] = 24
    mpl.rcParams['font.family'] = 'Arial'


init_global_params()


def do_snapshot_record_plot(session_id: str, snapshot_record_parameters: SnapshotRecordParameters):
    if snapshot_record_parameters.solver_type is None:
        solver_type = "None"
    else:
        solver_type = snapshot_record_parameters.solver_type.value

    filename = datetime.datetime.now().strftime(
        f"snapshot_record_{snapshot_record_parameters.scheduler_name}_{solver_type}_{snapshot_record_parameters.profit}_%Y-%m-%d-%H-%M-%S")
    json_filepath = os.path.join(get_session_dir(session_id), filename + ".json")
    fig_filepath = os.path.join(get_fig_dir(session_id), filename + ".pdf")
    logging.info(f"received record parameters, session_id = {session_id}, saving file to {json_filepath}")
    with open(json_filepath, 'w') as f:
        json.dump(snapshot_record_parameters.json(), f)
    plot_assignment(snapshot_record_parameters, filepath=fig_filepath)
    return {}


def plot_assignment(recorder_parameters: SnapshotRecordParameters, filepath: str):
    GPU_slots = list()
    GPU_ID_to_GPU_slot_idx = dict()
    GPU_ID_to_GPU_type = dict()

    for GPU_type, GPU_IDs in recorder_parameters.GPU_type_to_GPU_IDs.items():
        for GPU_ID in GPU_IDs:
            GPU_slots.append(GPU_ID)
    GPU_slots = sorted(GPU_slots)
    for slot_idx, GPU_ID in enumerate(GPU_slots):
        GPU_ID_to_GPU_slot_idx[GPU_ID] = slot_idx
    for GPU_type, GPU_IDs in recorder_parameters.GPU_type_to_GPU_IDs.items():
        for GPU_ID in GPU_IDs:
            GPU_ID_to_GPU_type[GPU_ID] = GPU_type

    GPU_count = len(GPU_slots)
    fig, ax = plt.subplots(figsize=(10 + GPU_count // 1.5, 8))
    GPU_type_to_comp_capacity = dict({GPU_type: comp_mem_capacity[0] for GPU_type, comp_mem_capacity in
                                      recorder_parameters.GPU_type_to_comp_mem_capacity.items()})
    GPU_type_to_mem_capacity = dict({GPU_type: comp_mem_capacity[1] for GPU_type, comp_mem_capacity in
                                     recorder_parameters.GPU_type_to_comp_mem_capacity.items()})
    all_tasks_list = sorted(list(recorder_parameters.task_comp_mem_requirements.keys()))
    tasks = set()
    task_ID_to_GPU_ID = dict()
    for GPU_ID, task_IDs in recorder_parameters.assignments.items():
        tasks.update(task_IDs)
        for task in task_IDs:
            task_ID_to_GPU_ID[task] = GPU_ID
    tasks_list = sorted(list(tasks))
    colors = ["lightblue", "lightgreen", "lightyellow", "lightcoral", "navajowhite", "thistle", "silver", "seashell",
              "lightcyan", "honeydew"]

    dist_job_IDs = list()
    task_ID_to_job_ID = dict()
    for dist_job, tasks_of_job in recorder_parameters.dist_job_to_tasks.items():
        dist_job_IDs.append(dist_job)
        for task_ID in tasks_of_job:
            task_ID_to_job_ID[task_ID] = dist_job
    dist_job_IDs = sorted(dist_job_IDs)
    for task in all_tasks_list:
        if task not in task_ID_to_job_ID:
            task_ID_to_job_ID[task] = task + "_job_ID"

    def generate_color_label_dicts():
        task_ID_to_label = dict()
        job_ID_to_label = dict()
        dist_job_ID_to_label_size = dict()
        job_ID_to_color = dict()
        next_color_idx = 0
        next_label_ord = ord('A')
        for task_ID, job_ID in task_ID_to_job_ID.items():
            if job_ID not in job_ID_to_color:
                job_ID_to_color[job_ID] = colors[next_color_idx % len(colors)]
                next_color_idx += 1
            if job_ID not in job_ID_to_label:
                job_ID_to_label[job_ID] = chr(next_label_ord)
                task_ID_to_label[task_ID] = chr(next_label_ord)
                next_label_ord += 1
        for dist_job in dist_job_IDs:
            task_IDs = recorder_parameters.dist_job_to_tasks[dist_job]
            for i, task_ID in enumerate(task_IDs):
                curr_label = job_ID_to_label[dist_job]
                task_ID_to_label[task_ID] = curr_label + f"$_{i + 1}$"
        return task_ID_to_label, job_ID_to_color

    task_ID_to_label, job_ID_to_color = generate_color_label_dicts()

    def task_comp(task_ID):
        comp_proportion = 0.1 * recorder_parameters.task_comp_mem_requirements[task_ID][0]
        GPU_ID = task_ID_to_GPU_ID[task_ID]
        GPU_type = GPU_ID_to_GPU_type[GPU_ID]
        GPU_comp_capacity = GPU_type_to_comp_capacity[GPU_type]
        return comp_proportion * GPU_comp_capacity

    def task_mem(task_ID):
        mem = recorder_parameters.task_comp_mem_requirements[task_ID][1]
        return mem

    task_ID_comp_data = dict()
    task_ID_mem_data = dict()
    for i, task in enumerate(tasks_list):
        GPU_ID = task_ID_to_GPU_ID[task]
        GPU_slot_idx = GPU_ID_to_GPU_slot_idx[GPU_ID]
        task_comp_data = np.zeros(len(GPU_slots))
        task_mem_data = np.zeros(len(GPU_slots))
        task_comp_value = task_comp(task)
        task_mem_value = task_mem(task)
        task_comp_data[GPU_slot_idx] = task_comp_value
        task_mem_data[GPU_slot_idx] = task_mem_value
        task_ID_comp_data[task] = task_comp_data
        task_ID_mem_data[task] = task_mem_data

    X = np.arange(len(GPU_slots))
    width = 0.4

    hatch = "/"

    bottom_comp_data_dict = dict()
    bottom_mem_data_dict = dict()

    def bottom_comp_mem_data():
        bottom_comp_data_ = np.zeros((len(GPU_slots),))
        bottom_mem_data_ = np.zeros((len(GPU_slots),))
        for _, data in bottom_comp_data_dict.items():
            bottom_comp_data_ += np.array(data)
        for _, data in bottom_mem_data_dict.items():
            bottom_mem_data_ += np.array(data)
        return bottom_comp_data_, bottom_mem_data_

    def plot_task(task):
        comp_data = task_ID_comp_data[task]
        mem_data = task_ID_mem_data[task]
        label_list = list()
        for value in comp_data:
            if value > 0.:
                label_list.append(task_ID_to_label[task])
            else:
                label_list.append("")
        bottom_comp_data_, bottom_mem_data_ = bottom_comp_mem_data()
        bar1 = ax.bar(X - width / 2, comp_data, edgecolor="black", bottom=bottom_comp_data_,
                      color=job_ID_to_color[task_ID_to_job_ID[task]],
                      label=task_ID_to_label[task], hatch=hatch, width=width)
        bar2 = ax.bar(X + width / 2, mem_data, edgecolor="black", bottom=bottom_mem_data_,
                      color=job_ID_to_color[task_ID_to_job_ID[task]],
                      label=task_ID_to_label[task], hatch=hatch, width=width)
        for bar in [bar1, bar2]:
            ax.bar_label(bar, labels=label_list, label_type="center", fontsize=24)

        bottom_comp_data_dict[task] = comp_data
        bottom_mem_data_dict[task] = mem_data

    for dist_job, tasks_of_job in recorder_parameters.dist_job_to_tasks.items():
        for task in tasks_of_job:
            assert task not in bottom_comp_data_dict
            plot_task(task)
    for task in tasks_list:
        if task in bottom_comp_data_dict:
            continue
        plot_task(task)

    bottom_comp_data_, bottom_mem_data_ = bottom_comp_mem_data()
    remaining_comp_data = list()
    remaining_mem_data = list()
    for i, GPU_ID in enumerate(GPU_slots):
        GPU_type = GPU_ID_to_GPU_type[GPU_ID]
        comp = GPU_type_to_comp_capacity[GPU_type]
        mem = GPU_type_to_mem_capacity[GPU_type]
        remaining_comp = comp - bottom_comp_data_[i]
        remaining_mem = mem - bottom_mem_data_[i]
        remaining_comp_data.append(remaining_comp)
        remaining_mem_data.append(remaining_mem)
    GPU_comp_bar = ax.bar(X - width / 2, remaining_comp_data, edgecolor="black", bottom=bottom_comp_data_,
                          color="white", width=width)
    ax.bar_label(GPU_comp_bar, labels=[GPUType.comp_power_label(GPU_ID_to_GPU_type[GPU_ID]) for GPU_ID in GPU_slots],
                 label_type="edge", fontsize=14)
    GPU_mem_bar = ax.bar(X + width / 2, remaining_mem_data, edgecolor="black", bottom=bottom_mem_data_, color="white",
                         width=width)
    ax.bar_label(GPU_mem_bar, labels=[GPUType.mem_label(GPU_ID_to_GPU_type[GPU_ID]) for GPU_ID in GPU_slots],
                 label_type="edge", fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks(X, [GPUType.label(GPU_ID_to_GPU_type[GPU_ID]) for GPU_ID in GPU_slots], rotation=45)
    ax.tick_params(axis=u'both', which=u'both', length=0)
    fig.tight_layout()
    # fig.show()
    fig.savefig(filepath, dpi=1200, format='pdf', bbox_inches='tight')


def read_recorder_parameters(filepath: str) -> SnapshotRecordParameters:
    recorder_parameters = pydantic.parse_file_as(path=filepath, type_=SnapshotRecordParameters)
    return recorder_parameters


def do_test():
    # recorder_parameters = read_recorder_parameters("sample_.json")
    dist_job_to_tasks = {
        "dist_job_1": ("task_1_job_1", "task_2_job_1"),
        "dist_job_2": ("task_1_job_5", "task_2_job_5")
    }
    GPU_type_to_comp_mem_capacity = {
        GPUType.Tesla_T4: (8, 15),
        GPUType.RTX2080_Ti: (13.5, 11)
    }
    task_comp_mem_requirements = {
        "task_1_job_1": (5, 5),
        "task_2_job_1": (5, 5),
        "task_1_job_2": (3, 10),
        "task_1_job_3": (2, 5),
        "task_1_job_4": (5, 8),
        "task_1_job_5": (5, 3),
        "task_2_job_5": (5, 3),
        "task_1_job_6": (7, 12),
        "task_1_job_7": (3, 4),
        "task_1_job_8": (2, 3),
        "task_1_job_9": (5, 6),
    }
    GPU_type_to_GPU_IDs = {
        GPUType.Tesla_T4: {"T4_1", "T4_2", "T4_3", "T4_4"},
        GPUType.RTX2080_Ti: {"2080Ti_1", "2080Ti_2", "2080Ti_3"}
    }
    assignment: Dict[str, Set[str]] = {
        "T4_1": {"task_1_job_1", "task_1_job_2"},
        "T4_2": {"task_2_job_1"},
        "T4_3": {"task_1_job_6", "task_1_job_8"},
        "T4_4": {"task_1_job_9"},
        "2080Ti_1": {"task_1_job_3", "task_1_job_7"},
        "2080Ti_2": {"task_1_job_4", "task_1_job_5"},
        "2080Ti_3": {"task_2_job_5"},
    }
    recorder_parameters = SnapshotRecordParameters(
        solver_type=SolverEnum.MMKP,
        GPU_type_to_comp_mem_capacity=GPU_type_to_comp_mem_capacity,
        GPU_type_to_GPU_IDs=GPU_type_to_GPU_IDs,
        dist_job_to_tasks=dist_job_to_tasks,
        task_comp_mem_requirements=task_comp_mem_requirements,
        assignment=assignment,
        profit=0,
        do_plot=True,
    )
    filepath = str(pathlib.Path(__file__).parent / "output" / "fig" / "fig_test.pdf")
    plot_assignment(recorder_parameters, filepath)


if __name__ == "__main__":
    do_test()
