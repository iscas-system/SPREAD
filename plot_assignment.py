import datetime
import json
import os.path
import pathlib
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pydantic
from matplotlib.patches import Patch

from cluster import Assignments
from common import get_fig_dir, get_json_dir
from log import info
from model import *
from object import CompCapacity


def init_global_params():
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.size'] = 24
    mpl.rcParams['font.family'] = 'Arial'


init_global_params()


def do_snapshot_record_plot(session_id: str, is_preemptive_interval: bool, snapshot_record_parameters: SnapshotRecordParameters):
    if snapshot_record_parameters.solver_type is None:
        solver_type = "None"
    else:
        solver_type = snapshot_record_parameters.solver_type

    filename = datetime.datetime.now().strftime(
        f"snapshot_record_{snapshot_record_parameters.scheduler_name}_{solver_type}_%Y-%m-%d_%H-%M-%S_{np.around(snapshot_record_parameters.profit, decimals=1)}_{is_preemptive_interval}")
    json_filepath = os.path.join(get_json_dir(session_id), filename + ".json")
    fig_filepath = os.path.join(get_fig_dir(session_id), filename + ".pdf")
    info(f"received record parameters, session_id = {session_id}, saving file to {json_filepath}")
    with open(json_filepath, 'w') as f:
        js = snapshot_record_parameters.json(indent='\t')
        f.write(js)
    if snapshot_record_parameters.do_plot:
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
            assert isinstance(GPU_type, str)
            GPU_ID_to_GPU_type[GPU_ID] = GPUType[GPU_type]

    GPU_count = len(GPU_slots)
    fig, ax = plt.subplots(figsize=(10 + GPU_count // 1.5, 8))
    all_tasks_list = sorted(list(recorder_parameters.task_comp_mem_requirements.keys()))
    tasks = set()
    task_ID_to_GPU_ID = dict()
    for GPU_ID, task_IDs in recorder_parameters.assignments.items():
        tasks.update(task_IDs)
        for task in task_IDs:
            task_ID_to_GPU_ID[task] = GPU_ID
    tasks_list = sorted(list(tasks))
    colors = ["lightblue", "lightgreen", "lightyellow", "lightcoral", "navajowhite", "thistle", "silver", "seashell",
              "lightcyan", "honeydew", "beige", "azure", "aliceblue", "snow", "floralwhite", "mintcream", "papayawhip", "palegreen"]

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
        comp_proportion = recorder_parameters.task_comp_mem_requirements[task_ID][0] / CompCapacity
        return comp_proportion

    def task_oversupply(task_ID):
        comp_proportion = recorder_parameters.task_comp_over_supply[task_ID] / CompCapacity
        return comp_proportion

    def task_lack_supply(task_ID):
        if task_ID not in recorder_parameters.task_comp_lack_supply:
            return 0
        lack_supply = recorder_parameters.task_comp_lack_supply[task_ID] / CompCapacity
        return lack_supply

    def task_mem(task_ID):
        GPU_ID = task_ID_to_GPU_ID[task_ID]
        GPU_type = GPU_ID_to_GPU_type[GPU_ID]
        n_mem = GPUType.normalized_memory(GPU_type)
        mem = recorder_parameters.task_comp_mem_requirements[task_ID][1]
        return mem / n_mem

    task_ID_comp_data = dict()
    task_ID_oversupply_data = dict()
    task_ID_mem_data = dict()
    task_ID_lack_supply_data = dict()
    for i, task in enumerate(tasks_list):
        GPU_ID = task_ID_to_GPU_ID[task]
        GPU_slot_idx = GPU_ID_to_GPU_slot_idx[GPU_ID]
        task_comp_data = np.zeros(len(GPU_slots))
        task_oversupply_data = np.zeros(len(GPU_slots))
        task_mem_data = np.zeros(len(GPU_slots))
        task_lack_supply_data = np.zeros(len(GPU_slots))
        task_comp_value = task_comp(task)
        task_oversupply_value = task_oversupply(task)
        task_mem_value = task_mem(task)
        task_lack_supply_value = task_lack_supply(task)
        task_comp_data[GPU_slot_idx] = task_comp_value
        task_oversupply_data[GPU_slot_idx] = task_oversupply_value
        task_mem_data[GPU_slot_idx] = task_mem_value
        task_lack_supply_data[GPU_slot_idx] = task_lack_supply_value
        task_ID_comp_data[task] = task_comp_data
        task_ID_oversupply_data[task] = task_oversupply_data
        task_ID_mem_data[task] = task_mem_data
        task_ID_lack_supply_data[task] = task_lack_supply_data

    X = np.arange(len(GPU_slots))
    width = 0.4
    split_bar_width = 0.01 / 2

    hatch = r"/"
    oversupply_hatch = r"**"
    lack_supply_hatch = r"oo"

    bottom_comp_data_dict = dict()
    bottom_oversupply_data_dict = dict()
    bottom_mem_data_dict = dict()
    bottom_lack_supply_data_dict = dict()

    def bottom_comp_mem_data():
        bottom_comp_data_ = np.zeros((len(GPU_slots),))
        bottom_mem_data_ = np.zeros((len(GPU_slots),))
        for _, data in bottom_comp_data_dict.items():
            bottom_comp_data_ += np.array(data)
        for _, data in bottom_oversupply_data_dict.items():
            bottom_comp_data_ += np.array(data)
        for _, data in bottom_mem_data_dict.items():
            bottom_mem_data_ += np.array(data)
        return bottom_comp_data_, bottom_mem_data_

    def bottom_lack_supply_data():
        bottom_comp_data_ = np.zeros((len(GPU_slots),))
        for _, data in bottom_lack_supply_data_dict.items():
            bottom_comp_data_ += np.array(data)
        return bottom_comp_data_

    def plot_task(task):
        comp_data = task_ID_comp_data[task]
        oversupply_data = task_ID_oversupply_data[task]
        mem_data = task_ID_mem_data[task]
        label_list = list()
        for value in comp_data:
            if value > 0.:
                label_list.append(task_ID_to_label[task])
            else:
                label_list.append("")
        bottom_comp_data_, bottom_mem_data_ = bottom_comp_mem_data()
        bar1 = ax.bar(X - width / 2 - split_bar_width, comp_data,
                      # edgecolor="black",
                      bottom=bottom_comp_data_,
                      color=job_ID_to_color[task_ID_to_job_ID[task]],
                      label=task_ID_to_label[task],
                      hatch=hatch,
                      width=width)
        ax.bar(X - width / 2 - split_bar_width, oversupply_data,
               bottom=np.array(bottom_comp_data_) + np.array(comp_data),
               color=job_ID_to_color[task_ID_to_job_ID[task]],
               label=task_ID_to_label[task],
               hatch=oversupply_hatch,
               width=width)
        bar2 = ax.bar(X + width / 2 + split_bar_width, mem_data,
                      # edgecolor="black",
                      bottom=bottom_mem_data_,
                      color=job_ID_to_color[task_ID_to_job_ID[task]],
                      label=task_ID_to_label[task],
                      hatch=hatch,
                      width=width)
        for bar in [bar1, bar2]:
            ax.bar_label(bar, labels=label_list, label_type="center", fontsize=24)

        bottom_comp_data_dict[task] = comp_data
        bottom_oversupply_data_dict[task] = oversupply_data
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
        comp = 1.
        mem = 1.
        remaining_comp = comp - bottom_comp_data_[i]
        remaining_mem = mem - bottom_mem_data_[i]
        remaining_comp_data.append(remaining_comp)
        remaining_mem_data.append(remaining_mem)
    under_utilized_color = "grey"
    GPU_comp_bar = ax.bar(X - width / 2 - split_bar_width,
                          remaining_comp_data,
                          # edgecolor="black",
                          bottom=bottom_comp_data_,
                          color=under_utilized_color,
                          width=width)
    # ax.bar_label(GPU_comp_bar, labels=[GPUType.comp_power_label(GPU_ID_to_GPU_type[GPU_ID]) for GPU_ID in GPU_slots],
    #              label_type="edge", fontsize=14)
    GPU_mem_bar = ax.bar(X + width / 2 + split_bar_width,
                         remaining_mem_data,
                         bottom=bottom_mem_data_,
                         color=under_utilized_color,
                         width=width)

    # ax.bar_label(GPU_mem_bar, labels=[GPUType.mem_label(GPU_ID_to_GPU_type[GPU_ID]) for GPU_ID in GPU_slots],
    #              label_type="edge", fontsize=14)

    def plot_lack_supply_task(task):
        lack_supply_data = task_ID_lack_supply_data[task]
        bottom_lack_supply_data_ = bottom_lack_supply_data()
        ax.bar(X - width / 2 - split_bar_width, lack_supply_data,
               bottom=np.array(bottom_lack_supply_data_) + np.ones_like(bottom_lack_supply_data_),
               color=job_ID_to_color[task_ID_to_job_ID[task]],
               label=task_ID_to_label[task],
               hatch=lack_supply_hatch,
               width=width)

        bottom_oversupply_data_dict[task] = lack_supply_data

    for task in tasks_list:
        plot_lack_supply_task(task)

    legend_elements = [
        Patch(edgecolor="black", facecolor="white",
              hatch=hatch,
              label='User-Requested Quota'),
        # Patch(edgecolor="black", facecolor="white",
        #       hatch=oversupply_hatch,
        #       label='Over-Supplied Resource'),
        # Patch(edgecolor="black", facecolor="white",
        #       hatch=lack_supply_hatch,
        #       label='Lack of Supply'),
        Patch(facecolor=under_utilized_color,
              label='Underutilized')
    ]
    ax.legend(handles=legend_elements, loc=(0.7, 1.05), fontsize=14)
    ax.bar(X, np.ones(len(GPU_slots)), color="black", width=2 * split_bar_width)

    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks(X, [f"{GPUType.label(GPU_ID_to_GPU_type[GPU_ID])} {idx}" for idx, GPU_ID in enumerate(GPU_slots)], rotation=45)
    ax.tick_params(axis=u'both', which=u'both', length=0)
    fig.tight_layout()
    # fig.show()
    fig.savefig(filepath, dpi=1200, format='pdf', bbox_inches='tight')
    plt.close(fig)


def read_recorder_parameters(filepath: str) -> SnapshotRecordParameters:
    recorder_parameters = pydantic.parse_file_as(path=filepath, type_=SnapshotRecordParameters)
    return recorder_parameters


def do_test():
    # recorder_parameters = read_recorder_parameters("sample_.json")
    dist_job_to_tasks = {
        "dist_job_1": ("job_1|task_1", "job_1|task_2"),
        "dist_job_2": ("job_5|task_1", "job_5|task_2")
    }
    GPU_type_to_comp_mem_capacity = {
        GPUType.RTX_2080Ti: (13.5, 11)
    }
    task_comp_mem_requirements = {
        "job_1|task_1": (5, 5),
        "job_1|task_2": (5, 5),
        "job_2|task_1": (3, 10),
        "job_3|task_1": (2, 5),
        "job_4|task_1": (5, 8),
        "job_5|task_1": (5, 3),
        "job_5|task_2": (5, 3),
        "job_6|task_1": (7, 12),
        "job_7|task_1": (3, 4),
        "job_8|task_1": (2, 3),
        "job_9|task_1": (5, 6),
    }
    GPU_type_to_GPU_IDs = {
        GPUType.RTX_2080Ti: {"2080Ti_1", "2080Ti_2", "2080Ti_3", "2080Ti_4"}
    }
    GPU_ID_to_GPU_type = {
        "2080Ti_1": GPUType.RTX_2080Ti,
        "2080Ti_2": GPUType.RTX_2080Ti,
        "2080Ti_3": GPUType.RTX_2080Ti,
        "2080Ti_4": GPUType.RTX_2080Ti,
    }
    assignments: Dict[str, Set[str]] = {
        "2080Ti_1": {"job_3|task_1", "job_7|task_1"},
        "2080Ti_2": {"job_4|task_1", "job_5|task_1"},
        "2080Ti_3": {"job_5|task_2", "job_1|task_2"},
        "2080Ti_4": {"job_1|task_1", "job_8|task_1"},
    }

    GPU_type_to_task_comp_mem_requirements = defaultdict(dict)
    for GPU_ID, task_IDs in assignments.items():
        GPU_type = GPU_ID_to_GPU_type[GPU_ID]
        for task_ID in task_IDs:
            GPU_type_to_task_comp_mem_requirements[GPU_type][task_ID] = task_comp_mem_requirements[task_ID]
    assignments_wrapped = Assignments.from_solver_assigment(GPU_ID_to_GPU_type=GPU_ID_to_GPU_type,
                                                            GPU_type_to_task_comp_mem_requirements=GPU_type_to_task_comp_mem_requirements,
                                                            solver_assignments=assignments)
    over_supplied_assignments_wrapped = assignments_wrapped.supplement_over_supply()
    job_over_supply, normalized_total_over_supply = over_supplied_assignments_wrapped.get_job_over_supply()
    task_over_supply = dict()
    for job_ID, task_assignments in over_supplied_assignments_wrapped.job_ID_to_task_assignments.items():
        worker_count = len(task_assignments)
        for task_assignment in task_assignments:
            task_over_supply[task_assignment.task.task_ID] = job_over_supply[job_ID] // worker_count

    task_comp_lack_supply = {
        "job_4|task_1": 4,
        "job_5|task_1": 2,
        "job_1|task_2": 2,
    }
    # scheduler_name: str
    #     scheduler_type: SchedulerEnum
    #     solver_type: Optional[SolverEnum]
    #     GPU_type_to_GPU_IDs: Dict[GPUType, Set[str]]
    #     dist_job_to_tasks: Dict[str, Tuple[str, ...]]
    #     task_comp_mem_requirements: Dict[str, Tuple[int, int]]
    #     task_comp_over_supply: Dict[str, int]
    #     assignments: Dict[str, Set[str]]
    #     profit: Union[int, float]
    #     do_plot: bool
    recorder_parameters = SnapshotRecordParameters(
        scheduler_name="MMKP_FCFS",
        scheduler_type=SchedulerEnum.MMKP,
        solver_type=SolverEnum.MMKP,
        GPU_type_to_GPU_IDs=GPU_type_to_GPU_IDs,
        dist_job_to_tasks=dist_job_to_tasks,
        task_comp_mem_requirements=task_comp_mem_requirements,
        task_comp_over_supply=task_over_supply,
        task_comp_lack_supply=task_comp_lack_supply,
        assignments=assignments,
        profit=0,
        do_plot=True,
    )
    filepath = str(pathlib.Path(__file__).parent / "output" / "fig" / "fig_test.pdf")
    plot_assignment(recorder_parameters, filepath)


if __name__ == "__main__":
    do_test()
