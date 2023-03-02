from collections import defaultdict
from collections import namedtuple
from typing import Tuple, Optional, Set, Dict, List, Any
from .best_fit import BestFitScheduler
from cluster import TaskAssignment, Assignments
from object import CompCapacity, GPUType, Task, PriorityType, Job
from scheduler import Scheduler
from schedulers.sorter import Sorter


class AFSScheduler(Scheduler):
    def _init_config(self):
        self.GPU_type = GPUType.RTX_2080Ti
        ...

    def do_assign(self, preemptive: bool, now: int, done_jobs_between_preemption: Set[Job]) -> Tuple[
        Assignments, Optional[Any]]:
        GPU_ID_to_task_assignments, job_IDs = self.prepare_assign_ctx(preemptive)

        for GPU_ID in self.cluster.cluster_config.GPU_IDs:
            if GPU_ID not in GPU_ID_to_task_assignments:
                GPU_ID_to_task_assignments[GPU_ID] = set()

        job_IDs = Sorter.sort(jobs=[self.cluster.get_job(job_ID) for job_ID in job_IDs], data_source=self.data_source,
                              priority_type=PriorityType.FCFS)
        GPU_ID_comp_mem_type = namedtuple(typename="GPU_ID_comp", field_names=["GPU_ID", "comp", "mem"])
        job_IDs = job_IDs[:128]

        empty_GPU_count = 0
        for GPU_ID, task_assignments in GPU_ID_to_task_assignments.items():
            if len(task_assignments) == 0:
                empty_GPU_count += 1
        if len(job_IDs) <= empty_GPU_count:
            return BestFitScheduler.best_fit_assign(self, GPU_type=self.GPU_type, GPU_ID_to_task_assignments=GPU_ID_to_task_assignments, job_IDs=job_IDs), None


        for job_ID in job_IDs:
            GPU_ID_to_remain_comp_mem = self.GPU_remain_comp_mem(GPU_ID_to_task_assignments=GPU_ID_to_task_assignments)
            remain_GPU_ID_comp_mem_list: List[GPU_ID_comp_mem_type] = list()
            for GPU_ID in self.cluster.cluster_config.GPU_IDs:
                remain_comp_mem = GPU_ID_to_remain_comp_mem[GPU_ID]
                comp, mem = remain_comp_mem
                remain_GPU_ID_comp_mem_list.append(GPU_ID_comp_mem_type(GPU_ID, comp, mem))
            remain_GPU_ID_comp_mem_list.sort(key=lambda t: t[0], reverse=True)

            def try_assign(worker_count_: int, cross_node_: bool) -> bool:
                window_size = worker_count_
                task_comp, _ = self.data_source.job_maximized_performance_comp(job_ID=job_ID, GPU_type=self.GPU_type, worker_count=worker_count_, cross_node=cross_node_)
                _, task_mem = self.data_source.get_job_task_memory(job_ID, worker_count=worker_count_)
                comp_enough_GPU_ID_com_mem_slice_list: List[Tuple[List[GPU_ID_comp_mem_type], float, bool]] = list()
                for i in range(len(remain_GPU_ID_comp_mem_list) - window_size + 1):
                    comp_enough = True
                    slice_GPU_ID_comp_mem = remain_GPU_ID_comp_mem_list[i: i + window_size]
                    slice_total_resource = 0
                    for sliding_idx, item in enumerate(slice_GPU_ID_comp_mem):
                        if item.comp < task_comp:
                            comp_enough = False
                            continue
                        slice_total_resource += item.comp / CompCapacity
                        if item.mem < task_mem:
                            comp_enough = False
                            continue
                        break
                    if not comp_enough:
                        continue
                    node_ids = set()
                    for GPU_ID_, _, _ in slice_GPU_ID_comp_mem:
                        node_ids.add(self.cluster.cluster_config.GPU_ID_to_node_id[GPU_ID_])
                    is_cross_node = len(node_ids) > 1
                    comp_enough_GPU_ID_com_mem_slice_list.append((slice_GPU_ID_comp_mem, slice_total_resource, is_cross_node))
                comp_enough_GPU_ID_com_mem_slice_list.sort(key=lambda slice_tuple: slice_tuple[1])
                if len(comp_enough_GPU_ID_com_mem_slice_list) == 0:
                    return False
                target_slice_GPU_ID_comp_mem = None
                for item in comp_enough_GPU_ID_com_mem_slice_list:
                    slice_GPU_ID_comp_mem, _, is_cross_node = item
                    if is_cross_node != cross_node_:
                        continue
                    target_slice_GPU_ID_comp_mem = slice_GPU_ID_comp_mem
                    break
                if target_slice_GPU_ID_comp_mem is None:
                    return False
                for task_idx in range(worker_count_):
                    GPU_ID_comp_mem = target_slice_GPU_ID_comp_mem[task_idx]
                    GPU_ID_, _, _ = GPU_ID_comp_mem
                    task_assignment = TaskAssignment(GPU_ID=GPU_ID_,
                                                     GPU_type=self.GPU_type,
                                                     task=Task(job_ID=job_ID, task_idx=task_idx),
                                                     comp_req=task_comp,
                                                     memory=task_mem)
                    GPU_ID_to_task_assignments[GPU_ID_].add(task_assignment)
                return True
            specs = [(2, False), (1, False)]
            for spec in specs:
                worker_count, cross_node = spec
                if try_assign(worker_count, cross_node):
                    break

        assignments = Assignments.from_GPU_ID_to_task_assignments(
            self.cluster.cluster_config,
            GPU_ID_to_task_assignments=GPU_ID_to_task_assignments)
        # oversupplied_assignments = assignments.supplement_over_supply()
        return assignments, None
