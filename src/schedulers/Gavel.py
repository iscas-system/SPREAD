from collections import defaultdict
from collections import namedtuple
from typing import Tuple, Optional, Set, List, Any

import numpy as np
from .best_fit import BestFitScheduler
from cluster import TaskAssignment, Assignments
from object import CompCapacity, GPUType, Task, Job
from scheduler import Scheduler


class GavelScheduler(Scheduler):
    def _init_config(self):
        self.strict = self.config.get("strict", True)
        self.GPU_type = GPUType.RTX_2080Ti

    def do_assign(self, preemptive: bool, now: int, done_jobs_between_preemption: Set[Job]) -> Tuple[
        Assignments, Optional[Any]]:
        GPU_ID_to_task_assignments, job_IDs = self.prepare_assign_ctx(preemptive)
        for GPU_ID in self.cluster.cluster_config.GPU_IDs:
            if GPU_ID not in GPU_ID_to_task_assignments:
                GPU_ID_to_task_assignments[GPU_ID] = set()

        # GPU_ID_comp_mem_type = namedtuple(typename="GPU_ID_comp", field_names=["GPU_ID", "comp", "mem"])
        GPU_mem = GPUType.normalized_memory(GPU_type=self.GPU_type)
        job_IDs = job_IDs[:128]
        # comp_req = CompCapacity // 2
        # assigned_job_IDs = set()
        empty_GPU_count = 0
        for GPU_ID, task_assignments in GPU_ID_to_task_assignments.items():
            if len(task_assignments) == 0:
                empty_GPU_count += 1
        if len(job_IDs) <= empty_GPU_count:
            return BestFitScheduler.best_fit_assign(self, GPU_type=self.GPU_type, GPU_ID_to_task_assignments=GPU_ID_to_task_assignments, job_IDs=job_IDs), None

        def job_mono_max_norm_throughput(job_ID_: str):
            _, iter_nano = self.data_source.job_maximized_performance_comp(job_ID=job_ID_, GPU_type=self.GPU_type,
                                                                           worker_count=1, cross_node=False)
            throughput = 1. / iter_nano
            max_iter_nano, _ = self.data_source.job_maximized_performance(job_ID=job_ID_, GPU_type=self.GPU_type)
            job_max_throughput = 1. / max_iter_nano
            norm_throughput = throughput / job_max_throughput
            return norm_throughput

        for job_ID in job_IDs:
            max_throughput = 0
            max_throughput_GPU_ID = None
            max_throughput_comp = None
            one_task_visited = False
            _, self_mem = self.data_source.get_job_task_memory(job_ID=job_ID, worker_count=1)
            self_comp, self_max_iter_nano = self.data_source.job_maximized_performance_comp(
                job_ID=job_ID,
                GPU_type=self.GPU_type,
                worker_count=1, cross_node=False)
            for GPU_ID, task_assignments in GPU_ID_to_task_assignments.items():
                if len(task_assignments) >= 2:
                    continue
                if len(task_assignments) == 0:
                    if one_task_visited:
                        continue
                    norm_throughput = job_mono_max_norm_throughput(job_ID_=job_ID)
                    one_task_visited = True
                    if norm_throughput > max_throughput:
                        max_throughput = norm_throughput
                        max_throughput_GPU_ID = GPU_ID
                        max_throughput_comp, _ = self.data_source.job_maximized_performance_comp(job_ID=job_ID,
                                                                                                     GPU_type=self.GPU_type,
                                                                                                     worker_count=1,
                                                                                                     cross_node=False)
                    continue
                assert len(task_assignments) == 1
                task_assignment = next(iter(task_assignments))
                _, other_mem = self.data_source.get_job_task_memory(job_ID=task_assignment.task.job_ID, worker_count=1)
                if self_mem + other_mem > GPU_mem:
                    continue
                other_comp, _ = self.data_source.job_maximized_performance_comp(job_ID=task_assignment.task.job_ID,
                                                                                   GPU_type=self.GPU_type,
                                                                                   worker_count=1, cross_node=False)
                self_comp_fix = self_comp
                if other_comp + self_comp > CompCapacity:
                    over_comp = other_comp + self_comp - CompCapacity
                    if self_comp <= over_comp:
                        continue
                    self_comp_fix -= over_comp
                self_throughput = 1. / self_max_iter_nano * (1. * self_comp_fix / self_comp)
                iter_nano, _ = self.data_source.job_maximized_performance(job_ID=job_ID, GPU_type=self.GPU_type)
                self_max_throughput = 1. / iter_nano
                self_norm_throughput = self_throughput / self_max_throughput
                norm_throughput = self_norm_throughput
                if norm_throughput > max_throughput:
                    max_throughput = norm_throughput
                    max_throughput_GPU_ID = GPU_ID
                    max_throughput_comp = self_comp_fix

            if max_throughput_GPU_ID is None:
                continue

            task_assignment = TaskAssignment(GPU_ID=max_throughput_GPU_ID,
                                             GPU_type=self.GPU_type,
                                             task=Task(job_ID=job_ID, task_idx=0),
                                             comp_req=max_throughput_comp,
                                             memory=self_mem)
            GPU_ID_to_task_assignments[max_throughput_GPU_ID].add(task_assignment)

        # while True:
        #     job_ID_to_comp_enough_GPU_ID_com_mem_slice = dict()
        #     for job_ID in job_IDs:
        #         if job_ID in assigned_job_IDs:
        #             continue
        #         GPU_ID_to_remain_comp_mem = self.GPU_remain_comp_mem(
        #             GPU_ID_to_task_assignments=GPU_ID_to_task_assignments)
        #         remain_GPU_ID_comp_mem_list: List[GPU_ID_comp_mem_type] = list()
        #         for GPU_ID in self.cluster.cluster_config.GPU_IDs:
        #             remain_comp_mem = GPU_ID_to_remain_comp_mem[GPU_ID]
        #             comp, mem = remain_comp_mem
        #             remain_GPU_ID_comp_mem_list.append(GPU_ID_comp_mem_type(GPU_ID, comp, mem))
        #         if len(remain_GPU_ID_comp_mem_list) == 0:
        #             break
        #         remain_GPU_ID_comp_mem_list.sort(key=lambda t: t[0], reverse=True)
        #         worker_count = 1
        #         window_size = worker_count
        #         _, task_mem = self.data_source.get_job_task_memory(job_ID, worker_count)
        #         comp_enough_GPU_ID_com_mem_slice_list: List[Tuple[List[GPU_ID_comp_mem_type], float]] = list()
        #         for i in range(len(remain_GPU_ID_comp_mem_list) - window_size + 1):
        #             comp_enough = True
        #             slice_GPU_ID_comp_mem = remain_GPU_ID_comp_mem_list[i: i + window_size]
        #             slice_total_resource = 0
        #             for sliding_idx, item in enumerate(slice_GPU_ID_comp_mem):
        #                 if item.comp < comp_req:
        #                     comp_enough = False
        #                     continue
        #                 if item.mem < task_mem:
        #                     comp_enough = False
        #                     continue
        #                 slice_total_resource += item.comp / CompCapacity + item.mem / GPU_mem
        #                 break
        #             if not comp_enough:
        #                 continue
        #             comp_enough_GPU_ID_com_mem_slice_list.append((slice_GPU_ID_comp_mem, slice_total_resource))
        #         comp_enough_GPU_ID_com_mem_slice_list.sort(key=lambda slice_tuple: slice_tuple[1])
        #         if len(comp_enough_GPU_ID_com_mem_slice_list) == 0:
        #             continue
        #         job_ID_to_comp_enough_GPU_ID_com_mem_slice[job_ID] = comp_enough_GPU_ID_com_mem_slice_list[0]
        #
        #     if len(job_ID_to_comp_enough_GPU_ID_com_mem_slice) == 0:
        #         break
        #
        #     min_total_resource = np.inf
        #     min_frag_job_ID = None
        #     for job_ID, com_mem_slice in job_ID_to_comp_enough_GPU_ID_com_mem_slice.items():
        #         total_resource = com_mem_slice[1]
        #         if total_resource < min_total_resource:
        #             min_total_resource = total_resource
        #             min_frag_job_ID = job_ID
        #     assigned_job_IDs.add(min_frag_job_ID)
        #     slice_GPU_ID_comp_mem = job_ID_to_comp_enough_GPU_ID_com_mem_slice[min_frag_job_ID][0]
        #     worker_count = 1
        #     _, task_mem = self.data_source.get_job_task_memory(min_frag_job_ID, worker_count=worker_count)
        #     for task_idx in range(worker_count):
        #         GPU_ID_comp_mem = slice_GPU_ID_comp_mem[task_idx]
        #         GPU_ID, _, _ = GPU_ID_comp_mem
        #         task_assignment = TaskAssignment(GPU_ID=GPU_ID,
        #                                          GPU_type=self.GPU_type,
        #                                          task=Task(job_ID=min_frag_job_ID, task_idx=task_idx),
        #                                          comp_req=comp_req,
        #                                          memory=task_mem)
        #         GPU_ID_to_task_assignments[GPU_ID].add(task_assignment)

        assignments = Assignments.from_GPU_ID_to_task_assignments(cluster_config=self.cluster.cluster_config,
                                                                  GPU_ID_to_task_assignments=GPU_ID_to_task_assignments)
        return assignments, None
