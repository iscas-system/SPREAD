import copy
import math
from collections import defaultdict
from collections import namedtuple
from itertools import chain
from typing import Tuple, Optional, List, Dict, Set, Any

from cluster import TaskAssignment, Assignments
from config import ClusterConfig, job_deploy_specs
from log import info
from object import GPUType, CompCapacity, Job, Task, PriorityType
from profit import get_profit_calculator
from scheduler import Scheduler
from .solver import do_MMKP_solve_SC_2, \
    SolverParametersSC2, do_partition_solve, SolverResultSC2, \
    PartitionSolverParameters, do_job_distribution_solve, JobDistributionSolverParameters
from .sorter import Sorter


class JobVariant:
    def __init__(self, job_ID: str, worker_count: int, comp: int, cross_node: bool):
        self.job_ID: str = job_ID
        self.worker_count: int = worker_count
        self.comp: int = comp
        self.cross_node: bool = cross_node
        self.variant_job_ID: str = JobVariant.trans_variant_job_ID(job_ID, worker_count, comp, cross_node)

    def to_dict(self) -> Dict:
        return {
            "job_ID": self.job_ID,
            "worker_count": self.worker_count,
            "comp": self.comp,
            "cross_node": self.cross_node,
            "variant_job_ID": self.variant_job_ID
        }

    @staticmethod
    def trans_variant_job_ID(job_ID: str, worker_count: int, comp: int, cross_node: bool):
        return f"{job_ID}|{worker_count}|{comp}|{cross_node}"

    @classmethod
    def from_variant_job_ID(cls, variant_job_ID: str) -> 'JobVariant':
        group = variant_job_ID.split("|")
        assert len(group) == 4
        job_ID = group[0]
        worker_count = int(group[1])
        comp = int(group[2])
        assert group[3] in ["True", "False"]
        cross_node = group[3] == "True"
        return cls(job_ID, worker_count, comp, cross_node)

    def variant_task_IDs(self):
        return [f"{self.variant_job_ID}|task_{i}" for i in range(1, self.worker_count + 1)]

    @staticmethod
    def from_variant_task_ID(variant_task_ID: str) -> 'JobVariant':
        group = variant_task_ID.rsplit("|", maxsplit=1)
        assert len(group) == 2
        return JobVariant.from_variant_job_ID(group[0])

    @staticmethod
    def variant_task_ID_idx(variant_task_ID: str) -> int:
        group = variant_task_ID.rsplit("|", maxsplit=1)
        assert len(group) == 2
        return int(group[1].rsplit("_", maxsplit=1)[1])


class MMKPScheduler(Scheduler):

    def _init_config(self):
        self.use_spread = self.config.get("use_spread", True)
        self.partition_strategy = self.config.get("partition_strategy", "heuristic")  # "heuristic", "round"
        self.partition_size = self.config.get("partition_size", 8)
        self.job_distributing_strategy = self.config.get("job_distributing_strategy",
                                                         "heuristic")  # "heuristic", "round"
        self.GPU_type = GPUType.RTX_2080Ti
        self.job_priority_policy = self.config.get("priority", "FCFS")  # "FCFS", "SRTF"
        self.timeout = self.config.get("timeout", 30)
        self.partition_solver_result = None

    def reuse_assignments(self):
        return self.cluster.assignments.clone(), MMKPScheduler.build_statistics()

    def do_assign(self, preemptive: bool, now: int, done_jobs_between_preemption: Set[Job]) -> Tuple[
        Assignments, Optional[Any]]:
        if self.partition_solver_result is None:
            self.partition_solver_result = do_partition_solve(PartitionSolverParameters(
                timeout=self.timeout,
                GPU_ID_to_node_id=self.cluster.cluster_config.GPU_ID_to_node_id,
                partition_size=self.partition_size,
                strategy=self.partition_strategy,
            ))

        GPU_ID_to_init_task_assignments, job_IDs = self.prepare_assign_ctx(preemptive=preemptive)
        job_mono_comp_mem_demands = self.job_mono_comp_mem_demands(job_IDs=job_IDs)

        partition_cluster_configs = {
            partition_ID: self.create_partition_cluster_config(partition_ID=partition_ID, partition_GPU_IDs=GPU_IDs) for
            partition_ID, GPU_IDs in
            self.partition_solver_result.partition_to_GPU_IDs.items()}

        partition_to_init_task_assignments = self.partition_assignments(
            partition_to_GPU_IDs=self.partition_solver_result.partition_to_GPU_IDs,
            GPU_ID_to_task_assignments=GPU_ID_to_init_task_assignments)
        partition_to_init_assignments = {
            partition_ID: Assignments.from_GPU_ID_to_task_assignments(partition_cluster_configs[partition_ID],
                                                                      partition_to_init_task_assignments[partition_ID])
            for partition_ID in self.partition_solver_result.partition_to_GPU_IDs.keys()
        }

        partition_to_final_assignments = dict()
        partition_to_schedule_statistics = dict()
        GPU_ID_to_final_task_assignments = copy.deepcopy(GPU_ID_to_init_task_assignments)

        unassigned_job_IDs = job_IDs
        done_partition_IDs = set()
        partition_IDs = list(partition_cluster_configs.keys())
        for partition_ID in partition_IDs:
            def not_assigning_partition():
                partition_to_final_assignments[partition_ID] = partition_to_init_assignments[partition_ID]
                partition_to_schedule_statistics[partition_ID] = None
                done_partition_IDs.add(partition_ID)

            unassigned_job_IDs = self.job_priority_sort(unassigned_job_IDs)
            if len(unassigned_job_IDs) == 0:
                not_assigning_partition()
                continue

            GPU_comp_mem_capacity = self.remain_GPU_resource_capacity(cluster_config=self.cluster.cluster_config,
                                                                      GPU_ID_to_task_assignments=GPU_ID_to_final_task_assignments)

            partition_to_GPU_IDs = copy.deepcopy(self.partition_solver_result.partition_to_GPU_IDs)
            for done_partition_ID in done_partition_IDs:
                partition_to_GPU_IDs.pop(done_partition_ID)

            job_distribution_result = do_job_distribution_solve(JobDistributionSolverParameters(
                partition_to_GPU_IDs=partition_to_GPU_IDs,
                GPU_comp_mem_capacity=GPU_comp_mem_capacity,
                GPU_comp_mem_total_capacity=(CompCapacity, GPUType.normalized_memory(self.GPU_type)),
                job_comp_mem_demand=job_mono_comp_mem_demands,
                job_priority=unassigned_job_IDs,
                strategy=self.job_distributing_strategy
            ))
            partition_job_IDs = job_distribution_result.partition_to_jobs[partition_ID]
            if len(partition_job_IDs) == 0:
                not_assigning_partition()
                continue

            solved_partition_assignments, assigned_job_IDs_of_partition, solved_partition_statistics = self.do_assign_on_partition(
                partition_cluster_config=partition_cluster_configs[partition_ID],
                partition_assignments=partition_to_init_assignments[partition_ID],
                preemptive=preemptive,
                job_IDs=partition_job_IDs)

            if not preemptive:
                assert len(solved_partition_assignments.job_ID_to_task_assignments) >= len(partition_to_init_assignments[partition_ID].job_ID_to_task_assignments)

            unassigned_job_IDs_set = set(unassigned_job_IDs)
            unassigned_job_IDs_set = unassigned_job_IDs_set.difference(assigned_job_IDs_of_partition)
            unassigned_job_IDs = list(unassigned_job_IDs_set)

            partition_to_final_assignments[partition_ID] = solved_partition_assignments
            partition_to_schedule_statistics[partition_ID] = solved_partition_statistics

            def update_GPU_ID_to_final_task_assignments():
                for GPU_ID, task_assignments in solved_partition_assignments.GPU_ID_to_task_assignments.items():
                    GPU_ID_to_final_task_assignments[GPU_ID].update(task_assignments)

            update_GPU_ID_to_final_task_assignments()

            GPU_comp_mem_capacity = self.remain_GPU_resource_capacity(cluster_config=self.cluster.cluster_config,
                                                                      GPU_ID_to_task_assignments=GPU_ID_to_final_task_assignments)
            for GPU_ID_, (comp, mem) in GPU_comp_mem_capacity.items():
                if comp < 0 or mem < 0:
                    info("impossible.")

            done_partition_IDs.add(partition_ID)

        merged_assignments = self.merge_partition_assignments(partition_to_final_assignments)
        merged_assignments.check_validity()

        GPU_comp_mem_capacity = self.remain_GPU_resource_capacity(cluster_config=self.cluster.cluster_config,
                                                                  GPU_ID_to_task_assignments=merged_assignments.GPU_ID_to_task_assignments)
        for GPU_ID_, (comp, mem) in GPU_comp_mem_capacity.items():
            if comp < 0 or mem < 0:
                info("impossible.")

        final_assignments = merged_assignments

        if self.use_spread:
            final_assignments = self.refill_assignments_bestfit(unassigned_job_IDs=unassigned_job_IDs,
                                                                curr_assignments=merged_assignments)
            final_assignments.check_validity()

        GPU_comp_mem_capacity = self.remain_GPU_resource_capacity(cluster_config=self.cluster.cluster_config,
                                                                  GPU_ID_to_task_assignments=final_assignments.GPU_ID_to_task_assignments)
        for GPU_ID_, (comp, mem) in GPU_comp_mem_capacity.items():
            if comp < 0 or mem < 0:
                info("impossible.")

        init_assignments = Assignments.from_GPU_ID_to_task_assignments(self.cluster.cluster_config, GPU_ID_to_init_task_assignments)
        if not preemptive:
            assert len(final_assignments.job_ID_to_task_assignments) >= len(init_assignments.job_ID_to_task_assignments)
        return final_assignments, partition_to_schedule_statistics

    def refill_assignments_bestfit(self, unassigned_job_IDs: List[str], curr_assignments: Assignments) -> Assignments:
        unassigned_job_variants = list()
        for unassigned_job_ID in unassigned_job_IDs:
            max_comp, _ = self.data_source.job_maximized_performance_comp(job_ID=unassigned_job_ID,
                                                                          GPU_type=self.GPU_type, worker_count=1,
                                                                          cross_node=False)
            unassigned_job_variants.append(
                JobVariant(job_ID=unassigned_job_ID, worker_count=1, comp=max_comp, cross_node=False))

        GPU_resource_capacity = self.remain_GPU_resource_capacity(cluster_config=self.cluster.cluster_config,
                                                                  GPU_ID_to_task_assignments=curr_assignments.GPU_ID_to_task_assignments)

        refilled_GPU_ID_to_task_assignments = copy.deepcopy(curr_assignments.GPU_ID_to_task_assignments)
        for v in unassigned_job_variants:
            max_comp = v.comp
            _, mem = self.data_source.get_job_task_memory(job_ID=v.job_ID, worker_count=v.worker_count)
            best_fit_GPU_ID = None
            best_fit_GPU_comp_diff = None
            for GPU_ID, (comp_cap, mem_cap) in GPU_resource_capacity.items():
                if comp_cap == 0:
                    continue
                comp_diff = abs(max_comp - comp_cap)
                if mem > mem_cap:
                    continue
                if best_fit_GPU_ID is None or best_fit_GPU_comp_diff is None or \
                        comp_diff < best_fit_GPU_comp_diff:
                    best_fit_GPU_ID = GPU_ID
                    best_fit_GPU_comp_diff = comp_diff
                    break
            if best_fit_GPU_ID is None:
                continue
            comp_cap, mem_cap = GPU_resource_capacity[best_fit_GPU_ID]
            assert comp_cap != 0
            comp_remain = max(0, comp_cap - max_comp)
            comp_used = min(comp_cap, max_comp)
            mem_remain = max(0, mem_cap - mem)
            mem_used = min(mem_cap, mem)
            assert mem_used == mem
            assert mem_cap >= mem
            GPU_resource_capacity[best_fit_GPU_ID] = (comp_remain, mem_remain)
            refilled_GPU_ID_to_task_assignments[best_fit_GPU_ID].add(TaskAssignment(
                GPU_ID=best_fit_GPU_ID,
                GPU_type=self.GPU_type,
                task=Task(job_ID=v.job_ID, task_idx=0),
                comp_req=comp_used,
                memory=mem_used
            ))
        assignments_ = Assignments.from_GPU_ID_to_task_assignments(self.cluster.cluster_config,
                                                                   refilled_GPU_ID_to_task_assignments)
        return assignments_

    def merge_partition_assignments(self, partition_to_assignments: Dict[str, Assignments]) -> Assignments:
        GPU_ID_to_task_assignments = defaultdict(set)
        for partition_id, assignments in partition_to_assignments.items():
            for GPU_ID, task_assignments in assignments.GPU_ID_to_task_assignments.items():
                GPU_ID_to_task_assignments[GPU_ID].update(task_assignments)
        return Assignments.from_GPU_ID_to_task_assignments(cluster_config=self.cluster.cluster_config,
                                                           GPU_ID_to_task_assignments=GPU_ID_to_task_assignments)

    @staticmethod
    def partition_assignments(partition_to_GPU_IDs: Dict[str, List[str]],
                              GPU_ID_to_task_assignments: Dict[str, Set[TaskAssignment]]) -> Dict[
        str, Dict[str, Set[TaskAssignment]]]:
        GPU_ID_to_partition_ID = dict()
        for partition_id, GPU_IDs in partition_to_GPU_IDs.items():
            for GPU_ID in GPU_IDs:
                GPU_ID_to_partition_ID[GPU_ID] = partition_id

        partition_id_to_task_assignments = defaultdict(lambda: defaultdict(set))
        for GPU_ID, task_assignments in GPU_ID_to_task_assignments.items():
            partition_id = GPU_ID_to_partition_ID[GPU_ID]
            partition_id_to_task_assignments[partition_id][GPU_ID].update(task_assignments)
        return partition_id_to_task_assignments

    def create_partition_cluster_config(self, partition_ID: str, partition_GPU_IDs: List[str]):
        GPU_ID_to_node_id = self.cluster.cluster_config.GPU_ID_to_node_id
        partition_GPU_ID_type_node_id = dict()
        for partition_GPU_ID in partition_GPU_IDs:
            GPU_type = self.cluster.cluster_config.get_GPU(partition_GPU_ID).GPU_type
            node_id = GPU_ID_to_node_id[partition_GPU_ID]
            partition_GPU_ID_type_node_id[partition_GPU_ID] = (GPU_type, node_id)
        partition_cluster_config = ClusterConfig.from_GPU_specs(name=partition_ID,
                                                                GPU_ID_type_node_id=partition_GPU_ID_type_node_id)
        return partition_cluster_config

    def job_priority_sort(self, job_IDs: List[str]) -> List[str]:
        jobs = list(self.cluster.get_job(job_ID=job_ID) for job_ID in job_IDs)
        if self.job_priority_policy == "FCFS":
            return Sorter.sort(jobs=jobs, data_source=self.data_source, priority_type=PriorityType.FCFS)
        assert False

    def job_mono_comp_mem_demands(self, job_IDs: List[str]) -> Dict[str, Tuple[int, int]]:
        return {j: self.job_mono_comp_mem_demand(job_ID=j) for j in job_IDs}

    def job_mono_comp_mem_demand(self, job_ID: str) -> Tuple[int, int]:
        comp, _ = self.data_source.job_maximized_performance_comp(job_ID=job_ID, GPU_type=self.GPU_type, worker_count=1,
                                                                  cross_node=False)
        _, mem = self.data_source.get_job_task_memory(job_ID=job_ID, worker_count=1)
        return comp, mem

    def do_assign_on_partition(self,
                               partition_cluster_config: ClusterConfig,
                               partition_assignments: Assignments,
                               preemptive: bool,
                               job_IDs: List[str]) -> Tuple[Assignments, Set[str], Optional[Any]]:
        info(f"MMKP starts do assign, preemptive: {preemptive}")

        profit_calculator = get_profit_calculator()

        def generate_job_variants(job_ID: str) -> List[JobVariant]:
            variants = list()
            node_ids = set(partition_cluster_config.GPU_ID_to_node_id.values())
            for spec in job_deploy_specs:
                cross_node, worker_count = spec
                comp, _ = self.data_source.job_maximized_performance_comp(job_ID=job_ID, GPU_type=self.GPU_type,
                                                                          worker_count=worker_count,
                                                                          cross_node=cross_node)
                if len(node_ids) == 1 and cross_node:
                    continue
                v = JobVariant(job_ID=job_ID, cross_node=cross_node, comp=comp, worker_count=worker_count)
                variants.append(v)
            return variants

        job_variants: List[JobVariant] = list(chain(*[generate_job_variants(job_ID=job_ID) for job_ID in job_IDs]))

        def precalculate_job_variant_profits(job_variants_: List[JobVariant]) -> Dict[str, float]:
            job_variant_profits_ = dict()
            for v in job_variants_:
                p = profit_calculator.calculate(
                    data_source=self.data_source,
                    job_ID=v.job_ID,
                    cluster_config=partition_cluster_config,
                    GPU_type=self.GPU_type,
                    worker_count=v.worker_count,
                    comp_req=v.comp,
                    cross_node=v.cross_node,
                )
                job_variant_profits_[v.variant_job_ID] = p
            return job_variant_profits_

        job_variant_profits = precalculate_job_variant_profits(job_variants)

        def prepare_solver_params(job_variants_: List[JobVariant]):
            # 1
            timeout = self.timeout
            # 2
            GPU_comp_mem_capacity = self.remain_GPU_resource_capacity(
                cluster_config=partition_cluster_config,
                GPU_ID_to_task_assignments=partition_assignments.GPU_ID_to_task_assignments)
            # 3
            job_ID_to_spread_job_IDs: Dict[str, List[str]] = defaultdict(list)
            for v in job_variants_:
                job_ID_to_spread_job_IDs[v.job_ID].append(v.variant_job_ID)
            # 4
            spread_job_ID_to_task_set: Dict[str, List[str]] = defaultdict(list)
            for v in job_variants_:
                spread_job_ID_to_task_set[v.variant_job_ID] = v.variant_task_IDs()
            # 5, 6
            in_node_spread_job_IDs: List[str] = list()
            cross_node_spread_job_IDs: List[str] = list()
            for v in job_variants_:
                if v.cross_node:
                    cross_node_spread_job_IDs.append(v.variant_job_ID)
                else:
                    in_node_spread_job_IDs.append(v.variant_job_ID)
            # 7, 8
            spread_job_task_resource_demands: Dict[str, Tuple[int, int]] = dict()
            spread_job_task_max_profit: Dict[str, float] = dict()
            for v in job_variants_:
                comp = v.comp
                _, mem = self.data_source.get_job_task_memory(v.job_ID, worker_count=v.worker_count)
                task_profit = job_variant_profits[v.variant_job_ID] / v.worker_count
                spread_job_task_resource_demands[v.variant_job_ID] = (comp, mem)
                spread_job_task_max_profit[v.variant_job_ID] = task_profit

            # 9
            GPU_ID_to_node_id: Dict[str, str] = partition_cluster_config.GPU_ID_to_node_id

            solver_params_ = SolverParametersSC2(
                timeout=timeout,
                job_ID_to_spread_job_IDs=job_ID_to_spread_job_IDs,
                spread_job_ID_to_task_set=spread_job_ID_to_task_set,
                spread_job_task_resource_demands=spread_job_task_resource_demands,
                spread_job_task_max_profit=spread_job_task_max_profit,
                GPU_comp_mem_capacity=GPU_comp_mem_capacity,
                in_node_spread_job_IDs=in_node_spread_job_IDs,
                cross_node_spread_job_IDs=cross_node_spread_job_IDs,
                GPU_ID_to_node_id=GPU_ID_to_node_id,
            )
            return solver_params_

        def solve() -> SolverResultSC2:
            SolvingSpec = namedtuple("SolvingSpec", field_names=["max_worker_count", "allow_cross_node"])

            def solve_for_spec(solving_spec_: SolvingSpec):
                max_worker_count_ = solving_spec_.max_worker_count
                allow_cross_node_ = solving_spec_.allow_cross_node
                job_variants_ = [v for v in job_variants if v.worker_count <= max_worker_count_]
                if not allow_cross_node_:
                    job_variants_ = [v for v in job_variants_ if not v.cross_node]
                solver_params_ = prepare_solver_params(job_variants_)
                return do_MMKP_solve_SC_2(solver_params=solver_params_)

            fallback_solving_specs = (
                # SolvingSpec(4, True),
                # SolvingSpec(4, False),
                SolvingSpec(2, True),
                SolvingSpec(2, False),
                SolvingSpec(1, False)
            )
            if not self.use_spread:
                fallback_solving_specs = (
                    SolvingSpec(1, False),
                )
            for solving_spec in fallback_solving_specs:
                solver_result_ = solve_for_spec(solving_spec_=solving_spec)
                info(f"MMKP scheduler starts solving with solving_spec = {solving_spec}.")
                if solver_result_ is not None:
                    info(
                        f"MMKP scheduler uses {solver_result_.duration / 1e9} secs to find the optimal placement with solving spec = {solving_spec}.")
                    return solver_result_
                else:
                    info(f"MMKP scheduler cannot find the optimal placement in "
                         f"{self.timeout} secs, using fallback solution.")
            assert "MMKP scheduler solving timeout with all fallback solutions"

        solver_result = solve()

        def build_assignments(solver_result_: SolverResultSC2):
            spread_job_task_resource_demands = dict()
            for variant_job_ID, comp_mem in solver_result_.solver_parameters_SC2.spread_job_task_resource_demands.items():
                comp, mem = comp_mem
                spread_job_task_resource_demands[variant_job_ID] = comp, mem

            GPU_type_to_task_assignments: Dict[GPUType, Dict[str, Set[TaskAssignment]]] = defaultdict(
                lambda: defaultdict(set))
            for GPU_ID, items in solver_result_.assignment.items():
                for item in items:
                    variant_task_ID_, max_comp = item
                    job_variant_ = JobVariant.from_variant_task_ID(variant_task_ID_)
                    GPU_type = self.GPU_type
                    _, mem_ = spread_job_task_resource_demands[job_variant_.variant_job_ID]

                    comp_ = round(max_comp)
                    if comp_ <= 0:
                        continue
                    job_ID = job_variant_.job_ID
                    task_idx = JobVariant.variant_task_ID_idx(variant_task_ID_)
                    task = Task(job_ID=job_ID, task_idx=task_idx)
                    task_assignment = TaskAssignment(GPU_ID=GPU_ID, GPU_type=GPU_type, task=task,
                                                     comp_req=comp_, memory=mem_)
                    task_assignments = GPU_type_to_task_assignments[GPU_type][job_ID]
                    assert task_assignment not in task_assignments
                    task_assignments.add(task_assignment)
            assignments_ = Assignments(cluster_config=partition_cluster_config,
                                       GPU_type_to_task_assignments=GPU_type_to_task_assignments)
            assignments_ = assignments_.merge(partition_assignments)

            def fix_assignments():
                assignments__ = None
                GPU_ID_to_remain_comp_mem = self.GPU_remain_comp_mem(assignments_.GPU_ID_to_task_assignments)
                with_fixed = False
                for GPU_ID_, (comp_cap, mem_cap) in GPU_ID_to_remain_comp_mem.items():
                    assert mem_cap >= 0
                    if comp_cap >= 0:
                        continue
                    if assignments__ is None:
                        assignments__ = assignments_.clone()
                    with_fixed = True
                    task_assignments_ = assignments__.GPU_ID_to_task_assignments[GPU_ID_]
                    for t_a in task_assignments_:
                        if t_a.comp_req + comp_cap < 1:
                            continue
                        comp_req = t_a.comp_req + comp_cap
                        for task_assignment_ in assignments__.job_ID_to_task_assignments[t_a.task.job_ID]:
                            task_assignment_.comp_req = comp_req
                        break
                if not with_fixed:
                    return assignments_
                return Assignments.from_job_ID_to_task_assignments(assignments__.cluster_config, assignments__.job_ID_to_task_assignments)

            assignments_ = fix_assignments()
            return assignments_

        assignments = build_assignments(solver_result)

        def extract_unassigned_job_IDs(curr_assignments: Assignments):
            assigned_job_IDs_ = set(curr_assignments.job_ID_to_task_assignments.keys())
            unassigned_job_IDs_ = list(set(job_IDs).difference(assigned_job_IDs_))
            unassigned_job_IDs_ = self.job_priority_sort(unassigned_job_IDs_)
            return unassigned_job_IDs_

        unassigned_job_IDs = extract_unassigned_job_IDs(curr_assignments=assignments)

        assigned_job_IDs = set(job_IDs).difference(unassigned_job_IDs)

        def extract_spread_job_variants(solver_result_: SolverResultSC2) -> List[JobVariant]:
            job_variant_IDs = set()
            for _, items in solver_result_.assignment.items():
                for item in items:
                    variant_task_ID, _ = item
                    job_variant = JobVariant.from_variant_task_ID(variant_task_ID)
                    job_variant_IDs.add(job_variant.variant_job_ID)
            return [JobVariant.from_variant_job_ID(job_variant_ID) for job_variant_ID in job_variant_IDs]

        statistics = MMKPScheduler.build_statistics(solver_duration=solver_result.duration,
                                                    spread_job_variants=extract_spread_job_variants(
                                                        solver_result_=solver_result))
        return assignments, assigned_job_IDs, statistics

    def remain_GPU_resource_capacity(self, cluster_config: ClusterConfig,
                                     GPU_ID_to_task_assignments: Dict[str, Set[TaskAssignment]]) -> Dict[
        str, Tuple[int, int]]:
        GPU_mem = GPUType.normalized_memory(self.GPU_type)
        GPU_comp_mem_capacity_: Dict[str, Tuple[int, int]] = {GPU_ID: (CompCapacity, GPU_mem) for GPU_ID in
                                                              cluster_config.GPU_IDs}
        for GPU_ID, task_assignments in GPU_ID_to_task_assignments.items():
            for task_assignment in task_assignments:
                comp_, mem_ = GPU_comp_mem_capacity_[GPU_ID]
                GPU_comp_mem_capacity_[GPU_ID] = comp_ - task_assignment.comp_req, mem_ - task_assignment.memory
        return GPU_comp_mem_capacity_

    @staticmethod
    def build_statistics(solver_duration: float, spread_job_variants: List[JobVariant]) -> Dict:
        return {
            "solver_duration": solver_duration,
            "spread_job_variants": [
                v.to_dict() for v in spread_job_variants
            ]
        }
