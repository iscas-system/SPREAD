from collections import defaultdict
from itertools import chain
from typing import Tuple, Optional, List, Dict, Set, Any

from cluster import TaskAssignment, Assignments
from config import ClusterConfig, job_deploy_specs
from log import info
from object import GPUType, CompCapacity, Job, Task
from profit import get_profit_calculator
from scheduler import Scheduler
from .solver import do_MMKP_solve_SC, \
    SolverParametersSC, do_partition_solve, SolverResult, \
    PartitionSolverParameters, do_job_distribution_solve, JobDistributionSolverParameters


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
        return int(group[1])


class MMKPScheduler(Scheduler):
    # class ScoreWeights:
    #     def __init__(self, effective_resource_score_range: Tuple[float, float], balance_weight: float,
    #                  resource_weight: float):
    #         self.effective_resource_score_range: Tuple[float, float] = effective_resource_score_range
    #         self.balance_weight: float = balance_weight
    #         self.resource_weight: float = resource_weigh

    def _init_config(self):
        self.use_split = self.config.get("use_split", True)
        self.partition_strategy = self.config.get("partition_strategy", "heuristic")  # "heuristic", "round"
        self.partition_size = self.config.get("partition_size", 8)
        self.job_distributing_strategy = self.config.get("job_distributing_strategy",
                                                         "heuristic")  # "heuristic", "round"
        self.job_priority_policy = self.config.get("priority", "FIFO")  # "FIFO", "SRTF"
        # self.strict = self.config.get("strict", True)
        # self.throughput_degrade_threshold = self.config.get("throughput_degrade_threshold", 0.03)
        self.GPU_type = GPUType.RTX_2080Ti
        # self.plan_comp_unit = self.config.get("plan_comp_unit", 1)
        # self.score_weights = [
        #     MMKPScheduler.ScoreWeights((0, 0.2), 4, 1),
        #     MMKPScheduler.ScoreWeights((0.2, np.inf), 4, 4)
        # ]
        # self.resource_score_upper_bound = self.config.get("resource_score_upper_bound", 0.6)
        # self.splitting_saturate_factor = self.config.get("splitting_saturate_factor", 0.5)
        # self.direct_saturate_factor = self.config.get("direct_saturate_factor", 2.5)
        # self.non_preemptive_direct_saturate_factor = self.config.get("non_preemptive_direct_saturate_factor", 128)
        # self.non_preemptive_splitting_saturate_factor = self.config.get("non_preemptive_splitting_saturate_factor", 32)
        # self.saturate_partitions = self.config.get("saturate_partitions", 5)

        # self.direct_saturate_partitions = self.config.get("direct_saturate_partitions", 20)
        # self.splitting_saturate_partitions = self.config.get("splitting_saturate_partitions", 5)
        # self.trail_best_splittable_max_depth = self.config.get("trail_best_splittable_max_depth", 10)
        # self.exploration_max_depth = self.config.get("exploration_max_depth", 20)
        # self.use_round_robin_resource_ratio = self.config.get("use_round_robin_resource_ratio", 0.75)
        # self.solver_duration_upper_bound = self.config.get("solver_duration_upper_bound", 60)
        # self.splitting_jobs_proportion = self.config.get("splitting_jobs_proportion", 1.5)
        self.timeout = self.config.get("timeout", 30)
        # self.rand_variants = self.config.get("rand_variants", False)

        self.selector = self.config.get("selector", "balance")

    class AssignmentPlan:
        def __init__(self,
                     job_ID: str,
                     worker_count: int,
                     comp_req: int,
                     mem_req: int,
                     GPU_type: GPUType,
                     score_weights: List['MMKPScheduler.ScoreWeights'],
                     cross_node: bool,
                     direct_plan: Optional['MMKPScheduler.AssignmentPlan'],
                     iteration_time: float,
                     ):
            self.job_ID: str = job_ID
            self.worker_count: int = worker_count
            self.comp_req: int = comp_req
            self.mem_req: int = mem_req
            self.GPU_type: GPUType = GPU_type
            self.score_weights: List['MMKPScheduler.ScoreWeights'] = score_weights
            self.direct_plan: Optional['MMKPScheduler.AssignmentPlan'] = direct_plan
            self.total_normalized_resource_req = self._total_normalized_resource_req()
            self.balance_score = self._balance_score()
            self.cross_node: bool = cross_node
            self.iteration_time: float = iteration_time
            # for splitting assignment plans
            self.resource_score = self._resource_score(direct_plan)
            self.comprehensive_score = self._comprehensive_score()

        def _total_normalized_resource_req(self) -> float:
            total_normalized_resource_req = (self.comp_req / CompCapacity) * self.worker_count + \
                                            (self.mem_req / GPUType.normalized_memory(
                                                self.GPU_type)) * self.worker_count
            return total_normalized_resource_req

        def _balance_score(self) -> float:
            return abs(self.comp_req / CompCapacity - self.mem_req / GPUType.normalized_memory(self.GPU_type))

        def _resource_score(self, direct: 'MMKPScheduler.AssignmentPlan') -> float:
            if self.direct_plan is None:
                return 1
            splitting_total_normalized_resource_req = self.total_normalized_resource_req
            direct_total_normalized_resource_req = direct.total_normalized_resource_req
            return splitting_total_normalized_resource_req / direct_total_normalized_resource_req - 1

        def _comprehensive_score(self) -> float:
            return self.resource_score

    def reuse_assignments(self):
        return self.cluster.assignments.clone(), MMKPScheduler.build_statistics()

    def do_assign(self, preemptive: bool, now: int, done_jobs_between_preemption: Set[Job]) -> Tuple[
        Assignments, Optional[Any]]:
        self.cluster.jobs.keys()
        partition_solver_result = do_partition_solve(PartitionSolverParameters(
            timeout=self.timeout,
            GPU_ID_to_node_id=self.cluster.cluster_config.GPU_ID_to_node_id,
            partition_size=self.partition_size,
            strategy=self.partition_strategy,
        ))
        GPU_ID_to_task_assignments, job_IDs = self.prepare_assign_ctx(preemptive=preemptive)
        GPU_comp_mem_capacity = self.remain_GPU_resource_capacity(cluster_config=self.cluster.cluster_config,
                                                                  GPU_ID_to_task_assignments=GPU_ID_to_task_assignments)
        job_original_comp_mem_demands = self.job_original_comp_mem_demands(job_IDs=job_IDs)

        job_distribution_result = do_job_distribution_solve(JobDistributionSolverParameters(
            partition_to_GPU_IDs=partition_solver_result.partition_to_GPU_IDs,
            GPU_comp_mem_capacity=GPU_comp_mem_capacity,
            GPU_comp_mem_total_capacity=(CompCapacity, GPUType.normalized_memory(self.GPU_type)),
            job_comp_mem_demand=job_original_comp_mem_demands,
            job_priority=self.job_priority_sort(job_IDs=job_IDs),
            strategy=self.job_distributing_strategy
        ))

        partition_cluster_configs = {partition_ID: GPU_IDs for partition_ID, GPU_IDs in
                                     partition_solver_result.partition_to_GPU_IDs.items()}

        partition_to_task_assignments = self.partition_assignments(
            partition_to_GPU_IDs=partition_solver_result.partition_to_GPU_IDs,
            GPU_ID_to_task_assignments=GPU_ID_to_task_assignments)
        partition_to_assignments = {
            partition_ID: Assignments.from_GPU_ID_to_task_assignments(partition_cluster_configs[partition_ID],
                                                                      partition_to_task_assignments[partition_ID])
            for partition_ID in partition_solver_result.partition_to_GPU_IDs.keys()
        }

        partition_to_final_assignments = dict()
        partition_to_statistics = dict()
        for partition_ID, partition_job_IDs in job_distribution_result.partition_to_jobs.items():
            solved_partition_assignments, solved_partition_statistics = self.do_assign_on_partition(
                partition_cluster_config=partition_cluster_configs[partition_ID],
                partition_assignments=partition_to_assignments[partition_ID],
                preemptive=preemptive,
                job_IDs=partition_job_IDs)
            partition_to_final_assignments[partition_ID] = solved_partition_assignments
            partition_to_statistics[partition_ID] = solved_partition_statistics
        final_assignments = self.merge_partition_assignments(partition_to_final_assignments)
        return final_assignments, partition_to_statistics

    @staticmethod
    def merge_partition_assignments(partition_to_assignments: Dict[str, Assignments]) -> Assignments:
        GPU_ID_to_task_assignments = defaultdict(set)
        for partition_id, assignments in partition_to_assignments.items():
            for GPU_ID, task_assignments in assignments.GPU_ID_to_task_assignments.items():
                GPU_ID_to_task_assignments[GPU_ID].update(task_assignments)
        return Assignments.from_GPU_ID_to_task_assignments(GPU_ID_to_task_assignments=GPU_ID_to_task_assignments)

    @staticmethod
    def partition_assignments(partition_to_GPU_IDs: Dict[str, List[str]],
                              GPU_ID_to_task_assignments: Dict[str, Set[TaskAssignment]]) -> Dict[
        str, Dict[str, Set[TaskAssignment]]]:
        GPU_ID_to_partition_ID = dict()
        for partition_id, GPU_IDs in partition_to_GPU_IDs:
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
        if self.job_priority_policy == "FIFO":
            return sorted(job_IDs, key=lambda j: self.data_source.get_job_spec(job_ID=j).submit_time_nano)
        assert False

    def job_original_comp_mem_demands(self, job_IDs: List[str]) -> Dict[str, Tuple[int, int]]:
        return {j: self.job_original_comp_mem_demand(job_ID=j) for j in job_IDs}

    def job_original_comp_mem_demand(self, job_ID: str) -> Tuple[int, int]:
        comp, _ = self.data_source.job_maximized_performance_comp(job_ID=job_ID, GPU_type=self.GPU_type, worker_count=1,
                                                                  cross_node=False)
        _, mem = self.data_source.get_job_task_memory(job_ID=job_ID, worker_count=1)
        return comp, mem

    # def GPU_resource_capacity(self, cluster_config: ClusterConfig,
    #                           curr_GPU_ID_to_task_assignments: Dict[str, Set[TaskAssignment]]) -> Dict[
    #     str, Tuple[int, int]]:
    #     GPU_mem = GPUType.normalized_memory(self.GPU_type)
    #     GPU_comp_mem_capacity: Dict[str, Tuple[int, int]] = dict()
    #     for GPU_ID in cluster_config.GPU_IDs:
    #         GPU_comp_mem_capacity[GPU_ID] = (CompCapacity, GPU_mem)
    #
    #     for GPU_ID, task_assignments in curr_GPU_ID_to_task_assignments.items():
    #         for task_assignment in task_assignments:
    #             comp, mem = GPU_comp_mem_capacity[GPU_ID]
    #             GPU_comp_mem_capacity[GPU_ID] = comp - task_assignment.comp_req, mem - task_assignment.memory
    #     return GPU_comp_mem_capacity

    def do_assign_on_partition(self,
                               partition_cluster_config: ClusterConfig,
                               partition_assignments: Assignments,
                               preemptive: bool,
                               job_IDs: List[str]) -> Tuple[Assignments, Optional[Any]]:
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

        job_variants: List[JobVariant] = list(chain(*generate_job_variants(job_ID=job_ID) for job_ID in job_IDs))

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
            solver_enum = self.solver_enum
            # 2
            timeout = self.timeout
            # 3
            GPU_comp_mem_capacity = self.remain_GPU_resource_capacity(
                cluster_config=partition_cluster_config,
                GPU_ID_to_task_assignments=partition_assignments.GPU_ID_to_task_assignments)
            # 4
            job_ID_to_spread_job_IDs: Dict[str, List[str]] = defaultdict(list)
            for v in job_variants_:
                job_ID_to_spread_job_IDs[v.job_ID].append(v.variant_job_ID)
            # 5
            spread_job_ID_to_task_sets: Dict[str, List[str]] = dict(list)
            for v in job_variants_:
                if v.worker_count == 1:
                    continue
                spread_job_ID_to_task_sets[v.variant_job_ID] = v.variant_task_IDs()
            # 6, 7
            in_node_spread_job_IDs: List[str] = list()
            cross_node_spread_job_IDs: List[str] = list()
            for v in job_variants_:
                if v.cross_node:
                    cross_node_spread_job_IDs.append(v.variant_job_ID)
                else:
                    in_node_spread_job_IDs.append(v.variant_job_ID)
            # 8
            dist_tasks: List[Tuple[str, ...]] = list()
            for v in job_variants_:
                if v.worker_count > 1:
                    dist_tasks.append(tuple(v.variant_task_IDs()))
            # 9
            task_comp_mem_requirements_and_profits: Dict[str, Tuple[int, int, float]] = dict()
            for v in job_variants_:
                comp = v.comp
                _, mem = self.data_source.get_job_task_memory(v.job_ID, worker_count=v.worker_count)
                task_profit = job_variant_profits[v.variant_job_ID] / v.worker_count
                for t in v.variant_task_IDs():
                    task_comp_mem_requirements_and_profits[t] = (comp, mem, task_profit)
            # 10
            GPU_ID_to_node_id: Dict[str, str] = partition_cluster_config.GPU_ID_to_node_id

            solver_params_ = SolverParametersSC(
                solver_type=solver_enum,
                timeout=timeout,
                job_ID_to_spread_job_IDs=job_ID_to_spread_job_IDs,
                spread_job_ID_to_task_sets=spread_job_ID_to_task_sets,
                GPU_comp_mem_capacity=GPU_comp_mem_capacity,
                in_node_spread_job_IDs=in_node_spread_job_IDs,
                cross_node_spread_job_IDs=cross_node_spread_job_IDs,
                dist_tasks=dist_tasks,
                task_comp_mem_requirements_and_profits=task_comp_mem_requirements_and_profits,
                GPU_ID_to_node_id=GPU_ID_to_node_id,
            )
            return solver_params_

        def solve() -> SolverResult:
            def solve_for_max_worker_count(max_worker_count_: int):
                job_variants_ = [v for v in job_variants if v.worker_count <= max_worker_count_]
                solver_params_ = prepare_solver_params(job_variants_)
                return do_MMKP_solve_SC(solver_params=solver_params_)

            for max_worker_count in (4, 2, 1):
                solver_result_ = solve_for_max_worker_count(max_worker_count_=max_worker_count)
                info(f"MMKP scheduler starts solving with maximum worker count = {max_worker_count}.")
                if solver_result_ is not None:
                    info(
                        f"MMKP scheduler uses {solver_result_.duration} secs to find the optimal placement with maximum worker count = {max_worker_count}.")
                    return solver_result_
                else:
                    info(f"MMKP scheduler cannot find the optimal placement in "
                         f"{self.timeout} secs, using fallback solution.")
            assert "MMKP scheduler solving timeout with all fallback solutions"

        solver_result = solve()

        def build_assignments(solver_result_: SolverResult):
            variant_task_comp_mem_demands = dict()
            for variant_task_ID_, comp_mem_profit in solver_result.solver_parameters_SC.task_comp_mem_requirements_and_profits.items():
                comp, mem, _ = comp_mem_profit
                variant_task_comp_mem_demands[variant_task_ID_] = comp, mem

            GPU_type_to_task_assignments: Dict[GPUType, Dict[str, Set[TaskAssignment]]] = defaultdict(
                lambda: defaultdict(set))
            for GPU_ID, variant_task_IDs in solver_result_.assignment.items():
                for variant_task_ID_ in variant_task_IDs:
                    job_variant_ = JobVariant.from_variant_task_ID(variant_task_ID_)
                    GPU_type = self.GPU_type
                    comp_, mem_ = variant_task_comp_mem_demands[variant_task_ID_]

                    job_ID = job_variant_.job_ID
                    task_idx = JobVariant.variant_task_ID_idx(variant_task_ID_)
                    task = Task(job_ID=job_ID, task_idx=task_idx)
                    task_assignment = TaskAssignment(GPU_ID=GPU_ID, GPU_type=GPU_type, task=task,
                                                     comp_req=comp_, memory=mem_)
                    task_assignments = GPU_type_to_task_assignments[GPU_type][job_ID]
                    assert task_assignment not in task_assignments
                    task_assignments.add(task_assignment)
            assignments_ = Assignments(cluster_config=partition_cluster_config, GPU_type_to_task_assignments=GPU_type_to_task_assignments)
            assignments_.merge(partition_assignments)
            return assignments_

        assignments = build_assignments(solver_result)

        def extract_unassigned_job_IDs(curr_assignments: Assignments):
            assigned_job_IDs = set(curr_assignments.job_ID_to_task_assignments.keys())
            unassigned_job_IDs = list(set(job_IDs).difference(assigned_job_IDs))
            unassigned_job_IDs = self.job_priority_sort(unassigned_job_IDs)
            return unassigned_job_IDs

        def refill_assignments_bestfit(curr_assignments: Assignments) -> Assignments:
            unassigned_job_IDs = extract_unassigned_job_IDs(curr_assignments)
            unassigned_job_variants = [job_variant_ for job_variant_ in job_variants if
                                       job_variant_.job_ID in unassigned_job_IDs and job_variant_.worker_count == 1]
            GPU_resource_capacity = self.remain_GPU_resource_capacity(cluster_config=partition_cluster_config,
                                                                      GPU_ID_to_task_assignments=curr_assignments.GPU_ID_to_task_assignments)

            refilled_GPU_ID_to_task_assignments = curr_assignments.GPU_ID_to_task_assignments
            for v in unassigned_job_variants:
                comp = v.comp
                _, mem = self.data_source.get_job_task_memory(job_ID=v.job_ID, worker_count=v.worker_count)
                target_GPU_ID = None
                GPU_resource_capacity_tuples = [(GPU_ID, comp_cap, mem_cap) for GPU_ID, (comp_cap, mem_cap) in
                                                GPU_resource_capacity.items()]
                GPU_resource_capacity_tuples.sort(key=lambda t_: t_[1])
                for t in GPU_resource_capacity_tuples:
                    GPU_ID, comp_cap, mem_cap = t
                    if comp < comp_cap and mem < mem_cap:
                        target_GPU_ID = GPU_ID
                        break
                if target_GPU_ID is None:
                    continue
                comp_cap, mem_cap = GPU_resource_capacity[target_GPU_ID]
                GPU_resource_capacity[target_GPU_ID] = (comp_cap - comp, mem_cap - mem)
                refilled_GPU_ID_to_task_assignments[target_GPU_ID].add(TaskAssignment(
                    GPU_ID=target_GPU_ID,
                    GPU_type=self.GPU_type,
                    task=Task(job_ID=v.job_ID, task_idx=0),
                    comp_req=comp,
                    memory=mem
                ))
            assignments_ = Assignments.from_GPU_ID_to_task_assignments(partition_cluster_config, refilled_GPU_ID_to_task_assignments)
            return assignments_

        assignments = refill_assignments_bestfit(curr_assignments=assignments)

        def extract_spread_job_variants(solver_result_: SolverResult) -> List[JobVariant]:
            job_variant_IDs = set()
            for _, task_IDs in solver_result_.assignment.items():
                for task_ID in task_IDs:
                    job_variant = JobVariant.from_variant_task_ID(task_ID)
                    job_variant_IDs.add(job_variant.variant_job_ID)
            return [JobVariant.from_variant_job_ID(job_variant_ID) for job_variant_ID in job_variant_IDs]

        statistics = MMKPScheduler.build_statistics(solver_duration=solver_result.duration,
                                                    spread_job_variants=extract_spread_job_variants(
                                                        solver_result_=solver_result))
        return assignments, statistics

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
