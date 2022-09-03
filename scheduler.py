import abc
from abc import ABC
from typing import Optional, Dict, Tuple, Set

from cluster import Cluster
from data_source import DataSource
from model import SnapshotRecordParameters
from profit import get_profit_calculator
from object import SchedulerEnum, SolverEnum, ProfitEnum, GPUType, MemoryUnit, CompCapacity


class Scheduler(ABC):
    def __init__(self,
                 name: str,
                 scheduler_enum: SchedulerEnum,
                 solver_enum: Optional[SolverEnum],
                 profit_enum: Optional[ProfitEnum],
                 data_source: DataSource,
                 cluster: Cluster,
                 config: Dict):
        self.name: str = name
        self.scheduler_enum: SchedulerEnum = scheduler_enum
        self.data_source: DataSource = data_source
        self.cluster: Cluster = cluster
        self.solver_enum: Optional[SolverEnum] = solver_enum
        self.profit_enum: Optional[ProfitEnum] = profit_enum
        self.config: Dict = config
        self.do_plot: bool = config.get("do_plot", True)
        self.__init_view_data()

    def __init_view_data(self):
        self.GPU_type_to_comp_mem_capacity: Dict[GPUType, Tuple[int, int]] = dict()
        self.GPU_type_to_GPU_IDs: Dict[GPUType, Set[str]] = dict()
        for GPU_type, gs in self.cluster.GPUs.items():
            comp, mem = CompCapacity, GPUType.memory(GPU_type=GPU_type) // MemoryUnit
            self.GPU_type_to_comp_mem_capacity[GPU_type] = (comp, mem)
            self.GPU_type_to_GPU_IDs[GPU_type] = {g.GPU_ID for g in gs}

    def build_snapshot_record_parameters(self) -> SnapshotRecordParameters:
        # RecordParameters(
        #     scheduler_type: SchedulerEnum
        #     solver_type: Optional[SolverEnum]
        #     GPU_type_to_comp_mem_capacity: Dict[GPUType, Tuple[Union[float, int], Union[float, int]]]
        #     GPU_type_to_GPU_IDs: Dict[GPUType, Set[str]]
        #     dist_job_to_tasks: Dict[str, Tuple[str, ...]]
        #     task_comp_mem_requirements: Dict[str, Tuple[int, int]]
        #     assignments: Dict[str, Set[str]]
        #     do_plot: bool
        # )
        dist_job_to_tasks: Dict[str, Tuple[str, ...]] = self.cluster.assignments.dist_job_to_tasks()
        task_comp_mem_requirements: Dict[str, Tuple[int, int]] = self.cluster.assignments.task_comp_mem_requirements()
        assignments = self.cluster.assignments.to_solver_assignments()

        return SnapshotRecordParameters(
            scheduler_name=self.name,
            scheduler_type=self.scheduler_enum,
            solver_type=self.solver_enum,
            GPU_type_to_comp_mem_capacity=self.GPU_type_to_comp_mem_capacity,
            GPU_type_to_GPU_IDs=self.GPU_type_to_GPU_IDs,
            dist_job_to_tasks=dist_job_to_tasks,
            task_comp_mem_requirements=task_comp_mem_requirements,
            assignments=assignments,
            profit=self.cluster.calc_profits(data_source=self.data_source, profit_calculator=get_profit_calculator(self.profit_enum)),
            do_plot=self.do_plot
        )

    @abc.abstractmethod
    def do_assign(self, preemptive: bool) -> Tuple[Optional[Tuple[int, ...]], ]:
        """

        :param preemptive:
        :return:
            Optional[Tuple[int, ...]]: Solver Overheads
        """
        ...
