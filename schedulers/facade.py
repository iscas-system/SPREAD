from typing import Callable, Dict, Optional, Set

from cluster import Cluster, TaskAssignment
from data_source import DataSource
from object import SchedulerEnum, ProfitEnum, GPUType, SolverEnum
from profit import ProfitComprehensiveUtilization
from scheduler import Scheduler

#                  name: str,
#                  scheduler_enum: SchedulerEnum,
#                  solver_enum: Optional[SolverEnum],
#                  profit_enum: Optional[ProfitEnum],
#                  data_source: DataSource,
#                  cluster: Cluster,
#                  config: Dict

scheduler_init_funcs: Dict[SchedulerEnum, Optional[Callable[[
                                                                str,
                                                                SchedulerEnum,
                                                                SolverEnum,
                                                                ProfitEnum,
                                                                DataSource,
                                                                Cluster,
                                                                Dict
                                                            ], Scheduler]]] = {
    SchedulerEnum.MMKP: None,
    SchedulerEnum.RR: None,
    SchedulerEnum.Themis: None,
    SchedulerEnum.Tiresias: None,
    SchedulerEnum.Optimus: None,
}


def init_scheduler(
        name: str,
        scheduler_enum: SchedulerEnum,
        solver_enum: SolverEnum,
        profit_enum: ProfitEnum,
        data_source: DataSource,
        cluster: Cluster,
        config: Dict):
    return scheduler_init_funcs[scheduler_enum](name, scheduler_enum, solver_enum, profit_enum, data_source, cluster,
                                                config)


def calculate_profit(profit_enum: ProfitEnum, data_source: DataSource, job_ID: str, GPU_type: GPUType,
                     task_assignments: Set[TaskAssignment]) -> float:
    return {
        ProfitEnum.ComprehensiveUtilization: ProfitComprehensiveUtilization
    }[profit_enum].calculate(data_source, job_ID, GPU_type, task_assignments)
