from typing import Callable, Dict, Optional

from cluster import Cluster
from data_source import DataSource
from object import SchedulerEnum, ProfitEnum, SolverEnum
from scheduler import Scheduler
from .MMKP import MMKPScheduler
from .RR import RRScheduler
from .kube_share import KubeShareScheduler

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
                                                                Optional[SolverEnum],
                                                                Optional[ProfitEnum],
                                                                DataSource,
                                                                Cluster,
                                                                Dict
                                                            ], Scheduler]]] = {
    SchedulerEnum.MMKP: MMKPScheduler,
    SchedulerEnum.RoundRobin: RRScheduler,
    SchedulerEnum.Themis: None,
    SchedulerEnum.Tiresias: None,
    SchedulerEnum.Optimus: None,
    SchedulerEnum.KubeShare: KubeShareScheduler
}


def init_scheduler(
        name: str,
        scheduler_enum: SchedulerEnum,
        solver_enum: Optional[SolverEnum],
        profit_enum: Optional[ProfitEnum],
        data_source: DataSource,
        cluster: Cluster,
        config: Dict):
    return scheduler_init_funcs[scheduler_enum](name, scheduler_enum, solver_enum, profit_enum, data_source, cluster,
                                                config)
