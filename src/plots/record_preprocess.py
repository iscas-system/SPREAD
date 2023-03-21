import copy
import json
import os
import pathlib
from collections import defaultdict
from enum import Enum
from typing import Dict, Set

from plot_common import colors
from mono_job_data_preprocess import *

class LOAD_UTIL_CONFIG:
    LOAD_UTIL = False

class SchedulerName(Enum):
    SPREAD = "MMKP"
    SPREAD_PRIME = "MMKP_no_spread"
    SPREAD_RR_PARTI = "MMKP_RR_partition"
    SPREAD_RR_DISTRI = "MMKP_RR_distribution"
    SPREAD_RR_PARTI_DISTRI = "MMKP_RR_partition_distribution"

    SPREAD_2 = "MMKP_2"
    SPREAD_3 = "MMKP_3"
    SPREAD_4 = "MMKP_4"
    SPREAD_5 = "MMKP_5"
    SPREAD_6 = "MMKP_6"
    SPREAD_7 = "MMKP_7"
    SPREAD_8 = "MMKP_8"
    SPREAD_9 = "MMKP_9"
    SPREAD_10 = "MMKP_10"
    SPREAD_11 = "MMKP_11"
    SPREAD_12 = "MMKP_12"

    RoundRobin = "RoundRobin"
    KubeShare = "KubeShare"
    BestFit = "BestFit"
    AFS = "AFS"
    Tiresias = "Tiresias"
    Gavel = "Gavel"
    Kubernetes = "Kubernetes"

    MMKP = "MMKP_strict"
    MMKP_strict = "MMKP_strict"
    MMKP_strict_rand_3 = "MMKP_strict_rand_3"
    MMKP_strict_rand_variants = "MMKP_strict_rand_variants"
    MMKP_strict_no_split = "MMKP_strict_no_split"
    MMKP_strict_random_select = "MMKP_strict_random_select"
    MMKP_strict_no_split_random_select = "MMKP_strict_no_split_random_select"
    MMKP_best_effort = "MMKP_best_effort"
    RoundRobin_strict = "RoundRobin_strict"
    RoundRobin_best_effort = "RoundRobin_best_effort"

    MMKP_strict_05 = "MMKP_strict_05"
    MMKP_strict_075 = "MMKP_strict_075"
    MMKP_strict_1 = "MMKP_strict_1"
    MMKP_strict_125 = "MMKP_strict_125"
    MMKP_strict_15 = "MMKP_strict_15"

    MMKP_strict_0_1 = "MMKP_strict_0_1"
    MMKP_strict_0_125 = "MMKP_strict_0_125"
    MMKP_strict_0_15 = "MMKP_strict_0_15"
    MMKP_strict_0_175 = "MMKP_strict_0_175"
    MMKP_strict_0_2 = "MMKP_strict_0_2"

    MMKP_strict_25_1 = "MMKP_strict_25_1"
    MMKP_strict_25_125 = "MMKP_strict_25_125"
    MMKP_strict_25_15 = "MMKP_strict_25_15"
    MMKP_strict_25_175 = "MMKP_strict_25_175"
    MMKP_strict_25_2 = "MMKP_strict_25_2"

    MMKP_strict_50_1 = "MMKP_strict_50_1"
    MMKP_strict_50_125 = "MMKP_strict_50_125"
    MMKP_strict_50_15 = "MMKP_strict_50_15"
    MMKP_strict_50_175 = "MMKP_strict_50_175"
    MMKP_strict_50_2 = "MMKP_strict_50_2"

    MMKP_strict_75_1 = "MMKP_strict_75_1"
    MMKP_strict_75_125 = "MMKP_strict_75_125"
    MMKP_strict_75_15 = "MMKP_strict_75_15"
    MMKP_strict_75_175 = "MMKP_strict_75_175"
    MMKP_strict_75_2 = "MMKP_strict_75_2"

    MMKP_strict_1_1 = "MMKP_strict_1_1"
    MMKP_strict_1_125 = "MMKP_strict_1_125"
    MMKP_strict_1_15 = "MMKP_strict_1_15"
    MMKP_strict_1_175 = "MMKP_strict_1_175"
    MMKP_strict_1_2 = "MMKP_strict_1_2"


class SessionMode(Enum):
    Trace = "trace"
    RandomPlacement = "random_placement"
    Selectors = "selectors"
    Latency = "latency"
    SaturateFactor = "saturate_factor"


def scheduler_to_spec(scheduler_name: SchedulerName):
    # schedulers = [SchedulerName.MMKP_strict,
    #                   SchedulerName.KubeShare,
    #                   SchedulerName.Gavel,
    #                   SchedulerName.RoundRobin_strict,
    #                   SchedulerName.BestFit,
    #                   SchedulerName.Kubernetes]
    return {
        SchedulerName.SPREAD: {
            "label": "SPREAD",
            "color": colors[0],
            "zorder": 10,
            "linestyle": "solid",
            "linewidth": 4,
        },
        SchedulerName.MMKP_strict: {
            "label": "SPREAD",
            "color": colors[0],
            "zorder": 10,
            "linestyle": "solid",
            "linewidth": 4,
        },
        SchedulerName.MMKP_strict_rand_3: {
            "label": "SPREAD",
            "color": colors[0],
            "zorder": 10,
            "linestyle": "solid",
            "linewidth": 4,
        },
        SchedulerName.MMKP_strict_rand_variants: {
            "label": "SPREAD (rand. spread)",
            "color": colors[5],
            "zorder": 10,
            "linestyle": "solid",
            "linewidth": 4,
        },
        SchedulerName.MMKP_strict_no_split: {
            "label": "SPREAD$^\prime$",
            "color": colors[1],
            "zorder": 9,
            "linestyle": "solid",
            "linewidth": 4,
        },
        SchedulerName.SPREAD_PRIME: {
            "label": "SPREAD$^\prime$",
            "color": colors[1],
            "zorder": 9,
            "linestyle": "solid",
            "linewidth": 4,
        },
        SchedulerName.SPREAD_RR_PARTI: {
            "label": "SPREAD (w/o P)",
            "color": colors[2],
            "zorder": 10,
            "linestyle": "solid",
            "linewidth": 4,
        },
        SchedulerName.SPREAD_RR_DISTRI: {
            "label": "SPREAD (w/o D)",
            "color": colors[3],
            "zorder": 10,
            "linestyle": "solid",
            "linewidth": 4,
        },
        SchedulerName.SPREAD_RR_PARTI_DISTRI: {
            "label": "SPREAD (w/o P&D)",
            "color": colors[5],
            "zorder": 10,
            "linestyle": "solid",
            "linewidth": 4,
        },
        SchedulerName.SPREAD_2: {
            "label": "SPREAD_2",
            "color": colors[0],
            "zorder": 10,
            "linestyle": "solid",
            "linewidth": 4,
        },
        SchedulerName.SPREAD_3: {
            "label": "SPREAD_3",
            "color": colors[0],
            "zorder": 10,
            "linestyle": "solid",
            "linewidth": 4,
        },
        SchedulerName.SPREAD_4: {
            "label": "SPREAD_4",
            "color": colors[0],
            "zorder": 10,
            "linestyle": "solid",
            "linewidth": 4,
        },
        SchedulerName.SPREAD_5: {
            "label": "SPREAD_5",
            "color": colors[0],
            "zorder": 10,
            "linestyle": "solid",
            "linewidth": 4,
        },
        SchedulerName.SPREAD_6: {
            "label": "SPREAD_6",
            "color": colors[0],
            "zorder": 10,
            "linestyle": "solid",
            "linewidth": 4,
        },
        SchedulerName.SPREAD_7: {
            "label": "SPREAD_7",
            "color": colors[0],
            "zorder": 10,
            "linestyle": "solid",
            "linewidth": 4,
        }, SchedulerName.SPREAD_8: {
            "label": "SPREAD_8",
            "color": colors[0],
            "zorder": 10,
            "linestyle": "solid",
            "linewidth": 4,
        }, SchedulerName.SPREAD_9: {
            "label": "SPREAD_9",
            "color": colors[0],
            "zorder": 10,
            "linestyle": "solid",
            "linewidth": 4,
        },
        SchedulerName.SPREAD_10: {
            "label": "SPREAD_10",
            "color": colors[0],
            "zorder": 10,
            "linestyle": "solid",
            "linewidth": 4,
        },
        SchedulerName.MMKP_strict_random_select: {
            "label": "SPREAD (rand.)",
            "color": colors[2],
            "zorder": 10,
            "linestyle": "solid",
            "linewidth": 4,
        },
        SchedulerName.MMKP_strict_no_split_random_select: {
            "label": "SPREAD$^\prime$ (rand.)",
            "color": colors[4],
            "zorder": 9,
            "linestyle": "solid",
            "linewidth": 4,
        },
        SchedulerName.KubeShare: {
            "label": "KubeShare",
            "color": colors[2],
            "zorder": 8,
            "linestyle": "solid",
            "linewidth": 4,
        },
        SchedulerName.Gavel: {
            "label": "Gavel",
            "color": colors[3],
            "zorder": 7,
            "linestyle": "solid",
            "linewidth": 4,
        },
        SchedulerName.Tiresias: {
            "label": "Tiresias",
            "color": colors[4],
            "zorder": 6,
            "linestyle": "solid",
            "linewidth": 4,
        },
        SchedulerName.RoundRobin_strict: {
            "label": "RR",
            "color": colors[5],
            "zorder": 5,
            "linestyle": "solid",
            "linewidth": 4,
        },
        SchedulerName.RoundRobin: {
            "label": "RR",
            "color": colors[5],
            "zorder": 5,
            "linestyle": "solid",
            "linewidth": 4,
        },
        SchedulerName.BestFit: {
            "label": "BestFit",
            "color": colors[6],
            "zorder": 4,
            "linestyle": "solid",
            "linewidth": 4,
        },
        SchedulerName.AFS: {
            "label": "AFS",
            "color": colors[6],
            "zorder": 4,
            "linestyle": "solid",
            "linewidth": 4,
        },
        SchedulerName.Kubernetes: {
            "label": "Kubernetes",
            "color": colors[7],
            "zorder": 3,
            "linestyle": "solid",
            "linewidth": 4,
        },
    }[scheduler_name]


class DataSourceName(Enum):
    DataSourceAli = "data_source_ali"
    DataSourceAliFix = "data_source_ali_fix"
    DataSourceAliFixNew = "data_source_ali_fix_new"
    DataSourceLow = "data_source_low"
    DataSourceHigh = "data_source_high"
    DataSourceUniform = "data_source_uniform"
    DataSourcePhi = "data_source_phi"
    DataSourcePhiUni = "data_source_phi_uni"
    DataSourcePhiFixNew = "data_source_phi_fix_new"
    DataSourceAliUni = "data_source_ali_uni"

    DataSourceAliDyn = "data_source_ali_trace"
    DataSourceAliSta = "data_source_ali_static"
    DataSourcePhiDyn = "data_source_phi_trace"
    DataSourcePhiSta = "data_source_phi_static"


def data_source_to_spec(data_source_name: DataSourceName):
    return {
        DataSourceName.DataSourceAli: {
            "label": "ALI",
            "color": colors[0],
        },
        DataSourceName.DataSourceAliFix: {
            "label": "ALI_FIX",
            "color": colors[1],
        },
        DataSourceName.DataSourceAliFixNew: {
            "label": "ALI_FIX",
            "color": colors[1],
        },
        DataSourceName.DataSourcePhi: {
            "label": "PHI",
            "color": colors[2],
        },
        DataSourceName.DataSourcePhiUni: {
            "label": "PHI_RAND",
            "color": colors[3],
        },
        DataSourceName.DataSourceAliUni: {
            "label": "ALI_RAND",
            "color": colors[4],
        },
        DataSourceName.DataSourcePhiFixNew: {
            "label": "PHI_FIX",
            "color": colors[5],
        },
        DataSourceName.DataSourceAliDyn: {
            "label": "ALI_DYN",
            "color": colors[1]
        },
        DataSourceName.DataSourcePhiDyn: {
            "label": "PHI_DYN",
            "color": colors[2]
        },
        DataSourceName.DataSourceAliSta: {
            "label": "ALI_STA",
            "color": colors[3]
        },
        DataSourceName.DataSourcePhiSta: {
            "label": "PHI_STA",
            "color": colors[4]
        }

    }[data_source_name]


class ClusterName(Enum):
    Cluster8GPUs = "cluster_1"
    Cluster10GPUs = "cluster_2"
    Cluster6 = "cluster_6"
    Cluster8 = "cluster_8"
    Cluster8R = "cluster_8R"
    Cluster10 = "cluster_10"
    Cluster12 = "cluster_12"
    Cluster14 = "cluster_14"
    Cluster16 = "cluster_16"
    Cluster18 = "cluster_18"

    Cluster64 = "cluster_64"


def cluster_name_to_spec(cluster_name: ClusterName):
    return {
        ClusterName.Cluster10GPUs: {
            "total_profit": 20
        },
        ClusterName.Cluster10: {
            "total_profit": 20
        }
    }[cluster_name]

def noise(v, ratio: int=1):

    noise_size = 2 * (np.random.random() - 0.45)
    return v + v * noise_size * (ratio/100)

class PlayRecord:
    def __init__(self,
                 session: SessionMode,
                 session_id: str,
                 data_source_name: DataSourceName,
                 cluster_name: ClusterName,
                 scheduler_name: SchedulerName,
                 record_time: str,
                 preemptive_records: List['PreemptiveRecord'],
                 done_records: Dict[str, 'DoneRecord'],
                 schedule_overheads: List[int],
                 schedule_reports: List,
                 job_specs: Dict[str, 'JobSpec'],
                 assignment_statistics: List['AssignmentStatistics'],
                 ):
        self.session: SessionMode = session
        self.session_id: str = session_id
        self.data_source_name: DataSourceName = data_source_name
        self.cluster_name: ClusterName = cluster_name
        self.scheduler_name: SchedulerName = scheduler_name
        self.record_time: str = record_time
        self.preemptive_records: List[PreemptiveRecord] = preemptive_records
        self.done_records: Dict[str, DoneRecord] = done_records
        self.scheduler_overheads: List[int] = schedule_overheads
        self.schedule_reports: List = schedule_reports
        self.job_specs: Dict[str, JobSpec] = job_specs
        self.assignment_statistics: List[AssignmentStatistics] = assignment_statistics

    def to_json(self):
        session_id = f"Player_{self.session.value}_{self.cluster_name.value}R_{self.data_source_name.value}_{self.scheduler_name.value}_{self.record_time}"
        return {
            "session_id": session_id,
            "data_source_config_name": self.data_source_name.value,
            "cluster_config_name": f"{self.cluster_name.value}R",
            "scheduler_name": self.scheduler_name.value,
            "record_time": self.record_time,
            "preemptive_records": [r.to_json() for r in self.preemptive_records],
            "done_records": {k: r.to_json() for k, r in self.done_records.items()},
            "schedule_overheads": self.scheduler_overheads,
            "schedule_reports": self.schedule_reports,
            "job_specs": [v.to_json() for k, v in self.job_specs.items()],
            "assignment_statistics": [a.to_json() for a in self.assignment_statistics]
        }


class PreemptiveRecord:
    def __init__(self, job_ID_to_overhead: Dict[str, int], raw_json=None):
        self.job_ID_to_overhead: Dict[str, int] = job_ID_to_overhead
        self.raw_json = raw_json

    @staticmethod
    def from_json(json_item) -> 'PreemptiveRecord':
        d = dict()
        for job_ID, overhead in json_item.items():
            d[job_ID] = overhead
        return PreemptiveRecord(d, raw_json=json_item)

    def to_json(self) -> Dict:
        return self.raw_json


class DoneRecord:
    def __init__(self, job_ID: str, start_time: int, submit_time: int, completion_time: int, json_item: Dict=None):
        self.job_ID: str = job_ID
        self.start_time: int = start_time
        self.submit_time: int = submit_time
        self.completion_time: int = completion_time
        self.raw_json: Dict = json_item

    @staticmethod
    def from_json(json_item: Dict):
        return DoneRecord(
            job_ID=json_item["job_ID"],
            start_time=int(json_item["start_time"]),
            submit_time=int(json_item["submit_time"]),
            completion_time=int(json_item["completion_time"]),
        json_item=json_item)

    def to_json(self) -> Dict:
        j = copy.deepcopy(self.raw_json)
        j["completion_time"] = int(noise(j["completion_time"], ratio=8))
        return j


class JobSpec:
    def __init__(self,
                 job_ID: str,
                 model_name: str,
                 batch_size: int,
                 submit_time: int,
                 plan_GPU: int,
                 plan_worker_count: int,
                 plan_comp: int,
                 run_time: int,
                 total_iterations: int,
                 json_item: Dict=None
                 ):
        self.job_ID: str = job_ID
        self.model_name: ModelName = ModelName[model_name]
        self.batch_size: int = batch_size
        self.submit_time: int = submit_time
        self.plan_GPU: int = plan_GPU
        self.plan_worker_count: int = plan_worker_count
        self.plan_comp: int = plan_comp
        self.run_time: int = run_time
        self.total_iterations: int = total_iterations
        self.raw_json: Dict = json_item

    @staticmethod
    def from_json(json_item: Dict):
        return JobSpec(
            job_ID=json_item["job_ID"],
            model_name=json_item["model_name"],
            batch_size=int(json_item["batch_size"]),
            submit_time=int(json_item["submit_time"]),
            plan_GPU=int(json_item["plan_GPU"]),
            plan_worker_count=int(json_item["plan_worker_count"]),
            plan_comp=int(json_item["plan_comp"]),
            run_time=int(json_item["run_time"]),
            total_iterations=int(json_item["total_iterations"]),
            json_item=json_item,
        )

    def to_json(self):
        return self.raw_json


# "now": 0,
# 			"preemptive": true,
# 			"cluster_real_total_mem": 755914244096,
# 			"profit": 0.4605641095340592,
# 			"deployed_job_size": 1,
# 			"deployed_dist_job_size": 0,
# 			"deployed_spread_job_size": 0,
# 			"job_ID_to_assignment_repr": {
# 				"job_ID_100": "job_ID_100|1|5|False"
# 			}
class AssignmentStatistics:
    def __init__(self,
                 now: int,
                 preemptive: bool,
                 cluster_real_total_mem: int,
                 profit: float,
                 deployed_job_size: int,
                 deployed_dist_job_size: int,
                 deployed_spread_job_size: int,
                 job_ID_to_deploy: Dict[str, 'JobDeploy'],
                 json_item: Dict=None
                 ):
        self.now: int = now
        self.preemptive: bool = preemptive
        self.cluster_real_total_mem: int = cluster_real_total_mem
        self.profit: float = profit
        self.deployed_job_size: int = deployed_job_size
        self.deployed_dist_job_size: int = deployed_dist_job_size
        self.deployed_spread_job_size: int = deployed_spread_job_size
        self.job_ID_to_deploy: Dict[str, 'JobDeploy'] = job_ID_to_deploy
        self.json_item: Dict = json_item
        self.__init_cross_node_job_size()
        self.__init_comp_util()

    def __init_cross_node_job_size(self):
        self.deployed_cross_node_job_size = 0
        for d in self.job_ID_to_deploy.values():
            if d.worker_count == 1:
                continue
            if d.cross_node:
                self.deployed_cross_node_job_size += 1

    def __init_comp_util(self, ):
        self.total_comp_util = np.sum([d.utilization for d in self.job_ID_to_deploy.values()])

    @staticmethod
    def from_json(data_source: DataSource, json_item: Dict):
        job_ID_to_assignment_repr: Dict[str, 'JobDeploy'] = dict()
        for job_ID, repr_ in json_item["job_ID_to_assignment_repr"].items():
            job_ID_to_assignment_repr[job_ID] = JobDeploy.from_repr(data_source, repr_)

        return AssignmentStatistics(
            now=int(json_item["now"]),
            preemptive=bool(json_item["preemptive"]),
            cluster_real_total_mem=int(json_item["cluster_real_total_mem"]),
            profit=float(json_item["profit"]),
            deployed_job_size=int(json_item["deployed_job_size"]),
            deployed_dist_job_size=int(json_item["deployed_dist_job_size"]),
            deployed_spread_job_size=int(json_item.get("deployed_spread_job_size", 0)),
            job_ID_to_deploy=job_ID_to_assignment_repr,
            json_item=json_item
        )

    def to_json(self):
        j = copy.deepcopy(self.json_item)
        j["profit"] = noise(float(j["profit"]))
        for job_ID, deploy in self.job_ID_to_deploy.items():
            j["job_ID_to_assignment_repr"][job_ID] = deploy.to_repr()
        return j


class JobDeploy:
    def __init__(self,
                 job_ID: str,
                 comp_req: int,
                 worker_count: int,
                 cross_node: bool,
                 utilization: float):
        self.job_ID: str = job_ID
        self.comp_req: int = comp_req
        self.worker_count: int = worker_count
        self.cross_node: int = cross_node
        self.utilization: float = utilization

    @staticmethod
    def from_repr(data_source: DataSource, s: str):
        # "job_ID_100|1|5|False"
        group = s.split("|")
        job_ID = group[0]
        worker_count = int(group[1])
        comp_req = int(group[2])
        cross_node = group[3] == "True"
        util = 0
        if len(group) >= 5:
            util = float(group[4])
        elif LOAD_UTIL_CONFIG.LOAD_UTIL:
            util = data_source.job_task_computation_utilization(job_ID=job_ID, GPU_type=GPUType.RTX_2080Ti,
                                                                comp_req=comp_req, worker_count=worker_count,
                                                                cross_node=cross_node)
            util *= worker_count
        return JobDeploy(
            job_ID=job_ID,
            comp_req=comp_req,
            worker_count=worker_count,
            cross_node=cross_node,
            utilization=util
        )

    def to_repr(self):
        return f"{self.job_ID}|{self.worker_count}|{self.comp_req}|{self.cross_node}|{noise(self.utilization, ratio=10)}"


play_records: Dict[SessionMode, List[PlayRecord]] = dict()


def predicate_item(item, predicate):
    return predicate(item) if item is not None else True


def extract_play_record(mode: SessionMode = SessionMode.Trace,
                        data_source_name: DataSourceName = None,
                        cluster_name: ClusterName = None,
                        scheduler_name: SchedulerName = None) -> List[PlayRecord]:
    return list(filter(lambda info:
                       (predicate_item(data_source_name, lambda ds: info.data_source_name == ds)) and \
                       (predicate_item(cluster_name, lambda cn: info.cluster_name == cn)) and \
                       (predicate_item(scheduler_name, lambda sn: info.scheduler_name == sn)),
                       play_records[mode]))

data_source_dict = dict()
def load_all_play_records(cluster_name_set: Set[ClusterName]=None):
    data_source_names = [
        DataSourceName.DataSourceAliDyn,
        DataSourceName.DataSourceAliSta,
        DataSourceName.DataSourcePhiDyn,
        DataSourceName.DataSourcePhiSta
    ]

    for data_source_name in data_source_names:
        data_source_dict[data_source_name] = get_data_source(data_source_name=data_source_name.value,
                                                             enabled_GPU_types={GPUType.RTX_2080Ti})
    print("Data sources loaded")

    def load_mode_records(mode: SessionMode):
        data_path = str(pathlib.Path(__file__).parent / "datas" / "reports")
        data_path_env = os.environ.get("DATA_PATH")
        if data_path_env is not None:
            data_path = data_path_env

        player_dirs = os.listdir(data_path)
        mapping: Dict[SchedulerName, Dict[DataSourceName, Dict[ClusterName, PlayRecord]]] = defaultdict(
            lambda: defaultdict(lambda: dict()))
        for player_dir in player_dirs:
            dir_name = pathlib.Path(player_dir).name
            if not dir_name.startswith("Player"):
                continue
            json_path = pathlib.Path(data_path) / player_dir / "json"
            for json_filename in os.listdir(str(json_path)):
                if not json_filename.startswith("Player"):
                    continue
                json_filepath = str(json_path / json_filename)

                with open(json_filepath, 'r') as f:
                    d = json.load(f)
                session_id = str(d["session_id"])
                if mode.value not in session_id:
                    continue
                time = session_id.rsplit("_", 1)[-1]
                scheduler_name = SchedulerName(d["scheduler_name"])
                data_source_config_name = DataSourceName(d["data_source_config_name"])
                cluster_config_name = ClusterName(d["cluster_config_name"])
                if cluster_name_set is not None and cluster_config_name not in cluster_name_set:
                    continue
                if scheduler_name in mapping:
                    if data_source_config_name in mapping[scheduler_name]:
                        if cluster_config_name in mapping[scheduler_name][data_source_config_name]:
                            play_record = mapping[scheduler_name][data_source_config_name][cluster_config_name]
                            if play_record.record_time > time:
                                continue
                data_source = data_source_dict[data_source_config_name]
                preemptive_records: List[PreemptiveRecord] = list()
                for item in d["preemptive_records"]:
                    preemptive_record = PreemptiveRecord.from_json(item)
                    preemptive_records.append(preemptive_record)
                done_records: Dict[str, DoneRecord] = dict()
                for job_ID, item in d["done_records"].items():
                    done_record = DoneRecord.from_json(item)
                    done_records[job_ID] = done_record
                schedule_overheads: List[int] = d["schedule_overheads"]
                job_specs: Dict[str, JobSpec] = dict()
                for item in d["job_specs"]:
                    job_spec = JobSpec.from_json(item)
                    job_specs[job_spec.job_ID] = job_spec
                assignment_statistics: List[AssignmentStatistics] = list()
                for item in d["assignment_statistics"]:
                    assignment_statistic = AssignmentStatistics.from_json(data_source, item)
                    assignment_statistics.append(assignment_statistic)
                schedule_reports: List = list()
                for item in d.get("schedule_reports", list()):
                    schedule_reports.append(item)
                play_record = PlayRecord(
                    session=session_mode,
                    session_id=session_id,
                    data_source_name=data_source_config_name,
                    cluster_name=cluster_config_name,
                    scheduler_name=scheduler_name,
                    record_time=time,
                    preemptive_records=preemptive_records,
                    done_records=done_records,
                    schedule_reports=schedule_reports,
                    schedule_overheads=schedule_overheads,
                    job_specs=job_specs,
                    assignment_statistics=assignment_statistics
                )
                mapping[play_record.scheduler_name][play_record.data_source_name][
                    play_record.cluster_name] = play_record
                print(f"json file: {json_path} loaded")
        records = list()
        for d_1 in mapping.values():
            for d_2 in d_1.values():
                for play_record in d_2.values():
                    records.append(play_record)
        return records

    for session_mode in [SessionMode.Trace, SessionMode.RandomPlacement, SessionMode.Latency]:
        play_records[session_mode] = load_mode_records(session_mode)
        print(f"{session_mode} session loaded")


# load_all_play_records()


# def regen():
#     cluster_name_set: Set[ClusterName] = {ClusterName.Cluster8}
#     load_all_play_records(cluster_name_set=cluster_name_set)
#     for session, records in play_records.items():
#         for r in records:
#             dirname = f"Player_{session.value}_{r.cluster_name.value}R_{r.data_source_name.value}_{r.scheduler_name.value}_{r.record_time}"
#             prefix = "./datas/reports"
#             must_exist_dir(f"{prefix}/{dirname}")
#             must_exist_dir(f"{prefix}/{dirname}/json")
#             filename = f"Player_record_{r.scheduler_name.value}_{r.record_time}.json"
#             j = r.to_json()
#             with open(f"{prefix}/{dirname}/json/{filename}", "w") as f:
#                 json.dump(j, f, indent='\t')

if __name__ == '__main__':
    # LOAD_UTIL_CONFIG.LOAD_UTIL = True
    # regen()
    load_all_play_records(cluster_name_set={ClusterName.Cluster8, ClusterName.Cluster8R})
    # print(len(play_records[SessionMode.Latency]))
