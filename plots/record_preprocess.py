import os
from typing import Dict, List
from mono_job_data_preprocess import *
from enum import Enum
import json
import pathlib
from collections import defaultdict
from common import colors


class SchedulerName(Enum):
    MMKP_strict = "MMKP_strict"
    MMKP_strict_rand_3 = "MMKP_strict_rand_3"
    MMKP_strict_rand_variants = "MMKP_strict_rand_variants"
    MMKP_strict_no_split = "MMKP_strict_no_split"
    MMKP_strict_random_select = "MMKP_strict_random_select"
    MMKP_strict_no_split_random_select = "MMKP_strict_no_split_random_select"
    MMKP_best_effort = "MMKP_best_effort"
    RoundRobin_strict = "RoundRobin_strict"
    RoundRobin_best_effort = "RoundRobin_best_effort"
    KubeShare = "KubeShare"
    BestFit = "BestFit"
    Tiresias = "Tiresias"
    Gavel = "Gavel"
    Kubernetes = "Kubernetes"

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
        SchedulerName.BestFit: {
            "label": "BestFit",
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
    }[data_source_name]


class ClusterName(Enum):
    Cluster8GPUs = "cluster_1"
    Cluster10GPUs = "cluster_2"
    Cluster6 = "cluster_6"
    Cluster8 = "cluster_8"
    Cluster10 = "cluster_10"
    Cluster12 = "cluster_12"
    Cluster14 = "cluster_14"
    Cluster16 = "cluster_16"
    Cluster18 = "cluster_18"


def cluster_name_to_spec(cluster_name: ClusterName):
    return {
        ClusterName.Cluster10GPUs: {
            "total_profit": 20
        },
        ClusterName.Cluster10: {
            "total_profit": 20
        }
    }[cluster_name]

class PlayRecord:
    def __init__(self,
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


class PreemptiveRecord:
    def __init__(self, job_ID_to_overhead: Dict[str, int]):
        self.job_ID_to_overhead: Dict[str, int] = job_ID_to_overhead

    @staticmethod
    def from_json(json_item) -> 'PreemptiveRecord':
        d = dict()
        for job_ID, overhead in json_item.items():
            d[job_ID] = overhead
        return PreemptiveRecord(d)


class DoneRecord:
    def __init__(self, job_ID: str, start_time: int, completion_time: int):
        self.job_ID: str = job_ID
        self.start_time: int = start_time
        self.completion_time: int = completion_time

    @staticmethod
    def from_json(json_item: Dict):
        return DoneRecord(
            job_ID=json_item["job_ID"],
            start_time=int(json_item["start_time"]),
            completion_time=int(json_item["completion_time"]))


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
                 total_iterations: int
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
            total_iterations=int(json_item["total_iterations"])
        )


class AssignmentStatistics:
    def __init__(self,
                 now: int,
                 preemptive: bool,
                 job_over_supply: Dict[str, int],
                 total_over_supply: int,
                 job_lack_supply: Dict[str, int],
                 total_lack_supply: int,
                 job_comp_util: Dict[str, float],
                 total_comp_util: float,
                 job_real_mem: Dict[str, int],
                 total_real_mem: int,
                 cluster_real_total_mem: int,
                 total_mem_utilization: float,
                 profit: float,
                 deployed_job_size: int,
                 deployed_dist_job_size: int,
                 deployed_spread_job_size: int,
                 job_ID_to_task_assignments: Dict[str, 'TaskAssignment']
                 ):
        self.now: int = now
        self.preemptive: bool = preemptive
        self.job_over_supply: Dict[str, int] = job_over_supply
        self.total_over_supply: int = total_over_supply
        self.job_lack_supply: Dict[str, int] = job_lack_supply
        self.total_lack_supply: int = total_lack_supply
        self.job_comp_util: Dict[str, float] = job_comp_util
        self.total_comp_util: float = total_comp_util
        self.job_real_mem: Dict[str, int] = job_real_mem
        self.total_real_mem: int = total_real_mem
        self.cluster_real_total_mem: int = cluster_real_total_mem
        self.total_mem_utilization: float = total_mem_utilization
        self.profit: float = profit
        self.deployed_job_size: int = deployed_job_size
        self.deployed_dist_job_size: int = deployed_dist_job_size
        self.deployed_spread_job_size: int = deployed_spread_job_size
        self.job_ID_to_task_assignments: Dict[str, 'TaskAssignment'] = job_ID_to_task_assignments

    @staticmethod
    def from_json(json_item: Dict):
        job_ID_to_task_assignments: Dict[str, 'TaskAssignment'] = dict()
        for job_ID, task_assignments in json_item["job_ID_to_task_assignments"].items():
            for task_assignment in task_assignments:
                task_assignment = TaskAssignment.from_json(task_assignment)
                job_ID_to_task_assignments[job_ID] = task_assignment
        return AssignmentStatistics(
            now=int(json_item["now"]),
            preemptive=bool(json_item["preemptive"]),
            job_over_supply=json_item["job_over_supply"],
            total_over_supply=int(json_item["total_over_supply"]),
            total_comp_util=int(json_item["total_comp_util"]),
            job_lack_supply=json_item["job_lack_supply"],
            total_lack_supply=int(json_item["total_lack_supply"]),
            job_comp_util={job_ID: float(comp_util) for job_ID, comp_util in json_item["job_comp_util"].items()},
            job_real_mem={job_ID: int(real_mem) for job_ID, real_mem in json_item["job_real_mem"].items()},
            total_real_mem=int(json_item["total_real_mem"]),
            cluster_real_total_mem=int(json_item["cluster_real_total_mem"]),
            total_mem_utilization=float(json_item["total_mem_utilization"]),
            profit=float(json_item["profit"]),
            deployed_job_size=int(json_item["deployed_job_size"]),
            deployed_dist_job_size=int(json_item["deployed_dist_job_size"]),
            deployed_spread_job_size=int(json_item.get("deployed_spread_job_size", 0)),
            job_ID_to_task_assignments=job_ID_to_task_assignments,
        )

class TaskAssignment:
    def __init__(self,
                 over_supplied: int,
                 comp_req: int,
                 memory: int,
                 task_ID: str,
                 job_ID: str,
                 task_idx: int):
        self.over_supplied: int = over_supplied
        self.comp_req: int = comp_req
        self.memory: int = memory
        self.task_ID: str = task_ID
        self.job_ID: str = job_ID
        self.task_idx: int = task_idx

    @staticmethod
    def from_json(json_item: Dict):
        return TaskAssignment(
            over_supplied=int(json_item["over_supplied"]),
            comp_req=int(json_item["comp_req"]),
            memory=int(json_item["memory"]),
            task_ID=json_item["task_ID"],
            job_ID=json_item["job_ID"],
            task_idx=int(json_item["task_idx"])
        )


play_records: Dict[SessionMode, List[PlayRecord]] = dict()


def predicate_item(item, predicate):
    return predicate(item) if item is not None else True


def extract_play_record(mode: SessionMode=SessionMode.Trace,
                        data_source_name: DataSourceName=None,
                        cluster_name: ClusterName=None,
                        scheduler_name: SchedulerName=None) -> List[PlayRecord]:
    return list(filter(lambda info:
                       (predicate_item(data_source_name, lambda ds: info.data_source_name == ds)) and \
                       (predicate_item(cluster_name, lambda cn: info.cluster_name == cn)) and \
                       (predicate_item(scheduler_name, lambda sn: info.scheduler_name == sn)),
                       play_records[mode]))


def load_all_play_records():
    def load_mode_records(mode: SessionMode):
        path = str(pathlib.Path(__file__).parent / "datas" / "reports")
        player_dirs = os.listdir(path)
        mapping: Dict[SchedulerName, Dict[DataSourceName, Dict[ClusterName, PlayRecord]]] = defaultdict(lambda : defaultdict(lambda :dict()))
        for player_dir in player_dirs:
            dir_name = pathlib.Path(player_dir).name
            if not dir_name.startswith("Player"):
                continue
            json_path = pathlib.Path(path) / player_dir / "json"
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
                if scheduler_name in mapping:
                    if data_source_config_name in mapping[scheduler_name]:
                        if cluster_config_name in mapping[scheduler_name][data_source_config_name]:
                            play_record = mapping[scheduler_name][data_source_config_name][cluster_config_name]
                            if play_record.record_time > time:
                                continue

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
                    assignment_statistic = AssignmentStatistics.from_json(item)
                    assignment_statistics.append(assignment_statistic)
                schedule_reports: List = list()
                for item in d.get("schedule_reports", list()):
                    schedule_reports.append(item)
                play_record = PlayRecord(
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
                mapping[play_record.scheduler_name][play_record.data_source_name][play_record.cluster_name] = play_record
        records = list()
        for d_1 in mapping.values():
            for d_2 in d_1.values():
                for play_record in d_2.values():
                    records.append(play_record)
        return records
    play_records[SessionMode.Trace] = load_mode_records(SessionMode.Trace)
    play_records[SessionMode.RandomPlacement] = load_mode_records(SessionMode.RandomPlacement)
    play_records[SessionMode.Selectors] = load_mode_records(SessionMode.Selectors)
    play_records[SessionMode.Latency] = load_mode_records(SessionMode.Latency)
    play_records[SessionMode.SaturateFactor] = load_mode_records(SessionMode.SaturateFactor)


load_all_play_records()

if __name__ == '__main__':
    print(len(play_records[SessionMode.SaturateFactor]))
