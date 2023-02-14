import json
import os
import pathlib
import re
from collections import namedtuple, defaultdict
from enum import Enum
from typing import List, Dict, Optional, Callable, Tuple, Set

import numpy as np
import pandas as pd
from scipy import stats
from log import info
from config import DataSourceConfig, get_config
from object import GPUType, JobSpec, ModelName, to_normalized_memory, CompCapacity

StableWarmupStartRatio = 3

NodeNames = ["dell01", "dell02", "dell03", "dell04", "dell05"]

class TrainOrInference(Enum):
    train = "train"
    inference = "inference"


class MonoJobExecInfo:
    def __init__(self,
                 model_name: ModelName,
                 GPU_type: GPUType,
                 worker_count: int,
                 train_or_inference: TrainOrInference,
                 batch_size: int,
                 computation_proportion: int,
                 time_str: str,
                 cross_node: bool,
                 raw_json: dict
                 ):
        self.model_name: ModelName = model_name
        self.GPU_type: GPUType = GPU_type
        self.worker_count: int = worker_count
        self.train_or_inference: TrainOrInference = train_or_inference
        self.batch_size: int = batch_size
        self.computation_proportion: int = computation_proportion
        self.time_str: str = time_str
        self.raw_json: dict = raw_json
        self.cross_node: bool = cross_node
        self.__parse_raw_json()

    def __parse_raw_json(self):
        self.iteration_count: int = self.raw_json["iteration_count"]
        self.iteration_intervals: List[int] = self.raw_json["iteration_intervals"]
        self.total_time_ns: int = self.raw_json["total_time_ns"]
        self.mem_infos: List[List[int]] = self.raw_json["mem_infos"]
        self.utilization: List[int] = self.raw_json["utilization"]
        memories = [mem_info[-1] - mem_info[0] for mem_info in self.mem_infos]
        self.max_memory_consumption: int = max(memories)
        self.most_memory_consumption: int = stats.mode(memories)[0][0]
        self.stabled_iteration_intervals: List[int] = self.iteration_intervals[
                                                      len(self.iteration_intervals) // StableWarmupStartRatio:]
        mean_iteration_intervals = np.mean(self.stabled_iteration_intervals)
        self.stabled_iteration_intervals = list(filter(lambda iteration_interval: iteration_interval < 50 * mean_iteration_intervals, self.stabled_iteration_intervals))
        self.avg_stabled_iteration_interval: int = int(np.mean(self.stabled_iteration_intervals))
        self.stabled_utilization: List[int] = self.utilization[len(self.utilization) // StableWarmupStartRatio:]
        self.avg_stabled_utilization: float = float(np.mean(self.stabled_utilization))


class MonoJobExecInfoLoader:
    @staticmethod
    def load_infos(data_dir: str) -> Dict[ModelName, List[MonoJobExecInfo]]:
        model_name_strs = [model_name.name for model_name in ModelName]
        files_in_data_dir = os.listdir(data_dir)
        session_dirs = list()
        for filename in files_in_data_dir:
            filepath = os.path.join(data_dir, filename)
            if filename.startswith("mono") and os.path.isdir(filepath):
                session_dirs.append(filepath)
        d = defaultdict(list)
        for session_dir in session_dirs:
            session_id = os.path.basename(session_dir)
            pattern = r"mono_([\w\d]+)_((?:train|inference))_.*"
            groups = re.match(pattern, session_id)
            assert groups is not None
            model_name_str = groups.group(1)
            if model_name_str not in model_name_strs:
                continue
            model_name = ModelName[model_name_str]
            train_or_inference = TrainOrInference(groups.group(2))
            if train_or_inference != TrainOrInference.train:
                continue
            worker_count = 0
            GPU_type = None
            for at in GPUType:
                if at.name in session_id:
                    worker_count = session_id.count(at.name)
                    GPU_type = at
            assert GPU_type is not None
            ExecIdentity = namedtuple(typename="ExecIdentity", field_names=["batch_size", "comp"])
            exec_infos = dict()
            for profiling_filename in os.listdir(session_dir):
                if not profiling_filename.endswith("json"):
                    continue
                pattern = rf"mono_{model_name.name}_{train_or_inference.name}_.*_batch_(\d+)_comp_(\d+)_rank_(\d+)_([\d-]+).json"
                groups = re.match(pattern, profiling_filename)
                assert groups is not None
                node_count = 0
                for node_name in NodeNames:
                    if node_name in profiling_filename:
                        node_count += 1
                cross_node = node_count > 1
                batch_size = int(groups.group(1))
                comp = int(groups.group(2))
                time_str = groups.group(3)
                exec_id = ExecIdentity(batch_size, comp)
                with open(os.path.join(session_dir, profiling_filename), 'r') as f:
                    raw_json = json.load(f)
                exec_info = MonoJobExecInfo(
                    model_name=model_name,
                    GPU_type=GPU_type,
                    worker_count=worker_count,
                    train_or_inference=train_or_inference,
                    batch_size=batch_size,
                    computation_proportion=comp,
                    time_str=time_str,
                    cross_node=cross_node,
                    raw_json=raw_json
                )
                if exec_id in exec_infos:
                    old_info = exec_infos[exec_id]
                    if time_str < old_info.time_str:
                        continue
                exec_infos[exec_id] = exec_info
            print(f"load over for {session_dir}, {exec_infos.values()}")
            d[model_name].extend(exec_infos.values())
        return d

    @staticmethod
    def batch_sizes(infos: List[MonoJobExecInfo]):
        batch_sizes_set = {info.batch_size for info in infos}
        batch_sizes = sorted(list(batch_sizes_set))
        return batch_sizes

    @staticmethod
    def extract(infos: List[MonoJobExecInfo],
                train_or_inference: Optional[TrainOrInference] = None,
                batch_size: Optional[int] = None,
                GPU_type: Optional[GPUType] = None,
                computation_proportion_predicate: Optional[Callable[[int], bool]] = None,
                computation_proportion: Optional[int] = None,
                worker_count: Optional[int] = None,
                cross_node: Optional[bool] = None,
                ):
        def predicate_item(item, predicate):
            return predicate(item) if item is not None else True

        return list(filter(lambda info:
                           (predicate_item(batch_size, lambda bs: info.batch_size == bs)) and \
                           (predicate_item(train_or_inference, lambda ti: info.train_or_inference == ti)) and \
                           (predicate_item(GPU_type, lambda gt: info.GPU_type == gt)) and \
                           (predicate_item(computation_proportion_predicate,
                                           lambda cpp: cpp(info.computation_proportion))) and \
                           (predicate_item(computation_proportion, lambda cp: info.computation_proportion == cp)) and \
                           (predicate_item(worker_count, lambda wc: info.worker_count == wc)) and \
                           (predicate_item(cross_node, lambda cn: info.cross_node == cross_node)),
                           infos))

    @staticmethod
    def extract_batch_size_with(infos: List[MonoJobExecInfo], batch_size: int) -> List[MonoJobExecInfo]:
        return list(filter(lambda info: info.batch_size == batch_size, infos))

    @staticmethod
    def sort_by_computation(infos: List[MonoJobExecInfo]):
        return sorted(infos, key=lambda info: info.computation_proportion)


class DataSource:

    mono_job_datas = dict()

    def __init__(self,
                 data_source_config: DataSourceConfig,
                 enabled_GPU_types=None,
                 ):
        if enabled_GPU_types is None:
            enabled_GPU_types = set(GPUType)
        self.data_source_config: DataSourceConfig = data_source_config
        self.enabled_GPU_types: List[GPUType] = list(enabled_GPU_types)
        self.__init_mono_job_data()
        self.__init_job_data()

    def __init_mono_job_data(self):
        if self.data_source_config.mono_job_data_path in self.mono_job_datas:
            self.mono_job_data = self.mono_job_datas[self.data_source_config.mono_job_data_path]
            return
        p = str(pathlib.Path(__file__).parent / self.data_source_config.mono_job_data_path)
        mono_job_data = MonoJobExecInfoLoader.load_infos(p)
        self.mono_job_datas[self.data_source_config.mono_job_data_path] = mono_job_data
        self.mono_job_data: Dict[ModelName, List[MonoJobExecInfo]] = self.mono_job_datas[self.data_source_config.mono_job_data_path]

    def __init_job_data(self):
        job_data_path = str(pathlib.Path(__file__).parent / self.data_source_config.submit_table_path)
        df = pd.read_csv(job_data_path)
        df = df[self.data_source_config.data_range[0]: self.data_source_config.data_range[1]]
        df["submit_time"] -= df.iloc[0]['submit_time'].item()
        self.job_specs: List[JobSpec] = list()
        self.job_specs_dict: Dict[str, JobSpec] = dict()
        np.random.seed(self.data_source_config.init_job_data_seed)
        c = get_config()
        model_names = list(c.model_configs.keys())
        comp_distribution = self.data_source_config.comp_distribution
        for _, row in df.iterrows():
            if len(self.job_specs) >= self.data_source_config.job_count:
                break
            job_ID = f"job_ID_{row['jobID']}"
            submit_time = row["submit_time"] if not self.data_source_config.submit_at_beginning else 0
            run_time = row["run_time"]
            run_time = DataSource.run_time_converter(run_time=run_time)
            plan_GPU = row["plan_gpu"]
            if plan_GPU > 100:
                plan_GPU = 100
            worker_count = 1
            plan_GPU = int(DataSource.plan_gpu_converter(comp_distribution=comp_distribution, plan_GPU=plan_GPU))
            if plan_GPU > 100:
                worker_count = 2
                computation_proportion = 100
            else:
                computation_proportion = plan_GPU
            comp_req = computation_proportion // (100 // CompCapacity)
            GPU_type = np.random.choice(self.enabled_GPU_types)
            model_name = np.random.choice(model_names)
            config = get_config()
            batch_sizes = config.model_configs[model_name].batch_sizes
            threshold = 16
            lower_threshold = 8
            batch_sizes_fixed = list()
            for batch_size in batch_sizes:
                if batch_size > threshold:
                    batch_sizes_fixed += [batch_size for _ in range(6)]
                elif batch_size > lower_threshold:
                    batch_sizes_fixed += [batch_size for _ in range(2)]
            batch_size = np.random.choice(batch_sizes_fixed)
            iteration_throughput = self.iteration_time(model_name,
                                                       batch_size,
                                                       GPU_type,
                                                       worker_count,
                                                       False,
                                                       comp_req)
            total_iterations = int(1e9 * run_time) // iteration_throughput
            job_spec = JobSpec(job_ID=job_ID,
                               model_name=model_name,
                               batch_size=batch_size,
                               submit_time=submit_time,
                               run_time=run_time,
                               plan_GPU=plan_GPU,
                               total_iterations=total_iterations,
                               worker_count=worker_count
                               )
            self.job_specs.append(job_spec)
            self.job_specs_dict[job_ID] = job_spec
        plan_GPUs_size = defaultdict(int)
        for job_spec in self.job_specs:
            plan_GPUs_size[job_spec.plan_GPU] += 1
        info(plan_GPUs_size.__str__())

    def get_job_spec(self, job_ID: str) -> JobSpec:
        return self.job_specs_dict[job_ID]

    def get_exec_info(self, model_name: ModelName, batch_size: int, GPU_type: GPUType, worker_count: int,
                      computation_proportion: int):
        batch_size = batch_size // worker_count
        info = MonoJobExecInfoLoader.extract(self.mono_job_data[model_name],
                                             GPU_type=GPU_type, batch_size=batch_size,
                                             computation_proportion=computation_proportion, worker_count=worker_count)
        assert len(info) == 1
        return info[0]

    def iteration_time(self, model_name: ModelName,
                       batch_size: int,
                       GPU_type: GPUType,
                       worker_count: int,
                       cross_node: bool,
                       comp_req: float):
        batch_size = batch_size // worker_count
        computation_proportion = int(comp_req * (100 // CompCapacity))
        info = MonoJobExecInfoLoader.extract(self.mono_job_data[model_name],
                                             GPU_type=GPU_type,
                                             batch_size=batch_size,
                                             computation_proportion=computation_proportion,
                                             cross_node=cross_node,
                                             worker_count=worker_count)
        if len(info) == 1:
            return info[0].avg_stabled_iteration_interval
        infos = MonoJobExecInfoLoader.extract(self.mono_job_data[model_name],
                                              batch_size=batch_size,
                                              GPU_type=GPU_type,
                                              cross_node=cross_node,
                                              worker_count=worker_count)
        if len(infos) == 0:
            return None
        factor = 1.0
        if len(infos) == 0:
            info_base = MonoJobExecInfoLoader.extract(self.mono_job_data[model_name], batch_size=batch_size,
                                                      GPU_type=GPUType.RTX_2080Ti, worker_count=1)
            info_self = MonoJobExecInfoLoader.extract(self.mono_job_data[model_name], batch_size=batch_size,
                                                      GPU_type=GPU_type, worker_count=1)
            assert len(info_base) == 1 and len(info_self) == 1
            infos = MonoJobExecInfoLoader.extract(self.mono_job_data[model_name], batch_size=batch_size, GPU_type=GPU_type,
                                                  worker_count=worker_count)
            factor = info_self[0].avg_stabled_iteration_interval / info_base[0].avg_stabled_iteration_interval
        infos = MonoJobExecInfoLoader.sort_by_computation(infos)
        greater_idx = 0
        for i, info in enumerate(infos):
            if info.computation_proportion > computation_proportion:
                greater_idx = i
                break
        less_idx = greater_idx - 1 if greater_idx > 0 else 0
        if less_idx == greater_idx:
            comp_k = (infos[-1].avg_stabled_iteration_interval - infos[-2].avg_stabled_iteration_interval) / (infos[-1].computation_proportion - infos[-2].computation_proportion)
            comp_b = infos[-1].avg_stabled_iteration_interval - comp_k * infos[-1].computation_proportion
            return comp_k * computation_proportion + comp_b
        less_comp = infos[less_idx].computation_proportion
        greater_comp = infos[greater_idx].computation_proportion
        iteration_interval_diff = infos[greater_idx].avg_stabled_iteration_interval - infos[
            less_idx].avg_stabled_iteration_interval
        comp_diff = greater_comp - less_comp
        k = iteration_interval_diff / comp_diff
        iteration_interval = infos[
            less_idx].avg_stabled_iteration_interval + k * (computation_proportion - less_comp)
        return iteration_interval * factor

    def computation_utilization(self, model_name: ModelName,
                       batch_size: int,
                       GPU_type: GPUType, worker_count: int,
                       comp_req: float) -> float:
        batch_size = batch_size // worker_count
        computation_proportion = int(comp_req * (100 // CompCapacity))
        info = MonoJobExecInfoLoader.extract(self.mono_job_data[model_name], GPU_type=GPU_type, batch_size=batch_size,
                                             computation_proportion=computation_proportion, worker_count=worker_count)
        if len(info) == 1:
            return info[0].avg_stabled_utilization
        infos = MonoJobExecInfoLoader.extract(self.mono_job_data[model_name], batch_size=batch_size, GPU_type=GPU_type,
                                              worker_count=worker_count)
        infos = MonoJobExecInfoLoader.sort_by_computation(infos)
        greater_idx = 0
        for i, info in enumerate(infos):
            if info.computation_proportion > computation_proportion:
                greater_idx = i
                break
        less_idx = greater_idx - 1 if greater_idx > 0 else 0
        if less_idx == greater_idx:
            comp = infos[less_idx].computation_proportion
            return infos[less_idx].avg_stabled_utilization / (computation_proportion / comp)
        less_comp = infos[less_idx].computation_proportion
        greater_comp = infos[greater_idx].computation_proportion
        utilization_diff = infos[greater_idx].avg_stabled_utilization - infos[
            less_idx].avg_stabled_utilization
        comp_diff = greater_comp - less_comp
        k = utilization_diff / comp_diff
        stabled_utilization = infos[
            less_idx].avg_stabled_utilization + k * (computation_proportion - less_comp)
        return stabled_utilization

    def get_job_task_memory(self,
                            job_ID: str,
                            worker_count: int) -> Tuple[int, int]:
        job_spec = self.job_specs_dict[job_ID]
        info = self.get_exec_info(model_name=job_spec.model_name,
                                  batch_size=job_spec.batch_size,
                                  GPU_type=GPUType.RTX_2080Ti,
                                  worker_count=worker_count,
                                  computation_proportion=50)
        normalized_memory = to_normalized_memory(info.most_memory_consumption)
        return info.most_memory_consumption, normalized_memory

    def remain_duration(self,
                        job_ID: str,
                        GPU_type: GPUType,
                        comp_req: float,
                        worker_count: int,
                        cross_node: bool,
                        remaining_iterations: float) -> int:
        iteration_time = self.job_iteration_time(job_ID, GPU_type, comp_req, worker_count, cross_node)
        remain_duration = int(remaining_iterations * iteration_time)
        return remain_duration

    def job_iteration_time(self,
                           job_ID: str,
                           GPU_type: GPUType,
                           comp_req: float,
                           worker_count: int,
                           cross_node: bool) -> float:
        job_spec = self.job_specs_dict[job_ID]
        iteration_time = self.iteration_time(
            model_name=job_spec.model_name,
            batch_size=job_spec.batch_size,
            GPU_type=GPU_type,
            worker_count=worker_count,
            cross_node=cross_node,
            comp_req=comp_req
        )
        return iteration_time

    def job_task_computation_utilization(self, job_ID: str, GPU_type: GPUType, comp_req: float, worker_count: int) -> float:
        job_spec = self.job_specs_dict[job_ID]
        utilization = self.computation_utilization(
            model_name=job_spec.model_name,
            batch_size=job_spec.batch_size,
            GPU_type=GPU_type,
            worker_count=worker_count,
            comp_req=comp_req
        )
        return utilization

    @staticmethod
    def plan_gpu_converter_ali_fix(plan_GPU: int):
        convert_dict: Dict[int, List] = {
            100: [65, 70, 75, 80, 85, 90, 95, 100, 100, 100, 100],
            50: [40, 45, 50, 55, 60],
            25: [15, 20, 25, 30, 35],
            20: [10, 15, 20, 25, 30],
            10: [5, 10, 15, 20],
            5: [5, 10, 15]
        }
        if plan_GPU in convert_dict:
            c = np.random.choice(convert_dict[plan_GPU])
            return c
        if plan_GPU % 5 != 0:
            return np.random.choice(np.arange(5, 105, 5))
        return plan_GPU

    @staticmethod
    def plan_gpu_converter_ali_fix_new(plan_GPU: int):
        convert_normal_distributions = [
            (0.36, [55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 100, 100, 95, 90, 85, 80, 75, 70, 65, 60, 55]),
            (0.36 + 0.26, [5, 10, 15, 20, 25, 25, 25, 30, 35, 40, 45]),
            (0.36 + 0.26 + 0.16, [30, 35, 40, 45, 50, 50, 50, 55, 60, 65, 70]),
            (0.36 + 0.26 + 0.16 + 0.15, [10, 10, 10, 5, 10, 10, 10, 15, 20, 25, 30]),
            (0.36 + 0.26 + 0.16 + 0.15 + 0.02, [5, 10, 15, 20, 20, 20, 25, 30, 35]),
            (1, [200])
        ]
        r = np.random.rand()
        dist_idx = None
        for idx in range(len(convert_normal_distributions)):
            if idx == 0:
                continue
            if idx == 1 and r < convert_normal_distributions[0][0]:
                dist_idx = 0
                break
            if convert_normal_distributions[idx - 1][0] < r <= convert_normal_distributions[idx][0]:
                dist_idx = idx
                break
        assert dist_idx is not None
        return DataSource.random_normal_idx(convert_normal_distributions[dist_idx][-1])


    @staticmethod
    def plan_gpu_converter_phi(plan_GPU: int):
        convert_normal_distributions = [
            (0.36, [100]),
            (0.36 + 0.26, [25, 25, 25]),
            (0.36 + 0.26 + 0.16, [50]),
            (0.36 + 0.26 + 0.16 + 0.15, [10]),
            (0.36 + 0.26 + 0.16 + 0.15 + 0.02, [20]),
            (1, [200])
        ]
        r = np.random.rand()
        dist_idx = None
        for idx in range(len(convert_normal_distributions)):
            if idx == 0:
                continue
            if idx == 1 and r < convert_normal_distributions[0][0]:
                dist_idx = 0
                break
            if convert_normal_distributions[idx - 1][0] < r <= convert_normal_distributions[idx][0]:
                dist_idx = idx
                break
        assert dist_idx is not None
        return DataSource.random_normal_idx(convert_normal_distributions[dist_idx][-1])

    @staticmethod
    def plan_gpu_converter_ali_original(plan_GPU: int):
        return plan_GPU

    @staticmethod
    def plan_gpu_converter_uniform(plan_GPU: int):
        distribution = list(reversed(np.arange(5, 105, 5)))
        distribution += [200]
        return np.random.choice(distribution)

    @staticmethod
    def random_normal_idx(distribution: List, std: Optional[float]=None):
        c = len(distribution)
        indices = np.arange(0, c)
        mean = np.mean(indices)
        if std is None:
            std = 2
        idx = np.random.normal(mean, std)
        idx = int(np.around(idx))
        if idx < 0:
            idx = 0
        if idx >= c:
            idx = c-1
        return distribution[idx]

    @staticmethod
    def plan_gpu_converter_low(plan_GPU: int):
        distribution = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
        return DataSource.random_normal_idx(distribution, std=3)

    @staticmethod
    def plan_gpu_converter_high(plan_GPU: int):
        distribution = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 200]
        return DataSource.random_normal_idx(distribution, std=3)

    @staticmethod
    def plan_gpu_converter_comp_all_100(plan_GPU: int):
        return 100

    @staticmethod
    def run_time_converter(run_time: int):
        while run_time < 30 * 60:
            run_time *= 2
        return run_time

    @staticmethod
    def plan_gpu_converter(comp_distribution: str, plan_GPU: int) -> int:
        return {
            "all_100": DataSource.plan_gpu_converter_comp_all_100,
            "ali": DataSource.plan_gpu_converter_ali_original,
            "ali_fix": DataSource.plan_gpu_converter_ali_fix,
            "ali_uni": DataSource.plan_gpu_converter_ali_fix,
            "uniform": DataSource.plan_gpu_converter_uniform,
            "low": DataSource.plan_gpu_converter_low,
            "high": DataSource.plan_gpu_converter_high,
            "ali_fix_new": DataSource.plan_gpu_converter_ali_fix_new,
            "phi_fix_new": DataSource.plan_gpu_converter_ali_fix_new,
            "phi": DataSource.plan_gpu_converter_phi
        }[comp_distribution](plan_GPU)


def do_test():
    c = get_config("./configs/MMKP_config.json")
    d = DataSource(data_source_config=c.data_source_configs["data_source_ali_fix_new"], enabled_GPU_types={GPUType.RTX_2080Ti})
    print(d.job_specs[0].total_iterations)
    e = DataSource(data_source_config=c.data_source_configs["data_source_phi_uni"], enabled_GPU_types={GPUType.RTX_2080Ti})
    print(e.job_specs[0].total_iterations)
    infos = MonoJobExecInfoLoader.extract(e.mono_job_data[ModelName.MEALV2], train_or_inference=TrainOrInference.train,
                                          batch_size=128, worker_count=4, cross_node=True)
    print(infos)


if __name__ == '__main__':
    do_test()
