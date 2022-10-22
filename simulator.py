import datetime
import json
import os.path
from time import time_ns
from typing import Dict, Tuple, Set, Optional, List, Any

import numpy as np

from cluster import Cluster, Assignments
from common import get_json_dir
from config import get_config, ClusterConfig, DataSourceConfig
from data_source import DataSource
from log import info
from object import Job, SchedulerEnum, ProfitEnum, SolverEnum, SimulatingMethod
from plot_assignment import do_snapshot_record_plot
from scheduler import Scheduler
from schedulers import init_scheduler


class Simulator:
    def __init__(self,
                 data_source_config: DataSourceConfig,
                 cluster_config: ClusterConfig):
        c = get_config()
        self.simulating_method: SimulatingMethod = c.simulating_method
        self.simulating_method_config: Dict = c.simulating_method_config
        self.data_source_config: DataSourceConfig = data_source_config
        self.data_source: DataSource = DataSource(data_source_config=data_source_config,
                                                  enabled_GPU_types=cluster_config.GPU_types)
        self.cluster_config: ClusterConfig = cluster_config
        enabled_schedulers = set(self.data_source_config.enabled_scheduler_names).intersection(
            set(self.cluster_config.enabled_scheduler_names))
        self.schedulers: List[Scheduler] = list()
        for scheduler_name in c.enabled_scheduler_names:
            scheduler_desc = c.schedulers[scheduler_name]
            scheduler_enum = scheduler_desc.scheduler_enum
            scheduler_config = scheduler_desc.config
            if scheduler_desc.name not in enabled_schedulers:
                continue
            scheduler = init_scheduler(
                name=scheduler_desc.name,
                scheduler_enum=scheduler_enum,
                solver_enum=SolverEnum[scheduler_config["solver_enum"]] if "solver_enum" in scheduler_config else None,
                profit_enum=ProfitEnum[scheduler_config.get("profit_enum", ProfitEnum.ComprehensiveUtilization.name)],
                data_source=self.data_source,
                cluster=Cluster(cluster_config=cluster_config),
                config=scheduler_config)
            self.schedulers.append(scheduler)
        info(f"Simulator Initialized, enabling schedulers: {[scheduler.name for scheduler in self.schedulers]}")
        self.now: int = 0
        self.next_job_idx: int = 0
        self.last_preemptive_time: int = 0

    def __init_play_status(self):
        self.now = 0
        self.next_job_idx = 0
        self.last_preemptive_time = 0

    def play_trace(self):
        for scheduler in self.schedulers:
            info(f"Simulator playing trace for scheduler: {scheduler.name}, data_source_name: {self.data_source_config.name}, cluster_config_name: {self.cluster_config.name}")
            self.__init_play_status()
            self.play_trace_for_scheduler(scheduler)

    def play_random_placement(self):
        for scheduler in self.schedulers:
            info(f"Simulator playing random placement for scheduler: {scheduler.name}, data_source_name: {self.data_source_config.name}, cluster_config_name: {self.cluster_config.name}")
            self.__init_play_status()
            self.play_random_placement_for_scheduler(scheduler)

    def play_trace_for_scheduler(self, scheduler: Scheduler):
        c = get_config()
        last_assignments = scheduler.cluster.assignments
        time_str = datetime.datetime.now().strftime(
            f"%Y-%m-%d-%H-%M-%S")
        session_id = f"Player_{c.session_id}_{self.cluster_config.name}_{self.data_source_config.name}_{scheduler.name}_{time_str}"
        record = PlayRecord(session_id=session_id, scheduler_name=scheduler.name,
                            scheduler_enum=scheduler.scheduler_enum, data_source_config=self.data_source_config,
                            data_source=self.data_source, cluster_config=self.cluster_config)

        iteration = 0
        done_jobs_between_preemption = set()
        simulation_rand_seed = scheduler.config.get("simulation_rand_seed", None)
        if simulation_rand_seed is not None:
            info(f"simulator: using random seed: {simulation_rand_seed}")
            np.random.seed(simulation_rand_seed)
        while True:
            info(f"Simulator: starts iteration: {iteration}")
            iteration += 1
            now_assignments = scheduler.cluster.assignments
            self.add_preemptive_overheads(cluster=scheduler.cluster, record=record, last_assignments=last_assignments,
                                          curr_assignments=now_assignments)
            next_events = self.next_events(scheduler)
            if next_events is None:
                break
            duration, submit_job_IDs, done_job_IDs, is_preemptive_interval = next_events
            done_jobs: Set[Job] = set()
            for job_ID in done_job_IDs:
                scheduler.cluster.done(job_ID=job_ID, now=self.now + duration)
                done_jobs.add(scheduler.cluster.get_job(job_ID))
            done_jobs_between_preemption.update(done_jobs)
            record.add_done_jobs(done_jobs=done_jobs)
            scheduler.cluster.assignments = scheduler.cluster.assignments.remove_jobs(
                job_IDs={job.job_ID for job in done_jobs})
            self.pass_duration(cluster=scheduler.cluster, duration=duration)
            if self.now == 0:
                is_preemptive_interval = True
            if is_preemptive_interval:
                self.last_preemptive_time = self.now
            submit_jobs: Set[Job] = set()
            for job_ID in submit_job_IDs:
                job_spec = self.data_source.job_specs_dict[job_ID]
                job = Job(job_ID=job_ID, remaining_iterations=job_spec.total_iterations)
                scheduler.cluster.submit(job)
                submit_jobs.add(job)
            last_assignments = now_assignments
            before = time_ns()
            assignments, scheduler_reports = scheduler.do_assign(preemptive=is_preemptive_interval, now=self.now, done_jobs_between_preemption=done_jobs_between_preemption)
            end = time_ns()
            if is_preemptive_interval:
                done_jobs_between_preemption.clear()
            scheduler.cluster.assignments = assignments
            scheduler.cluster.ensure_start(self.now)

            info(f"Simulator scheduler {scheduler.name} do assign done with {(end - before) / 1e9} seconds.")
            snapshot_record_parameters = scheduler.build_snapshot_record_parameters()
            enable_plot = self.cluster_config.enable_plot and self.data_source_config.enable_plot
            snapshot_record_parameters.do_plot = enable_plot

            do_snapshot_record_plot(session_id=session_id, is_preemptive_interval=is_preemptive_interval, snapshot_record_parameters=snapshot_record_parameters)
            running_status = scheduler.cluster.assignments.running_status(data_source=self.data_source)

            record.add_schedule_overhead(end - before)
            record.add_schedule_reports(scheduler_reports)
            record.add_assignments_statistics(
                scheduler.cluster.assignments.statistics(preemptive=is_preemptive_interval, now=self.now, cluster=scheduler.cluster, data_source=self.data_source))

            info(f"Simulator: running status: {running_status}, assignment profit: {snapshot_record_parameters.profit}")
            info(f"Simulator: done jobs size: {len(scheduler.cluster.done_jobs)}, undone jobs size: {len(scheduler.cluster.jobs)}")
            job_remaining_duration_seconds = {job_ID: remaining_duration / 1e9 for job_ID, remaining_duration in
                                              self.job_remaining_durations(scheduler.cluster).items()}
            info(f"Simulator: running job remaining durations (second): {job_remaining_duration_seconds}")
        record.save()

    def play(self):
        if self.simulating_method == SimulatingMethod.Trace:
            self.play_trace()
        elif self.simulating_method == SimulatingMethod.RandomPlacement:
            self.play_random_placement()

    def play_random_placement_for_scheduler(self, scheduler: Scheduler):
        c = get_config()
        time_str = datetime.datetime.now().strftime(
            f"%Y-%m-%d-%H-%M-%S")
        session_id = f"Player_{c.session_id}_{self.cluster_config.name}_{self.data_source_config.name}_{scheduler.name}_{time_str}"
        record = PlayRecord(session_id=session_id, scheduler_name=scheduler.name,
                            scheduler_enum=scheduler.scheduler_enum, data_source_config=self.data_source_config,
                            data_source=self.data_source, cluster_config=self.cluster_config)
        np.random.seed(1)
        simulation_rand_seed = scheduler.config.get("simulation_rand_seed", None)
        if simulation_rand_seed is not None:
            info(f"simulator: using random seed: {simulation_rand_seed}")
            np.random.seed(simulation_rand_seed)
        sample_job_size = self.simulating_method_config.get("job_size", 30)
        for i in range(self.simulating_method_config.get("repeat", 100)):
            scheduler.cluster.assignments = Assignments()
            scheduler.cluster.jobs.clear()
            info(f"Simulator play random placement: starts iteration: {i}")
            job_specs = list(self.data_source.job_specs)
            np.random.shuffle(job_specs)
            job_specs = job_specs[:sample_job_size]
            submit_jobs: Set[Job] = set()
            for job_spec in job_specs:
                job = Job(job_ID=job_spec.job_ID, remaining_iterations=job_spec.total_iterations)
                scheduler.cluster.submit(job)
                submit_jobs.add(job)
            submit_job_IDs = [job.job_ID for job in submit_jobs]
            info(f"Simulator submit job IDs = {submit_job_IDs}")
            before = time_ns()
            assignments, scheduler_reports = scheduler.do_assign(preemptive=True, now=self.now, done_jobs_between_preemption=set())
            end = time_ns()
            info(f"Simulator random placement scheduler {scheduler.name} do assign done with {(end - before) / 1e9} seconds.")
            scheduler.cluster.assignments = assignments
            scheduler.cluster.ensure_start(self.now)

            snapshot_record_parameters = scheduler.build_snapshot_record_parameters()
            enable_plot = self.cluster_config.enable_plot and self.data_source_config.enable_plot
            snapshot_record_parameters.do_plot = enable_plot

            do_snapshot_record_plot(session_id=session_id, is_preemptive_interval=True, snapshot_record_parameters=snapshot_record_parameters)
            running_status = scheduler.cluster.assignments.running_status(data_source=self.data_source)

            record.add_schedule_overhead(end - before)
            record.add_schedule_reports(scheduler_reports)
            record.add_assignments_statistics(
                scheduler.cluster.assignments.statistics(preemptive=True, now=self.now, cluster=scheduler.cluster, data_source=self.data_source))

            info(f"Simulator: running status: {running_status}, assignment profit: {snapshot_record_parameters.profit}")
            info(f"Simulator: done jobs size: {len(scheduler.cluster.done_jobs)}, undone jobs size: {len(scheduler.cluster.jobs)}")
            job_remaining_duration_seconds = {job_ID: remaining_duration / 1e9 for job_ID, remaining_duration in
                                              self.job_remaining_durations(scheduler.cluster).items()}
            info(f"Simulator: running job remaining durations (second): {job_remaining_duration_seconds}")
        record.save()

    def add_preemptive_overheads(self, cluster: Cluster, record: 'PlayRecord', last_assignments: Assignments,
                                 curr_assignments: Assignments):
        job_to_overheads = Assignments.preemptive_overheads(self.data_source, last_assignments, curr_assignments)
        record.add_preemptive_overheads(job_to_overheads)
        for job_ID, overhead in job_to_overheads.items():
            throughput = curr_assignments.job_iteration_time(data_source=self.data_source, job_ID=job_ID)
            overhead_iterations = overhead / throughput
            cluster.add_preemptive_overhead(job_ID=job_ID, overhead_iterations=overhead_iterations)

    def next_events(self, scheduler: Scheduler) -> Optional[Tuple[int, Set[str], Set[str], bool]]:
        def next_submit_jobs() -> Tuple[Set[str], int]:
            s: Set[str] = set()
            st: Optional[int] = None
            while self.next_job_idx < len(self.data_source.job_specs) and \
                    (st is None or self.data_source.job_specs[self.next_job_idx].submit_time == st):
                job_spec = self.data_source.job_specs[self.next_job_idx]
                if st is None:
                    st = job_spec.submit_time
                s.add(job_spec.job_ID)
                self.next_job_idx += 1
            return s, np.inf if st is None else st

        def next_done_jobs() -> Tuple[Set[str], int]:
            job_to_remaining_durations = self.job_remaining_durations(cluster=scheduler.cluster)
            min_remain = np.inf
            min_remain_jobs: Set[str] = set()
            for job_ID, remaining_duration in job_to_remaining_durations.items():
                if remaining_duration < min_remain:
                    min_remain = remaining_duration
                    min_remain_jobs.clear()
                if remaining_duration == min_remain:
                    min_remain_jobs.add(job_ID)
            return min_remain_jobs, min_remain

        submit_jobs, submit_time = next_submit_jobs()
        done_jobs, done_time = next_done_jobs()
        if len(submit_jobs) == 0 and len(done_jobs) == 0:
            return None
        c = get_config()
        scheduler_preemptive_interval = int(
            1e9 * scheduler.config.get("preemptive_interval", c.default_scheduling_preemptive_interval))
        next_preemptive_time = (scheduler_preemptive_interval + self.last_preemptive_time)
        next_preemptive_duration = (next_preemptive_time - self.now) if scheduler_preemptive_interval != 0 else np.inf
        min_time = int(np.min([submit_time, done_time, next_preemptive_duration]))
        submit_jobs = [] if min_time != submit_time else submit_jobs
        done_jobs = [] if min_time != done_time else done_jobs
        next_preemptive_interval = next_preemptive_duration == min_time or scheduler_preemptive_interval == 0
        if min_time == np.inf:
            return None
        return min_time, submit_jobs, done_jobs, next_preemptive_interval

    def pass_duration(self, cluster: Cluster, duration: int):
        job_ID_to_throughput = self.jobs_iteration_time(cluster=cluster)
        for job_ID_to_task_assignments in cluster.assignments.GPU_type_to_task_assignments.values():
            # Dict[str, Set[TaskAssignment]]
            for job_ID in job_ID_to_task_assignments:
                if job_ID in cluster.done_jobs:
                    continue
                job = cluster.get_undone_job(job_ID=job_ID)
                remaining_duration = job.remaining_iterations * job_ID_to_throughput[job_ID]
                remaining_duration -= duration
                remaining_iterations = remaining_duration / job_ID_to_throughput[job_ID]
                job.remaining_iterations = remaining_iterations
        self.now += duration

    def job_remaining_durations(self, cluster: Cluster) -> Dict[str, int]:
        d: Dict[str, int] = dict()
        job_ID_to_iteration_time = self.jobs_iteration_time(cluster=cluster)
        for job_ID_to_task_assignments in cluster.assignments.GPU_type_to_task_assignments.values():
            # Dict[str, Set[TaskAssignment]]
            for job_ID in job_ID_to_task_assignments:
                job = cluster.get_undone_job(job_ID=job_ID)
                remaining_duration = job.remaining_iterations * job_ID_to_iteration_time[job_ID]
                d[job_ID] = int(remaining_duration)
        return d

    def jobs_iteration_time(self, cluster: Cluster) -> Dict[str, float]:
        return cluster.assignments.jobs_iteration_time(data_source=self.data_source)


class PlayRecord:
    def __init__(self,
                 session_id: str,
                 scheduler_name: str,
                 scheduler_enum: SchedulerEnum,
                 data_source_config: DataSourceConfig,
                 data_source: DataSource,
                 cluster_config: ClusterConfig,
                 ):
        self.session_id: str = session_id
        self.scheduler_name: str = scheduler_name
        self.scheduler_enum: SchedulerEnum = scheduler_enum
        self.data_source_config: DataSourceConfig = data_source_config
        self.data_source: DataSource = data_source
        self.cluster_config: ClusterConfig = cluster_config
        self.preemptive_records: List[Dict[str, int]] = list()
        self.done_records: Dict[str, Job] = dict()
        self.schedule_overheads: List[int] = list()
        self.schedule_reports: List[Optional[Any]] = list()
        self.assignments_statistics: List[Dict] = list()

    def add_preemptive_overheads(self, preemptive_overheads: Dict[str, int]):
        self.preemptive_records.append(preemptive_overheads)

    def add_done_jobs(self, done_jobs: Set[Job]):
        done_jobs = {job.job_ID: Job(job_ID=job.job_ID,
                                     remaining_iterations=0,
                                     start_time=job.start_time,
                                     completion_time=job.completion_time) for job in done_jobs}
        for done_job_ID, done_job in done_jobs.items():
            self.done_records[done_job_ID] = done_job

    def add_schedule_overhead(self, overhead: int):
        self.schedule_overheads.append(overhead)

    def add_schedule_reports(self, schedule_reports: Optional[Any]):
        self.schedule_reports.append(schedule_reports)

    def add_assignments_statistics(self, data: Dict):
        self.assignments_statistics.append(data)

    def save(self):
        json_dir = get_json_dir(self.session_id)
        filename = datetime.datetime.now().strftime(
            f"Player_record_{self.scheduler_name}_%Y-%m-%d-%H-%M-%S.json")
        filepath = os.path.join(json_dir, filename)
        d = dict()
        done_records = dict()
        for job_ID, job in self.done_records.items():
            done_records[job_ID] = {
                "job_ID": job.job_ID,
                "start_time": job.start_time,
                "completion_time": job.completion_time,
            }
        d["session_id"] = self.session_id
        d["scheduler_name"] = self.scheduler_name
        d["scheduler_enum"] = self.scheduler_enum.name
        d["data_source_config_name"] = self.data_source_config.name
        d["cluster_config_name"] = self.cluster_config.name
        d["preemptive_records"] = self.preemptive_records
        d["done_records"] = done_records
        d["schedule_overheads"] = self.schedule_overheads
        d["schedule_reports"] = self.schedule_reports
        d["job_specs"] = [job_spec.to_dict() for job_spec in self.data_source.job_specs]
        d["assignment_statistics"] = self.assignments_statistics

        def np_encoder(object):
            if isinstance(object, np.generic):
                return object.item()
        with open(filepath, 'w') as f:
            json.dump(d, f, default=np_encoder, indent='\t')
