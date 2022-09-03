# import argparse
# import json
# import os
# import sys
#
# import uvicorn
# import datetime
# import pathlib
# from fastapi import FastAPI, Path
#
# from log import logging
# from schedulers.solver import profit
# from model import *
# from plot_assignment import plot_assignment
#
# app = FastAPI()
#
# global_settings = dict()
#
#
# def get_session_dir(session_id: str) -> str:
#     return str(pathlib.Path(global_settings["data-dir-path"]) / "json" / session_id)
#
# def get_fig_dir(session_id: str) -> str:
#     return str(pathlib.Path(global_settings["data-dir-path"]) / "fig" / session_id)
#
#
# @app.post("/solve/{session_id}")
# def solve(*,
#           session_id: str = Path(..., title="session id"),
#           solver_params: SolverParameters):
#     return do_solve(session_id, solver_params)
#
#
# def do_solve(session_id: str,
#           solver_params: SolverParameters):
#     filename = datetime.datetime.now().strftime(f"{session_id}_{solver_params.solver_type.value}_%Y-%m-%d-%H-%M-%S.json")
#     session_dir = get_session_dir(session_id)
#     if not os.path.exists(session_dir):
#         os.mkdir(session_dir)
#     filepath = os.path.join(session_dir, filename)
#     logging.info(f"received solver parameters, session_id = {session_id}, saving file to {filepath}")
#     solver = SolverEnum.get_solver(solver_params.solver_type)
#     solve_raw_res = solver(dist_job_to_tasks=solver_params.dist_job_to_tasks,
#            GPU_comp_mem_capacity=solver_params.GPU_comp_mem_capacity,
#            task_comp_mem_requirements_and_profits=solver_params.task_comp_mem_requirements_and_profits)
#     if solve_raw_res is None:
#         return None
#     assignment, duration = solve_raw_res
#     pf = profit(task_comp_mem_requirements_and_profits=solver_params.task_comp_mem_requirements_and_profits,
#            assignment=assignment)
#     solver_result = SolverResult(solver_parameters=solver_params, duration=duration, profit=pf, assignment=assignment)
#     with open(filepath, 'w') as f:
#         json.dump(solver_result.json(), f)
#     return solver_result
#
#
# @app.post("/record_plot/{session_id}")
# def record_plot(*,
#           session_id: str = Path(..., title="session id"),
#           recorder_parameters: RecordParameters):
#     do_record_plot(session_id, recorder_parameters)
#
#
# def do_record_plot(session_id: str, recorder_parameters: RecordParameters):
#     filename = datetime.datetime.now().strftime(
#         f"{session_id}_{recorder_parameters.solver_type.value}_record_%Y-%m-%d-%H-%M-%S.json")
#     json_filepath = os.path.join(get_session_dir(session_id), filename)
#     fig_filepath = os.path.join(get_fig_dir(session_id), filename)
#     logging.info(f"received record parameters, session_id = {session_id}, saving file to {json_filepath}")
#     with open(json_filepath, 'w') as f:
#         json.dump(recorder_parameters.json(), f)
#     plot_assignment(recorder_parameters, save_dir=fig_filepath)
#     return {}
#
#
# @app.get("/health")
# def health():
#     return "I'm healthy"
#
#
# def run_server():
#     parser = argparse.ArgumentParser(description='MMKP solver (gurobi backend required)')
#     parser.add_argument('--data-dir-path',
#                         type=str,
#                         required=False,
#                         default="",
#                         help="data target dir path")
#     parser.add_argument('--host',
#                         type=str,
#                         default="0.0.0.0",
#                         help="host")
#     parser.add_argument('--port',
#                         type=int,
#                         default=80,
#                         help="port")
#     parser.add_argument('--license-location',
#                         type=str,
#                         default="",
#                         help="license location, using env variable in default")
#     args = parser.parse_args()
#
#     if args.data_dir_path != "":
#         assert os.path.exists(args.data_dir_path) and os.path.isdir(
#             args.data_dir_path), "data dir path not exists or is not a dir!"
#     else:
#         args.data_dir_path = pathlib.Path(__file__).parent / "output"
#
#     global_settings["data-dir-path"] = args.data_dir_path
#
#     GRB_LICENSE_FILE_ENV_KEY = "GRB_LICENSE_FILE"
#     if args.license_location != "":
#         if not os.path.exists(args.license_location):
#             logging.fatal(f"license_location: {args.license_location} not exists.")
#             sys.exit(-1)
#         os.environ[GRB_LICENSE_FILE_ENV_KEY] = args.license_location
#         logging.info(f"using specified license file in {args.license_location}")
#     else:
#         logging.info(f"using license file specified in environment variable {os.environ[GRB_LICENSE_FILE_ENV_KEY]}")
#     uvicorn.run(app, host=args.host, port=args.port, access_log=False)
