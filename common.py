import os.path
import pathlib

def must_exist_dir(dir_path: str):
    if os.path.exists(dir_path):
        assert os.path.isdir(dir_path)
        return
    os.mkdir(dir_path)


def get_session_dir(session_id: str) -> str:
    path_str = str(pathlib.Path(__file__).parent / "output" / session_id)
    must_exist_dir(path_str)
    return path_str


def get_json_dir(session_id: str) -> str:
    path_str = str(pathlib.Path(get_session_dir(session_id=session_id)) / "json")
    must_exist_dir(path_str)
    return path_str


def get_fig_dir(session_id: str) -> str:
    path_str = str(pathlib.Path(get_session_dir(session_id=session_id)) / "fig")
    must_exist_dir(path_str)
    return path_str
