import os.path
import pathlib
import threading

mkdir_lock = threading.Lock()

def must_exist_dir(dir_path: str):
    if os.path.exists(dir_path):
        assert os.path.isdir(dir_path)
        return
    try:
        mkdir_lock.acquire()
        if os.path.exists(dir_path):
            assert os.path.isdir(dir_path)
            return
        else:
            os.mkdir(dir_path)
            return
    finally:
        mkdir_lock.release()



def get_session_dir(session_id: str) -> str:
    reports_path = os.environ.get("DATA_PATH")
    if reports_path is not None:
        path_str = str(pathlib.Path(reports_path) / session_id)
        must_exist_dir(path_str)
        return path_str

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
