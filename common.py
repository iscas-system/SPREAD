import pathlib


def get_session_dir(session_id: str) -> str:
    return str(pathlib.Path(__file__).parent / "json" / session_id)


def get_fig_dir(session_id: str) -> str:
    return str(pathlib.Path(__file__).parent / "fig" / session_id)
