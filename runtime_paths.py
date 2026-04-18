import os
from typing import Optional


_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def _read_bool_env(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return raw not in {"0", "false", "no", "off", ""}


def project_root() -> str:
    return _CURRENT_DIR


def resolve_path(path: str, base_dir: Optional[str] = None) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir or _CURRENT_DIR, path)


def runtime_output_root() -> str:
    raw = str(os.getenv("RUN_OUTPUT_ROOT", "")).strip()
    if raw:
        return resolve_path(raw)
    return os.path.join(_CURRENT_DIR, "test")


def ensure_runtime_output_root() -> str:
    root = runtime_output_root()
    os.makedirs(root, exist_ok=True)
    return root


def output_dir(name: str) -> str:
    return os.path.join(runtime_output_root(), name)


def ensure_output_dir(name: str) -> str:
    path = output_dir(name)
    os.makedirs(path, exist_ok=True)
    return path


def runtime_path(*parts: str) -> str:
    return os.path.join(runtime_output_root(), *parts)


def batch_mode_enabled() -> bool:
    return _read_bool_env("BATCH_MODE", False)


def console_logging_enabled() -> bool:
    return _read_bool_env("APP_ENABLE_CONSOLE_LOG", True)


def trace_live_enabled() -> bool:
    default_enabled = not batch_mode_enabled()
    return _read_bool_env("RED_TRACE_ENABLE_LIVE", default_enabled)
