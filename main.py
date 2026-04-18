import argparse
import logging
import logging.handlers
import os
import sys
import threading
import time
from datetime import datetime
from threading import Event
from typing import Type

import dotenv

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(current_dir, ".env")
    if not os.path.exists(dotenv_path):
        dotenv_path = os.path.join(current_dir, "..", ".env")
    if os.path.exists(dotenv_path):
        dotenv.load_dotenv(dotenv_path)
except Exception:
    pass
from jsqlsim.enum import FactionEnum
from jsqlsim.world.game import Game

from base_commander import BaseCommander
from blue_commander import BlueCommander
from llm_manager import LLMManager
from red_commander import RedCommander
from runtime_paths import console_logging_enabled, ensure_output_dir





class MuteEngineFilter(logging.Filter):
    def filter(self, record):
        allowed_files = [
            "main.py",
            "base_commander.py",
            "red_commander.py",
            "blue_commander.py",
            "graph.py",
            "rule_book.py",
            "llm_manager.py",
            "situation_summarizer.py",
            "event_logger.py",
            "reflection_agent.py",
        ]
        return record.filename in allowed_files


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        return float(raw)
    except Exception:
        logging.warning("Invalid %s=%s, fallback to %s", name, raw, default)
        return default


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        return int(raw)
    except Exception:
        logging.warning("Invalid %s=%s, fallback to %s", name, raw, default)
        return default


def _read_log_level_env(name: str, default: str) -> int:
    raw = str(os.getenv(name, default)).strip().upper()
    value = getattr(logging, raw, None)
    if isinstance(value, int):
        return value
    logging.warning("Invalid %s=%s, fallback to %s", name, raw, default)
    return getattr(logging, str(default).upper(), logging.INFO)


def _safe_simtime(value) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return 0


def _resolve_stop_simtime(end_simtime: int) -> int:
    auto_stop_simtime = _read_int_env("AUTO_STOP_SIMTIME_SECONDS", 1800)
    candidates = [int(end_simtime)] if end_simtime and end_simtime > 0 else []
    if auto_stop_simtime > 0:
        candidates.append(auto_stop_simtime)
    if not candidates:
        return 0
    return min(candidates)


def _wait_until_stop_condition(game: Game, shutdown_event: Event, stop_simtime: int, poll_interval: float):
    last_logged_simtime = -1
    while not shutdown_event.is_set():
        red_simtime = _safe_simtime(getattr(game, "red_simtime", 0))
        blue_simtime = _safe_simtime(getattr(game, "blue_simtime", 0))

        if stop_simtime > 0 and red_simtime >= stop_simtime and blue_simtime >= stop_simtime:
            logging.info(
                "auto_stop_reached stop_simtime=%s red_simtime=%s blue_simtime=%s",
                stop_simtime,
                red_simtime,
                blue_simtime,
            )
            return "auto_stop"

        current_minute_mark = min(red_simtime, blue_simtime)
        if current_minute_mark >= 0 and current_minute_mark // 60 > last_logged_simtime // 60:
            logging.info(
                "simtime_progress red_simtime=%s blue_simtime=%s stop_simtime=%s",
                red_simtime,
                blue_simtime,
                stop_simtime,
            )
            last_logged_simtime = current_minute_mark

        time.sleep(max(0.1, poll_interval))

    return "shutdown_event"


def _join_commander_threads(threads, timeout_seconds: float):
    thread_names = [thread.name for thread in threads]
    logging.info("shutdown_join_start timeout=%.1fs threads=%s", timeout_seconds, thread_names)
    deadline = time.time() + timeout_seconds
    alive_threads = []
    for thread in threads:
        remaining = max(0.0, deadline - time.time())
        thread.join(timeout=remaining)
        if thread.is_alive():
            alive_threads.append(thread.name)

    if alive_threads:
        logging.warning(
            "shutdown_join_timeout timeout=%.1fs alive_threads=%s",
            timeout_seconds,
            alive_threads,
        )
        logging.warning(
            "shutdown_exit_forced alive_threads=%s note=process will exit and may lose post-battle reflection",
            alive_threads,
        )
    else:
        logging.info("shutdown_join_done timeout=%.1fs threads=%s", timeout_seconds, thread_names)
    return alive_threads


def setup_logging():
    log_format = logging.Formatter("%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s")
    logs_dir = ensure_output_dir("logs")
    log_filename = os.path.join(logs_dir, f"wargame_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    file_level = _read_log_level_env("APP_FILE_LOG_LEVEL", "INFO")
    console_level = _read_log_level_env("APP_CONSOLE_LOG_LEVEL", "WARNING")
    enable_console = console_logging_enabled()

    logger = logging.getLogger()
    logger.setLevel(file_level if not enable_console else min(file_level, console_level))

    for handler in list(logger.handlers):
        if getattr(handler, "_wargame_file_handler", False) or getattr(handler, "_wargame_console_handler", False):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_filename,
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
    )
    file_handler._wargame_file_handler = True
    logger.addHandler(file_handler)

    console_handler = None
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler._wargame_console_handler = True
        console_handler.addFilter(MuteEngineFilter())
        logger.addHandler(console_handler)

    file_handler.setFormatter(log_format)
    file_handler.setLevel(file_level)
    if console_handler is not None:
        console_handler.setFormatter(log_format)
        console_handler.setLevel(console_level)

    logging.info("Logging initialized. file=%s console_enabled=%s", log_filename, enable_console)
    return log_filename


def _configure_external_loggers():
    logger_levels = {
        "httpx": _read_log_level_env("HTTPX_LOG_LEVEL", "WARNING"),
        "httpcore": _read_log_level_env("HTTPX_LOG_LEVEL", "WARNING"),
        "openai": _read_log_level_env("OPENAI_LOG_LEVEL", "WARNING"),
        "jsqlsim.llm.model.chat_model": _read_log_level_env("CHAT_MODEL_LOG_LEVEL", "WARNING"),
    }
    for logger_name, level in logger_levels.items():
        logging.getLogger(logger_name).setLevel(level)


def _run_commander_thread(
    agent_cls: Type[BaseCommander],
    faction_enum: FactionEnum,
    shutdown_event: Event,
    llm_manager: LLMManager,
    game_faction,
):
    threading.current_thread().name = f"{faction_enum.name}CommanderThread"
    logging.info("--- initializing [%s] commander ---", faction_enum.name)
    agent = agent_cls(faction_enum, shutdown_event, llm_manager)
    agent.faction = game_faction
    logging.info("--- [%s] commander started ---", faction_enum.name)
    agent.run()
    logging.info("--- [%s] commander stopped ---", faction_enum.name)


def main(end_simtime: int = 2000, engine_host: str = "127.0.0.1", step_interval: float = 0.1):
    log_filename = setup_logging()
    _configure_external_loggers()
    logging.info("Environment loaded.")

    logging.info("Initializing LLMManager...")
    try:
        llm_manager = LLMManager()
    except Exception as e:
        logging.critical("LLMManager init failed: %s", e, exc_info=True)
        sys.exit(1)

    shutdown_event = Event()
    stop_simtime = _resolve_stop_simtime(end_simtime)
    os.environ["AUTO_STOP_SIMTIME_SECONDS"] = str(stop_simtime)
    logging.info(
        "Runtime config: LANGGRAPH_ENABLED_SIDES=%s BLUE_DECISION_MODE=%s BLUE_LLM_INTERVAL=%s",
        os.getenv("LANGGRAPH_ENABLED_SIDES", "RED"),
        os.getenv("BLUE_DECISION_MODE", "rule"),
        os.getenv("BLUE_LLM_INTERVAL", "60"),
    )

    game = Game.new(
        sim_control_red_endpoint=f"{engine_host}:30001",
        sim_control_blue_endpoint=f"{engine_host}:40001",
        sim_platform_endpoint=f"{engine_host}:50005",
        step_interval=step_interval,
    )
    red_faction = game.faction(FactionEnum.RED)
    blue_faction = game.faction(FactionEnum.BLUE)

    for logger_name in logging.root.manager.loggerDict:
        lower = logger_name.lower()
        if any(kw in lower for kw in ["faction", "jsqlsim", "engine", "simulation", "game"]):
            target_logger = logging.getLogger(logger_name)
            target_logger.setLevel(logging.WARNING)
            target_logger.propagate = False

    red_thread = threading.Thread(
        target=_run_commander_thread,
        args=(RedCommander, FactionEnum.RED, shutdown_event, llm_manager, red_faction),
        name="RedCommanderThread",
        daemon=True,
    )
    blue_thread = threading.Thread(
        target=_run_commander_thread,
        args=(BlueCommander, FactionEnum.BLUE, shutdown_event, llm_manager, blue_faction),
        name="BlueCommanderThread",
        daemon=True,
    )

    logging.info("Starting RED commander thread...")
    red_thread.start()
    time.sleep(2)

    logging.info("Starting BLUE commander thread...")
    blue_thread.start()

    join_timeout_seconds = _read_float_env("COMMANDER_JOIN_TIMEOUT_SECONDS", 600.0)
    poll_interval_seconds = _read_float_env("MAIN_POLL_INTERVAL_SECONDS", 0.5)
    logging.info(
        "Main loop config: requested_end_simtime=%s auto_stop_simtime=%s poll_interval=%.1fs",
        end_simtime,
        stop_simtime,
        poll_interval_seconds,
    )

    try:
        stop_reason = _wait_until_stop_condition(game, shutdown_event, stop_simtime, poll_interval_seconds)
        logging.info("main_wait_finished reason=%s stop_simtime=%s", stop_reason, stop_simtime)
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received. Requesting shutdown...")
    finally:
        shutdown_event.set()

    alive_threads = _join_commander_threads([red_thread, blue_thread], join_timeout_seconds)
    if alive_threads:
        logging.warning("Battle finished with unfinished commander threads: %s", alive_threads)
    else:
        logging.info("Battle finished. All commander threads exited.")
    return {
        "log_path": log_filename,
        "stop_reason": stop_reason if "stop_reason" in locals() else "keyboard_interrupt",
        "alive_threads": alive_threads,
        "stop_simtime": stop_simtime,
    }


def _build_cli_parser():
    parser = argparse.ArgumentParser(description="Run a single wargame simulation.")
    parser.add_argument("--end-simtime", type=int, default=3000, dest="end_simtime")
    parser.add_argument("--engine-host", type=str, default="127.0.0.1", dest="engine_host")
    parser.add_argument("--step-interval", type=float, default=0.1, dest="step_interval")
    return parser


if __name__ == "__main__":
    args = _build_cli_parser().parse_args()
    main(end_simtime=args.end_simtime, engine_host=args.engine_host, step_interval=args.step_interval)
