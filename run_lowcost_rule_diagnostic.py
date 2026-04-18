import argparse
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from threading import Event
from typing import List, Optional

import dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(CURRENT_DIR, ".env")
if os.path.exists(dotenv_path):
    dotenv.load_dotenv(dotenv_path)

from jsqlsim.enum import FactionEnum, WeaponNameEnum
from jsqlsim.geo import Position
from jsqlsim.world.action import Action, MoveAction
from jsqlsim.world.game import Game
from jsqlsim.world.missions.mission_guide_attack import MissionGuideAttack
from jsqlsim.world.missions.mission_move_to_engage import MissionMoveToEngage
from jsqlsim.world.policy.oneshot import OneshotPolicy
from jsqlsim.world.unit_tasks.oneshot_task import OneShotTask

import main as app_main
from blue_commander import BlueCommander
from red_commander import RedCommander


logger = logging.getLogger(__name__)

VALID_STRATEGIES = {
    "direct_salvo",
    "guide_then_salvo",
    "move_close_then_salvo",
    "guide_and_move_then_salvo",
    "scout_refresh_then_salvo",
    "guide_and_scout_refresh_then_salvo",
    "guide_refresh_move_then_salvo",
    "guide_and_scout_refresh_move_then_salvo",
}


@dataclass
class DiagnosticStats:
    lock: threading.Lock = field(default_factory=threading.Lock)
    lowcost_launches: int = 0
    last_lowcost_launch_time: float = -1e9

    def record_lowcost_launches(self, sim_time: float, count: int) -> None:
        if count <= 0:
            return
        with self.lock:
            self.lowcost_launches += int(count)
            self.last_lowcost_launch_time = max(float(sim_time), float(self.last_lowcost_launch_time))

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "lowcost_launches": int(self.lowcost_launches),
                "last_lowcost_launch_time": float(self.last_lowcost_launch_time),
            }


class DummyLLMModel:
    def __init__(self):
        self.history = []

    def chat(self):
        return ""


class DummyLLMManager:
    def get_llm_model(self, faction_name: str):
        return DummyLLMModel()


class NoReflectionMixin:
    def _cleanup_battle(self):
        with self._cleanup_lock:
            if self._cleanup_done:
                logger.info("[%s] cleanup already completed, skipping duplicate call", self.faction_name)
                return
            self._cleanup_done = True

        logger.info("[%s] --- diagnostic battle ended ---", self.faction_name)
        if self.trace_logger is not None:
            try:
                logger.info("[%s] trace_finalize_start", self.faction_name)
                self.trace_logger.finalize_all(status="shutdown")
                logger.info("[%s] trace_finalize_done", self.faction_name)
            except Exception:
                logger.error("[%s] trace_finalize_error", self.faction_name, exc_info=True)

        try:
            if self.event_logger is not None:
                if self.previous_state is not None:
                    try:
                        self.event_logger.log_event(
                            "SCORE_SNAPSHOT",
                            f"Final score snapshot for {self.faction_name}",
                            extra={
                                "stage": "battle_end",
                                "side_score": self.score,
                                "score_breakdown": self.previous_state.raw_state_dict.get("score", {}),
                            },
                            sim_time=self.previous_state.simtime(),
                        )
                    except Exception:
                        logger.debug("[%s] final score snapshot log failed", self.faction_name, exc_info=True)
                logger.info("[%s] battle_log_finalize_start path=%s", self.faction_name, self.event_logger.filepath)
                self.event_logger.save_log()
                logger.info("[%s] diagnostic_cleanup_done final_score=%s", self.faction_name, self.score)
        finally:
            if getattr(self, "trajectory_recorder", None) is not None:
                try:
                    self.trajectory_recorder.close()
                except Exception:
                    logger.debug("[%s] trajectory recorder close failed", self.faction_name, exc_info=True)


class DiagnosticBlueCommander(NoReflectionMixin, BlueCommander):
    def __init__(self, faction: FactionEnum, shutdown_event: Event, llm_manager: DummyLLMManager):
        super().__init__(faction, shutdown_event, llm_manager)
        self.blue_move_interval = max(
            5.0, float(os.getenv("LOWCOST_DIAGNOSTIC_BLUE_MOVE_INTERVAL_SECONDS", "20"))
        )
        self.blue_move_offset_lon = float(os.getenv("LOWCOST_DIAGNOSTIC_BLUE_MOVE_OFFSET_LON", "0.05"))
        self.blue_move_offset_lat = float(os.getenv("LOWCOST_DIAGNOSTIC_BLUE_MOVE_OFFSET_LAT", "0.035"))
        self._blue_last_move_time = -1e9
        self._blue_motion_anchors = {}
        logger.info(
            "[BLUE] moving diagnostic enabled interval=%.1fs offset_lon=%.4f offset_lat=%.4f",
            self.blue_move_interval,
            self.blue_move_offset_lon,
            self.blue_move_offset_lat,
        )

    def execute_rules(self, state) -> List[Action]:
        base_actions = super().execute_rules(state)
        sim_time = state.simtime()
        if sim_time < self._blue_last_move_time + self.blue_move_interval:
            return base_actions

        patrol_actions = self._build_patrol_move_actions(state, sim_time, base_actions)
        if not patrol_actions:
            return base_actions

        self._blue_last_move_time = sim_time
        merged = self._merge_and_dedupe_actions(base_actions, patrol_actions)
        logger.info(
            "[BLUE] moving diagnostic sim_time=%s patrol_move_count=%s base_action_count=%s merged_action_count=%s",
            sim_time,
            len(patrol_actions),
            len(base_actions),
            len(merged),
        )
        return merged

    def _build_patrol_move_actions(self, state, sim_time: int, base_actions: List[Action]) -> List[Action]:
        move_locked_units = {
            str(getattr(action, "id", getattr(action, "unit_id", "")) or "")
            for action in base_actions or []
            if isinstance(action, MoveAction)
        }
        combatants = []
        for type_name in ("Flagship_Surface", "Cruiser_Surface", "Destroyer_Surface"):
            try:
                combatants.extend(state.find_units(type_name))
            except Exception:
                continue
        if not combatants:
            return []

        phase_index = int(max(0, sim_time) // self.blue_move_interval)
        pattern = [
            (self.blue_move_offset_lon, 0.0),
            (0.0, self.blue_move_offset_lat),
            (-self.blue_move_offset_lon, 0.0),
            (0.0, -self.blue_move_offset_lat),
        ]
        actions: List[Action] = []
        for idx, unit in enumerate(combatants):
            unit_id = str(getattr(unit, "id", "") or "")
            if not unit_id or unit_id in move_locked_units:
                continue
            try:
                pos = unit.position()
            except Exception:
                continue
            if pos is None:
                continue
            anchor = self._blue_motion_anchors.setdefault(
                unit_id,
                (float(pos.lon), float(pos.lat), float(getattr(pos, "alt", 0.0) or 0.0)),
            )
            dx, dy = pattern[(phase_index + idx) % len(pattern)]
            actions.append(
                MoveAction(
                    unit_id,
                    Position(anchor[0] + dx, anchor[1] + dy, float(getattr(pos, "alt", anchor[2]) or anchor[2])),
                )
            )
        return actions


class LowCostOnlyOneshotPolicy(OneshotPolicy):
    def _weapon_select(self, unit_id, state, ds):  # type: ignore[override]
        return WeaponNameEnum.LOW_COST_ATTACK_MISSLE


class DiagnosticRedCommander(NoReflectionMixin, RedCommander):
    def __init__(self, faction, shutdown_event, llm_manager, diagnostic_stats: Optional[DiagnosticStats] = None):
        super().__init__(faction, shutdown_event, llm_manager)
        self.diagnostic_stats = diagnostic_stats
        self.strategy = str(os.getenv("LOWCOST_DIAGNOSTIC_STRATEGY", "guide_and_move_then_salvo")).strip()
        if self.strategy not in VALID_STRATEGIES:
            self.strategy = "guide_and_move_then_salvo"
        self.salvo_size = max(1, int(os.getenv("LOWCOST_DIAGNOSTIC_SALVO_SIZE", "8")))
        self.salvo_interval = max(5.0, float(os.getenv("LOWCOST_DIAGNOSTIC_SALVO_INTERVAL_SECONDS", "45")))
        self.move_interval = max(10.0, float(os.getenv("LOWCOST_DIAGNOSTIC_MOVE_INTERVAL_SECONDS", "45")))
        self.guide_interval = max(10.0, float(os.getenv("LOWCOST_DIAGNOSTIC_GUIDE_INTERVAL_SECONDS", "60")))
        self.guide_units = max(1, int(os.getenv("LOWCOST_DIAGNOSTIC_GUIDE_UNITS", "1")))
        self.guide_hold_seconds = max(0.0, float(os.getenv("LOWCOST_DIAGNOSTIC_GUIDE_HOLD_SECONDS", "0")))
        self.scout_interval = max(10.0, float(os.getenv("LOWCOST_DIAGNOSTIC_SCOUT_INTERVAL_SECONDS", "60")))
        self.track_refresh_threshold = max(
            0.0, float(os.getenv("LOWCOST_DIAGNOSTIC_TRACK_REFRESH_THRESHOLD_SECONDS", "35"))
        )
        self._last_salvo_time = -1e9
        self._last_move_time = -1e9
        self._last_guide_time = -1e9
        self._last_scout_time = -1e9
        self._guide_ready_after = -1e9
        self._lowcost_policy = LowCostOnlyOneshotPolicy()
        logger.info(
            "[RED] lowcost_diagnostic strategy=%s salvo_size=%s salvo_interval=%.1fs move_interval=%.1fs guide_interval=%.1fs guide_units=%s guide_hold=%.1fs scout_interval=%.1fs refresh_threshold=%.1fs",
            self.strategy,
            self.salvo_size,
            self.salvo_interval,
            self.move_interval,
            self.guide_interval,
            self.guide_units,
            self.guide_hold_seconds,
            self.scout_interval,
            self.track_refresh_threshold,
        )

    def execute_rules(self, state) -> List[Action]:
        sim_time = state.simtime()
        graph_context = self._ensure_graph_context(state, sim_time)
        target_table = graph_context.get("target_table", []) or []
        unit_roster = graph_context.get("unit_roster", []) or []

        actions: List[Action] = []
        used_units = set()

        if not target_table or not unit_roster:
            if sim_time >= self._last_scout_time + self.scout_interval:
                scout_actions, scout_tasks = self._build_fast_scout_actions(state, sim_time, unit_roster, target_table, used_units)
                if scout_actions:
                    actions.extend(scout_actions)
                    for task in scout_tasks:
                        used_units.update(task.get("unit_ids", []))
                    self._last_scout_time = sim_time
            return actions

        top_target = target_table[0]
        target_id = str(top_target.get("target_id", "")).strip()
        if not target_id:
            return []

        attack_window = str(top_target.get("attack_window", "")).strip()
        track_staleness_sec = float(top_target.get("track_staleness_sec", 0.0) or 0.0)
        in_range_trucks = [str(unit_id) for unit_id in top_target.get("in_range_trucks", []) or [] if str(unit_id).strip()]
        move_then_fire_trucks = [
            str(unit_id) for unit_id in top_target.get("move_then_fire_trucks", []) or [] if str(unit_id).strip()
        ]

        use_guide = "guide" in self.strategy
        use_move = "move" in self.strategy
        use_scout_refresh = "scout_refresh" in self.strategy or self.strategy == "scout_refresh_then_salvo"

        if use_scout_refresh:
            if track_staleness_sec >= self.track_refresh_threshold and sim_time >= self._last_scout_time + self.scout_interval:
                scout_actions, scout_tasks = self._build_fast_scout_actions(state, sim_time, unit_roster, target_table, used_units)
                if scout_actions:
                    actions.extend(scout_actions)
                    for task in scout_tasks:
                        used_units.update(task.get("unit_ids", []))
                    self._last_scout_time = sim_time

        if use_guide and sim_time >= self._last_guide_time + self.guide_interval:
            guide_actions, guide_tasks = self._build_fast_guide_actions(state, sim_time, unit_roster, top_target, used_units)
            if guide_actions:
                actions.extend(guide_actions)
                for task in guide_tasks:
                    used_units.update(task.get("unit_ids", []))
                self._last_guide_time = sim_time
                if self.guide_hold_seconds > 0 and self._guide_ready_after <= sim_time:
                    self._guide_ready_after = sim_time + self.guide_hold_seconds

        if use_move and attack_window != "fire_now" and move_then_fire_trucks:
            if sim_time >= self._last_move_time + self.move_interval:
                selected_move_units = move_then_fire_trucks[: self.salvo_size]
                move_actions = self._build_move_to_engage_actions(state, sim_time, selected_move_units, target_id)
                if move_actions:
                    actions.extend(move_actions)
                    used_units.update(selected_move_units)
                    self._last_move_time = sim_time

        salvo_blocked_by_guide_hold = use_guide and sim_time < self._guide_ready_after
        if in_range_trucks and not salvo_blocked_by_guide_hold and sim_time >= self._last_salvo_time + self.salvo_interval:
            selected_units = [unit_id for unit_id in in_range_trucks if unit_id not in used_units][: self.salvo_size]
            salvo_actions = self._build_lowcost_salvo_actions(state, sim_time, selected_units, target_id)
            if salvo_actions:
                actions.extend(salvo_actions)
                lowcost_launch_count = sum(
                    1
                    for action in salvo_actions
                    if str(getattr(action, "weapon_name", getattr(action, "weapon", ""))) == WeaponNameEnum.LOW_COST_ATTACK_MISSLE.value
                    or str(getattr(action, "weapon_name", getattr(action, "weapon", ""))) == "LowCostAttackMissile"
                    or "LowCostAttackMissile" in str(action)
                )
                if self.diagnostic_stats is not None:
                    self.diagnostic_stats.record_lowcost_launches(sim_time=sim_time, count=lowcost_launch_count)
                self._last_salvo_time = sim_time

        if actions:
            logger.info(
                "[RED] lowcost_rule_diag sim_time=%s strategy=%s target=%s window=%s stale=%.1f action_count=%s guide_hold_until=%.1f",
                sim_time,
                self.strategy,
                target_id,
                attack_window,
                track_staleness_sec,
                len(actions),
                self._guide_ready_after,
            )
        return actions

    def _build_fast_guide_actions(
        self,
        state,
        sim_time: int,
        unit_roster: List[dict],
        top_target: dict,
        used_units: set,
    ) -> tuple[List[Action], List[dict]]:
        if not top_target or not str(top_target.get("target_id", "")).strip():
            return [], []
        guide_units = self._available_units(unit_roster, role="guide", used_units=used_units)
        if not guide_units:
            return [], []

        selected_units = [str(unit["unit_id"]) for unit in guide_units[: self.guide_units]]
        mission = MissionGuideAttack(
            id=f"{self.faction_name}_diag_guide_{sim_time}_{selected_units[0]}",
            unit_ids=selected_units,
            target_id=str(top_target["target_id"]),
            sim_start_time=sim_time,
            sim_end_time=sim_time + 240,
            loiter_radius_m=25000.0,
        )
        actions = self._expand_tasks_to_actions(mission.run(state), state)
        if not actions:
            return [], []
        return actions, [
            {
                "task_type": "GuideAttack",
                "unit_ids": selected_units,
                "target_id": str(top_target["target_id"]),
                "role": "guide",
            }
        ]

    def _build_move_to_engage_actions(self, state, sim_time: int, unit_ids: List[str], target_id: str) -> List[Action]:
        if not unit_ids or not target_id:
            return []
        mission = MissionMoveToEngage(
            id=f"{self.faction_name}_diag_move_{sim_time}",
            unit_ids=unit_ids,
            target_id=target_id,
            sim_start_time=sim_time,
            sim_end_time=sim_time + 180,
        )
        return self._expand_tasks_to_actions(mission.run(state), state)

    def _build_lowcost_salvo_actions(self, state, sim_time: int, unit_ids: List[str], target_id: str) -> List[Action]:
        if not unit_ids or not target_id:
            return []
        history = None
        try:
            history = self.faction.get_state_history()
        except Exception:
            history = None

        actions: List[Action] = []
        for unit_id in unit_ids:
            task = OneShotTask(
                mission_id=f"{self.faction_name}_diag_lowcost_{sim_time}_{unit_id}",
                unit_id=unit_id,
                target_id=target_id,
                sim_start_time=sim_time,
                sim_end_time=sim_time + 180,
            )
            task.set_policy(self._lowcost_policy)
            if history is not None:
                task.set_history(history)
            task.set_launch_time(sim_time)
            result = task.run(state)
            if not result:
                continue
            if isinstance(result, list):
                actions.extend(result)
            else:
                actions.append(result)
        return actions


def _run_commander_thread(agent_cls, faction_enum, shutdown_event, llm_manager, game_faction, diagnostic_stats=None):
    threading.current_thread().name = f"{faction_enum.name}CommanderThread"
    logging.info("--- initializing [%s] commander ---", faction_enum.name)
    if diagnostic_stats is None:
        agent = agent_cls(faction_enum, shutdown_event, llm_manager)
    else:
        agent = agent_cls(faction_enum, shutdown_event, llm_manager, diagnostic_stats)
    agent.faction = game_faction
    logging.info("--- [%s] diagnostic commander started ---", faction_enum.name)
    agent.run()
    logging.info("--- [%s] diagnostic commander stopped ---", faction_enum.name)


def _wait_until_stop_condition_or_sample_goal(
    game: Game,
    shutdown_event: Event,
    stop_simtime: int,
    poll_interval: float,
    diagnostic_stats: Optional[DiagnosticStats],
    desired_launches: int,
    post_launch_tail_seconds: float,
):
    last_logged_simtime = -1
    while not shutdown_event.is_set():
        red_simtime = app_main._safe_simtime(getattr(game, "red_simtime", 0))
        blue_simtime = app_main._safe_simtime(getattr(game, "blue_simtime", 0))

        if stop_simtime > 0 and red_simtime >= stop_simtime and blue_simtime >= stop_simtime:
            logging.info(
                "auto_stop_reached stop_simtime=%s red_simtime=%s blue_simtime=%s",
                stop_simtime,
                red_simtime,
                blue_simtime,
            )
            return "auto_stop"

        if diagnostic_stats is not None and desired_launches > 0:
            snapshot = diagnostic_stats.snapshot()
            if snapshot["lowcost_launches"] >= desired_launches:
                tail_target = snapshot["last_lowcost_launch_time"] + float(post_launch_tail_seconds)
                if red_simtime >= tail_target and blue_simtime >= tail_target:
                    logging.info(
                        "diagnostic_sample_goal_reached launches=%s tail_target=%.1f red_simtime=%s blue_simtime=%s",
                        snapshot["lowcost_launches"],
                        tail_target,
                        red_simtime,
                        blue_simtime,
                    )
                    return "sample_goal"

        current_minute_mark = min(red_simtime, blue_simtime)
        if current_minute_mark >= 0 and current_minute_mark // 60 > last_logged_simtime // 60:
            extra = diagnostic_stats.snapshot() if diagnostic_stats is not None else {}
            logging.info(
                "simtime_progress red_simtime=%s blue_simtime=%s stop_simtime=%s lowcost_launches=%s",
                red_simtime,
                blue_simtime,
                stop_simtime,
                extra.get("lowcost_launches"),
            )
            last_logged_simtime = current_minute_mark

        time.sleep(max(0.1, poll_interval))

    return "shutdown_event"


def run_diagnostic(end_simtime: int, engine_host: str, step_interval: float) -> dict:
    app_main.setup_logging()
    app_main._configure_external_loggers()

    shutdown_event = Event()
    llm_manager = DummyLLMManager()
    diagnostic_stats = DiagnosticStats()
    desired_launches = max(0, int(os.getenv("LOWCOST_DIAGNOSTIC_DESIRED_LAUNCHES", "0")))
    post_launch_tail_seconds = max(0.0, float(os.getenv("LOWCOST_DIAGNOSTIC_POST_LAUNCH_TAIL_SECONDS", "150")))

    stop_simtime = app_main._resolve_stop_simtime(end_simtime)
    os.environ["AUTO_STOP_SIMTIME_SECONDS"] = str(stop_simtime)
    os.environ["LANGGRAPH_ENABLED_SIDES"] = ""
    os.environ["BLUE_DECISION_MODE"] = "rule"
    os.environ["TRAJECTORY_DIAGNOSTICS_ENABLED"] = "1"

    game = Game.new(
        sim_control_red_endpoint=f"{engine_host}:30001",
        sim_control_blue_endpoint=f"{engine_host}:40001",
        sim_platform_endpoint=f"{engine_host}:50005",
        step_interval=step_interval,
    )
    red_faction = game.faction(FactionEnum.RED)
    blue_faction = game.faction(FactionEnum.BLUE)

    red_thread = threading.Thread(
        target=_run_commander_thread,
        args=(DiagnosticRedCommander, FactionEnum.RED, shutdown_event, llm_manager, red_faction, diagnostic_stats),
        name="RedCommanderThread",
        daemon=True,
    )
    blue_thread = threading.Thread(
        target=_run_commander_thread,
        args=(DiagnosticBlueCommander, FactionEnum.BLUE, shutdown_event, llm_manager, blue_faction),
        name="BlueCommanderThread",
        daemon=True,
    )

    logging.info("Starting RED diagnostic commander thread...")
    red_thread.start()
    time.sleep(2)
    logging.info("Starting BLUE diagnostic commander thread...")
    blue_thread.start()

    join_timeout_seconds = 60.0
    poll_interval_seconds = 0.5
    try:
        stop_reason = _wait_until_stop_condition_or_sample_goal(
            game=game,
            shutdown_event=shutdown_event,
            stop_simtime=stop_simtime,
            poll_interval=poll_interval_seconds,
            diagnostic_stats=diagnostic_stats,
            desired_launches=desired_launches,
            post_launch_tail_seconds=post_launch_tail_seconds,
        )
        logging.info("diagnostic_wait_finished reason=%s stop_simtime=%s", stop_reason, stop_simtime)
    finally:
        shutdown_event.set()

    alive_threads = app_main._join_commander_threads([red_thread, blue_thread], join_timeout_seconds)
    stats_snapshot = diagnostic_stats.snapshot()
    return {
        "stop_reason": stop_reason if "stop_reason" in locals() else "shutdown_event",
        "alive_threads": alive_threads,
        "stop_simtime": stop_simtime,
        "lowcost_launches": stats_snapshot["lowcost_launches"],
        "last_lowcost_launch_time": stats_snapshot["last_lowcost_launch_time"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a pure-rule low-cost missile diagnostic battle.")
    parser.add_argument("--end-simtime", type=int, default=600, dest="end_simtime")
    parser.add_argument("--engine-host", type=str, default="127.0.0.1", dest="engine_host")
    parser.add_argument("--step-interval", type=float, default=0.1, dest="step_interval")
    parser.add_argument("--strategy", type=str, default="guide_and_move_then_salvo", choices=sorted(VALID_STRATEGIES))
    parser.add_argument("--salvo-size", type=int, default=8, dest="salvo_size")
    parser.add_argument("--salvo-interval", type=float, default=45.0, dest="salvo_interval")
    parser.add_argument("--move-interval", type=float, default=45.0, dest="move_interval")
    parser.add_argument("--guide-interval", type=float, default=60.0, dest="guide_interval")
    parser.add_argument("--guide-units", type=int, default=1, dest="guide_units")
    parser.add_argument("--guide-hold-seconds", type=float, default=0.0, dest="guide_hold_seconds")
    parser.add_argument("--scout-interval", type=float, default=60.0, dest="scout_interval")
    parser.add_argument("--refresh-threshold", type=float, default=35.0, dest="refresh_threshold")
    parser.add_argument("--desired-launches", type=int, default=18, dest="desired_launches")
    parser.add_argument("--post-launch-tail-seconds", type=float, default=120.0, dest="post_launch_tail_seconds")
    parser.add_argument("--diagnostic-interval-seconds", type=float, default=3.0, dest="diagnostic_interval_seconds")
    parser.add_argument("--blue-move-interval", type=float, default=20.0, dest="blue_move_interval")
    parser.add_argument("--blue-move-offset-lon", type=float, default=0.05, dest="blue_move_offset_lon")
    parser.add_argument("--blue-move-offset-lat", type=float, default=0.035, dest="blue_move_offset_lat")
    parser.add_argument("--oneshot-debug", action="store_true", dest="oneshot_debug")
    return parser


def main():
    args = _parser().parse_args()
    os.environ["LOWCOST_DIAGNOSTIC_STRATEGY"] = str(args.strategy)
    os.environ["LOWCOST_DIAGNOSTIC_SALVO_SIZE"] = str(args.salvo_size)
    os.environ["LOWCOST_DIAGNOSTIC_SALVO_INTERVAL_SECONDS"] = str(args.salvo_interval)
    os.environ["LOWCOST_DIAGNOSTIC_MOVE_INTERVAL_SECONDS"] = str(args.move_interval)
    os.environ["LOWCOST_DIAGNOSTIC_GUIDE_INTERVAL_SECONDS"] = str(args.guide_interval)
    os.environ["LOWCOST_DIAGNOSTIC_GUIDE_UNITS"] = str(args.guide_units)
    os.environ["LOWCOST_DIAGNOSTIC_GUIDE_HOLD_SECONDS"] = str(args.guide_hold_seconds)
    os.environ["LOWCOST_DIAGNOSTIC_SCOUT_INTERVAL_SECONDS"] = str(args.scout_interval)
    os.environ["LOWCOST_DIAGNOSTIC_TRACK_REFRESH_THRESHOLD_SECONDS"] = str(args.refresh_threshold)
    os.environ["LOWCOST_DIAGNOSTIC_DESIRED_LAUNCHES"] = str(args.desired_launches)
    os.environ["LOWCOST_DIAGNOSTIC_POST_LAUNCH_TAIL_SECONDS"] = str(args.post_launch_tail_seconds)
    os.environ["TRAJECTORY_DIAGNOSTICS_INTERVAL_SECONDS"] = str(args.diagnostic_interval_seconds)
    os.environ["LOWCOST_DIAGNOSTIC_BLUE_MOVE_INTERVAL_SECONDS"] = str(args.blue_move_interval)
    os.environ["LOWCOST_DIAGNOSTIC_BLUE_MOVE_OFFSET_LON"] = str(args.blue_move_offset_lon)
    os.environ["LOWCOST_DIAGNOSTIC_BLUE_MOVE_OFFSET_LAT"] = str(args.blue_move_offset_lat)
    if args.oneshot_debug:
        os.environ["JSQLSIM_ONESHOT_DEBUG"] = "1"
    else:
        os.environ.pop("JSQLSIM_ONESHOT_DEBUG", None)
    result = run_diagnostic(args.end_simtime, args.engine_host, args.step_interval)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
