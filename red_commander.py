import json
import logging
import os
from collections import Counter
from threading import Event
from typing import Dict, List, Optional

import dotenv
from jsqlsim import GameState
from jsqlsim.enum import FactionEnum
from jsqlsim.geo import Area, Position
from jsqlsim.world.action import Action
from jsqlsim.world.missions.mission_focus_fire import MissionFocusFire
from jsqlsim.world.missions.mission_guide_attack import MissionGuideAttack
from jsqlsim.world.missions.mission_move_and_attack import MissionMoveAndAttack
from jsqlsim.world.missions.mission_move_to_engage import MissionMoveToEngage
from jsqlsim.world.missions.mission_scout import MissionScout

from action_semantics import filter_red_actions_semantically
from base_commander import BaseCommander
from llm_manager import LLMManager
from red_trace_helper import summarize_engine_actions, summarize_json_actions
from state_access import (
    distance_between_positions,
    get_detected_target_ids,
    get_unit_hp,
    get_unit_position,
    get_target_position,
    get_unit_type_name,
    get_weapon_inventory,
    infer_unit_type,
    iter_platform_units,
)

logger = logging.getLogger(__name__)

RED_TASK_TTLS = {
    "ScoutArea": 90,
    "GuideAttack": 90,
    "MoveToEngage": 60,
    "FocusFire": 45,
    "ShootAndScoot": 45,
}

RED_FIRE_ACTIONS = {"FocusFire", "ShootAndScoot"}
RED_ATTACK_WEAPON_SPEED_MPS = {
    "HighCostAttackMissile": 1200.0,
    "LowCostAttackMissile": 500.0,
    "ShipToGround_CruiseMissile": 250.0,
}
RED_ATTACK_WEAPON_COST = {
    "HighCostAttackMissile": 8.0,
    "LowCostAttackMissile": 3.0,
    "ShipToGround_CruiseMissile": 10.0,
}
SURFACE_TARGET_TYPES = {"Flagship_Surface", "Cruiser_Surface", "Destroyer_Surface"}

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(current_dir, ".env")
    if not os.path.exists(dotenv_path):
        dotenv_path = os.path.join(current_dir, "..", ".env")
    if os.path.exists(dotenv_path):
        dotenv.load_dotenv(dotenv_path)
except Exception:
    pass


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        return float(raw)
    except Exception:
        logger.warning("Invalid %s=%s, fallback to %s", name, raw, default)
        return default


def _read_bool_env(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return raw not in {"0", "false", "no", "off", ""}


class RedCommander(BaseCommander):
    def __init__(self, faction: FactionEnum, shutdown_event: Event, llm_manager: LLMManager):
        super().__init__(faction, shutdown_event, llm_manager)
        self.faction_name = "RED"
        self._distance_api_warning_emitted = False
        self.red_fast_interval_seconds = _read_float_env("RED_FAST_INTERVAL_SECONDS", 5.0)
        self.red_slow_base_interval_seconds = _read_float_env("RED_SLOW_BASE_INTERVAL_SECONDS", 30.0)
        self.red_opening_enable_probe_fire = _read_bool_env("RED_OPENING_ENABLE_PROBE_FIRE", True)
        self.red_endgame_launch_guard_enabled = _read_bool_env("RED_ENDGAME_LAUNCH_GUARD_ENABLED", True)
        self.red_endgame_min_impact_margin_seconds = _read_float_env("RED_ENDGAME_MIN_IMPACT_MARGIN_SECONDS", 45.0)
        self.red_endgame_stop_simtime = _read_float_env("AUTO_STOP_SIMTIME_SECONDS", 0.0)
        self.red_low_cost_track_stale_threshold_seconds = _read_float_env("RED_LOW_COST_TRACK_STALE_THRESHOLD_SECONDS", 30.0)
        self.red_low_cost_max_flight_time_seconds = _read_float_env("RED_LOW_COST_MAX_FLIGHT_TIME_SECONDS", 180.0)
        self.red_guidance_settle_seconds = _read_float_env("RED_GUIDANCE_SETTLE_SECONDS", 12.0)
        self._opening_window_seconds = max(12.0, self.red_fast_interval_seconds * 4.0)
        self._latest_graph_context_sim_time = -1
        self._latest_graph_context = {}
        self._last_fast_run_time = -9999.0
        self._slow_started = False
        self._last_event_trigger_time = -9999.0
        self._slow_trigger_snapshot = {
            "target_windows": {},
            "task_board_units": set(),
            "seen_high_value_targets": set(),
        }
        logger.info("[%s] Red commander initialized in async dual-system mode.", self.faction_name)
        logger.info(
            "[%s] RED fast/slow config fast_interval=%.1fs slow_base_interval=%.1fs opening_probe_fire=%s",
            self.faction_name,
            self.red_fast_interval_seconds,
            self.red_slow_base_interval_seconds,
            self.red_opening_enable_probe_fire,
        )
        logger.info(
            "[%s] RED endgame guard enabled=%s stop_simtime=%.1f impact_margin=%.1fs",
            self.faction_name,
            self.red_endgame_launch_guard_enabled,
            self.red_endgame_stop_simtime,
            self.red_endgame_min_impact_margin_seconds,
        )
        logger.info(
            "[%s] RED low-cost policy stale_threshold=%.1fs max_flight=%.1fs guidance_settle=%.1fs",
            self.faction_name,
            self.red_low_cost_track_stale_threshold_seconds,
            self.red_low_cost_max_flight_time_seconds,
            self.red_guidance_settle_seconds,
        )

    def execute_rules(self, state: GameState) -> List[Action]:
        sim_time = state.simtime()
        graph_context = self._ensure_graph_context(state, sim_time)
        if sim_time < self._last_fast_run_time + self.red_fast_interval_seconds:
            return []
        actions, planned_tasks, reasons = self._run_fast_layer(state, sim_time, graph_context)
        if planned_tasks:
            self._remember_fast_tasks(planned_tasks, sim_time)
        if actions:
            self._last_fast_run_time = sim_time
            logger.info(
                "[RED] fast_layer sim_time=%s reasons=%s action_count=%s summary=%s",
                sim_time,
                ",".join(reasons) or "none",
                len(actions),
                summarize_engine_actions(actions),
            )
        return actions

    def _get_system2_base_interval(self) -> float:
        return self.red_slow_base_interval_seconds

    def _get_system2_trigger_reason(self, state: GameState, sim_time: int) -> Optional[str]:
        graph_context = self._ensure_graph_context(state, sim_time)
        task_board = graph_context.get("task_board", {}) or {}
        slow_signature = graph_context.get("slow_context_signature", {})
        target_windows = slow_signature.get("target_windows", {})
        seen_high_value_targets = set(slow_signature.get("seen_high_value_targets", []))

        if not self._slow_started:
            return "battle_start"
        if sim_time < self._last_event_trigger_time + max(8.0, self.red_fast_interval_seconds * 2.0):
            return None
        previous_windows = self._slow_trigger_snapshot.get("target_windows", {})
        if target_windows != previous_windows:
            return "window_shift"
        previous_hvt = self._slow_trigger_snapshot.get("seen_high_value_targets", set())
        if seen_high_value_targets - previous_hvt:
            return "high_value_target_spotted"
        previous_task_units = self._slow_trigger_snapshot.get("task_board_units", set())
        current_task_units = set(task_board.keys())
        if previous_task_units and not current_task_units:
            return "task_board_empty"
        return None

    def _on_system2_scheduled(self, state: GameState, sim_time: int, reason: str, graph_context: dict) -> None:
        task_board = graph_context.get("task_board", {}) or {}
        slow_signature = graph_context.get("slow_context_signature", {})
        self._slow_started = True
        self._last_event_trigger_time = sim_time
        self._slow_trigger_snapshot = {
            "target_windows": dict(slow_signature.get("target_windows", {})),
            "task_board_units": set(task_board.keys()),
            "seen_high_value_targets": set(slow_signature.get("seen_high_value_targets", [])),
        }

    def _get_current_slow_context_signature(self, state: GameState, sim_time: int) -> Optional[dict]:
        return self._ensure_graph_context(state, sim_time).get("slow_context_signature", {})

    def parse_system2_actions(self, actions_json: List[dict], state: GameState) -> List[Action]:
        if actions_json:
            try:
                pretty = json.dumps({"actions": actions_json}, ensure_ascii=False, indent=2)
                self._ui_show(f"\n--- [System 2 async actions] ---\n{pretty}")
            except Exception:
                pass
        return self._parse_actions_json(actions_json, state)

    def _ui_show(self, text: str):
        try:
            if self.faction:
                self.faction.show_llm_msg(text)
        except Exception:
            logger.debug("[RED] show_llm_msg failed", exc_info=True)

    def filter_actions_before_submit(self, actions: List[Action], state: GameState, sim_time: int) -> tuple[List[Action], dict]:
        if not self.red_endgame_launch_guard_enabled or self.red_endgame_stop_simtime <= 0:
            return actions, {"dropped_count": 0, "drop_reasons": {}}

        remaining_simtime = float(self.red_endgame_stop_simtime) - float(sim_time)
        if remaining_simtime <= 0:
            return actions, {"dropped_count": 0, "drop_reasons": {}}

        filtered: List[Action] = []
        dropped = Counter()
        examples = []
        for action in actions:
            drop_reason = self._late_attack_drop_reason(action, state, remaining_simtime)
            if drop_reason:
                dropped[drop_reason] += 1
                if len(examples) < 3:
                    examples.append(self._action_summary_text(action))
                continue
            filtered.append(action)

        if dropped:
            logger.info(
                "[RED] endgame_launch_guard sim_time=%s remaining=%.1fs dropped=%s examples=%s",
                sim_time,
                remaining_simtime,
                dict(dropped),
                examples,
            )
        return filtered, {"dropped_count": sum(dropped.values()), "drop_reasons": dict(dropped)}

    def _late_attack_drop_reason(self, action: Action, state: GameState, remaining_simtime: float) -> Optional[str]:
        payload = self._action_payload(action)
        if not isinstance(payload, dict):
            return None
        if str(payload.get("Type") or "").strip() != "Launch":
            return None

        weapon_type = str(payload.get("WeaponType") or "").strip()
        speed_mps = RED_ATTACK_WEAPON_SPEED_MPS.get(weapon_type)
        if speed_mps is None:
            return None

        unit_id = str(payload.get("Id") or "").strip()
        if not unit_id:
            return None
        firing_pos = get_unit_position(state=state, unit_id=unit_id)
        if firing_pos is None:
            return None

        target_pos = self._payload_target_position(payload)
        if target_pos is None:
            return None

        distance_m = distance_between_positions(firing_pos, target_pos)
        if distance_m is None or distance_m <= 0:
            return None

        estimated_impact_seconds = float(distance_m) / float(speed_mps)
        if estimated_impact_seconds + self.red_endgame_min_impact_margin_seconds > remaining_simtime:
            return "impact_after_stop"
        return None

    def _payload_target_position(self, payload: dict):
        try:
            lon = float(payload.get("Lon"))
            lat = float(payload.get("Lat"))
            alt = float(payload.get("Alt", 0) or 0)
            return Position(lat=lat, lon=lon, alt=alt)
        except Exception:
            return None

    def _action_payload(self, action: Action):
        if isinstance(action, dict):
            return action
        if callable(getattr(action, "to_cmd_dict", None)):
            try:
                return action.to_cmd_dict()
            except Exception:
                return None
        return None

    def _action_summary_text(self, action: Action) -> str:
        payload = self._action_payload(action)
        if not isinstance(payload, dict):
            return str(action)
        return (
            f"{payload.get('Type')}:{payload.get('Id')}->{payload.get('WeaponType')}"
            f"@({payload.get('Lon')},{payload.get('Lat')})"
        )

    def _summarize_state(self, state: GameState) -> str:
        summary = self.summarizer.summarize_state(state, self.faction_name)
        failed_events = []
        if hasattr(state, "recent_events"):
            try:
                failed_events = [e for e in state.recent_events() if e.get("name") == "RED_FIRE_FAILED"]
            except Exception as e:
                logger.warning("[%s] recent_events() failed: %s", self.faction_name, e)

        if failed_events:
            failed_count = len(failed_events)
            units = ", ".join(ev.get("unit_id", "?") for ev in failed_events[-3:])
            summary += (
                f"\nRecent fire failures: {failed_count}. Example units: {units}."
                "\nPossible causes: out of range / terrain masking / bad coordinates."
                "\nSuggestion: close in before firing."
            )
        return summary

    def _expand_tasks_to_actions(self, tasks, state: GameState) -> List[Action]:
        actions = []
        for task in tasks or []:
            try:
                result = task.run(state)
                if not result:
                    continue
                if isinstance(result, list):
                    actions.extend(result)
                else:
                    actions.append(result)
            except Exception as e:
                logger.warning("[%s] Mission.run failed: %s", self.faction_name, e, exc_info=True)
        return actions

    def _trace_id(self):
        context = getattr(self, "_active_trace_context", None) or {}
        return context.get("trace_id")

    def _ensure_graph_context(self, state: GameState, sim_time: int) -> dict:
        if self._latest_graph_context_sim_time == sim_time and self._latest_graph_context:
            return self._latest_graph_context
        self._prune_task_board(sim_time, state)
        target_table = self._build_target_table(state, sim_time)
        engagement_payload = self._inject_engagement_memory(target_table, sim_time)
        unit_roster = self._build_unit_roster(state, sim_time, target_table)
        task_board = self._snapshot_task_board()
        slow_context_signature = self._build_slow_context_signature(target_table, task_board)
        self._latest_graph_context = {
            "unit_roster": unit_roster,
            "target_table": target_table,
            "task_board": task_board,
            "engagement_summary": engagement_payload.get("engagement_summary", ""),
            "engagement_memory": engagement_payload.get("engagement_memory", []),
            "repeat_high_cost_targets": engagement_payload.get("repeat_high_cost_targets", []),
            "pending_bda_targets": engagement_payload.get("pending_bda_targets", []),
            "recent_attack_cost": engagement_payload.get("recent_attack_cost", 0.0),
            "slow_context_signature": slow_context_signature,
        }
        self._latest_graph_context_sim_time = sim_time
        return self._latest_graph_context

    def _build_slow_context_signature(self, target_table: List[dict], task_board: dict) -> dict:
        return {
            "target_windows": {
                str(target.get("target_id")): str(target.get("attack_window", ""))
                for target in target_table
                if isinstance(target, dict) and str(target.get("target_id", "")).strip()
            },
            "seen_high_value_targets": sorted(
                str(target.get("target_id"))
                for target in target_table
                if isinstance(target, dict) and str(target.get("target_type")) in {"Flagship_Surface", "Cruiser_Surface"}
            ),
            "task_board_units": sorted(str(unit_id) for unit_id in task_board.keys()),
        }

    def _build_graph_context(self, state: GameState, sim_time: int, memory_packet: dict) -> dict:
        return self._ensure_graph_context(state, sim_time)

    def _inject_engagement_memory(self, target_table: List[dict], sim_time: int) -> dict:
        if self.engagement_manager is None:
            return {
                "engagement_summary": "",
                "engagement_memory": [],
                "repeat_high_cost_targets": [],
                "pending_bda_targets": [],
                "recent_attack_cost": 0.0,
            }
        payload = self.engagement_manager.build_prompt_payload(sim_time=sim_time, target_table=target_table)
        memory_by_target = {
            str(item.get("target_id")): item
            for item in payload.get("engagement_memory", [])
            if isinstance(item, dict) and str(item.get("target_id", "")).strip()
        }
        for target in target_table:
            if not isinstance(target, dict):
                continue
            memory = memory_by_target.get(str(target.get("target_id", "")))
            if not memory:
                continue
            target["recent_high_cost_launches"] = memory.get("recent_high_cost_launches", 0)
            target["recent_low_cost_launches"] = memory.get("recent_low_cost_launches", 0)
            target["recent_attack_cost"] = memory.get("recent_attack_cost", 0.0)
            target["recent_score_gain"] = memory.get("recent_score_gain", 0.0)
            target["guidance_used_recently"] = memory.get("guidance_used_recently", False)
            target["pending_bda"] = memory.get("pending_bda", False)
            target["repeat_high_cost_risk"] = memory.get("repeat_high_cost_risk", "none")
            target["engagement_recommendation"] = memory.get("engagement_recommendation", "")
        return payload

    def _run_fast_layer(self, state: GameState, sim_time: int, graph_context: dict) -> tuple[List[Action], List[dict], List[str]]:
        target_table = graph_context.get("target_table", []) or []
        unit_roster = graph_context.get("unit_roster", []) or []
        if not unit_roster:
            return [], [], []

        active_roles = self._active_fast_roles(sim_time)
        if (
            sim_time > self._opening_window_seconds
            and {"fire", "guide", "scout"} <= active_roles
        ):
            return [], [], []

        actions: List[Action] = []
        planned_tasks: List[dict] = []
        reasons: List[str] = []
        used_units = set()

        if "scout" not in active_roles:
            scout_actions, scout_tasks = self._build_fast_scout_actions(state, sim_time, unit_roster, target_table, used_units)
            if scout_actions:
                actions.extend(scout_actions)
                planned_tasks.extend(scout_tasks)
                reasons.append("opening_scout")
                for task in scout_tasks:
                    used_units.update(task.get("unit_ids", []))

        top_target = target_table[0] if target_table else None
        if top_target and "guide" not in active_roles:
            guide_actions, guide_tasks = self._build_fast_guide_actions(state, sim_time, unit_roster, top_target, used_units)
            if guide_actions:
                actions.extend(guide_actions)
                planned_tasks.extend(guide_tasks)
                reasons.append("fast_guidance")
                for task in guide_tasks:
                    used_units.update(task.get("unit_ids", []))
        guide_started_this_tick = bool(
            top_target
            and str(top_target.get("target_type") or "") in SURFACE_TARGET_TYPES
            and any(str(task.get("task_type") or "") == "GuideAttack" for task in planned_tasks)
        )

        if top_target and "fire" not in active_roles:
            if guide_started_this_tick:
                reasons.append("wait_for_guidance_settle")
                return actions, planned_tasks, reasons
            fire_actions, fire_tasks, fire_reason = self._build_fast_fire_actions(
                state,
                sim_time,
                unit_roster,
                top_target,
                used_units,
            )
            if fire_actions:
                actions.extend(fire_actions)
                planned_tasks.extend(fire_tasks)
                reasons.append(fire_reason)
                for task in fire_tasks:
                    used_units.update(task.get("unit_ids", []))

        return actions, planned_tasks, reasons

    def _active_fast_roles(self, sim_time: int) -> set:
        roles = set()
        for task in self.red_task_board.values():
            if not isinstance(task, dict):
                continue
            expires_at = int(task.get("expires_at", 0) or 0)
            if expires_at and sim_time >= expires_at:
                continue
            role = str(task.get("role", "")).strip()
            if role:
                roles.add(role)
        return roles

    def _available_units(self, unit_roster: List[dict], role: str, used_units: set) -> List[dict]:
        return [
            unit
            for unit in unit_roster
            if isinstance(unit, dict)
            and unit.get("role") == role
            and unit.get("alive", True)
            and unit.get("available", True)
            and str(unit.get("unit_id", "")) not in used_units
        ]

    def _build_fast_scout_actions(
        self,
        state: GameState,
        sim_time: int,
        unit_roster: List[dict],
        target_table: List[dict],
        used_units: set,
    ) -> tuple[List[Action], List[dict]]:
        scout_units = self._available_units(unit_roster, role="scout", used_units=used_units)
        if not scout_units:
            return [], []

        areas = self._build_fast_scout_areas(unit_roster, target_table, count=min(2, len(scout_units)))
        actions: List[Action] = []
        planned_tasks: List[dict] = []
        for unit, area in zip(scout_units, areas):
            mission = MissionScout(
                id=f"{self.faction_name}_fast_scout_{sim_time}_{unit['unit_id']}",
                unit_ids=[unit["unit_id"]],
                area=area,
                sim_start_time=sim_time,
                sim_end_time=sim_time + 180,
            )
            mission_actions = self._expand_tasks_to_actions(mission.run(state), state)
            if not mission_actions:
                continue
            actions.extend(mission_actions)
            planned_tasks.append(
                {
                    "task_type": "ScoutArea",
                    "unit_ids": [unit["unit_id"]],
                    "target_id": "",
                    "role": "scout",
                }
            )
        return actions, planned_tasks

    def _build_fast_guide_actions(
        self,
        state: GameState,
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

        guide_unit = guide_units[0]
        mission = MissionGuideAttack(
            id=f"{self.faction_name}_fast_guide_{sim_time}_{guide_unit['unit_id']}",
            unit_ids=[guide_unit["unit_id"]],
            target_id=top_target["target_id"],
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
                "unit_ids": [guide_unit["unit_id"]],
                "target_id": top_target["target_id"],
                "role": "guide",
            }
        ]

    def _build_fast_fire_actions(
        self,
        state: GameState,
        sim_time: int,
        unit_roster: List[dict],
        top_target: dict,
        used_units: set,
    ) -> tuple[List[Action], List[dict], str]:
        target_id = str(top_target.get("target_id", "")).strip()
        attack_window = str(top_target.get("attack_window", "")).strip()
        if not target_id or attack_window == "observe_only":
            return [], [], ""
        target_type = str(top_target.get("target_type") or "")
        track_staleness_sec = float(top_target.get("track_staleness_sec") or 0.0)
        low_cost_policy = str(top_target.get("low_cost_surface_policy") or "")
        guidance_recently = bool(top_target.get("guidance_used_recently"))
        if target_type in SURFACE_TARGET_TYPES:
            if track_staleness_sec >= self.red_low_cost_track_stale_threshold_seconds:
                return [], [], ""
            if low_cost_policy in {"refresh_and_guide_first", "guide_then_consider_low_cost", "move_closer_then_consider_low_cost"} and not guidance_recently:
                return [], [], ""

        available_fire_units = {
            str(unit.get("unit_id")): unit
            for unit in self._available_units(unit_roster, role="fire", used_units=used_units)
        }
        if not available_fire_units:
            return [], [], ""

        if attack_window == "fire_now" and self.red_opening_enable_probe_fire:
            candidate_ids = [
                unit_id
                for unit_id in top_target.get("in_range_trucks", []) or []
                if unit_id in available_fire_units
            ]
            candidate_ids.sort(
                key=lambda unit_id: (
                    0 if available_fire_units[unit_id].get("can_fire_now") else 1,
                    -int(available_fire_units[unit_id].get("ammo_high_cost", 0)),
                    -int(available_fire_units[unit_id].get("ammo_low_cost", 0)),
                )
            )
            selected_ids = candidate_ids[:2]
            if not selected_ids:
                return [], [], ""
            mission = MissionFocusFire(
                id=f"{self.faction_name}_fast_probe_{sim_time}_{target_id}",
                unit_ids=selected_ids,
                target_id=target_id,
                sim_start_time=sim_time,
                sim_end_time=sim_time + 180,
            )
            actions = self._expand_tasks_to_actions(mission.run(state), state)
            if not actions:
                return [], [], ""
            return (
                actions,
                [
                    {
                        "task_type": "FocusFire",
                        "unit_ids": selected_ids,
                        "target_id": target_id,
                        "role": "fire",
                    }
                ],
                "probe_fire",
            )

        if attack_window == "move_then_fire":
            candidate_ids = [
                unit_id
                for unit_id in top_target.get("move_then_fire_trucks", []) or []
                if unit_id in available_fire_units
            ]
            if not candidate_ids:
                candidate_ids = [
                    unit_id
                    for unit_id, unit in available_fire_units.items()
                    if target_id in (unit.get("candidate_targets") or [])
                ]
            selected_ids = candidate_ids[:2]
            if not selected_ids:
                return [], [], ""
            mission = MissionMoveToEngage(
                id=f"{self.faction_name}_fast_move_{sim_time}_{target_id}",
                unit_ids=selected_ids,
                target_id=target_id,
                sim_start_time=sim_time,
                sim_end_time=sim_time + 180,
            )
            actions = self._expand_tasks_to_actions(mission.run(state), state)
            if not actions:
                return [], [], ""
            return (
                actions,
                [
                    {
                        "task_type": "MoveToEngage",
                        "unit_ids": selected_ids,
                        "target_id": target_id,
                        "role": "fire",
                    }
                ],
                "fast_move_then_fire",
            )

        return [], [], ""

    def _remember_fast_tasks(self, planned_tasks: List[dict], sim_time: int) -> None:
        for task in planned_tasks:
            if not isinstance(task, dict):
                continue
            task_type = str(task.get("task_type", "")).strip()
            ttl = RED_TASK_TTLS.get(task_type, 45)
            role = str(task.get("role", "")).strip() or self._role_for_action_type(task_type)
            target_id = str(task.get("target_id", "")).strip()
            for unit_id in task.get("unit_ids", []) or []:
                if not isinstance(unit_id, str) or not unit_id:
                    continue
                self.red_task_board[unit_id] = {
                    "task_type": task_type,
                    "target_id": target_id,
                    "role": role,
                    "assigned_at": sim_time,
                    "expires_at": sim_time + ttl,
                    "reassignable": True,
                    "source_trace_id": f"fast@{sim_time}",
                }
        if self._latest_graph_context_sim_time == sim_time and self._latest_graph_context:
            self._latest_graph_context["task_board"] = self._snapshot_task_board()

    def _build_fast_scout_areas(self, unit_roster: List[dict], target_table: List[dict], count: int) -> List[Area]:
        centers = []
        for target in target_table[:count]:
            pos = target.get("position") if isinstance(target, dict) else None
            if isinstance(pos, dict) and pos.get("lon") is not None and pos.get("lat") is not None:
                centers.append((float(pos["lon"]), float(pos["lat"])))

        if len(centers) < count:
            centroid = self._friendly_centroid(unit_roster)
            if centroid is not None:
                base_lon, base_lat = centroid
                offsets = [(0.45, 0.30), (0.75, 0.0), (0.45, -0.30), (-0.15, 0.0)]
                for delta_lon, delta_lat in offsets:
                    centers.append((base_lon + delta_lon, base_lat + delta_lat))
                    if len(centers) >= count:
                        break

        areas = []
        for lon, lat in centers[:count]:
            areas.append(self._area_around(lon, lat, half_lon=0.25, half_lat=0.18))
        return areas

    def _friendly_centroid(self, unit_roster: List[dict]) -> Optional[tuple[float, float]]:
        points = []
        for unit in unit_roster or []:
            pos = unit.get("position") if isinstance(unit, dict) else None
            if not isinstance(pos, dict):
                continue
            lon = pos.get("lon")
            lat = pos.get("lat")
            if lon is None or lat is None:
                continue
            points.append((float(lon), float(lat)))
        if not points:
            return None
        return (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )

    def _area_around(self, center_lon: float, center_lat: float, half_lon: float, half_lat: float) -> Area:
        top_left = Position(
            lon=max(min(center_lon - half_lon, 180.0), -180.0),
            lat=max(min(center_lat + half_lat, 90.0), -90.0),
        )
        bottom_right = Position(
            lon=max(min(center_lon + half_lon, 180.0), -180.0),
            lat=max(min(center_lat - half_lat, 90.0), -90.0),
        )
        return Area(top_left, bottom_right)

    def record_system2_assignments(self, parse_context: dict, sim_time: int, trace_id: Optional[str]):
        accepted_actions = parse_context.get("accepted_actions_json") or []
        if not accepted_actions:
            return

        self._prune_task_board(sim_time)
        for action in accepted_actions:
            if not isinstance(action, dict):
                continue
            action_type = str(action.get("Type", "")).strip()
            ttl = RED_TASK_TTLS.get(action_type)
            if ttl is None:
                continue
            target_id = str(action.get("Target_Id", "")).strip()
            role = self._role_for_action_type(action_type)
            reassignable = action_type == "ScoutArea"
            for unit_id in action.get("UnitIds", []) or []:
                if not isinstance(unit_id, str) or not unit_id:
                    continue
                self.red_task_board[unit_id] = {
                    "task_type": action_type,
                    "target_id": target_id,
                    "role": role,
                    "assigned_at": sim_time,
                    "expires_at": sim_time + ttl,
                    "reassignable": reassignable,
                    "source_trace_id": trace_id or "",
                }
        if self._latest_graph_context:
            self._latest_graph_context["task_board"] = self._snapshot_task_board()

    def _role_for_action_type(self, action_type: str) -> str:
        if action_type in RED_FIRE_ACTIONS or action_type == "MoveToEngage":
            return "fire"
        if action_type == "GuideAttack":
            return "guide"
        if action_type == "ScoutArea":
            return "scout"
        return "support"

    def _snapshot_task_board(self) -> Dict[str, dict]:
        snapshot: Dict[str, dict] = {}
        for unit_id, task in self.red_task_board.items():
            if not isinstance(task, dict):
                continue
            snapshot[unit_id] = dict(task)
        return snapshot

    def _prune_task_board(self, sim_time: int, state: Optional[GameState] = None):
        valid_units = None
        if state is not None:
            valid_units = {str(getattr(unit, "id", "")) for unit in getattr(state, "unit_states", []) or []}
        expired = []
        for unit_id, task in self.red_task_board.items():
            if not isinstance(task, dict):
                expired.append(unit_id)
                continue
            expires_at = int(task.get("expires_at", 0))
            if expires_at and sim_time >= expires_at:
                expired.append(unit_id)
                continue
            if valid_units is not None and unit_id not in valid_units:
                expired.append(unit_id)
                continue
            target_id = str(task.get("target_id", "")).strip()
            if state is not None and target_id:
                try:
                    if state.get_target_position(target_id) is None:
                        expired.append(unit_id)
                        continue
                except Exception:
                    pass
        for unit_id in expired:
            self.red_task_board.pop(unit_id, None)

    def _build_target_table(self, state: GameState, sim_time: int) -> List[dict]:
        detected_ids = get_detected_target_ids(state)

        trucks = self._collect_truck_info(state)
        guide_candidates = self._collect_guide_candidates(state)
        targets = []
        for target_id in detected_ids:
            target_type = infer_unit_type(target_id) or str(target_id).split("-")[0]
            if target_type not in SURFACE_TARGET_TYPES:
                continue

            target_pos = None
            try:
                target_pos = state.get_target_position(target_id)
                if callable(target_pos):
                    target_pos = target_pos()
            except Exception:
                target_pos = None

            in_range_trucks = []
            move_then_fire_trucks = []
            min_distance_low_cost_km = None
            min_distance_high_cost_km = None
            for truck in trucks:
                if target_pos is None or truck["max_range_km"] <= 0:
                    continue
                dist_m = distance_between_positions(truck["position"], target_pos)
                if dist_m is None:
                    if not self._distance_api_warning_emitted:
                        logger.warning(
                            "[RED] target window distance calc unavailable: truck_pos lacks distance/distance_to; "
                            "skipping affected truck-target pairs."
                        )
                        self._distance_api_warning_emitted = True
                    continue
                dist_km = float(dist_m) / 1000.0
                if int(truck.get("ammo_low_cost", 0)) > 0 and (
                    min_distance_low_cost_km is None or dist_km < min_distance_low_cost_km
                ):
                    min_distance_low_cost_km = dist_km
                if int(truck.get("ammo_high_cost", 0)) > 0 and (
                    min_distance_high_cost_km is None or dist_km < min_distance_high_cost_km
                ):
                    min_distance_high_cost_km = dist_km
                if dist_km <= truck["max_range_km"]:
                    in_range_trucks.append(truck["unit_id"])
                elif dist_km <= truck["max_range_km"] + 120:
                    move_then_fire_trucks.append(truck["unit_id"])

            if in_range_trucks:
                attack_window = "fire_now"
            elif move_then_fire_trucks:
                attack_window = "move_then_fire"
            else:
                attack_window = "observe_only"

            value = 150 if target_type == "Flagship_Surface" else 80 if target_type == "Cruiser_Surface" else 50
            priority = 1 if target_type == "Flagship_Surface" else 2 if target_type == "Cruiser_Surface" else 3
            last_seen_time = (
                self.summarizer.last_seen_targets.get(self.faction_name, {}).get(target_id, sim_time)
                if hasattr(self.summarizer, "last_seen_targets")
                else sim_time
            )
            try:
                track_staleness_sec = max(0, int(sim_time) - int(last_seen_time))
            except Exception:
                track_staleness_sec = 0
            min_low_cost_flight_time_sec = (
                round((float(min_distance_low_cost_km) * 1000.0) / RED_ATTACK_WEAPON_SPEED_MPS["LowCostAttackMissile"], 1)
                if min_distance_low_cost_km is not None
                else None
            )
            min_high_cost_flight_time_sec = (
                round((float(min_distance_high_cost_km) * 1000.0) / RED_ATTACK_WEAPON_SPEED_MPS["HighCostAttackMissile"], 1)
                if min_distance_high_cost_km is not None
                else None
            )
            low_cost_surface_policy = self._surface_low_cost_policy(
                track_staleness_sec=track_staleness_sec,
                min_low_cost_flight_time_sec=min_low_cost_flight_time_sec,
                has_guidance=bool(guide_candidates),
            )
            targets.append(
                {
                    "target_id": target_id,
                    "target_type": target_type,
                    "value": value,
                    "priority": priority,
                    "position": self._position_dict(target_pos),
                    "visible": True,
                    "last_seen_time": last_seen_time,
                    "track_staleness_sec": track_staleness_sec,
                    "in_range_trucks": in_range_trucks,
                    "move_then_fire_trucks": move_then_fire_trucks,
                    "guide_candidates": guide_candidates,
                    "attack_window": attack_window,
                    "min_distance_low_cost_km": min_distance_low_cost_km,
                    "min_distance_high_cost_km": min_distance_high_cost_km,
                    "min_low_cost_flight_time_sec": min_low_cost_flight_time_sec,
                    "min_high_cost_flight_time_sec": min_high_cost_flight_time_sec,
                    "surface_motion_risk": "high",
                    "low_cost_surface_policy": low_cost_surface_policy,
                    "low_cost_surface_recommended": low_cost_surface_policy == "guided_low_cost_viable",
                }
            )

        targets.sort(
            key=lambda item: (
                0 if item["attack_window"] == "fire_now" else 1 if item["attack_window"] == "move_then_fire" else 2,
                item["priority"],
                -item["value"],
            )
        )
        return targets

    def _surface_low_cost_policy(
        self,
        *,
        track_staleness_sec: int,
        min_low_cost_flight_time_sec: Optional[float],
        has_guidance: bool,
    ) -> str:
        if min_low_cost_flight_time_sec is None:
            return "move_closer_then_consider_low_cost"
        if track_staleness_sec >= self.red_low_cost_track_stale_threshold_seconds and has_guidance:
            return "refresh_and_guide_first"
        if track_staleness_sec >= self.red_low_cost_track_stale_threshold_seconds:
            return "refresh_track_first"
        if min_low_cost_flight_time_sec > self.red_low_cost_max_flight_time_seconds and has_guidance:
            return "guide_then_consider_low_cost"
        if min_low_cost_flight_time_sec > self.red_low_cost_max_flight_time_seconds:
            return "move_closer_then_consider_low_cost"
        if has_guidance:
            return "guided_low_cost_viable"
        return "unguided_low_cost_risky"

    def _build_unit_roster(self, state: GameState, sim_time: int, target_table: List[dict]) -> List[dict]:
        roster = []
        target_windows = {t["target_id"]: t for t in target_table}
        for unit in self._friendly_units(state):
            unit_id = str(getattr(unit, "id", ""))
            unit_type = get_unit_type_name(unit) or infer_unit_type(unit_id) or unit_id.split("-")[0]
            role = self._role_for_unit(unit_type)
            if role is None:
                continue

            pos = self._unit_position(state, unit_id, unit)
            ammo_high, ammo_low = self._truck_ammo(unit)
            task = self.red_task_board.get(unit_id, {})
            alive = self._unit_alive(state, unit, unit_id)
            available = alive and not (task and not task.get("reassignable", False) and int(task.get("expires_at", 0)) > sim_time)
            candidate_targets = self._candidate_targets_for_unit(unit_id, role, pos, target_table)
            can_fire_now = role == "fire" and any(unit_id in target_windows[target_id].get("in_range_trucks", []) for target_id in candidate_targets)

            roster.append(
                {
                    "unit_id": unit_id,
                    "unit_type": unit_type,
                    "role": role,
                    "alive": alive,
                    "available": available,
                    "ammo_high_cost": ammo_high,
                    "ammo_low_cost": ammo_low,
                    "position": self._position_dict(pos),
                    "last_task": task.get("task_type"),
                    "last_task_target": task.get("target_id"),
                    "last_task_assigned_at": task.get("assigned_at"),
                    "can_fire_now": can_fire_now,
                    "candidate_targets": candidate_targets,
                }
            )
        return roster

    def _friendly_units(self, state: GameState) -> List[object]:
        return list(iter_platform_units(state))

    def _collect_truck_info(self, state: GameState) -> List[dict]:
        trucks = []
        for unit in self._friendly_units(state):
            unit_id = str(getattr(unit, "id", ""))
            if (get_unit_type_name(unit) or "").strip() != "Truck_Ground":
                continue
            pos = self._unit_position(state, unit_id, unit)
            ammo_high, ammo_low = self._truck_ammo(unit)
            max_range_km = 400 if ammo_high > 0 else 350 if ammo_low > 0 else 0
            if pos is None:
                continue
            trucks.append(
                {
                    "unit_id": unit_id,
                    "position": pos,
                    "ammo_high_cost": ammo_high,
                    "ammo_low_cost": ammo_low,
                    "max_range_km": max_range_km,
                }
            )
        return trucks

    def _collect_guide_candidates(self, state: GameState) -> List[str]:
        ids = []
        for unit in self._friendly_units(state):
            unit_id = str(getattr(unit, "id", ""))
            if (get_unit_type_name(unit) or "").strip() == "Guide_Ship_Surface":
                ids.append(unit_id)
        return ids

    def _candidate_targets_for_unit(self, unit_id: str, role: str, pos, target_table: List[dict]) -> List[str]:
        if role == "fire":
            ordered = sorted(
                target_table,
                key=lambda item: (
                    0 if unit_id in item.get("in_range_trucks", []) else 1 if item.get("attack_window") == "move_then_fire" else 2,
                    item.get("priority", 99),
                ),
            )
            return [item["target_id"] for item in ordered[:3]]
        return [item["target_id"] for item in target_table[:3]]

    def _role_for_unit(self, unit_type: str) -> Optional[str]:
        if unit_type == "Truck_Ground":
            return "fire"
        if unit_type == "Guide_Ship_Surface":
            return "guide"
        if unit_type == "Recon_UAV_FixWing":
            return "scout"
        return None

    def _position_dict(self, pos) -> Optional[dict]:
        if pos is None:
            return None
        try:
            return {"lon": float(pos.lon), "lat": float(pos.lat)}
        except Exception:
            return None

    def _unit_position(self, state: GameState, unit_id: str, unit) -> Optional[object]:
        return get_unit_position(state=state, unit_id=unit_id, unit=unit)

    def _truck_ammo(self, unit) -> tuple[int, int]:
        inventory = get_weapon_inventory(unit)
        return inventory.get("HighCostAttackMissile", 0), inventory.get("LowCostAttackMissile", 0)

    def _unit_alive(self, state: GameState, unit, unit_id: str) -> bool:
        hp = get_unit_hp(state=state, unit_id=unit_id, unit=unit)
        return hp is None or float(hp) > 0

    def _existing_guide_units_by_target(self, task_board: dict) -> Dict[str, List[str]]:
        guides: Dict[str, List[str]] = {}
        for unit_id, task in (task_board or {}).items():
            if not isinstance(task, dict):
                continue
            if str(task.get("task_type") or "") != "GuideAttack":
                continue
            target_id = str(task.get("target_id") or "").strip()
            if not target_id:
                continue
            guides.setdefault(target_id, []).append(str(unit_id))
        return guides

    def _engagement_target_meta(self, target_table: List[dict]) -> Dict[str, dict]:
        return {
            str(target.get("target_id")): target
            for target in (target_table or [])
            if isinstance(target, dict) and str(target.get("target_id", "")).strip()
        }

    def _launch_actions_summary(self, mission_actions: List[Action], state: GameState, target_id: str, sim_time: int) -> dict:
        target_pos = get_target_position(state, target_id)
        launches = []
        high_cost_launch_count = 0
        low_cost_launch_count = 0
        estimated_attack_cost = 0.0
        estimated_impact_until = 0.0
        for action in mission_actions or []:
            payload = self._action_payload(action)
            if not isinstance(payload, dict):
                continue
            if str(payload.get("Type") or "").strip() != "Launch":
                continue
            weapon_type = str(payload.get("WeaponType") or payload.get("weapon_type") or "").strip()
            if weapon_type not in RED_ATTACK_WEAPON_SPEED_MPS:
                continue
            if weapon_type == "HighCostAttackMissile":
                high_cost_launch_count += 1
            elif weapon_type == "LowCostAttackMissile":
                low_cost_launch_count += 1
            estimated_cost = float(RED_ATTACK_WEAPON_COST.get(weapon_type, 0.0))
            estimated_attack_cost += estimated_cost
            impact_until = 0.0
            if target_pos is not None:
                firing_unit_id = str(payload.get("Id") or "").strip()
                firing_pos = get_unit_position(state=state, unit_id=firing_unit_id) if firing_unit_id else None
                distance_m = distance_between_positions(firing_pos, target_pos) if firing_pos is not None else None
                speed_mps = RED_ATTACK_WEAPON_SPEED_MPS.get(weapon_type)
                if distance_m and speed_mps:
                    impact_until = float(sim_time) + float(distance_m) / float(speed_mps)
                    estimated_impact_until = max(estimated_impact_until, impact_until)
            launches.append(
                {
                    "action_id": id(action),
                    "weapon_type": weapon_type,
                    "estimated_cost": estimated_cost,
                    "impact_until": impact_until,
                }
            )
        return {
            "launch_actions": launches,
            "high_cost_launch_count": high_cost_launch_count,
            "low_cost_launch_count": low_cost_launch_count,
            "estimated_attack_cost": round(estimated_attack_cost, 2),
            "estimated_impact_until": estimated_impact_until,
        }

    def _build_engagement_event(
        self,
        *,
        action_type: str,
        act: dict,
        target_meta: Optional[dict],
        mission_actions: List[Action],
        state: GameState,
        sim_time: int,
        guide_units_by_target: Dict[str, List[str]],
    ) -> Optional[dict]:
        target_id = str(act.get("Target_Id") or "").strip()
        if not target_id:
            return None
        guide_unit_ids = sorted({str(unit_id) for unit_id in guide_units_by_target.get(target_id, []) if str(unit_id).strip()})
        track_staleness_sec = 0
        attack_window = ""
        target_type = infer_unit_type(target_id) or target_id.split("-")[0]
        if isinstance(target_meta, dict):
            target_type = str(target_meta.get("target_type") or target_type)
            attack_window = str(target_meta.get("attack_window") or "")
            try:
                track_staleness_sec = int(target_meta.get("track_staleness_sec") or 0)
            except Exception:
                track_staleness_sec = 0
        launch_summary = self._launch_actions_summary(mission_actions, state, target_id, sim_time)
        guidance_used = bool(guide_unit_ids) or action_type == "GuideAttack"
        scout_support_recommended = track_staleness_sec >= 90 or attack_window == "observe_only"
        if action_type not in {"GuideAttack", "MoveToEngage"} and not launch_summary["launch_actions"]:
            return None
        return {
            "target_id": target_id,
            "target_type": target_type,
            "action_type": action_type,
            "unit_ids": [str(unit_id) for unit_id in act.get("UnitIds", []) or [] if str(unit_id).strip()],
            "guide_unit_ids": guide_unit_ids,
            "guidance_used": guidance_used,
            "scout_support_recommended": scout_support_recommended,
            "track_staleness_sec": track_staleness_sec,
            "attack_window": attack_window,
            **launch_summary,
        }

    def _parse_actions_json(self, actions_json: List[dict], state: GameState) -> List[Action]:
        parsed_actions: List[Action] = []
        ignored_counter = Counter()
        if not actions_json:
            self._trace_step(
                self._trace_id(),
                "Parsed Actions",
                json_action_count=0,
                json_summary="none",
                parsed_action_count=0,
                parsed_summary="none",
                ignored_summary={},
            )
            return parsed_actions

        sim_time = state.simtime()
        validation_context = {
            "allocation_plan": (self._active_trace_context or {}).get("allocation_plan", {}),
            "unit_roster": (self._active_trace_context or {}).get("unit_roster", []),
            "target_table": (self._active_trace_context or {}).get("target_table", []),
            "task_board": (self._active_trace_context or {}).get("task_board", {}),
        }
        target_meta_by_id = self._engagement_target_meta(validation_context.get("target_table", []))
        filtered_actions_json, semantic_ignored = filter_red_actions_semantically(
            actions_json,
            validation_context=validation_context,
        )
        semantic_force_replan = bool(actions_json and not filtered_actions_json and semantic_ignored)
        trace_id = self._trace_id()
        guide_units_by_target = self._existing_guide_units_by_target(validation_context.get("task_board", {}))
        for accepted_action in filtered_actions_json:
            if not isinstance(accepted_action, dict):
                continue
            if str(accepted_action.get("Type") or "").strip() != "GuideAttack":
                continue
            target_id = str(accepted_action.get("Target_Id") or "").strip()
            if not target_id:
                continue
            guide_units_by_target.setdefault(target_id, [])
            for unit_id in accepted_action.get("UnitIds", []) or []:
                unit_id_text = str(unit_id).strip()
                if unit_id_text:
                    guide_units_by_target[target_id].append(unit_id_text)
        for target_id, unit_ids in guide_units_by_target.items():
            guide_units_by_target[target_id] = sorted(set(unit_ids))
        engagement_events: List[dict] = []

        if self._active_trace_context is not None:
            self._active_trace_context["semantic_reject_summary"] = dict(semantic_ignored)
            self._active_trace_context["semantic_force_replan"] = semantic_force_replan
            self._active_trace_context["semantic_rejected_count"] = len(actions_json) - len(filtered_actions_json)
            self._active_trace_context["accepted_actions_json"] = list(filtered_actions_json)
            self._active_trace_context["engagement_events"] = engagement_events

        if semantic_ignored:
            logger.info(
                "[RED] semantic_filter trace_id=%s json_action_count=%s rejected_count=%s reasons=%s force_replan=%s",
                trace_id,
                len(actions_json),
                len(actions_json) - len(filtered_actions_json),
                dict(semantic_ignored),
                semantic_force_replan,
            )

        for act in filtered_actions_json:
            if not isinstance(act, dict):
                ignored_counter["invalid_action_object"] += 1
                continue

            action_type = str(act.get("Type", "")).strip()
            t_lower = action_type.lower()

            try:
                if t_lower in {"move", "launch", "changestate", "setradar", "setjammer", "launchinterceptor"}:
                    ignored_counter["low_level_rejected"] += 1
                    logger.warning("[RED] ignored invalid low-level action: %s", act)
                    self._ui_show(f"[WARN] Ignored invalid RED low-level action: {action_type}")
                    continue

                if t_lower == "scoutarea":
                    area_dict = act.get("Area")
                    unit_ids = act.get("UnitIds", [])
                    if not area_dict or not isinstance(unit_ids, list) or not unit_ids:
                        ignored_counter["scoutarea_missing_fields"] += 1
                        continue
                    top_left = area_dict.get("TopLeft")
                    bottom_right = area_dict.get("BottomRight")
                    if not top_left or not bottom_right:
                        ignored_counter["scoutarea_invalid_area"] += 1
                        continue
                    area = Area(Position(**top_left), Position(**bottom_right))
                    mission = MissionScout(
                        id=f"{self.faction_name}_scout_{sim_time}",
                        unit_ids=unit_ids,
                        area=area,
                        sim_start_time=sim_time,
                        sim_end_time=sim_time + 300,
                    )
                    mission_actions = self._expand_tasks_to_actions(mission.run(state), state)
                    parsed_actions.extend(mission_actions)
                    continue

                if t_lower == "movetoengage":
                    unit_ids = act.get("UnitIds", [])
                    target_id = act.get("Target_Id")
                    if not unit_ids or not target_id:
                        ignored_counter["movetoengage_missing_fields"] += 1
                        self._ui_show("[WARN] MoveToEngage missing UnitIds or Target_Id")
                        continue
                    mission = MissionMoveToEngage(
                        id=f"{self.faction_name}_movetoengage_{sim_time}",
                        unit_ids=unit_ids,
                        target_id=target_id,
                        sim_start_time=sim_time,
                        sim_end_time=sim_time + 300,
                    )
                    mission_actions = self._expand_tasks_to_actions(mission.run(state), state)
                    parsed_actions.extend(mission_actions)
                    engagement_event = self._build_engagement_event(
                        action_type="MoveToEngage",
                        act=act,
                        target_meta=target_meta_by_id.get(target_id),
                        mission_actions=mission_actions,
                        state=state,
                        sim_time=sim_time,
                        guide_units_by_target=guide_units_by_target,
                    )
                    if engagement_event:
                        engagement_events.append(engagement_event)
                    continue

                if t_lower == "guideattack":
                    unit_ids = act.get("UnitIds", [])
                    target_id = act.get("Target_Id")
                    radius_km = act.get("Loiter_Radius_km", 30)
                    if not unit_ids or not target_id:
                        ignored_counter["guideattack_missing_fields"] += 1
                        self._ui_show("[WARN] GuideAttack missing UnitIds or Target_Id")
                        continue
                    mission = MissionGuideAttack(
                        id=f"{self.faction_name}_guideattack_{sim_time}",
                        unit_ids=unit_ids,
                        target_id=target_id,
                        sim_start_time=sim_time,
                        sim_end_time=sim_time + 600,
                        loiter_radius_m=float(radius_km) * 1000.0,
                    )
                    mission_actions = self._expand_tasks_to_actions(mission.run(state), state)
                    parsed_actions.extend(mission_actions)
                    engagement_event = self._build_engagement_event(
                        action_type="GuideAttack",
                        act=act,
                        target_meta=target_meta_by_id.get(target_id),
                        mission_actions=mission_actions,
                        state=state,
                        sim_time=sim_time,
                        guide_units_by_target=guide_units_by_target,
                    )
                    if engagement_event:
                        engagement_events.append(engagement_event)
                    continue

                if t_lower == "focusfire":
                    target_id = act.get("Target_Id")
                    unit_ids = act.get("UnitIds", [])
                    if not target_id or not unit_ids:
                        ignored_counter["focusfire_missing_fields"] += 1
                        self._ui_show("[WARN] FocusFire missing UnitIds or Target_Id")
                        continue
                    mission = MissionFocusFire(
                        id=f"{self.faction_name}_focusfire_{sim_time}",
                        unit_ids=unit_ids,
                        target_id=target_id,
                        sim_start_time=sim_time,
                        sim_end_time=sim_time + 300,
                    )
                    mission_actions = self._expand_tasks_to_actions(mission.run(state), state)
                    parsed_actions.extend(mission_actions)
                    engagement_event = self._build_engagement_event(
                        action_type="FocusFire",
                        act=act,
                        target_meta=target_meta_by_id.get(target_id),
                        mission_actions=mission_actions,
                        state=state,
                        sim_time=sim_time,
                        guide_units_by_target=guide_units_by_target,
                    )
                    if engagement_event:
                        engagement_events.append(engagement_event)
                    continue

                if t_lower == "shootandscoot":
                    unit_ids = act.get("UnitIds", [])
                    target_id = act.get("Target_Id")
                    if not unit_ids or not target_id:
                        ignored_counter["shootandscoot_missing_fields"] += 1
                        self._ui_show("[WARN] ShootAndScoot missing UnitIds or Target_Id")
                        continue

                    move_area = None
                    area_dict = act.get("Move_Area")
                    if area_dict:
                        top_left = area_dict.get("TopLeft")
                        bottom_right = area_dict.get("BottomRight")
                        if top_left and bottom_right:
                            move_area = Area(Position(**top_left), Position(**bottom_right))
                    if move_area is None:
                        unit_state = state.get_unit_state(unit_ids[0])
                        if unit_state is None:
                            ignored_counter["shootandscoot_missing_unit"] += 1
                            continue
                        move_area = MissionMoveAndAttack.generate_area_around_position(unit_state.position())

                    mission = MissionMoveAndAttack(
                        id=f"{self.faction_name}_shootscoot_{sim_time}",
                        unit_ids=unit_ids,
                        area=move_area,
                        sim_start_time=sim_time,
                        sim_end_time=sim_time + 300,
                    )
                    mission.forced_target_id = target_id
                    mission_actions = self._expand_tasks_to_actions(mission.run(state), state)
                    parsed_actions.extend(mission_actions)
                    engagement_event = self._build_engagement_event(
                        action_type="ShootAndScoot",
                        act=act,
                        target_meta=target_meta_by_id.get(target_id),
                        mission_actions=mission_actions,
                        state=state,
                        sim_time=sim_time,
                        guide_units_by_target=guide_units_by_target,
                    )
                    if engagement_event:
                        engagement_events.append(engagement_event)
                    continue

                ignored_counter["unknown_type"] += 1
                logger.warning("[RED] unknown action ignored: %s", act)
                self._ui_show(f"[WARN] Unknown action ignored: {act}")

            except Exception as e:
                ignored_counter["parse_exception"] += 1
                logger.error("[%s] failed to parse action '%s': %s", self.faction_name, action_type, e, exc_info=True)

        combined_ignored = semantic_ignored + ignored_counter
        if self._active_trace_context is not None:
            self._active_trace_context["engagement_events"] = engagement_events
        self._trace_step(
            trace_id,
            "Parsed Actions",
            json_action_count=len(actions_json),
            json_summary=summarize_json_actions(actions_json),
            semantic_filtered_action_count=len(filtered_actions_json),
            parsed_action_count=len(parsed_actions),
            parsed_summary=summarize_engine_actions(parsed_actions),
            ignored_summary=dict(combined_ignored),
            force_replan=semantic_force_replan,
        )
        logger.info("[%s] parsed %d engine actions from %d JSON actions.", self.faction_name, len(parsed_actions), len(actions_json))
        return parsed_actions
