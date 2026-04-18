import json
import logging
import os
from threading import Event
from typing import List, Tuple

from jsqlsim import GameState
from jsqlsim.enum import FactionEnum, WeaponNameEnum
from jsqlsim.geo import Area, Position
from jsqlsim.world.action import (
    Action,
    JammerControlAction,
    LaunchInterceptorAction,
    MoveAction,
    RadarControlAction,
)
from jsqlsim.world.missions.mission_anti_missile import MissionAntiMissile
from jsqlsim.world.missions.mission_focus_fire import MissionFocusFire
from jsqlsim.world.missions.mission_move_and_attack import MissionMoveAndAttack
from jsqlsim.world.missions.mission_scout import MissionScout
from jsqlsim.world.unit_tasks.oneshot_task import OneShotTask

from base_commander import BaseCommander
from custom_actions import ChangeStateAction, RawPlatformAction
from llm_manager import LLMManager
from rule_book import RuleBook

logger = logging.getLogger(__name__)

VALID_BLUE_MODES = {"rule", "hybrid", "llm"}


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        return int(raw)
    except Exception:
        logger.warning("Invalid %s=%s, fallback to %s", name, raw, default)
        return default


class BlueCommander(BaseCommander):
    def __init__(self, faction: FactionEnum, shutdown_event: Event, llm_manager: LLMManager):
        super().__init__(faction, shutdown_event, llm_manager)
        self.rule_book = RuleBook()

        mode_raw = os.getenv("BLUE_DECISION_MODE", "rule").strip().lower()
        if mode_raw not in VALID_BLUE_MODES:
            logger.warning("Invalid BLUE_DECISION_MODE=%s, fallback to rule", mode_raw)
            mode_raw = "rule"
        self.blue_decision_mode = mode_raw

        self.blue_llm_interval = _read_int_env("BLUE_LLM_INTERVAL", 60)
        self.llm_call_interval = float(self.blue_llm_interval)
        self.last_llm_trigger_time = 5.0 - self.llm_call_interval
        self.rule_memory = {}

        logger.info(
            "[%s] Blue commander initialized. mode=%s llm_interval=%ss scout=%s focus=%s reposition=%s reserve=%s",
            self.faction_name,
            self.blue_decision_mode,
            self.blue_llm_interval,
            self.rule_book.blue_rule_scout_interval,
            self.rule_book.blue_rule_focus_interval,
            self.rule_book.blue_rule_reposition_interval,
            self.rule_book.blue_min_intercept_reserve,
        )

    def _ui_show(self, text: str):
        try:
            if self.faction:
                self.faction.show_llm_msg(text)
        except Exception:
            logger.debug("[BLUE] show_llm_msg failed", exc_info=True)

    def parse_system2_actions(self, actions_json: List[dict], state: GameState) -> List[Action]:
        if actions_json:
            try:
                pretty = json.dumps({"actions": actions_json}, ensure_ascii=False, indent=2)
                self._ui_show(f"\n--- [System2 async actions] ---\n{pretty}")
            except Exception:
                pass
        return self._parse_actions_json(actions_json, state)

    def execute_rules(self, state: GameState) -> List[Action]:
        if self.blue_decision_mode == "rule":
            return self._run_rule_mode(state)
        if self.blue_decision_mode == "hybrid":
            return self._run_hybrid_mode(state)
        if self.blue_decision_mode == "llm":
            return self._run_llm_mode(state)

        logger.warning(
            "[BLUE] mode=%s is invalid at runtime, fallback to rule",
            self.blue_decision_mode,
        )
        self.blue_decision_mode = "rule"
        return self._run_rule_mode(state)

    def _run_rule_mode(self, state: GameState) -> List[Action]:
        now = state.simtime()
        actions = self.rule_book.execute_blue_rules(state, now, self.rule_memory)
        log_fn = logger.info if actions else logger.debug
        log_fn("[BLUE] mode=rule action_count=%s sim_time=%s", len(actions), now)
        return actions

    def _run_hybrid_mode(self, state: GameState) -> List[Action]:
        now = state.simtime()
        rule_actions = self.rule_book.execute_blue_rules(state, now, self.rule_memory)

        llm_actions: List[Action] = []
        if now >= self.last_llm_trigger_time + self.llm_call_interval:
            self.last_llm_trigger_time = now
            _, llm_actions = self._decide_llm_strategy(state, self._summarize_state(state))
        else:
            logger.debug(
                "[BLUE] mode=hybrid reason=llm_cooldown remaining=%s",
                (self.last_llm_trigger_time + self.llm_call_interval) - now,
            )

        merged_actions = self._merge_and_dedupe_actions(rule_actions, llm_actions)
        logger.info(
            "[BLUE] mode=hybrid rule_action_count=%s llm_action_count=%s merged_action_count=%s sim_time=%s",
            len(rule_actions),
            len(llm_actions),
            len(merged_actions),
            now,
        )
        return merged_actions

    def _run_llm_mode(self, state: GameState) -> List[Action]:
        now = state.simtime()
        if now < self.last_llm_trigger_time + self.llm_call_interval:
            logger.debug(
                "[BLUE] mode=llm reason=llm_cooldown remaining=%s",
                (self.last_llm_trigger_time + self.llm_call_interval) - now,
            )
            return []

        self.last_llm_trigger_time = now
        _, llm_actions = self._decide_llm_strategy(state, self._summarize_state(state))
        logger.info(
            "[BLUE] mode=llm llm_action_count=%s sim_time=%s",
            len(llm_actions),
            now,
        )
        return llm_actions

    def _merge_and_dedupe_actions(self, *action_groups: List[Action]) -> List[Action]:
        merged: List[Action] = []
        seen = set()
        for group in action_groups:
            for action in group or []:
                try:
                    key = str(action.to_cmd_dict()) if hasattr(action, "to_cmd_dict") else str(action)
                except Exception:
                    key = str(action)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(action)
        return merged

    def _summarize_state(self, state: GameState) -> str:
        return self.summarizer.summarize_state(state, self.faction_name)

    def _expand_tasks_to_actions(self, tasks, state: GameState) -> List[Action]:
        actions: List[Action] = []
        for t in tasks or []:
            try:
                result = t.run(state)
                if not result:
                    continue
                if isinstance(result, list):
                    actions.extend(result)
                else:
                    actions.append(result)
            except Exception as e:
                logger.warning(
                    "[%s] task %s.run() failed: %s",
                    self.faction_name,
                    type(t).__name__,
                    e,
                    exc_info=True,
                )
        return actions

    def _parse_actions_json(self, actions_json: List[dict], state: GameState) -> List[Action]:
        parsed_actions: List[Action] = []
        if not actions_json:
            return parsed_actions

        sim_time = state.simtime()

        for act in actions_json:
            if not isinstance(act, dict):
                continue

            action_type = act.get("Type", "").strip()
            if not action_type:
                logger.warning("[%s] action missing Type: %s", self.faction_name, act)
                continue

            t_lower = action_type.lower()
            unit_id = act.get("Id")

            try:
                if t_lower == "move":
                    lon, lat, alt = act.get("lon"), act.get("lat"), act.get("alt", 0.0)
                    if isinstance(lon, (int, float)) and isinstance(lat, (int, float)):
                        parsed_actions.append(MoveAction(unit_id, Position(lon, lat, alt)))
                    continue

                elif t_lower == "smartlaunchontarget":
                    launch_unit = act.get("UnitId")
                    target_id = act.get("Target_Id")
                    if launch_unit and target_id:
                        task = OneShotTask(
                            mission_id=f"blue_smartlaunch_target_{sim_time}",
                            unit_id=launch_unit,
                            target_id=target_id,
                            sim_start_time=sim_time,
                            sim_end_time=sim_time + 60,
                        )
                        parsed_actions.extend(self._expand_tasks_to_actions([task], state))
                    continue

                elif t_lower == "changestate":
                    is_hide = act.get("isHideOn")
                    if is_hide in ["0", "1"]:
                        parsed_actions.append(ChangeStateAction(unit_id, is_hide_on=(is_hide == "1")))
                    continue

                elif t_lower == "setradar":
                    is_on = act.get("isHideOn")
                    if is_on in ["0", "1"]:
                        parsed_actions.append(RadarControlAction(unit_id, radar_on=(is_on == "1")))
                    continue

                elif t_lower == "setjammer":
                    is_on = act.get("Pattern")
                    if is_on in ["0", "1"]:
                        parsed_actions.append(JammerControlAction(unit_id, jammer_on=(is_on == "1")))
                    continue

                elif t_lower == "launchinterceptor":
                    weapon_name = act.get("weapon_type")
                    target_id = act.get("target_ID")
                    if unit_id and weapon_name and target_id:
                        weapon_enum = WeaponNameEnum.from_name(weapon_name)
                        if weapon_enum:
                            parsed_actions.append(LaunchInterceptorAction(unit_id, weapon_enum, target_id))
                    continue

                elif t_lower in ["scoutarea", "aircraftscout"]:
                    area_dict = act.get("Area")
                    if not area_dict:
                        parsed_actions.append(RawPlatformAction(act))
                        continue

                    top_left = area_dict.get("TopLeft")
                    bottom_right = area_dict.get("BottomRight")
                    if not top_left or not bottom_right:
                        parsed_actions.append(RawPlatformAction(act))
                        continue

                    area = Area(Position(**top_left), Position(**bottom_right))
                    unit_ids = act.get("UnitIds", [])

                    if any("Shipboard_Aircraft_FixWing" in u for u in unit_ids):
                        try:
                            from jsqlsim.world.missions.mission_scout import MissionAircraftScout

                            mission = MissionAircraftScout(
                                id=f"{self.faction_name}_air_scout_{sim_time}",
                                unit_ids=unit_ids,
                                area=area,
                                sim_start_time=sim_time,
                                sim_end_time=sim_time + 300,
                            )
                        except ImportError:
                            mission = MissionScout(
                                id=f"{self.faction_name}_scout_{sim_time}",
                                unit_ids=unit_ids,
                                area=area,
                                sim_start_time=sim_time,
                                sim_end_time=sim_time + 300,
                            )
                    else:
                        mission = MissionScout(
                            id=f"{self.faction_name}_scout_{sim_time}",
                            unit_ids=unit_ids,
                            area=area,
                            sim_start_time=sim_time,
                            sim_end_time=sim_time + 300,
                        )

                    parsed_actions.extend(self._expand_tasks_to_actions(mission.run(state), state))
                    continue

                elif t_lower == "antimissiledefense":
                    area_dict = act.get("Protected_Area")
                    if not area_dict:
                        continue

                    try:
                        threats = state.find_detector_states(target_id_contain_str="AttackMissile") or []
                        if not threats:
                            threats = state.find_detector_states(target_id_contain_str="CruiseMissile") or []
                    except Exception:
                        threats = []

                    if not threats:
                        logger.info(
                            "[%s] no active incoming missile detected, skip AntiMissileDefense.",
                            self.faction_name,
                        )
                        continue

                    area = Area(Position(**area_dict["TopLeft"]), Position(**area_dict["BottomRight"]))
                    mission = MissionAntiMissile(
                        id=f"{self.faction_name}_antimissile_{sim_time}",
                        unit_ids=act.get("UnitIds", []),
                        area=area,
                        target_position=None,
                        sim_start_time=sim_time,
                        sim_end_time=sim_time + 300,
                    )
                    parsed_actions.extend(self._expand_tasks_to_actions(mission.run(state), state))
                    continue

                elif t_lower == "focusfire":
                    target_id = act.get("Target_Id")
                    if target_id:
                        mission = MissionFocusFire(
                            id=f"{self.faction_name}_focusfire_{sim_time}",
                            unit_ids=act.get("UnitIds", []),
                            target_id=target_id,
                            sim_start_time=sim_time,
                            sim_end_time=sim_time + 300,
                        )
                        parsed_actions.extend(self._expand_tasks_to_actions(mission.run(state), state))
                    elif act.get("Target_Lon") is not None or act.get("Target_Lat") is not None:
                        logger.warning(
                            "[%s] rejected legacy BLUE FocusFire coordinates payload: %s",
                            self.faction_name,
                            act,
                        )
                    continue

                elif t_lower == "engageandreposition":
                    area_dict = act.get("Move_Area")
                    if not area_dict:
                        continue

                    top_left = area_dict.get("TopLeft")
                    bottom_right = area_dict.get("BottomRight")
                    if not top_left or not bottom_right:
                        continue

                    unit_ids = act.get("UnitIds", [])
                    target_id = act.get("Target_Id")

                    try:
                        if target_id and not state.get_unit_state(target_id):
                            continue
                    except Exception:
                        if target_id:
                            continue

                    area = Area(Position(**top_left), Position(**bottom_right))
                    mission = MissionMoveAndAttack(
                        id=f"{self.faction_name}_engage_repo_{sim_time}",
                        unit_ids=unit_ids,
                        area=area,
                        sim_start_time=sim_time,
                        sim_end_time=sim_time + 300,
                    )
                    parsed_actions.extend(self._expand_tasks_to_actions(mission.run(state), state))
                    continue

                else:
                    parsed_actions.append(RawPlatformAction(act))

            except Exception as e:
                logger.error("[%s] failed to parse action %s: %s", self.faction_name, action_type, e, exc_info=True)

        logger.info("[%s] generated %d actions from JSON.", self.faction_name, len(parsed_actions))
        return parsed_actions

    def _decide_llm_strategy(self, state: GameState, state_summary: str) -> Tuple[str, List[Action]]:
        sim_time = state.simtime()
        llm_response_json = self.llm_manager.get_llm_decision(
            state=state,
            faction_name=self.faction_name,
            sim_time=sim_time,
            show_fn=self._ui_show,
        )

        if not llm_response_json:
            self._ui_show("\n[WARN] No valid JSON decision from LLM in this round.")
            logger.warning("[%s] no LLM decision returned.", self.faction_name)
            return "No analysis (LLM failed).", []

        analysis = llm_response_json.get("analysis", "No analysis found.")
        actions_json = llm_response_json.get("actions", [])
        if not isinstance(actions_json, list):
            logger.error("[%s] actions is not a list.", self.faction_name)
            actions_json = []

        actions_list = self._parse_actions_json(actions_json, state)

        try:
            pretty = json.dumps({"analysis": analysis, "actions": actions_json}, ensure_ascii=False, indent=2)
            self._ui_show("\n--- structured result ---\n" + pretty)
        except Exception:
            pass

        logger.info(
            "[%s] LLM round finished: analysis_len=%d actions=%d",
            self.faction_name,
            len(analysis),
            len(actions_list),
        )
        return analysis, actions_list
