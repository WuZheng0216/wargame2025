import logging
import os
from typing import Dict, List, Optional, Set, Tuple

from jsqlsim.enum import WeaponNameEnum
from jsqlsim.geo import Area, Position
from jsqlsim.world.action import (
    Action,
    JammerControlAction,
    LaunchInterceptorAction,
    MoveAction,
    RadarControlAction,
)
from jsqlsim.world.game_state import GameState
from jsqlsim.world.missions.mission_focus_fire import MissionFocusFire
from jsqlsim.world.missions.mission_move_and_attack import MissionMoveAndAttack
from jsqlsim.world.missions.mission_scout import MissionScout

from custom_actions import ChangeStateAction

logger = logging.getLogger(__name__)


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        return int(raw)
    except Exception:
        logger.warning("Invalid %s=%s, fallback to %s", name, raw, default)
        return default


class RuleBook:
    def __init__(self):
        self.blue_intercept_targets: Set[str] = set()
        self.red_truck_fired: Dict[str, int] = {}

        self.blue_rule_scout_interval = _read_int_env("BLUE_RULE_SCOUT_INTERVAL", 90)
        self.blue_rule_focus_interval = _read_int_env("BLUE_RULE_FOCUS_INTERVAL", 45)
        self.blue_rule_reposition_interval = _read_int_env("BLUE_RULE_REPOSITION_INTERVAL", 120)
        self.blue_rule_sensor_refresh_interval = _read_int_env("BLUE_RULE_SENSOR_REFRESH_INTERVAL", 30)
        self.blue_rule_detection_stale_time = _read_int_env("BLUE_RULE_DETECTION_STALE_TIME", 60)
        self.blue_rule_min_attackable_targets = _read_int_env("BLUE_RULE_MIN_ATTACKABLE_TARGETS", 2)
        self.blue_rule_sensing_escalation_cooldown = _read_int_env("BLUE_RULE_SENSING_ESCALATION_COOLDOWN", 45)
        self.blue_rule_air_task_ttl = _read_int_env("BLUE_RULE_AIR_TASK_TTL", 120)
        self.blue_rule_air_reassign_gap = _read_int_env("BLUE_RULE_AIR_REASSIGN_GAP", 45)
        self.blue_rule_target_cooldown = _read_int_env("BLUE_RULE_TARGET_COOLDOWN", 90)
        self.blue_rule_surface_fire_package_size = _read_int_env("BLUE_RULE_SURFACE_FIRE_PACKAGE_SIZE", 2)
        self.blue_rule_air_fire_package_size = _read_int_env("BLUE_RULE_AIR_FIRE_PACKAGE_SIZE", 2)
        self.blue_rule_reposition_move_count = _read_int_env("BLUE_RULE_REPOSITION_MOVE_COUNT", 2)
        self.blue_min_intercept_reserve = _read_int_env("BLUE_MIN_INTERCEPT_RESERVE", 2)
        self.blue_air_threat_radius_m = 180000
        self.blue_hot_area_half_span_deg = 0.45
        self.blue_air_role_offsets = {
            "cap_inner": (0.15, 0.15, 0.18),
            "cap_outer": (-0.28, 0.24, 0.24),
            "forward_hot": (0.22, -0.18, 0.22),
            "air_reserve": (-0.18, -0.18, 0.16),
        }

        self._blue_scout_areas: List[Tuple[Tuple[float, float], Tuple[float, float]]] = [
            ((49.2, 15.8), (47.6, 14.2)),
            ((47.8, 15.5), (46.2, 13.9)),
            ((46.9, 14.8), (45.4, 13.2)),
            ((48.8, 14.1), (47.0, 12.5)),
        ]

        logger.info(
            "RuleBook initialized: scout_interval=%s focus_interval=%s reposition_interval=%s sensor_refresh=%s "
            "detection_stale=%s min_attackable=%s sensing_cooldown=%s reserve=%s air_ttl=%s target_cd=%s",
            self.blue_rule_scout_interval,
            self.blue_rule_focus_interval,
            self.blue_rule_reposition_interval,
            self.blue_rule_sensor_refresh_interval,
            self.blue_rule_detection_stale_time,
            self.blue_rule_min_attackable_targets,
            self.blue_rule_sensing_escalation_cooldown,
            self.blue_min_intercept_reserve,
            self.blue_rule_air_task_ttl,
            self.blue_rule_target_cooldown,
        )

    # ------------------------------
    # BLUE high-coverage rule mode
    # ------------------------------
    def execute_blue_rules(self, state: GameState, now: int, memory: dict) -> List[Action]:
        actions: List[Action] = []
        try:
            actions.extend(self._blue_sensor_posture_policy(state, now, memory))
        except Exception as e:
            logger.error(
                "[BLUE][rule] rule_type=sensor_posture reason=exception error=%s existing_action_count=%s",
                e,
                len(actions),
                exc_info=True,
            )

        ammo_guard = {"blocked_units": set()}
        try:
            ammo_guard = self._blue_ammo_guard_policy(state, now, memory)
        except Exception as e:
            logger.error(
                "[BLUE][rule] rule_type=ammo_guard reason=exception error=%s existing_action_count=%s",
                e,
                len(actions),
                exc_info=True,
            )

        strategies = [
            ("intercept", lambda: self._blue_intercept_policy(state, now, memory, ammo_guard)),
            ("scout", lambda: self._blue_scout_policy(state, now, memory)),
            ("focus_fire", lambda: self._blue_focus_fire_policy(state, now, memory)),
            ("reposition", lambda: self._blue_reposition_policy(state, now, memory)),
        ]
        for rule_type, runner in strategies:
            try:
                produced = runner() or []
                actions.extend(produced)
            except Exception as e:
                logger.error(
                    "[BLUE][rule] rule_type=%s reason=exception error=%s existing_action_count=%s",
                    rule_type,
                    e,
                    len(actions),
                    exc_info=True,
                )

        deduped = self._dedupe_actions(actions)
        aggregate_log = logger.info if deduped else logger.debug
        aggregate_log(
            "[BLUE][rule] rule_type=aggregate action_count=%s deduped_count=%s",
            len(actions),
            len(deduped),
        )
        return deduped

    def _blue_sensor_posture_policy(self, state: GameState, now: int, memory: dict) -> List[Action]:
        actions: List[Action] = []
        desired_state = memory.setdefault("blue_sensor_desired_state", {})

        detected_targets = self._collect_detected_targets(state)
        current_target_ids = {target_id for target_id, _ in detected_targets}
        previous_target_ids = set(memory.get("blue_last_detected_target_ids", []))
        detection_changed = current_target_ids != previous_target_ids
        if detection_changed or "blue_last_detection_change_time" not in memory:
            memory["blue_last_detection_change_time"] = now

        memory["blue_last_detected_target_ids"] = sorted(current_target_ids)
        memory["blue_last_detected_target_count"] = len(current_target_ids)

        attackable_target_count = self._count_attackable_targets(state, detected_targets)
        memory["blue_attackable_target_count"] = attackable_target_count

        hot_target = self._highest_priority_detected_target(detected_targets)
        if hot_target is not None:
            hot_target_id, hot_target_pos = hot_target
            previous_hot_target_id = str(memory.get("blue_last_hot_target_id", "")).strip()
            memory["blue_last_hot_target_id"] = hot_target_id
            if hot_target_pos is not None:
                memory["blue_last_hot_target_position"] = {
                    "lon": hot_target_pos.lon,
                    "lat": hot_target_pos.lat,
                    "alt": hot_target_pos.alt if hot_target_pos.alt is not None else 0.0,
                }
            if previous_hot_target_id and previous_hot_target_id != hot_target_id:
                memory["blue_air_role_dirty"] = True

        stale_detection = self._should_escalate_sensing(
            now,
            memory,
            len(current_target_ids),
            attackable_target_count,
        )
        last_escalation_time = int(memory.get("blue_last_sensing_escalation_time", -10**9))
        stale_escalation_due = stale_detection and (
            now - last_escalation_time >= self.blue_rule_sensing_escalation_cooldown
        )
        if stale_escalation_due:
            memory["blue_last_sensing_escalation_time"] = now
            memory["blue_force_scout_until"] = now + self.blue_rule_sensor_refresh_interval
            memory["blue_force_scout_reason"] = "detection_stale"
            memory["blue_air_role_dirty"] = True

        missile_threats = self._find_missile_threats(state)
        near_air_threat = self._has_air_or_guide_threat(state)
        refresh_due = now - int(memory.get("blue_last_sensor_refresh_time", -10**9)) >= self.blue_rule_sensor_refresh_interval

        flagships = self._find_units(state, "Flagship_Surface")
        destroyers = self._find_units(state, "Destroyer_Surface")
        cruisers = self._find_units(state, "Cruiser_Surface")
        surface_units = flagships + destroyers + cruisers

        radar_units = {u.id for u in flagships + destroyers}
        if stale_detection or attackable_target_count < self.blue_rule_min_attackable_targets or missile_threats or near_air_threat:
            radar_units.update(u.id for u in cruisers)

        jammer_units: Set[str] = set()
        if missile_threats or near_air_threat:
            for unit in surface_units:
                unit_state = self._get_unit_state(state, unit.id)
                if self._unit_has_jammer(unit_state):
                    jammer_units.add(unit.id)

        reason = "radar_refresh"
        if now - int(memory.get("blue_last_detection_change_time", now)) >= self.blue_rule_detection_stale_time:
            reason = "detection_stale"
        elif missile_threats or near_air_threat:
            reason = "threat_jammer_on"
        elif detection_changed:
            reason = "tracking_update"

        for unit in surface_units:
            unit_state = self._get_unit_state(state, unit.id)
            if unit_state is None:
                continue

            unit_desired = desired_state.setdefault(unit.id, {})
            radar_on = unit.id in radar_units
            previous_radar = unit_desired.get("radar_on")
            if previous_radar is None:
                unit_desired["radar_on"] = radar_on
                if radar_on:
                    actions.append(RadarControlAction(unit.id, radar_on))
                    logger.info(
                        "[BLUE][rule] rule_type=sensor_posture unit_id=%s radar_on=%s jammer_on=%s reason=%s",
                        unit.id,
                        radar_on,
                        unit_desired.get("jammer_on"),
                        reason,
                    )
            elif previous_radar != radar_on:
                actions.append(RadarControlAction(unit.id, radar_on))
                logger.info(
                    "[BLUE][rule] rule_type=sensor_posture unit_id=%s radar_on=%s jammer_on=%s reason=%s",
                    unit.id,
                    radar_on,
                    unit_desired.get("jammer_on"),
                    reason,
                )
                unit_desired["radar_on"] = radar_on

            if self._unit_has_jammer(unit_state):
                jammer_on = unit.id in jammer_units
                previous_jammer = unit_desired.get("jammer_on")
                if previous_jammer is None:
                    unit_desired["jammer_on"] = jammer_on
                elif previous_jammer != jammer_on:
                    actions.append(JammerControlAction(unit.id, jammer_on))
                    logger.info(
                        "[BLUE][rule] rule_type=sensor_posture unit_id=%s radar_on=%s jammer_on=%s reason=%s",
                        unit.id,
                        unit_desired.get("radar_on", radar_on),
                        jammer_on,
                        reason,
                    )
                    unit_desired["jammer_on"] = jammer_on

        if refresh_due or detection_changed or stale_escalation_due or actions:
            memory["blue_last_sensor_refresh_time"] = now
            logger.info(
                "[BLUE][rule] rule_type=sensor_posture detected_target_count=%s attackable_target_count=%s reason=%s action_count=%s",
                len(current_target_ids),
                attackable_target_count,
                reason,
                len(actions),
            )

        return actions

    def _blue_ammo_guard_policy(self, state: GameState, now: int, memory: dict) -> dict:
        blocked_units: Set[str] = set()
        ships = self._find_units(state, "Destroyer_Surface") + self._find_units(state, "Cruiser_Surface")
        for ship in ships:
            ship_state = self._get_unit_state(state, ship.id)
            if ship_state is None:
                continue
            total_interceptors = self._count_interceptor_stock(ship_state)
            if total_interceptors <= self.blue_min_intercept_reserve:
                blocked_units.add(ship.id)

        memory["blue_ammo_guard_blocked_units"] = blocked_units
        previous_blocked = set(memory.get("blue_last_ammo_guard_blocked_units", []))
        log_times = memory.setdefault("blue_ammo_guard_log_times", {})
        if blocked_units:
            for unit_id in sorted(blocked_units):
                log_key = f"{unit_id}:interceptor_reserve"
                last_log_time = int(log_times.get(log_key, -10**9))
                if unit_id not in previous_blocked or now - last_log_time >= 60:
                    logger.debug(
                        "[BLUE][rule] rule_type=ammo_guard target_id=%s reason=interceptor_reserve action_count=0",
                        unit_id,
                    )
                    log_times[log_key] = now
        elif previous_blocked:
            clear_key = "clear:interceptor_reserve"
            last_log_time = int(log_times.get(clear_key, -10**9))
            if now - last_log_time >= 60:
                logger.debug("[BLUE][rule] rule_type=ammo_guard reason=clear action_count=0")
                log_times[clear_key] = now
        memory["blue_last_ammo_guard_blocked_units"] = sorted(blocked_units)
        return {"blocked_units": blocked_units}

    def _blue_intercept_policy(self, state: GameState, now: int, memory: dict, ammo_guard: dict) -> List[Action]:
        actions: List[Action] = []
        blocked_units: Set[str] = set(ammo_guard.get("blocked_units", set()))
        threats = self._find_missile_threats(state)
        if not threats:
            return actions

        ships = [u for u in (self._find_units(state, "Destroyer_Surface") + self._find_units(state, "Cruiser_Surface")) if u.id not in blocked_units]
        if not ships:
            logger.debug("[BLUE][rule] rule_type=intercept reason=no_available_ship action_count=0")
            return actions

        handled = memory.setdefault("blue_handled_intercepts", {})
        self._prune_old_entries(handled, now, ttl=600)

        for target_id, target_pos in threats:
            last_t = handled.get(target_id)
            if isinstance(last_t, int) and now - last_t < 80:
                continue

            best_ship = None
            best_dist = float("inf")
            best_weapon = None

            for ship in ships:
                ship_state = self._get_unit_state(state, ship.id)
                ship_pos = self._get_position(ship)
                if ship_state is None or ship_pos is None or target_pos is None:
                    continue
                if not self._launcher_ready(ship_state):
                    continue

                dist = self._distance(ship_pos, target_pos)
                weapon = self._choose_interceptor_weapon(ship_state, dist)
                if weapon is None:
                    continue
                if dist < best_dist:
                    best_dist = dist
                    best_ship = ship
                    best_weapon = weapon

            if best_ship is None or best_weapon is None:
                logger.debug(
                    "[BLUE][rule] rule_type=intercept target_id=%s reason=no_weapon_or_range action_count=0",
                    target_id,
                )
                continue

            actions.append(LaunchInterceptorAction(best_ship.id, best_weapon, target_id))
            handled[target_id] = now
            logger.info(
                "[BLUE][rule] rule_type=intercept target_id=%s reason=engage action_count=1",
                target_id,
            )

        return actions

    def _blue_scout_policy(self, state: GameState, now: int, memory: dict) -> List[Action]:
        last_scout = int(memory.get("blue_last_scout_time", -10**9))
        force_scout = now <= int(memory.get("blue_force_scout_until", -10**9))
        if not force_scout and now - last_scout < self.blue_rule_scout_interval:
            return []

        uav_units = self._find_units(state, "Recon_UAV_FixWing")
        aircraft_units = self._find_units(state, "Shipboard_Aircraft_FixWing")
        if not uav_units and not aircraft_units:
            logger.debug("[BLUE][rule] rule_type=scout reason=no_scout_unit action_count=0")
            return []

        actions: List[Action] = []
        primary_area, reason, area_label = self._select_blue_scout_area(memory, force_scout)
        uav_assignments = self._build_uav_scout_assignments(memory, primary_area, area_label, force_scout)
        for assignment, unit in zip(uav_assignments, uav_units[: len(uav_assignments)]):
            mission = MissionScout(
                id=f"blue_rule_uav_scout_{now}_{assignment['area_label']}_{unit.id}",
                unit_ids=[unit.id],
                area=assignment["area"],
                sim_start_time=now,
                sim_end_time=now + 300,
            )
            produced = self._expand_mission_actions(mission.run(state), state)
            actions.extend(produced)
            logger.info(
                "[BLUE][rule] rule_type=scout unit_group=uav role=%s area_label=%s reason=%s action_count=%s",
                assignment["role"],
                assignment["area_label"],
                assignment["reason"],
                len(produced),
            )

        for assignment in self._assign_blue_air_roles(state, now, memory):
            produced = self._run_air_scout_assignment(state, now, assignment)
            actions.extend(produced)
            logger.info(
                "[BLUE][rule] rule_type=scout unit_group=aircraft role=%s area_label=%s reason=%s action_count=%s",
                assignment["role"],
                assignment["area_label"],
                assignment["reason"],
                len(produced),
            )

        memory["blue_last_scout_time"] = now
        if force_scout:
            memory["blue_force_scout_until"] = -10**9
            memory.pop("blue_force_scout_reason", None)
        logger.info(
            "[BLUE][rule] rule_type=scout target_id=%s reason=%s action_count=%s",
            area_label,
            reason,
            len(actions),
        )
        return actions

    def _blue_focus_fire_policy(self, state: GameState, now: int, memory: dict) -> List[Action]:
        last_focus = int(memory.get("blue_last_focus_time", -10**9))
        if now - last_focus < self.blue_rule_focus_interval:
            return []

        if not (self._blue_surface_strike_units(state) or self._blue_air_attack_units(state)):
            logger.debug("[BLUE][rule] rule_type=focus_fire reason=no_attacker action_count=0")
            return []

        memory["blue_last_focus_time"] = now
        actions: List[Action] = []
        fallback_target_id = None
        fallback_target_pos = None
        fallback_reason = None

        candidates = self._pick_focus_fire_targets(state, memory, now)
        if not candidates.get("surface") and not candidates.get("air"):
            memory["blue_reposition_urgent_until"] = now + self.blue_rule_focus_interval
            logger.debug("[BLUE][rule] rule_type=focus_fire reason=no_effective_target action_count=0")
            return []

        for platform_mode, package_size, base_reason in (
            ("surface", self.blue_rule_surface_fire_package_size, "surface_package"),
            ("air", self.blue_rule_air_fire_package_size, "air_package"),
        ):
            candidate = candidates.get(platform_mode)
            if not candidate:
                continue

            target_id = candidate["target_id"]
            target_pos = candidate["target_pos"]
            attack_window = candidate["attack_window"]
            selected_attackers = self._select_focus_fire_attackers(state, target_id, platform_mode=platform_mode)
            if not selected_attackers:
                continue

            if attack_window == "observe_only":
                memory["blue_force_scout_until"] = now + self.blue_rule_sensor_refresh_interval
                memory["blue_force_scout_reason"] = "observe_only_target"
                continue

            if attack_window == "move_then_fire":
                if fallback_target_pos is None:
                    fallback_target_id = target_id
                    fallback_target_pos = target_pos
                    fallback_reason = f"{platform_mode}_move_then_fire"
                continue

            mission = MissionFocusFire(
                id=f"blue_rule_focus_{platform_mode}_{now}",
                unit_ids=[u.id for u in selected_attackers[:package_size]],
                target_id=target_id,
                sim_start_time=now,
                sim_end_time=now + 180,
            )
            produced = self._expand_mission_actions(mission.run(state), state)
            if not produced:
                if fallback_target_pos is None:
                    fallback_target_id = target_id
                    fallback_target_pos = target_pos
                    fallback_reason = f"{platform_mode}_focus_failed"
                continue

            actions.extend(produced)
            memory["blue_last_nonzero_focus_time"] = now
            memory["blue_last_focus_target_id"] = target_id
            self._record_target_selection(memory, platform_mode, target_id, attack_window, now)
            if platform_mode == "surface":
                memory["blue_surface_strike_target_id"] = target_id
            else:
                memory["blue_air_strike_target_id"] = target_id
            logger.info(
                "[BLUE][rule] rule_type=focus_fire target_id=%s attack_window=%s attacker_mix=%s reason=%s action_count=%s",
                target_id,
                attack_window,
                self._summarize_unit_mix(selected_attackers[:package_size]),
                "retargeted" if candidate.get("retargeted") else base_reason,
                len(produced),
            )

        if actions:
            return actions

        if fallback_target_pos is not None:
            memory["blue_reposition_urgent_until"] = now + self.blue_rule_focus_interval
            memory["blue_reposition_target_id"] = fallback_target_id
            memory["blue_reposition_target_position"] = {
                "lon": fallback_target_pos.lon,
                "lat": fallback_target_pos.lat,
                "alt": fallback_target_pos.alt if fallback_target_pos.alt is not None else 0.0,
            }
            memory["blue_reposition_reason"] = fallback_reason or "move_then_fire_target"
            logger.debug(
                "[BLUE][rule] rule_type=focus_fire target_id=%s reason=%s action_count=0",
                fallback_target_id,
                fallback_reason or "move_then_fire_target",
            )
            return []

        logger.debug("[BLUE][rule] rule_type=focus_fire reason=no_effective_target action_count=0")
        return actions

    def _blue_reposition_policy(self, state: GameState, now: int, memory: dict) -> List[Action]:
        last_move = int(memory.get("blue_last_reposition_time", -10**9))
        urgent_reposition = now <= int(memory.get("blue_reposition_urgent_until", -10**9))
        if not urgent_reposition and now - last_move < self.blue_rule_reposition_interval:
            return []

        destroyers = self._find_units(state, "Destroyer_Surface")
        cruisers = self._find_units(state, "Cruiser_Surface")
        units = destroyers + cruisers
        if not units:
            logger.debug("[BLUE][rule] rule_type=reposition reason=no_surface_unit action_count=0")
            return []

        anchor = self._get_reference_position(state, memory)
        if anchor is None:
            logger.info("[BLUE][rule] rule_type=reposition reason=no_anchor action_count=0")
            return []

        target_id = str(memory.get("blue_reposition_target_id", "")).strip()
        target_pos_dict = memory.get("blue_reposition_target_position")
        target_pos = None
        if isinstance(target_pos_dict, dict):
            target_pos = Position(target_pos_dict.get("lon"), target_pos_dict.get("lat"), target_pos_dict.get("alt", 0.0))

        selected_units, phase = self._select_reposition_units(memory, destroyers, cruisers)
        selected_units = selected_units[: self.blue_rule_reposition_move_count]
        area = self._build_reposition_area(anchor)
        reason = "interval_trigger"
        target_label = "anchor"
        fallback = "mission"
        if urgent_reposition and target_pos is not None:
            area = self._build_standoff_area(anchor, target_pos, int(memory.get("blue_surface_reposition_phase", 0)))
            reason = str(memory.get("blue_reposition_reason", "move_then_fire_target")) or "move_then_fire_target"
            target_label = target_id or "standoff"

        mission = MissionMoveAndAttack(
            id=f"blue_rule_reposition_{now}",
            unit_ids=[u.id for u in selected_units],
            area=area,
            sim_start_time=now,
            sim_end_time=now + 240,
        )
        actions = self._expand_mission_actions(mission.run(state), state)
        if not actions:
            fallback = "move_action"
            actions = self._fallback_reposition_moves(selected_units, area)

        memory["blue_last_reposition_time"] = now
        memory["blue_surface_reposition_phase"] = (phase + 1) % 3
        memory["blue_reposition_urgent_until"] = -10**9
        memory.pop("blue_reposition_target_id", None)
        memory.pop("blue_reposition_target_position", None)
        memory.pop("blue_reposition_reason", None)
        if actions:
            memory["blue_last_nonzero_reposition_time"] = now
            if urgent_reposition:
                memory["blue_last_focus_time"] = min(
                    int(memory.get("blue_last_focus_time", now)),
                    now - max(1, self.blue_rule_focus_interval // 2),
                )
        logger.info(
            "[BLUE][rule] rule_type=reposition target_id=%s reason=%s phase=%s selected_units=%s fallback=%s action_count=%s",
            target_label,
            reason if urgent_reposition else "interval_trigger",
            phase,
            [u.id for u in selected_units],
            fallback,
            len(actions),
        )
        return actions

    # ------------------------------
    # RED rule compatibility
    # ------------------------------
    def execute_red_rules(self, state: GameState) -> List[Action]:
        return self._handle_red_shoot_and_scoot(state)

    def _handle_red_shoot_and_scoot(self, state: GameState) -> List[Action]:
        actions: List[Action] = []
        sim_time = state.simtime()
        my_trucks = self._find_units(state, "Truck_Ground")

        for truck in my_trucks:
            vehicle_state = getattr(truck, "vechicle_state", None)
            if not vehicle_state:
                continue

            is_exposed = not getattr(vehicle_state, "is_hide_on", True)
            if is_exposed and truck.id not in self.red_truck_fired:
                self.red_truck_fired[truck.id] = sim_time

            if truck.id in self.red_truck_fired and is_exposed:
                actions.append(ChangeStateAction(unit_id=truck.id, is_hide_on=True))
            elif truck.id in self.red_truck_fired and not is_exposed:
                self.red_truck_fired.pop(truck.id, None)

        return actions

    # ------------------------------
    # Internal helpers
    # ------------------------------
    def _find_units(self, state: GameState, id_contain_str: str):
        try:
            return state.find_units(id_contain_str=id_contain_str) or []
        except Exception:
            return []

    def _get_unit_state(self, state: GameState, unit_id: str):
        try:
            return state.get_unit_state(unit_id)
        except Exception:
            return None

    def _get_position(self, unit) -> Optional[Position]:
        try:
            return unit.position()
        except Exception:
            return None

    def _distance(self, pos_a: Position, pos_b: Position) -> float:
        try:
            d = pos_a.distance(pos_b)
            if d is None:
                return float("inf")
            return float(d)
        except Exception:
            return float("inf")

    def _expand_mission_actions(self, tasks, state: GameState) -> List[Action]:
        actions: List[Action] = []
        for task in tasks or []:
            try:
                result = task.run(state)
                if not result:
                    continue
                if isinstance(result, list):
                    actions.extend(result)
                else:
                    actions.append(result)
            except Exception:
                continue
        return actions

    def _extract_target_id(self, detector_state) -> Optional[str]:
        if detector_state is None:
            return None
        if isinstance(detector_state, dict):
            for key in ("real_id", "target_id", "id", "Id"):
                value = detector_state.get(key)
                if isinstance(value, str) and value:
                    return value
            return None

        for attr in ("real_id", "target_id", "id"):
            value = getattr(detector_state, attr, None)
            if isinstance(value, str) and value:
                return value
        return None

    def _get_target_position(self, state: GameState, target_id: str) -> Optional[Position]:
        if not target_id:
            return None
        try:
            pos = state.get_target_position(target_id)
            if pos is not None:
                return pos
        except Exception:
            pass
        try:
            us = state.get_unit_state(target_id)
            if us is not None and hasattr(us, "position"):
                return us.position()
        except Exception:
            pass
        return None

    def _find_missile_threats(self, state: GameState) -> List[Tuple[str, Position]]:
        detector_states = []
        for key in ("AttackMissile", "CruiseMissile", "Missile"):
            try:
                detector_states.extend(state.find_detector_states(target_id_contain_str=key) or [])
            except Exception:
                continue

        seen: Set[str] = set()
        threats: List[Tuple[str, Position]] = []
        for ds in detector_states:
            tid = self._extract_target_id(ds)
            if not tid or tid in seen:
                continue
            pos = self._get_target_position(state, tid)
            if pos is None:
                continue
            seen.add(tid)
            threats.append((tid, pos))
        return threats

    def _count_interceptor_stock(self, ship_state) -> int:
        total = 0
        for ws in getattr(ship_state, "weapon_states", []) or []:
            name = str(getattr(ws, "name", "")).upper()
            if "INTERCEPT" in name and "MISSILE" in name:
                total += int(getattr(ws, "num", 0) or 0)
        return total

    def _count_weapon_stock(self, unit_state, keyword: str) -> int:
        total = 0
        if unit_state is None:
            return total
        token = str(keyword or "").upper()
        for ws in getattr(unit_state, "weapon_states", []) or []:
            name = str(getattr(ws, "name", "")).upper()
            if token in name:
                total += int(getattr(ws, "num", 0) or 0)
        return total

    def _unit_has_jammer(self, unit_state) -> bool:
        return getattr(unit_state, "jammer_state", None) is not None

    def _blue_attackers(self, state: GameState):
        return (
            self._find_units(state, "Flagship_Surface")
            + self._find_units(state, "Cruiser_Surface")
            + self._find_units(state, "Destroyer_Surface")
            + self._find_units(state, "Shipboard_Aircraft_FixWing")
        )

    def _blue_surface_strike_units(self, state: GameState):
        candidates = (
            self._find_units(state, "Flagship_Surface")
            + self._find_units(state, "Cruiser_Surface")
            + self._find_units(state, "Destroyer_Surface")
        )
        units = []
        for unit in candidates:
            unit_state = self._get_unit_state(state, unit.id)
            if self._count_weapon_stock(unit_state, "CRUISE") > 0:
                units.append(unit)
        return units

    def _blue_air_attack_units(self, state: GameState):
        candidates = self._find_units(state, "Shipboard_Aircraft_FixWing")
        units = []
        for unit in candidates:
            unit_state = self._get_unit_state(state, unit.id)
            if self._count_weapon_stock(unit_state, "AIM") > 0 or self._count_weapon_stock(unit_state, "JDAM") > 0:
                units.append(unit)
        return units

    def _launcher_ready(self, ship_state) -> bool:
        launcher_states = getattr(ship_state, "launcher_states", []) or []
        if not launcher_states:
            return True
        for launcher in launcher_states:
            try:
                if int(getattr(launcher, "cd", 0)) == 0:
                    return True
            except Exception:
                continue
        return False

    def _weapon_enum_by_candidates(self, candidates: List[str]):
        for name in candidates:
            if hasattr(WeaponNameEnum, name):
                return getattr(WeaponNameEnum, name)
            try:
                enum_value = WeaponNameEnum.from_name(name)
                if enum_value is not None:
                    return enum_value
            except Exception:
                continue
        return None

    def _choose_interceptor_weapon(self, ship_state, distance_m: float):
        short_count = 0
        long_count = 0
        for ws in getattr(ship_state, "weapon_states", []) or []:
            name = str(getattr(ws, "name", "")).upper()
            num = int(getattr(ws, "num", 0) or 0)
            if "INTERCEPT" not in name or "MISSILE" not in name:
                continue
            if "SHORT" in name:
                short_count += num
            elif "LONG" in name:
                long_count += num

        prefer_long = distance_m >= 70000
        short_enum = self._weapon_enum_by_candidates(
            ["SHORT_RANGE_INTERCEPT_MISSILE", "SHORT_RANGE_INTERCEPTOR", "INTERCEPTOR_SHORT"]
        )
        long_enum = self._weapon_enum_by_candidates(
            ["LONG_RANGE_INTERCEPT_MISSILE", "LONG_RANGE_INTERCEPTOR", "INTERCEPTOR_LONG"]
        )

        if prefer_long:
            if long_enum is not None and long_count > self.blue_min_intercept_reserve:
                return long_enum
            if short_enum is not None and short_count > self.blue_min_intercept_reserve:
                return short_enum
        else:
            if short_enum is not None and short_count > self.blue_min_intercept_reserve:
                return short_enum
            if long_enum is not None and long_count > self.blue_min_intercept_reserve:
                return long_enum
        return None

    def _target_priority_score(self, target_id: str) -> int:
        target = (target_id or "").upper()
        ranking = [
            ("TRUCK_GROUND", 120),
            ("GUIDE_SHIP_SURFACE", 105),
            ("RECON_UAV_FIXWING", 90),
            ("SHIPBOARD_AIRCRAFT_FIXWING", 80),
            ("MERCHANT_SHIP_SURFACE", -1000),
        ]
        for key, score in ranking:
            if key in target:
                return score
        return 20

    def _collect_detected_targets(self, state: GameState) -> List[Tuple[str, Optional[Position]]]:
        detector_states = []
        try:
            detector_states = state.find_detector_states() or []
        except Exception:
            detector_states = []

        results: List[Tuple[str, Optional[Position]]] = []
        seen: Set[str] = set()
        for ds in detector_states:
            target_id = self._extract_target_id(ds)
            if not target_id or target_id in seen or "Merchant_Ship_Surface" in target_id:
                continue
            seen.add(target_id)
            results.append((target_id, self._get_target_position(state, target_id)))
        return results

    def _highest_priority_detected_target(
        self, detected_targets: List[Tuple[str, Optional[Position]]]
    ) -> Optional[Tuple[str, Optional[Position]]]:
        best_item = None
        best_score = float("-inf")
        for target_id, target_pos in detected_targets:
            score = self._target_priority_score(target_id)
            if target_pos is not None:
                score += 1
            if score > best_score:
                best_score = score
                best_item = (target_id, target_pos)
        return best_item

    def _is_air_target(self, target_id: str) -> bool:
        return "Recon_UAV_FixWing" in target_id or "Shipboard_Aircraft_FixWing" in target_id

    def _is_surface_strike_target(self, target_id: str) -> bool:
        return "Truck_Ground" in target_id or "Guide_Ship_Surface" in target_id

    def _has_air_or_guide_threat(self, state: GameState) -> bool:
        targets = self._collect_detected_targets(state)
        if not targets:
            return False

        surface_units = (
            self._find_units(state, "Flagship_Surface")
            + self._find_units(state, "Destroyer_Surface")
            + self._find_units(state, "Cruiser_Surface")
        )
        surface_positions = [self._get_position(unit) for unit in surface_units]
        surface_positions = [pos for pos in surface_positions if pos is not None]
        if not surface_positions:
            return False

        for target_id, target_pos in targets:
            if target_pos is None:
                continue
            if not (
                "Recon_UAV_FixWing" in target_id
                or "Shipboard_Aircraft_FixWing" in target_id
                or "Guide_Ship_Surface" in target_id
            ):
                continue
            if min(self._distance(pos, target_pos) for pos in surface_positions) <= self.blue_air_threat_radius_m:
                return True
        return False

    def _count_attackable_targets(
        self, state: GameState, detected_targets: List[Tuple[str, Optional[Position]]]
    ) -> int:
        attackers = self._blue_attackers(state)
        count = 0
        for target_id, target_pos in detected_targets:
            if target_pos is None:
                continue
            if self._is_attackable_target_for_blue(state, target_id, target_pos, attackers) != "observe_only":
                count += 1
        return count

    def _should_escalate_sensing(
        self, now: int, memory: dict, detected_target_count: int, attackable_target_count: int
    ) -> bool:
        last_change = int(memory.get("blue_last_detection_change_time", now))
        if now - last_change >= self.blue_rule_detection_stale_time:
            return True
        if attackable_target_count < self.blue_rule_min_attackable_targets:
            return True
        return detected_target_count == 0

    def _select_blue_scout_area(self, memory: dict, force_scout: bool) -> Tuple[Area, str, str]:
        hot_pos = memory.get("blue_last_hot_target_position")
        if force_scout and isinstance(hot_pos, dict):
            center = Position(
                hot_pos.get("lon"),
                hot_pos.get("lat"),
                hot_pos.get("alt", 0.0),
            )
            half = self.blue_hot_area_half_span_deg
            area = Area(
                Position(center.lon - half, center.lat + half, 0.0),
                Position(center.lon + half, center.lat - half, 0.0),
            )
            target_id = str(memory.get("blue_last_hot_target_id", "hotspot"))
            return area, str(memory.get("blue_force_scout_reason", "detection_stale")), f"hot_{target_id}"

        idx = int(memory.get("blue_scout_area_idx", 0)) % len(self._blue_scout_areas)
        (tl_lon, tl_lat), (br_lon, br_lat) = self._blue_scout_areas[idx]
        memory["blue_scout_area_idx"] = idx + 1
        area = Area(Position(tl_lon, tl_lat), Position(br_lon, br_lat))
        return area, "interval_trigger", f"area_{idx}"

    def _build_uav_scout_assignments(
        self, memory: dict, primary_area: Area, area_label: str, force_scout: bool
    ) -> List[dict]:
        reason = str(memory.get("blue_force_scout_reason", "interval_trigger")) if force_scout else "uav_broad_scan"
        assignments = [
            {
                "role": "uav_primary",
                "area": primary_area,
                "area_label": area_label,
                "reason": reason,
            },
            {
                "role": "uav_flank_left",
                "area": self._shift_area(primary_area, lon_offset=-0.22, lat_offset=0.12, scale=0.9),
                "area_label": f"{area_label}_left",
                "reason": "hot_flank_scan" if force_scout else "broad_left",
            },
            {
                "role": "uav_flank_right",
                "area": self._shift_area(primary_area, lon_offset=0.24, lat_offset=-0.1, scale=0.9),
                "area_label": f"{area_label}_right",
                "reason": "hot_flank_scan" if force_scout else "broad_right",
            },
        ]
        return assignments

    def _assign_blue_air_roles(self, state: GameState, now: int, memory: dict) -> List[dict]:
        aircraft_units = sorted(self._find_units(state, "Shipboard_Aircraft_FixWing"), key=lambda u: u.id)
        if not aircraft_units:
            return []

        anchor = self._get_reference_position(state, memory)
        if anchor is None:
            return []

        hot_pos = None
        hot_pos_dict = memory.get("blue_last_hot_target_position")
        if isinstance(hot_pos_dict, dict):
            hot_pos = Position(
                hot_pos_dict.get("lon"),
                hot_pos_dict.get("lat"),
                hot_pos_dict.get("alt", 0.0),
            )
        hot_target_id = str(memory.get("blue_last_hot_target_id", "")).strip()

        role_state = memory.setdefault("blue_air_role_state", {})
        last_assignment = int(memory.get("blue_air_last_assignment_time", -10**9))
        dirty = bool(memory.get("blue_air_role_dirty"))

        reusable_assignments: List[dict] = []
        if role_state and not dirty and now - last_assignment < self.blue_rule_air_task_ttl:
            for unit in aircraft_units:
                cached = role_state.get(unit.id)
                if not isinstance(cached, dict):
                    reusable_assignments = []
                    break
                if int(cached.get("expires_at", -10**9)) < now:
                    reusable_assignments = []
                    break
                if str(cached.get("hot_target_id", "")).strip() != hot_target_id:
                    reusable_assignments = []
                    break
                top_left = cached.get("top_left")
                bottom_right = cached.get("bottom_right")
                if not isinstance(top_left, dict) or not isinstance(bottom_right, dict):
                    reusable_assignments = []
                    break
                reusable_assignments.append(
                    {
                        "unit_id": unit.id,
                        "role": str(cached.get("role", "cap_inner")),
                        "area": Area(Position(**top_left), Position(**bottom_right)),
                        "area_label": str(cached.get("area_label", f"hold_{unit.id}")),
                        "reason": "hold_role",
                    }
                )
            if len(reusable_assignments) == len(aircraft_units):
                return reusable_assignments

        roles = self._resolve_blue_air_roles(len(aircraft_units), now)
        new_state = {}
        assignments: List[dict] = []
        for idx, unit in enumerate(aircraft_units):
            role = roles[idx] if idx < len(roles) else ("forward_hot" if idx % 2 else "cap_outer")
            area = self._build_air_role_area(anchor, hot_pos, role, idx)
            area_label = f"{role}_{idx}"
            reason = "role_refresh"
            if dirty:
                reason = "role_reassign"
            elif now - last_assignment < self.blue_rule_air_reassign_gap:
                reason = "hold_gap"
            assignments.append(
                {
                    "unit_id": unit.id,
                    "role": role,
                    "area": area,
                    "area_label": area_label,
                    "reason": reason,
                }
            )
            new_state[unit.id] = {
                "role": role,
                "area_label": area_label,
                "assigned_at": now,
                "expires_at": now + self.blue_rule_air_task_ttl,
                "hot_target_id": hot_target_id,
                "top_left": {
                    "lon": area.top_left.lon,
                    "lat": area.top_left.lat,
                    "alt": area.top_left.alt if area.top_left.alt is not None else 0.0,
                },
                "bottom_right": {
                    "lon": area.bottom_right.lon,
                    "lat": area.bottom_right.lat,
                    "alt": area.bottom_right.alt if area.bottom_right.alt is not None else 0.0,
                },
            }

        memory["blue_air_role_state"] = new_state
        memory["blue_air_last_assignment_time"] = now
        memory["blue_air_role_dirty"] = False
        return assignments

    def _resolve_blue_air_roles(self, aircraft_count: int, now: int) -> List[str]:
        if aircraft_count >= 4:
            roles = ["cap_inner", "cap_outer", "forward_hot", "air_reserve"]
            for idx in range(4, aircraft_count):
                roles.append("forward_hot" if idx % 2 else "cap_outer")
            return roles
        if aircraft_count == 3:
            return ["cap_inner", "cap_outer", "forward_hot"]
        if aircraft_count == 2:
            return ["cap_inner", "forward_hot"]
        if aircraft_count == 1:
            return ["cap_inner" if (now // 60) % 2 == 0 else "forward_hot"]
        return []

    def _build_air_role_area(
        self,
        anchor: Position,
        hot_pos: Optional[Position],
        role: str,
        slot_idx: int,
    ) -> Area:
        base = hot_pos if role == "forward_hot" and hot_pos is not None else anchor
        offset_lon, offset_lat, half_span = self.blue_air_role_offsets.get(role, (0.0, 0.0, 0.18))
        jitter_lon = 0.04 * (slot_idx % 3)
        jitter_lat = 0.03 * ((slot_idx + 1) % 3)
        if slot_idx % 2 == 1:
            jitter_lon *= -1
            jitter_lat *= -1
        center_lon = base.lon + offset_lon + jitter_lon
        center_lat = base.lat + offset_lat + jitter_lat
        span = max(0.12, half_span + 0.01 * slot_idx)
        return self._area_from_center(center_lon, center_lat, span, span)

    def _run_air_scout_assignment(self, state: GameState, now: int, assignment: dict) -> List[Action]:
        unit_id = str(assignment.get("unit_id", "")).strip()
        area = assignment.get("area")
        if not unit_id or not isinstance(area, Area):
            return []
        try:
            from jsqlsim.world.missions.mission_scout import MissionAircraftScout

            mission = MissionAircraftScout(
                id=f"blue_aircraft_scout_{now}_{assignment.get('role', 'role')}_{unit_id}",
                unit_ids=[unit_id],
                area=area,
                sim_start_time=now,
                sim_end_time=now + 300,
            )
        except ImportError:
            mission = MissionScout(
                id=f"blue_aircraft_scout_{now}_{assignment.get('role', 'role')}_{unit_id}",
                unit_ids=[unit_id],
                area=area,
                sim_start_time=now,
                sim_end_time=now + 300,
            )
        return self._expand_mission_actions(mission.run(state), state)

    def _select_focus_fire_attackers(
        self,
        state: GameState,
        target_id: str,
        platform_mode: Optional[str] = None,
    ):
        target = (target_id or "").upper()
        surface_units = self._blue_surface_strike_units(state)
        air_units = self._blue_air_attack_units(state)

        def _air_units_for_target() -> List:
            selected = []
            for unit in air_units:
                unit_state = self._get_unit_state(state, unit.id)
                aim_stock = self._count_weapon_stock(unit_state, "AIM")
                jdam_stock = self._count_weapon_stock(unit_state, "JDAM")
                if "RECON_UAV_FIXWING" in target or "SHIPBOARD_AIRCRAFT_FIXWING" in target:
                    if aim_stock > 0:
                        selected.append(unit)
                elif "GUIDE_SHIP_SURFACE" in target:
                    if jdam_stock > 0 or aim_stock > 0:
                        selected.append(unit)
                elif "TRUCK_GROUND" in target:
                    if jdam_stock > 0:
                        selected.append(unit)
            return selected

        if platform_mode == "surface":
            if "TRUCK_GROUND" in target or "GUIDE_SHIP_SURFACE" in target:
                return surface_units
            return []

        if platform_mode == "air":
            return _air_units_for_target()

        if "TRUCK_GROUND" in target or "GUIDE_SHIP_SURFACE" in target:
            return surface_units or _air_units_for_target()

        if "RECON_UAV_FIXWING" in target or "SHIPBOARD_AIRCRAFT_FIXWING" in target:
            return _air_units_for_target()

        return surface_units or _air_units_for_target()

    def _is_attackable_target_for_blue(
        self,
        state: GameState,
        target_id: str,
        target_pos: Position,
        attackers,
    ) -> str:
        if "Merchant_Ship_Surface" in target_id:
            return "observe_only"

        compatible_attackers = attackers or self._select_focus_fire_attackers(state, target_id)
        if not compatible_attackers:
            return "observe_only"

        attacker_positions = [self._get_position(attacker) for attacker in compatible_attackers]
        attacker_positions = [pos for pos in attacker_positions if pos is not None]
        if not attacker_positions:
            return "observe_only"

        if self._is_surface_strike_target(target_id):
            fire_now_range = 450000
            move_then_fire_range = 650000
        elif self._is_air_target(target_id):
            fire_now_range = 300000
            move_then_fire_range = 500000
        else:
            fire_now_range = 500000
            move_then_fire_range = 700000

        min_distance = min(self._distance(attacker_pos, target_pos) for attacker_pos in attacker_positions)
        if min_distance <= fire_now_range:
            return "fire_now"
        if min_distance <= move_then_fire_range:
            return "move_then_fire"
        return "observe_only"

    def _platform_target_priority_score(self, target_id: str, platform_mode: str) -> int:
        target = (target_id or "").upper()
        if "MERCHANT_SHIP_SURFACE" in target:
            return -1000
        if platform_mode == "surface":
            ranking = [
                ("TRUCK_GROUND", 120),
                ("GUIDE_SHIP_SURFACE", 110),
                ("RECON_UAV_FIXWING", 80),
                ("SHIPBOARD_AIRCRAFT_FIXWING", 70),
            ]
        else:
            ranking = [
                ("RECON_UAV_FIXWING", 120),
                ("GUIDE_SHIP_SURFACE", 108),
                ("TRUCK_GROUND", 95),
                ("SHIPBOARD_AIRCRAFT_FIXWING", 70),
            ]
        for key, score in ranking:
            if key in target:
                return score
        return 20

    def _target_window_rank(self, attack_window: str) -> int:
        ranking = {
            "fire_now": 3,
            "move_then_fire": 2,
            "observe_only": 1,
        }
        return ranking.get(str(attack_window or "").strip(), 0)

    def _is_target_on_cooldown(
        self,
        memory: dict,
        platform_mode: str,
        target_id: str,
        attack_window: str,
        now: int,
    ) -> bool:
        cooldowns = memory.get("blue_target_cooldowns", {})
        if not isinstance(cooldowns, dict):
            return False
        entry = cooldowns.get(f"{platform_mode}:{target_id}")
        if not isinstance(entry, dict):
            return False
        selected_at = int(entry.get("selected_at", -10**9))
        if now - selected_at >= self.blue_rule_target_cooldown:
            return False
        previous_window = str(entry.get("attack_window", ""))
        if attack_window == "fire_now" and previous_window != "fire_now":
            return False
        return True

    def _pick_focus_fire_targets(self, state: GameState, memory: dict, now: int) -> Dict[str, dict]:
        detected_targets = self._collect_detected_targets(state)
        results: Dict[str, dict] = {}
        if not detected_targets:
            return results

        for platform_mode in ("surface", "air"):
            preferred: List[dict] = []
            cooled: List[dict] = []
            for target_id, target_pos in detected_targets:
                if target_pos is None:
                    continue
                attackers = self._select_focus_fire_attackers(state, target_id, platform_mode=platform_mode)
                if not attackers:
                    continue
                attack_window = self._is_attackable_target_for_blue(state, target_id, target_pos, attackers)
                attacker_positions = [self._get_position(attacker) for attacker in attackers]
                attacker_positions = [pos for pos in attacker_positions if pos is not None]
                if not attacker_positions:
                    continue
                min_distance = min(self._distance(pos, target_pos) for pos in attacker_positions)
                score = self._platform_target_priority_score(target_id, platform_mode) * 100000.0 - min_distance
                score += self._target_window_rank(attack_window) * 50000.0
                candidate = {
                    "target_id": target_id,
                    "target_pos": target_pos,
                    "attack_window": attack_window,
                    "score": score,
                    "retargeted": False,
                }
                if self._is_target_on_cooldown(memory, platform_mode, target_id, attack_window, now):
                    cooled.append(candidate)
                else:
                    preferred.append(candidate)

            candidate_pool = preferred or cooled
            if not candidate_pool:
                continue

            candidate_pool.sort(
                key=lambda item: (self._target_window_rank(item["attack_window"]), item["score"]),
                reverse=True,
            )
            streak_key = f"blue_{platform_mode}_target_streak"
            last_key = f"blue_last_{platform_mode}_selected_target_id"
            last_target_id = str(memory.get(last_key, "")).strip()
            target_streak = int(memory.get(streak_key, 0))
            chosen = candidate_pool[0]
            if last_target_id and chosen["target_id"] == last_target_id and target_streak >= 3:
                for alternative in candidate_pool[1:]:
                    if alternative["target_id"] != last_target_id:
                        chosen = dict(alternative)
                        chosen["retargeted"] = True
                        break
            results[platform_mode] = chosen
        return results

    def _record_target_selection(
        self,
        memory: dict,
        platform_mode: str,
        target_id: str,
        attack_window: str,
        now: int,
    ):
        cooldowns = memory.setdefault("blue_target_cooldowns", {})
        if not isinstance(cooldowns, dict):
            cooldowns = {}
            memory["blue_target_cooldowns"] = cooldowns
        ttl = max(300, self.blue_rule_target_cooldown * 3)
        for key, value in list(cooldowns.items()):
            selected_at = value.get("selected_at") if isinstance(value, dict) else value
            if isinstance(selected_at, int) and now - selected_at > ttl:
                cooldowns.pop(key, None)
        cooldowns[f"{platform_mode}:{target_id}"] = {
            "selected_at": now,
            "attack_window": attack_window,
        }
        last_key = f"blue_last_{platform_mode}_selected_target_id"
        streak_key = f"blue_{platform_mode}_target_streak"
        last_target_id = str(memory.get(last_key, "")).strip()
        if last_target_id == target_id:
            memory[streak_key] = int(memory.get(streak_key, 0)) + 1
        else:
            memory[streak_key] = 1
        memory[last_key] = target_id

    def _summarize_unit_mix(self, units) -> str:
        counters: Dict[str, int] = {}
        for unit in units or []:
            unit_id = str(getattr(unit, "id", unit))
            if "Destroyer_Surface" in unit_id:
                key = "destroyer"
            elif "Cruiser_Surface" in unit_id:
                key = "cruiser"
            elif "Flagship_Surface" in unit_id:
                key = "flagship"
            elif "Shipboard_Aircraft_FixWing" in unit_id:
                key = "aircraft"
            else:
                key = "other"
            counters[key] = counters.get(key, 0) + 1
        return ",".join(f"{key}:{counters[key]}" for key in sorted(counters)) or "none"

    def _pick_high_value_target(self, state: GameState, attackers) -> Tuple[Optional[str], Optional[Position], str]:
        detected_targets = self._collect_detected_targets(state)
        if not detected_targets:
            return None, None, "observe_only"

        bucket_best = {
            "fire_now": (None, None, float("-inf")),
            "move_then_fire": (None, None, float("-inf")),
            "observe_only": (None, None, float("-inf")),
        }

        for target_id, target_pos in detected_targets:
            if target_pos is None:
                continue
            attack_window = self._is_attackable_target_for_blue(state, target_id, target_pos, attackers)

            compatible_attackers = self._select_focus_fire_attackers(state, target_id)
            attacker_positions = [self._get_position(attacker) for attacker in compatible_attackers]
            attacker_positions = [pos for pos in attacker_positions if pos is not None]
            if not attacker_positions:
                continue

            distance_penalty = min(self._distance(pos, target_pos) for pos in attacker_positions)
            score = self._target_priority_score(target_id) * 100000.0 - distance_penalty
            _, _, best_score = bucket_best[attack_window]
            if score > best_score:
                bucket_best[attack_window] = (target_id, target_pos, score)

        for bucket in ("fire_now", "move_then_fire", "observe_only"):
            target_id, target_pos, score = bucket_best[bucket]
            if target_id and target_pos is not None and score > float("-inf"):
                return target_id, target_pos, bucket
        return None, None, "observe_only"

    def _select_reposition_units(self, memory: dict, destroyers, cruisers) -> Tuple[List, int]:
        phase = int(memory.get("blue_surface_reposition_phase", 0)) % 3
        destroyers = sorted(destroyers or [], key=lambda unit: unit.id)
        cruisers = sorted(cruisers or [], key=lambda unit: unit.id)
        destroyer_group_a = [unit for idx, unit in enumerate(destroyers) if idx % 2 == 0]
        destroyer_group_b = [unit for idx, unit in enumerate(destroyers) if idx % 2 == 1]

        if phase == 0:
            selected = destroyer_group_a or destroyers[:1]
        elif phase == 1:
            selected = destroyer_group_b or destroyers[:1]
        else:
            selected = cruisers

        if not selected:
            selected = destroyers[: self.blue_rule_reposition_move_count] or cruisers[: self.blue_rule_reposition_move_count]
        return selected, phase

    def _fallback_reposition_moves(self, selected_units, area: Area) -> List[Action]:
        if not selected_units or not isinstance(area, Area):
            return []
        center_lon = (area.top_left.lon + area.bottom_right.lon) / 2.0
        center_lat = (area.top_left.lat + area.bottom_right.lat) / 2.0
        offsets = [(0.0, 0.0), (0.12, 0.1), (-0.12, -0.08), (0.08, -0.12)]
        actions: List[Action] = []
        for idx, unit in enumerate(selected_units):
            offset_lon, offset_lat = offsets[idx % len(offsets)]
            actions.append(
                MoveAction(
                    unit.id,
                    Position(center_lon + offset_lon, center_lat + offset_lat, 0.0),
                )
            )
        return actions

    def _build_standoff_area(self, origin: Position, target: Position, phase: int = 0) -> Area:
        center_lon = (origin.lon + target.lon) / 2.0
        center_lat = (origin.lat + target.lat) / 2.0
        quadrant_offsets = [
            (0.28, 0.22),
            (-0.28, 0.22),
            (-0.28, -0.22),
            (0.28, -0.22),
        ]
        offset_lon, offset_lat = quadrant_offsets[int(phase) % len(quadrant_offsets)]
        top_left = Position(center_lon + offset_lon - 0.24, center_lat + offset_lat + 0.24, 0.0)
        bottom_right = Position(center_lon + offset_lon + 0.24, center_lat + offset_lat - 0.24, 0.0)
        return Area(top_left, bottom_right)

    def _area_from_center(self, center_lon: float, center_lat: float, half_lon: float, half_lat: float) -> Area:
        return Area(
            Position(center_lon - half_lon, center_lat + half_lat, 0.0),
            Position(center_lon + half_lon, center_lat - half_lat, 0.0),
        )

    def _shift_area(self, area: Area, lon_offset: float, lat_offset: float, scale: float = 1.0) -> Area:
        center_lon = (area.top_left.lon + area.bottom_right.lon) / 2.0 + lon_offset
        center_lat = (area.top_left.lat + area.bottom_right.lat) / 2.0 + lat_offset
        half_lon = max(0.08, abs(area.bottom_right.lon - area.top_left.lon) * 0.5 * scale)
        half_lat = max(0.08, abs(area.top_left.lat - area.bottom_right.lat) * 0.5 * scale)
        return self._area_from_center(center_lon, center_lat, half_lon, half_lat)

    def _get_reference_position(self, state: GameState, memory: dict) -> Optional[Position]:
        last_tid = memory.get("blue_last_focus_target_id")
        if isinstance(last_tid, str):
            pos = self._get_target_position(state, last_tid)
            if pos is not None:
                return pos

        hot_pos = memory.get("blue_last_hot_target_position")
        if isinstance(hot_pos, dict):
            return Position(hot_pos.get("lon"), hot_pos.get("lat"), hot_pos.get("alt", 0.0))

        ships = self._find_units(state, "Destroyer_Surface") + self._find_units(state, "Cruiser_Surface")
        positions = [self._get_position(u) for u in ships]
        positions = [p for p in positions if p is not None]
        if not positions:
            return None

        lon = sum(p.lon for p in positions) / len(positions)
        lat = sum(p.lat for p in positions) / len(positions)
        return Position(lon, lat, 0.0)

    def _build_reposition_area(self, anchor: Position) -> Area:
        # Keep a compact move box around anchor to avoid extreme relocation.
        top_left = Position(anchor.lon - 0.6, anchor.lat + 0.6, 0.0)
        bottom_right = Position(anchor.lon + 0.6, anchor.lat - 0.6, 0.0)
        return Area(top_left, bottom_right)

    def _prune_old_entries(self, mapping: dict, now: int, ttl: int):
        remove_keys = []
        for k, t in mapping.items():
            if isinstance(t, int) and now - t > ttl:
                remove_keys.append(k)
        for k in remove_keys:
            mapping.pop(k, None)

    def _dedupe_actions(self, actions: List[Action]) -> List[Action]:
        deduped: List[Action] = []
        seen: Set[str] = set()
        for action in actions:
            try:
                key = str(action.to_cmd_dict()) if hasattr(action, "to_cmd_dict") else str(action)
            except Exception:
                key = str(action)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(action)
        return deduped
