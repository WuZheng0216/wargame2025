import json
from pathlib import Path
from typing import Any, Dict, Optional


INTERCEPT_WEAPONS = {
    "Short_Range_InterceptMissile",
    "Long_Range_InterceptMissile",
}

ATTACK_WEAPONS = {
    "HighCostAttackMissile",
    "LowCostAttackMissile",
    "ShipToGround_CruiseMissile",
    "JDAM",
    "AIM_AirMissile",
}

UNIT_PREFIXES = (
    "Flagship_Surface",
    "Cruiser_Surface",
    "Destroyer_Surface",
    "Truck_Ground",
    "Guide_Ship_Surface",
    "Recon_UAV_FixWing",
    "Shipboard_Aircraft_FixWing",
    "Merchant_Ship_Surface",
    "HighCostAttackMissile",
    "LowCostAttackMissile",
)

HIGH_VALUE_BY_SIDE = {
    "RED": {"Truck_Ground", "Guide_Ship_Surface"},
    "BLUE": {"Flagship_Surface", "Cruiser_Surface", "Destroyer_Surface"},
}


def infer_unit_type(identifier: str) -> Optional[str]:
    text = str(identifier or "")
    for prefix in UNIT_PREFIXES:
        if prefix in text:
            return prefix
    return None


def is_high_value_type(side: str, unit_type: Optional[str]) -> bool:
    if not unit_type:
        return False
    return unit_type in HIGH_VALUE_BY_SIDE.get(str(side or "").upper(), set())


def parse_side_battle_log(log_path: Optional[Path], side: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "new_target_event_count": 0,
        "unique_targets_detected": 0,
        "high_value_targets_detected": 0,
        "unit_loss_count": 0,
        "high_value_unit_loss_count": 0,
        "decision_count": 0,
        "submitted_action_count": 0,
        "first_action_simtime": None,
        "last_action_simtime": None,
        "intercept_launch_count": 0,
        "attack_launch_count": 0,
        "score_change_event_count": 0,
        "score_gain_total": 0.0,
        "score_loss_total": 0.0,
        "positive_score_events": 0,
        "final_score": None,
    }
    if not log_path or not log_path.exists():
        return result

    unique_targets = set()
    high_value_targets = set()
    final_score = None

    for raw_line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue

        event_type = str(record.get("type") or "")
        extra = record.get("extra") or {}

        if event_type == "NEW_TARGET_DETECTED":
            result["new_target_event_count"] += 1
            target_id = str(extra.get("target_id") or "")
            if target_id:
                unique_targets.add(target_id)
                target_type = infer_unit_type(target_id)
                enemy_side = "BLUE" if str(side).upper() == "RED" else "RED"
                if is_high_value_type(enemy_side, target_type):
                    high_value_targets.add(target_id)

        elif event_type == "UNIT_LOST":
            result["unit_loss_count"] += 1
            unit_id = str(extra.get("unit_id") or "")
            unit_type = str(extra.get("unit_type") or infer_unit_type(unit_id) or "")
            if is_high_value_type(side, unit_type):
                result["high_value_unit_loss_count"] += 1

        elif event_type == "SCORE_CHANGED":
            result["score_change_event_count"] += 1
            try:
                old_score = float(extra.get("old_score"))
                new_score = float(extra.get("new_score"))
                delta = new_score - old_score
            except Exception:
                delta = 0.0
                try:
                    new_score = float(extra.get("new_score"))
                except Exception:
                    new_score = final_score
            final_score = new_score
            if delta > 0:
                result["score_gain_total"] += delta
                result["positive_score_events"] += 1
            elif delta < 0:
                result["score_loss_total"] += abs(delta)
        elif event_type == "SCORE_SNAPSHOT":
            try:
                final_score = float(extra.get("side_score"))
            except Exception:
                pass

        elif event_type == "DECISION":
            result["decision_count"] += 1
            actions = record.get("actions") or []
            if isinstance(actions, list) and actions:
                result["submitted_action_count"] += len(actions)
                sim_time = record.get("sim_time")
                try:
                    sim_time_value = int(float(sim_time))
                except Exception:
                    sim_time_value = None
                if sim_time_value is not None:
                    if result["first_action_simtime"] is None:
                        result["first_action_simtime"] = sim_time_value
                    result["last_action_simtime"] = sim_time_value
            for action in actions:
                if not isinstance(action, dict):
                    continue
                if str(action.get("Type") or "") != "Launch":
                    continue
                weapon_type = str(action.get("WeaponType") or "")
                if weapon_type in INTERCEPT_WEAPONS:
                    result["intercept_launch_count"] += 1
                elif weapon_type in ATTACK_WEAPONS:
                    result["attack_launch_count"] += 1

    result["unique_targets_detected"] = len(unique_targets)
    result["high_value_targets_detected"] = len(high_value_targets)
    result["final_score"] = final_score
    return result


def extract_run_battle_metrics(red_log_path: Optional[Path], blue_log_path: Optional[Path]) -> Dict[str, Any]:
    red = parse_side_battle_log(red_log_path, "RED")
    blue = parse_side_battle_log(blue_log_path, "BLUE")
    return {
        "red_target_detection_events": red["new_target_event_count"],
        "blue_target_detection_events": blue["new_target_event_count"],
        "red_unique_targets_detected": red["unique_targets_detected"],
        "blue_unique_targets_detected": blue["unique_targets_detected"],
        "red_high_value_targets_detected": red["high_value_targets_detected"],
        "blue_high_value_targets_detected": blue["high_value_targets_detected"],
        "red_units_lost": red["unit_loss_count"],
        "blue_units_lost": blue["unit_loss_count"],
        "red_high_value_units_lost": red["high_value_unit_loss_count"],
        "blue_high_value_units_lost": blue["high_value_unit_loss_count"],
        "red_decision_count": red["decision_count"],
        "blue_decision_count": blue["decision_count"],
        "red_submitted_action_count": red["submitted_action_count"],
        "blue_submitted_action_count": blue["submitted_action_count"],
        "first_red_action_simtime": red["first_action_simtime"],
        "first_blue_action_simtime": blue["first_action_simtime"],
        "last_red_action_simtime": red["last_action_simtime"],
        "last_blue_action_simtime": blue["last_action_simtime"],
        "red_targets_destroyed": blue["unit_loss_count"],
        "blue_targets_destroyed": red["unit_loss_count"],
        "red_high_value_targets_destroyed": blue["high_value_unit_loss_count"],
        "blue_high_value_targets_destroyed": red["high_value_unit_loss_count"],
        "red_intercept_launch_count": red["intercept_launch_count"],
        "blue_intercept_launch_count": blue["intercept_launch_count"],
        "red_attack_launch_count": red["attack_launch_count"],
        "blue_attack_launch_count": blue["attack_launch_count"],
        "red_score_change_event_count": red["score_change_event_count"],
        "blue_score_change_event_count": blue["score_change_event_count"],
        "red_score_gain_total": red["score_gain_total"],
        "blue_score_gain_total": blue["score_gain_total"],
        "red_score_loss_total": red["score_loss_total"],
        "blue_score_loss_total": blue["score_loss_total"],
        "red_positive_score_events": red["positive_score_events"],
        "blue_positive_score_events": blue["positive_score_events"],
        "final_score_red": red["final_score"],
        "final_score_blue": blue["final_score"],
    }
