import logging
from collections import defaultdict
from typing import List, Optional

from jsqlsim import GameState

from state_access import (
    PLATFORM_UNIT_PREFIXES,
    distance_between_positions,
    extract_score_dict,
    get_detected_target_ids,
    get_detector_states,
    get_target_position,
    get_unit_hp,
    get_unit_position,
    get_unit_type_name,
    get_unit_velocity,
    get_weapon_inventory,
    infer_unit_type,
    incoming_threat_count,
    is_unit_active,
    is_unit_hidden,
    iter_platform_units,
    safe_simtime,
)


logger = logging.getLogger(__name__)

TARGET_VALUE_MAP = {
    "Flagship_Surface": 150,
    "Cruiser_Surface": 80,
    "Destroyer_Surface": 50,
}

RED_TARGET_TYPES = tuple(TARGET_VALUE_MAP.keys())


def infer_value_from_id(target_id: str) -> int:
    target_type = infer_unit_type(target_id) or str(target_id or "").split("-")[0]
    return TARGET_VALUE_MAP.get(target_type, 0)


class SituationSummarizer:
    """
    统一基于 jsqlsim 的 GameState 接口生成态势摘要。

    重点使用：
    - state.simtime()
    - state.find_units()
    - state.find_all_detect_targets()
    - state.get_unit_state() / state.get_unit_position()
    - state.get_target_position()
    - state.find_detector_states()
    - raw_state_dict["score"]
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.last_seen_targets = defaultdict(dict)
        self.current_faction = None
        self.launchers: List[dict] = []

    def summarize_state(self, state: GameState, faction_name: str) -> str:
        self.current_faction = faction_name
        sim_time = safe_simtime(state) or 0

        parts = [f"--- 态势概要 (时间: {sim_time}s, 阵营: {faction_name}) ---"]
        parts.append(self._summarize_score(state))
        parts.append(self._summarize_friendlies(state, faction_name))

        if str(faction_name).upper() == "RED":
            parts.append(self._summarize_targets_red(state, faction_name, sim_time))
        else:
            parts.append(self._summarize_targets_basic(state, faction_name, sim_time))

        parts.append(self._summarize_threats(state))
        return "\n\n".join(part for part in parts if part)

    def _summarize_score(self, state: GameState) -> str:
        score = extract_score_dict(state)
        if not score:
            return "**当前得分信息:**\n- 仿真平台未提供 score 字段。"

        ordered_keys = [
            "redScore",
            "redDestroyScore",
            "redCost",
            "blueScore",
            "blueDestroyScore",
            "blueCost",
        ]
        seen = set()
        lines = ["**当前得分信息:**"]
        for key in ordered_keys:
            if key in score:
                lines.append(f"- {key}: {score[key]}")
                seen.add(key)
        for key in sorted(score.keys()):
            if key in seen:
                continue
            lines.append(f"- {key}: {score[key]}")
        return "\n".join(lines)

    def _summarize_friendlies(self, state: GameState, faction_name: str) -> str:
        friendlies = [unit for unit in iter_platform_units(state) if get_unit_type_name(unit) in PLATFORM_UNIT_PREFIXES]
        if not friendlies:
            return "**我方单位状态:**\n- 未发现我方作战平台。"

        self.launchers = []
        lines = [f"**我方单位状态:**（数量: {len(friendlies)}）"]
        for unit in friendlies:
            unit_id = str(getattr(unit, "id", "") or "?")
            unit_type = get_unit_type_name(unit) or unit_id.split("-")[0]
            position = get_unit_position(state=state, unit_id=unit_id, unit=unit)
            hp = get_unit_hp(state=state, unit_id=unit_id, unit=unit)
            active = is_unit_active(unit)
            hidden = is_unit_hidden(unit)
            velocity = get_unit_velocity(unit)
            inventory = get_weapon_inventory(unit)

            pos_text = self._format_position(position)
            hp_text = f"{hp:.1f}" if hp is not None else "?"
            active_text = "active" if active else "inactive" if active is not None else "unknown"
            hidden_text = "hidden" if hidden else "visible" if hidden is not None else "unknown"
            velocity_text = f"{velocity:.1f}" if velocity is not None else "?"

            status = [
                f"ID: {unit_id}",
                f"类型: {unit_type}",
                f"HP: {hp_text}",
                f"位置: {pos_text}",
                f"状态: {active_text}",
                f"隐蔽: {hidden_text}",
                f"速度: {velocity_text}",
            ]
            if inventory:
                weapons_text = ", ".join(f"{name}:{count}" for name, count in sorted(inventory.items()))
                status.append(f"武器: {weapons_text}")
            lines.append("- " + " | ".join(status))

            if str(faction_name).upper() == "RED" and unit_type == "Truck_Ground" and position is not None:
                high_cost = inventory.get("HighCostAttackMissile", 0)
                low_cost = inventory.get("LowCostAttackMissile", 0)
                max_range_km = 400 if high_cost > 0 else 350 if low_cost > 0 else 0
                self.launchers.append(
                    {
                        "id": unit_id,
                        "pos": position,
                        "max_range_km": max_range_km,
                        "high_cost_num": high_cost,
                        "low_cost_num": low_cost,
                    }
                )

        if str(faction_name).upper() == "RED" and not self.launchers:
            lines.append("- 提示: 当前没有具备攻击弹药的发射车。")

        return "\n".join(lines)

    def _summarize_targets_red(self, state: GameState, faction_name: str, sim_time: int) -> str:
        detected_ids = get_detected_target_ids(state)
        entries = []
        for target_id in detected_ids:
            target_type = infer_unit_type(target_id) or str(target_id).split("-")[0]
            if target_type not in RED_TARGET_TYPES:
                continue

            position = get_target_position(state, target_id)
            value = infer_value_from_id(target_id)
            min_dist_km = None
            max_range_km = None
            in_range = None

            for launcher in self.launchers:
                dist_m = distance_between_positions(launcher["pos"], position)
                if dist_m is None:
                    continue
                dist_km = dist_m / 1000.0
                if min_dist_km is None or dist_km < min_dist_km:
                    min_dist_km = dist_km
                    max_range_km = launcher["max_range_km"]
            if min_dist_km is not None and max_range_km:
                in_range = min_dist_km <= max_range_km

            self.last_seen_targets[faction_name][target_id] = sim_time
            entries.append(
                {
                    "target_id": target_id,
                    "target_type": target_type,
                    "value": value,
                    "position": position,
                    "min_dist_km": min_dist_km,
                    "max_range_km": max_range_km,
                    "in_range": in_range,
                }
            )

        if not entries:
            return "**敌方态势:**\n- 暂未发现可打击的蓝方高价值水面目标。"

        entries.sort(key=lambda item: (-item["value"], item["min_dist_km"] if item["min_dist_km"] is not None else 1e9))
        lines = [
            "**敌方态势（高价值目标 + 射程评估）:**",
            "- 目标价值基线: Flagship=150, Cruiser=80, Destroyer=50",
        ]
        for item in entries:
            lines.append(
                "- ID: {target_id} | 类型: {target_type} | 价值: {value} | 位置: {position}".format(
                    target_id=item["target_id"],
                    target_type=item["target_type"],
                    value=item["value"],
                    position=self._format_position(item["position"]),
                )
            )
            if item["min_dist_km"] is not None:
                lines.append(
                    "  射程评估: 最近发射车距离={dist:.1f}km | 可用最大射程={max_range}km | 在射程内={in_range}".format(
                        dist=item["min_dist_km"],
                        max_range=item["max_range_km"],
                        in_range=item["in_range"],
                    )
                )
        return "\n".join(lines)

    def _summarize_targets_basic(self, state: GameState, faction_name: str, sim_time: int) -> str:
        detected_ids = get_detected_target_ids(state)
        if not detected_ids:
            return "**敌方态势:**\n- 暂未发现任何敌方目标。"

        lines = ["**敌方态势（已探测目标）:**"]
        for target_id in detected_ids:
            position = get_target_position(state, target_id)
            target_type = infer_unit_type(target_id) or str(target_id).split("-")[0]
            self.last_seen_targets[faction_name][target_id] = sim_time
            lines.append(
                f"- ID: {target_id} | 类型: {target_type} | 位置: {self._format_position(position)}"
            )
        return "\n".join(lines)

    def _summarize_threats(self, state: GameState) -> str:
        threat_count = incoming_threat_count(state)
        if threat_count <= 0:
            return "**当前威胁:**\n- 未发现明确来袭导弹威胁。"

        threats = get_detector_states(state, "AttackMissile")
        if not threats:
            threats = get_detector_states(state, "CruiseMissile")

        lines = [f"**当前威胁:**\n- 来袭导弹目标数: {threat_count}"]
        for threat in threats[:8]:
            try:
                target_id = str(getattr(threat, "target_id", "") or "")
                target_type = getattr(threat, "target_type", None)
                position = getattr(threat, "target_position", None)
                lines.append(
                    f"- 威胁ID: {target_id} | 类型: {target_type} | 位置: {self._format_position(position)}"
                )
            except Exception:
                continue
        return "\n".join(lines)

    def _format_position(self, position) -> str:
        if position is None:
            return "(未知位置)"
        try:
            return f"({float(position.lon):.6f}, {float(position.lat):.6f})"
        except Exception:
            return "(未知位置)"
