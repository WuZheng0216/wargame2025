import json
import logging
import math
import operator
import os
import re
import dotenv
from time import perf_counter
from typing import Annotated, Dict, List, Tuple, TypedDict

from langgraph.graph import END, StateGraph

from action_semantics import _is_red_ship_target, validate_red_action_semantics, validate_red_plan_semantics
from red_prompt_compiler import (
    ALLOCATOR_OUTPUT_SCHEMA,
    OPERATOR_OUTPUT_SCHEMA,
    compile_allocator_prompt,
    compile_analyst_prompt,
    compile_commander_prompt,
    compile_operator_prompt,
)
from red_trace_helper import summarize_json_actions, truncate_text

logger = logging.getLogger(__name__)

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(current_dir, ".env")
    if not os.path.exists(dotenv_path):
        dotenv_path = os.path.join(current_dir, "..", ".env")
    if os.path.exists(dotenv_path):
        dotenv.load_dotenv(dotenv_path)
except Exception:
    pass

class AgentState(TypedDict):
    sim_time: int
    faction_name: str
    trace_id: str
    state_summary: str
    window_summary: str
    event_timeline: List[dict]
    ltm_lessons: str
    ltm_lessons_structured: List[dict]
    memory_window: dict
    unit_roster: List[dict]
    target_table: List[dict]
    task_board: dict
    engagement_summary: str
    engagement_memory: List[dict]
    key_findings: str
    intent: str
    allocation_plan: dict
    allocation_summary: str
    critique: str
    planned_at_sim_time: int
    actions_json: List[dict]
    is_valid: bool
    retry_count: int
    retry_stage: str
    logs: Annotated[List[str], operator.add]
    trace_events: Annotated[List[dict], operator.add]


FACTION_ACTION_SCHEMAS: Dict[str, str] = {
    "RED": """
你是 RED 方 operator agent。
只返回 JSON 数组。
每个元素的 Type 只能是：
- ScoutArea: Type, UnitIds, Area{TopLeft{lon,lat}, BottomRight{lon,lat}}
- MoveToEngage: Type, UnitIds, Target_Id
- GuideAttack: Type, UnitIds, Target_Id, Loiter_Radius_km(optional)
- FocusFire: Type, UnitIds, Target_Id
- ShootAndScoot: Type, UnitIds, Target_Id, Move_Area{TopLeft,BottomRight}(optional)
不要输出底层动作（如 Move / Launch / ChangeState 等）。
""".strip(),
    "BLUE": """
你是 BLUE 方 operator agent。
只返回 JSON 数组。
每个元素的 Type 只能是：
- ScoutArea: Type, UnitIds, Area{TopLeft{lon,lat}, BottomRight{lon,lat}}
- SmartLaunchOnTarget: Type, UnitId, Target_Id
- AntiMissileDefense: Type, UnitIds, Protected_Area{TopLeft{lon,lat}, BottomRight{lon,lat}}
- FocusFire: Type, UnitIds, Target_Id
- EngageAndReposition: Type, UnitIds, Move_Area{TopLeft{lon,lat}, BottomRight{lon,lat}}, Target_Id(optional)
不要输出底层动作（如 Move / Launch / ChangeState 等）。
""".strip(),
}

FACTION_ACTION_GUARDRAILS: Dict[str, str] = {
    "RED": """
RED 方语义规则：
- MoveToEngage / FocusFire / ShootAndScoot 的 UnitIds 必须全部是 Truck_Ground-*。
- GuideAttack 的 UnitIds 必须全部是 Guide_Ship_Surface-*。
- ScoutArea 的 UnitIds 必须全部是 Recon_UAV_FixWing-*。
- 不要把 HighCostAttackMissile-* 或 LowCostAttackMissile-* 当作 UnitIds。
- RED 方攻击目标只能是海上舰船：Flagship_Surface-* / Cruiser_Surface-* / Destroyer_Surface-*。
- 每个 RED 单位在同一轮规划中只能承担一个主任务。
- 保留约三分之一的 Truck_Ground 作为 reserve，且至少保留 1 辆。
- 如果存在有效的 fire_now 海上目标，至少包含一个 truck fire action。
- 如果目标是 move_then_fire，只能使用 MoveToEngage，且投入 trucks 不要超过约三分之一。
- 如果所有可见目标都是 observe_only，不要输出 FocusFire 或 ShootAndScoot。
- 不要把所有平台都用相同任务打到同一个目标上。
""".strip(),
    "BLUE": """
BLUE 方语义规则：
- FocusFire 必须使用 Target_Id，而不是目标坐标。
- AntiMissileDefense 只用于处理当前有效的 incoming missiles。
""".strip(),
}

ALLOWED_TYPES: Dict[str, set] = {
    "RED": {"ScoutArea", "MoveToEngage", "GuideAttack", "FocusFire", "ShootAndScoot"},
    "BLUE": {"ScoutArea", "SmartLaunchOnTarget", "AntiMissileDefense", "FocusFire", "EngageAndReposition"},
}

RED_BUCKET_DEFAULT_TASK_TYPES = {
    "main_attack": "FocusFire",
    "support_guidance": "GuideAttack",
    "scout_tasks": "ScoutArea",
}

RED_BUCKET_ALIASES = {
    "main_attack": "main_attack",
    "attack": "main_attack",
    "attack_tasks": "main_attack",
    "fire": "main_attack",
    "fires": "main_attack",
    "fire_tasks": "main_attack",
    "support_guidance": "support_guidance",
    "guidance": "support_guidance",
    "guidance_tasks": "support_guidance",
    "guide": "support_guidance",
    "guide_tasks": "support_guidance",
    "scout_tasks": "scout_tasks",
    "scouting": "scout_tasks",
    "scouting_tasks": "scout_tasks",
    "scout": "scout_tasks",
    "recon": "scout_tasks",
    "recon_tasks": "scout_tasks",
}

RED_TASK_TYPE_ALIASES = {
    "focusfire": "FocusFire",
    "focus_fire": "FocusFire",
    "fire": "FocusFire",
    "fire_now": "FocusFire",
    "attack": "FocusFire",
    "main_attack": "FocusFire",
    "fire_high_cost": "FocusFire",
    "high_cost_fire": "FocusFire",
    "highcost_fire": "FocusFire",
    "high_cost_focus_fire": "FocusFire",
    "fire_low_cost": "FocusFire",
    "low_cost_fire": "FocusFire",
    "lowcost_fire": "FocusFire",
    "shootandscoot": "ShootAndScoot",
    "shoot_and_scoot": "ShootAndScoot",
    "hit_and_run": "ShootAndScoot",
    "movetoengage": "MoveToEngage",
    "move_to_engage": "MoveToEngage",
    "move_then_fire": "MoveToEngage",
    "advance_and_fire": "MoveToEngage",
    "engage": "MoveToEngage",
    "reposition_and_fire": "MoveToEngage",
    "guideattack": "GuideAttack",
    "guide_attack": "GuideAttack",
    "guide": "GuideAttack",
    "guidance": "GuideAttack",
    "support_guidance": "GuideAttack",
    "support_guide": "GuideAttack",
    "scoutarea": "ScoutArea",
    "scout_area": "ScoutArea",
    "scout": "ScoutArea",
    "recon": "ScoutArea",
    "recon_area": "ScoutArea",
    "refresh_track": "ScoutArea",
    "refresh_tracks": "ScoutArea",
    "track_refresh": "ScoutArea",
    "refresh_and_observe": "ScoutArea",
}

MAX_RETRY = 1


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        return int(raw)
    except Exception:
        logger.warning("Invalid %s=%s, fallback to %s", name, raw, default)
        return default


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        return float(raw)
    except Exception:
        logger.warning("Invalid %s=%s, fallback to %s", name, raw, default)
        return default


RED_MAX_ATTACK_GROUP_SIZE = _read_int_env("RED_MAX_ATTACK_GROUP_SIZE", 4)
RED_FLAGSHIP_MAX_ATTACK_GROUP_SIZE = _read_int_env("RED_FLAGSHIP_MAX_ATTACK_GROUP_SIZE", 6)
RED_MOVE_THEN_FIRE_COMMIT_RATIO = _read_float_env("RED_MOVE_THEN_FIRE_COMMIT_RATIO", 0.34)


def _normalize_faction(faction_name: str) -> str:
    name = (faction_name or "").upper()
    return "RED" if "RED" in name else "BLUE"


def _red_group_size_limit(target_id: str, attack_window: str) -> int:
    if str(attack_window) == "fire_now" and str(target_id or "").startswith("Flagship_Surface"):
        return max(1, RED_FLAGSHIP_MAX_ATTACK_GROUP_SIZE)
    return max(1, RED_MAX_ATTACK_GROUP_SIZE)


def _allowed_move_then_fire_commit(truck_count: int) -> int:
    if truck_count <= 0:
        return 0
    return max(1, int(math.ceil(truck_count * RED_MOVE_THEN_FIRE_COMMIT_RATIO)))


def _target_window_rank(window: str) -> int:
    mapping = {"fire_now": 0, "move_then_fire": 1, "observe_only": 2}
    return mapping.get(str(window or ""), 3)


def _locked_task_entries(task_board: dict) -> dict:
    locked = {}
    if not isinstance(task_board, dict):
        return locked
    for unit_id, task in task_board.items():
        if isinstance(task, dict) and task.get("reassignable") is False:
            locked[str(unit_id)] = task
    return locked


def _is_area(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    return isinstance(value.get("TopLeft"), dict) and isinstance(value.get("BottomRight"), dict)


def _normalize_bucket_name(raw_bucket: object) -> str:
    text = str(raw_bucket or "").strip().lower()
    if not text:
        return ""
    key = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return RED_BUCKET_ALIASES.get(key, text if text in RED_BUCKET_DEFAULT_TASK_TYPES else "")


def _get_allocation_bucket(plan: dict, bucket_name: str):
    if not isinstance(plan, dict):
        return None
    if bucket_name in plan:
        return plan.get(bucket_name)
    for raw_key, raw_value in plan.items():
        if _normalize_bucket_name(raw_key) == bucket_name:
            return raw_value
    return None


def _normalize_red_task_type(raw_task_type: object, bucket_name: str = "") -> str:
    raw = str(raw_task_type or "").strip()
    if not raw:
        return RED_BUCKET_DEFAULT_TASK_TYPES.get(bucket_name, "")
    if raw in ALLOWED_TYPES["RED"]:
        return raw
    key = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    canonical = RED_TASK_TYPE_ALIASES.get(key)
    if canonical:
        return canonical
    compact = key.replace("_", "")
    canonical = RED_TASK_TYPE_ALIASES.get(compact)
    if canonical:
        return canonical
    return raw


def _format_unit_roster(units: List[dict]) -> str:
    if not units:
        return "No unit roster available."
    lines = []
    for unit in units[:24]:
        lines.append(
            " | ".join(
                [
                    f"unit_id={unit.get('unit_id')}",
                    f"type={unit.get('unit_type')}",
                    f"role={unit.get('role')}",
                    f"available={unit.get('available')}",
                    f"ammo_high={unit.get('ammo_high_cost', 0)}",
                    f"ammo_low={unit.get('ammo_low_cost', 0)}",
                    f"can_fire_now={unit.get('can_fire_now')}",
                    f"last_task={unit.get('last_task') or 'none'}",
                    f"candidates={','.join(unit.get('candidate_targets', [])[:3]) or 'none'}",
                ]
            )
        )
    return "\n".join(lines)


def _format_target_table(targets: List[dict]) -> str:
    if not targets:
        return "No targets in table."
    lines = []
    for target in targets[:16]:
        lines.append(
            " | ".join(
                [
                    f"target_id={target.get('target_id')}",
                    f"type={target.get('target_type')}",
                    f"value={target.get('value')}",
                    f"priority={target.get('priority')}",
                    f"window={target.get('attack_window')}",
                    f"in_range_trucks={','.join(target.get('in_range_trucks', [])[:4]) or 'none'}",
                    f"guides={','.join(target.get('guide_candidates', [])[:4]) or 'none'}",
                ]
            )
        )
    return "\n".join(lines)


def _format_task_board(task_board: dict) -> str:
    if not isinstance(task_board, dict) or not task_board:
        return "No active task board entries."
    lines = []
    for unit_id, task in sorted(task_board.items())[:24]:
        if not isinstance(task, dict):
            continue
        lines.append(
            f"{unit_id}: task={task.get('task_type')} target={task.get('target_id')} role={task.get('role')} "
            f"assigned_at={task.get('assigned_at')} expires_at={task.get('expires_at')} "
            f"reassignable={task.get('reassignable')}"
        )
    return "\n".join(lines) if lines else "No active task board entries."


def _default_allocation_plan() -> dict:
    return {
        "main_attack": [],
        "support_guidance": [],
        "scout_tasks": [],
        "reserve_units": [],
        "withheld_units": [],
        "allocation_notes": [],
    }


def _normalize_allocation_plan(plan: object) -> dict:
    normalized = _default_allocation_plan()
    if not isinstance(plan, dict):
        return normalized

    nested_plan = plan.get("allocation_plan")
    if isinstance(nested_plan, dict):
        plan = nested_plan

    for key in ("main_attack", "support_guidance", "scout_tasks"):
        raw_items = _get_allocation_bucket(plan, key)
        if isinstance(raw_items, dict):
            raw_items = [raw_items]
        elif isinstance(raw_items, list) and raw_items and all(isinstance(item, str) for item in raw_items):
            default_task_type = RED_BUCKET_DEFAULT_TASK_TYPES.get(key, "")
            raw_items = [
                {
                    "task_type": default_task_type,
                    "unit_ids": [item for item in raw_items if isinstance(item, str) and item],
                    "target_id": "",
                    "attack_window": "",
                }
            ]
        if not isinstance(raw_items, list):
            continue
        grouped_items = {}
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            task_type_raw = item.get("task_type") or item.get("action") or item.get("action_type") or ""
            task_type = _normalize_red_task_type(task_type_raw, key)
            target_id = str(item.get("target_id", "")).strip()
            attack_window = str(item.get("attack_window", "")).strip()
            area = item.get("area") if isinstance(item.get("area"), dict) else item.get("Area") if isinstance(item.get("Area"), dict) else None
            unit_ids = [str(v) for v in item.get("unit_ids", []) if isinstance(v, str) and v]
            if isinstance(item.get("UnitIds"), list):
                unit_ids.extend(str(v) for v in item.get("UnitIds", []) if isinstance(v, str) and v)
            single_unit = item.get("unit_id") or item.get("UnitId")
            if isinstance(single_unit, str) and single_unit:
                unit_ids.append(single_unit)
            key_id = (
                task_type,
                target_id,
                attack_window,
                json.dumps(area, ensure_ascii=False, sort_keys=True) if area else "",
            )
            grouped = grouped_items.setdefault(
                key_id,
                {
                    "task_type": task_type,
                    "unit_ids": [],
                    "target_id": target_id,
                    "attack_window": attack_window,
                    "loiter_radius_km": item.get("loiter_radius_km", item.get("Loiter_Radius_km", 30)),
                    "area": area,
                    "purpose": str(item.get("purpose", "")).strip(),
                    "notes": str(item.get("notes", "")).strip(),
                },
            )
            for unit_id in unit_ids:
                if unit_id and unit_id not in grouped["unit_ids"]:
                    grouped["unit_ids"].append(unit_id)
        normalized[key] = list(grouped_items.values())

    reserve_candidates = plan.get("reserve_units")
    if reserve_candidates is None:
        reserve_candidates = plan.get("reserve")
    withheld_candidates = plan.get("withheld_units")
    if withheld_candidates is None:
        withheld_candidates = plan.get("withheld")

    for key, raw_values in (("reserve_units", reserve_candidates), ("withheld_units", withheld_candidates)):
        values = []
        if isinstance(raw_values, str):
            raw_values = [raw_values]
        elif isinstance(raw_values, dict):
            raw_values = [raw_values]
        if isinstance(raw_values, list):
            for value in raw_values:
                if isinstance(value, str) and value:
                    values.append(value)
                elif isinstance(value, dict):
                    unit_id = value.get("unit_id") or value.get("UnitId")
                    if isinstance(unit_id, str) and unit_id:
                        values.append(unit_id)
        normalized[key] = values
    raw_notes = plan.get("allocation_notes", [])
    if isinstance(raw_notes, str):
        raw_notes = [raw_notes]
    elif isinstance(raw_notes, dict):
        raw_notes = [
            str(raw_notes.get("notes") or raw_notes.get("summary") or raw_notes.get("text") or "").strip()
        ]
    normalized["allocation_notes"] = [str(v).strip() for v in raw_notes if isinstance(v, str) and str(v).strip()]
    return normalized


def _default_scout_area_dict(state: AgentState, index: int) -> dict:
    unit_roster = state.get("unit_roster", []) or []
    points = []
    for unit in unit_roster:
        if not isinstance(unit, dict):
            continue
        pos = unit.get("position")
        if not isinstance(pos, dict):
            continue
        lon = pos.get("lon")
        lat = pos.get("lat")
        if lon is None or lat is None:
            continue
        points.append((float(lon), float(lat)))
    if points:
        center_lon = sum(point[0] for point in points) / len(points)
        center_lat = sum(point[1] for point in points) / len(points)
    else:
        center_lon = 48.0
        center_lat = 13.0
    offsets = [(0.45, 0.30), (0.75, 0.0), (0.45, -0.30), (-0.15, 0.0)]
    delta_lon, delta_lat = offsets[index % len(offsets)]
    center_lon += delta_lon
    center_lat += delta_lat
    return {
        "TopLeft": {"lon": center_lon - 0.25, "lat": center_lat + 0.18},
        "BottomRight": {"lon": center_lon + 0.25, "lat": center_lat - 0.18},
    }


def _hydrate_allocation_plan(plan: dict, state: AgentState) -> dict:
    hydrated = _normalize_allocation_plan(plan)
    truck_units = [
        str(unit.get("unit_id"))
        for unit in state.get("unit_roster", []) or []
        if isinstance(unit, dict) and unit.get("role") == "fire" and unit.get("alive", True)
    ]
    target_windows = {
        str(target.get("target_id")): str(target.get("attack_window", ""))
        for target in state.get("target_table", []) or []
        if isinstance(target, dict) and str(target.get("target_id", "")).strip()
    }
    sorted_targets = [
        target
        for target in state.get("target_table", []) or []
        if isinstance(target, dict) and str(target.get("target_id", "")).strip()
    ]
    sorted_targets.sort(
        key=lambda item: (
            0 if str(item.get("attack_window", "")) == "fire_now" else 1 if str(item.get("attack_window", "")) == "move_then_fire" else 2,
            -int(item.get("priority", 0) or 0),
            -(float(item.get("value", 0.0) or 0.0)),
        )
    )
    default_attack_target = str(sorted_targets[0].get("target_id")) if sorted_targets else ""

    for item in hydrated.get("main_attack", []) or []:
        if not isinstance(item, dict):
            continue
        if not str(item.get("target_id", "")).strip():
            item["target_id"] = default_attack_target
        if not str(item.get("task_type", "")).strip():
            item["task_type"] = "FocusFire"

    preferred_guidance_target = next(
        (
            str(item.get("target_id"))
            for item in hydrated.get("main_attack", []) or []
            if isinstance(item, dict) and str(item.get("target_id", "")).strip()
        ),
        default_attack_target,
    )
    for item in hydrated.get("support_guidance", []) or []:
        if not isinstance(item, dict):
            continue
        if not str(item.get("target_id", "")).strip():
            item["target_id"] = preferred_guidance_target
        if not str(item.get("task_type", "")).strip():
            item["task_type"] = "GuideAttack"

    scout_target_cycle = [
        str(target.get("target_id"))
        for target in sorted_targets
        if str(target.get("target_id", "")).strip()
    ]
    scout_index = 0
    for item in hydrated.get("scout_tasks", []) or []:
        if not isinstance(item, dict):
            continue
        if not str(item.get("target_id", "")).strip() and scout_target_cycle:
            item["target_id"] = scout_target_cycle[scout_index % len(scout_target_cycle)]
            scout_index += 1
        if not str(item.get("task_type", "")).strip():
            item["task_type"] = "ScoutArea"

    overflow_units = []
    for item in hydrated.get("main_attack", []) or []:
        if not isinstance(item, dict):
            continue
        target_id = str(item.get("target_id", "")).strip()
        attack_window = str(item.get("attack_window", "")).strip() or target_windows.get(target_id, "")
        item["attack_window"] = attack_window
        unit_ids = [unit_id for unit_id in item.get("unit_ids", []) or [] if isinstance(unit_id, str)]
        group_limit = _red_group_size_limit(target_id, attack_window)
        if attack_window == "move_then_fire":
            group_limit = min(group_limit, _allowed_move_then_fire_commit(len(truck_units)))
        if len(unit_ids) > group_limit:
            overflow_units.extend(unit_ids[group_limit:])
            unit_ids = unit_ids[:group_limit]
        item["unit_ids"] = unit_ids
    committed = {
        unit_id
        for item in hydrated.get("main_attack", []) or []
        if isinstance(item, dict)
        for unit_id in item.get("unit_ids", []) or []
        if isinstance(unit_id, str)
    }
    reserve_candidates = [unit_id for unit_id in truck_units if unit_id not in committed]
    reserve_min = max(1, len(truck_units) // 3) if truck_units else 0
    merged_reserve = []
    for unit_id in list(hydrated.get("reserve_units", []) or []) + overflow_units + reserve_candidates:
        if isinstance(unit_id, str) and unit_id and unit_id not in merged_reserve:
            merged_reserve.append(unit_id)
    if reserve_min:
        hydrated["reserve_units"] = merged_reserve[: max(reserve_min, len(overflow_units))]
    else:
        hydrated["reserve_units"] = merged_reserve
    for index, item in enumerate(hydrated.get("scout_tasks", []) or []):
        if not isinstance(item, dict):
            continue
        if not _is_area(item.get("area")):
            item["area"] = _default_scout_area_dict(state, index)
    return hydrated


def _translate_allocation_plan_to_actions(state: AgentState) -> List[dict]:
    plan = _hydrate_allocation_plan(state.get("allocation_plan") or {}, state)
    target_windows = {
        str(target.get("target_id")): str(target.get("attack_window", ""))
        for target in state.get("target_table", []) or []
        if isinstance(target, dict) and str(target.get("target_id", "")).strip()
    }
    actions = []
    used_units = set()

    for item in plan.get("main_attack", []) or []:
        if not isinstance(item, dict):
            continue
        unit_ids = [unit_id for unit_id in item.get("unit_ids", []) or [] if isinstance(unit_id, str) and unit_id and unit_id not in used_units]
        target_id = str(item.get("target_id", "")).strip()
        if not unit_ids or not target_id:
            continue
        attack_window = str(item.get("attack_window", "")).strip() or target_windows.get(target_id, "")
        if attack_window == "observe_only":
            continue
        task_type = str(item.get("task_type", "")).strip()
        if attack_window == "move_then_fire":
            action_type = "MoveToEngage"
        else:
            action_type = task_type if task_type in {"FocusFire", "ShootAndScoot"} else "FocusFire"
        actions.append(
            {
                "Type": action_type,
                "UnitIds": unit_ids,
                "Target_Id": target_id,
            }
        )
        used_units.update(unit_ids)

    for item in plan.get("support_guidance", []) or []:
        if not isinstance(item, dict):
            continue
        unit_ids = [unit_id for unit_id in item.get("unit_ids", []) or [] if isinstance(unit_id, str) and unit_id and unit_id not in used_units]
        target_id = str(item.get("target_id", "")).strip()
        if not unit_ids or not target_id:
            continue
        actions.append(
            {
                "Type": "GuideAttack",
                "UnitIds": unit_ids,
                "Target_Id": target_id,
                "Loiter_Radius_km": item.get("loiter_radius_km", 30),
            }
        )
        used_units.update(unit_ids)

    for index, item in enumerate(plan.get("scout_tasks", []) or []):
        if not isinstance(item, dict):
            continue
        unit_ids = [unit_id for unit_id in item.get("unit_ids", []) or [] if isinstance(unit_id, str) and unit_id and unit_id not in used_units]
        if not unit_ids:
            continue
        area = item.get("area") if _is_area(item.get("area")) else _default_scout_area_dict(state, index)
        actions.append(
            {
                "Type": "ScoutArea",
                "UnitIds": unit_ids,
                "Area": area,
            }
        )
        used_units.update(unit_ids)
    return actions


def _normalize_operator_actions_output(raw_actions: object, state: AgentState) -> List[dict]:
    actions = raw_actions
    if isinstance(actions, dict):
        actions = actions.get("actions", [])
    normalized = []
    if isinstance(actions, list):
        for index, action in enumerate(actions):
            if not isinstance(action, dict):
                continue
            action_type_raw = action.get("Type") or action.get("action") or action.get("action_type") or ""
            action_type = _normalize_red_task_type(action_type_raw)
            if not action_type:
                continue
            unit_ids = [str(v) for v in action.get("UnitIds", []) if isinstance(v, str) and v]
            if isinstance(action.get("unit_ids"), list):
                unit_ids.extend(str(v) for v in action.get("unit_ids", []) if isinstance(v, str) and v)
            single_unit = action.get("unit_id") or action.get("UnitId")
            if isinstance(single_unit, str) and single_unit:
                unit_ids.append(single_unit)
            item = {
                "Type": action_type,
            }
            if unit_ids:
                item["UnitIds"] = list(dict.fromkeys(unit_ids))
            target_id = str(action.get("Target_Id") or action.get("target_id") or "").strip()
            if target_id:
                item["Target_Id"] = target_id
            area = (
                action.get("Area")
                if isinstance(action.get("Area"), dict)
                else action.get("area")
                if isinstance(action.get("area"), dict)
                else action.get("target_area")
                if isinstance(action.get("target_area"), dict)
                else None
            )
            move_area = action.get("Move_Area") if isinstance(action.get("Move_Area"), dict) else action.get("move_area") if isinstance(action.get("move_area"), dict) else None
            if action_type == "ScoutArea":
                item["Area"] = area if _is_area(area) else _default_scout_area_dict(state, index)
            elif _is_area(move_area):
                item["Move_Area"] = move_area
            radius = action.get("Loiter_Radius_km", action.get("loiter_radius_km"))
            if radius is not None:
                item["Loiter_Radius_km"] = radius
            normalized.append(item)
    if normalized and all(isinstance(item, dict) and item.get("Type") for item in normalized):
        return normalized
    return _translate_allocation_plan_to_actions(state)


def _summarize_allocation_plan(plan: dict) -> str:
    if not isinstance(plan, dict):
        return "allocation_plan=invalid"
    attack = len(plan.get("main_attack", []) or [])
    guidance = len(plan.get("support_guidance", []) or [])
    scout = len(plan.get("scout_tasks", []) or [])
    reserve = len(plan.get("reserve_units", []) or [])
    withheld = len(plan.get("withheld_units", []) or [])
    notes = "; ".join((plan.get("allocation_notes") or [])[:3]) or "none"
    return (
        f"main_attack={attack} support_guidance={guidance} scout_tasks={scout} "
        f"reserve_units={reserve} withheld_units={withheld} notes={truncate_text(notes, 120)}"
    )


def _validate_red_allocation_plan(state: AgentState) -> List[str]:
    plan = _normalize_allocation_plan(state.get("allocation_plan"))
    target_table = state.get("target_table") or []
    target_map = {
        str(target.get("target_id")): target
        for target in target_table
        if isinstance(target, dict) and str(target.get("target_id", "")).strip()
    }
    fire_now_targets = {
        str(target.get("target_id"))
        for target in target_table
        if isinstance(target, dict) and str(target.get("attack_window")) == "fire_now"
    }
    move_then_fire_targets = {
        str(target.get("target_id"))
        for target in target_table
        if isinstance(target, dict) and str(target.get("attack_window")) == "move_then_fire"
    }
    observe_only_targets = {
        str(target.get("target_id"))
        for target in target_table
        if isinstance(target, dict) and str(target.get("attack_window")) == "observe_only"
    }
    locked_tasks = _locked_task_entries(state.get("task_board") or {})

    issues: List[str] = []
    used_units = set()
    duplicate_found = False
    attack_targets = set()
    attack_units = set()
    scout_entries = plan.get("scout_tasks", []) or []
    guidance_entries = plan.get("support_guidance", []) or []
    fire_attack_entries = []
    available_guides = {
        str(unit.get("unit_id"))
        for unit in state.get("unit_roster", []) or []
        if isinstance(unit, dict) and unit.get("role") == "guide" and unit.get("alive", True) and unit.get("available", True)
    }

    for bucket in ("main_attack", "support_guidance", "scout_tasks"):
        for item in plan.get(bucket, []) or []:
            if not isinstance(item, dict):
                continue
            task_type = str(item.get("task_type", "")).strip()
            target_id = str(item.get("target_id", "")).strip()
            attack_window = str(item.get("attack_window", "")).strip() or str(
                target_map.get(target_id, {}).get("attack_window", "")
            ).strip()
            unit_ids = [str(unit_id) for unit_id in item.get("unit_ids", []) if isinstance(unit_id, str)]
            if task_type not in {"FocusFire", "ShootAndScoot", "MoveToEngage", "GuideAttack", "ScoutArea"}:
                issues.append(f"invalid_allocation_task_type:{task_type or 'missing'}")
            for unit_id in unit_ids:
                if unit_id in used_units:
                    duplicate_found = True
                used_units.add(unit_id)
                locked_task = locked_tasks.get(unit_id)
                if locked_task and (
                    str(locked_task.get("task_type", "")).strip() != task_type
                    or str(locked_task.get("target_id", "")).strip() != target_id
                ):
                    issues.append("locked_task_reassigned")
            if bucket == "main_attack":
                if task_type not in {"FocusFire", "ShootAndScoot", "MoveToEngage"}:
                    issues.append("invalid_role_bucket_assignment")
                if any(not unit_id.startswith("Truck_Ground") for unit_id in unit_ids):
                    issues.append("invalid_role_bucket_assignment")
                if target_id in attack_targets:
                    issues.append("unauthorized_target_rewrite")
                attack_targets.add(target_id)
                attack_units.update(unit_ids)
                if unit_ids and len(unit_ids) > _red_group_size_limit(target_id, attack_window):
                    issues.append("over_sized_attack_group")
                if task_type in {"FocusFire", "ShootAndScoot"}:
                    fire_attack_entries.append(item)
                if attack_window == "observe_only" and task_type in {"FocusFire", "ShootAndScoot"}:
                    issues.append("observe_only_fire_action")
                if attack_window == "move_then_fire" and task_type != "MoveToEngage":
                    issues.append("invalid_attack_window_policy")
            elif bucket == "support_guidance":
                if task_type != "GuideAttack" or any(not unit_id.startswith("Guide_Ship_Surface") for unit_id in unit_ids):
                    issues.append("invalid_role_bucket_assignment")
            elif bucket == "scout_tasks":
                if task_type != "ScoutArea" or any(not unit_id.startswith("Recon_UAV_FixWing") for unit_id in unit_ids):
                    issues.append("invalid_role_bucket_assignment")

    if duplicate_found:
        issues.append("duplicate_unit_assignment")

    truck_units = {
        str(unit.get("unit_id"))
        for unit in state.get("unit_roster", []) or []
        if isinstance(unit, dict) and unit.get("role") == "fire" and unit.get("alive", True)
    }
    reserve_units = set(plan.get("reserve_units", []) or []) & truck_units
    reserve_min = max(1, len(truck_units) // 3) if truck_units else 0
    if reserve_min and len(reserve_units) < reserve_min:
        issues.append("missing_reserve_force")
    if truck_units:
        max_commit = max(0, len(truck_units) - reserve_min)
        committed = attack_units & truck_units
        if len(committed) > max_commit:
            issues.append("over_committed_trucks")
        if len(committed) > 1 and len({t for t in attack_targets if t}) == 1 and committed == truck_units:
            issues.append("all_trucks_single_target")
        move_then_fire_committed = {
            unit_id
            for item in plan.get("main_attack", []) or []
            if isinstance(item, dict)
            and str(item.get("target_id", "")).strip() in move_then_fire_targets
            and str(item.get("task_type", "")).strip() == "MoveToEngage"
            for unit_id in item.get("unit_ids", []) or []
            if isinstance(unit_id, str) and unit_id in truck_units
        }
        if len(move_then_fire_committed) > _allowed_move_then_fire_commit(len(truck_units)):
            issues.append("over_committed_trucks")

    if fire_now_targets:
        fire_capable = any(
            str(item.get("target_id", "")).strip() in fire_now_targets
            and str(item.get("task_type", "")).strip() in {"FocusFire", "ShootAndScoot"}
            for item in fire_attack_entries
        )
        if not fire_capable:
            issues.append("missing_fire_action_with_valid_window")
        guidance_targets = {
            str(item.get("target_id", "")).strip()
            for item in guidance_entries
            if isinstance(item, dict) and str(item.get("target_id", "")).strip()
        }
        guided_primary_targets = {
            str(item.get("target_id", "")).strip()
            for item in fire_attack_entries
            if isinstance(item, dict)
            and str(item.get("target_id", "")).strip() in fire_now_targets
            and _is_red_ship_target(str(item.get("target_id", "")).strip())
        }
        if available_guides and guided_primary_targets and not (guidance_targets & guided_primary_targets):
            issues.append("missing_guidance_support_for_ship_attack")

    if observe_only_targets and len(observe_only_targets) == len(target_map):
        if fire_attack_entries:
            issues.append("observe_only_fire_action")

    guidance_targets = {
        str(item.get("target_id", "")).strip()
        for item in guidance_entries
        if isinstance(item, dict) and str(item.get("target_id", "")).strip()
    }
    if guidance_targets and len(guidance_targets) == 1 and not scout_entries:
        issues.append("missing_lateral_scout")

    deduped: List[str] = []
    for issue in issues:
        if issue not in deduped:
            deduped.append(issue)
    return deduped


def _validate_red_operator_output(state: AgentState) -> List[str]:
    actions_json = state.get("actions_json") or []
    allocation_plan = _normalize_allocation_plan(state.get("allocation_plan") or {})
    task_board = state.get("task_board") or {}
    locked_tasks = _locked_task_entries(task_board)
    issues = validate_red_plan_semantics(
        actions_json,
        validation_context={
            "allocation_plan": allocation_plan,
            "unit_roster": state.get("unit_roster") or [],
            "target_table": state.get("target_table") or [],
            "task_board": task_board,
        },
    )
    if issues:
        return issues

    plan_bucket_by_unit = {}
    plan_targets_by_bucket = {"main_attack": set(), "support_guidance": set(), "scout_tasks": set()}
    for bucket_name, item_key in (("main_attack", "unit_ids"), ("support_guidance", "unit_ids"), ("scout_tasks", "unit_ids")):
        for item in allocation_plan.get(bucket_name, []) or []:
            if not isinstance(item, dict):
                continue
            for unit_id in item.get(item_key, []) or []:
                if isinstance(unit_id, str) and unit_id:
                    plan_bucket_by_unit[unit_id] = bucket_name
            target_id = str(item.get("target_id", "")).strip()
            if target_id:
                plan_targets_by_bucket[bucket_name].add(target_id)

    scout_actions = [a for a in actions_json if isinstance(a, dict) and a.get("Type") == "ScoutArea"]
    guide_actions = [a for a in actions_json if isinstance(a, dict) and a.get("Type") == "GuideAttack"]
    fire_actions = [a for a in actions_json if isinstance(a, dict) and a.get("Type") in {"FocusFire", "ShootAndScoot"}]
    move_actions = [a for a in actions_json if isinstance(a, dict) and a.get("Type") == "MoveToEngage"]

    for action in actions_json:
        if not isinstance(action, dict):
            continue
        action_type = str(action.get("Type", "")).strip()
        target_id = str(action.get("Target_Id", "")).strip()
        units = [unit_id for unit_id in action.get("UnitIds", []) or [] if isinstance(unit_id, str)]
        if action_type in {"FocusFire", "ShootAndScoot", "MoveToEngage"}:
            expected_bucket = "main_attack"
        elif action_type == "GuideAttack":
            expected_bucket = "support_guidance"
        elif action_type == "ScoutArea":
            expected_bucket = "scout_tasks"
        else:
            expected_bucket = ""

        for unit_id in units:
            if expected_bucket and plan_bucket_by_unit.get(unit_id) != expected_bucket:
                return ["invalid_role_bucket_assignment"]
            locked_task = locked_tasks.get(unit_id)
            if locked_task and (
                str(locked_task.get("task_type", "")).strip() != action_type
                or str(locked_task.get("target_id", "")).strip() != target_id
            ):
                return ["locked_task_reassigned"]

        if target_id and expected_bucket and target_id not in plan_targets_by_bucket.get(expected_bucket, set()):
            return ["unauthorized_target_rewrite"]

    if fire_actions:
        fire_targets = {str(a.get("Target_Id", "")).strip() for a in fire_actions if str(a.get("Target_Id", "")).strip()}
        truck_units = {
            unit_id
            for action in fire_actions
            for unit_id in action.get("UnitIds", [])
            if isinstance(unit_id, str) and unit_id.startswith("Truck_Ground")
        }
        roster_trucks = {
            str(unit.get("unit_id"))
            for unit in state.get("unit_roster", []) or []
            if isinstance(unit, dict) and unit.get("role") == "fire" and unit.get("alive", True)
        }
        if roster_trucks and truck_units == roster_trucks and len(fire_targets) == 1:
            return ["all_trucks_single_target"]
        for action in fire_actions:
            target_id = str(action.get("Target_Id", "")).strip()
            target_info = next(
                (
                    target
                    for target in state.get("target_table", []) or []
                    if isinstance(target, dict) and str(target.get("target_id", "")).strip() == target_id
                ),
                {},
            )
            attack_window = str(target_info.get("attack_window", "")).strip()
            if len([u for u in action.get("UnitIds", []) or [] if isinstance(u, str)]) > _red_group_size_limit(target_id, attack_window):
                return ["over_sized_attack_group"]
            if attack_window == "observe_only":
                return ["observe_only_fire_action"]

    fire_now_targets = {
        str(target.get("target_id"))
        for target in state.get("target_table", []) or []
        if isinstance(target, dict) and str(target.get("attack_window")) == "fire_now"
    }
    if fire_now_targets and not fire_actions and (scout_actions or guide_actions):
        return ["missing_fire_action_with_valid_window"]
    move_then_fire_targets = {
        str(target.get("target_id"))
        for target in state.get("target_table", []) or []
        if isinstance(target, dict) and str(target.get("attack_window")) == "move_then_fire"
    }
    roster_trucks = {
        str(unit.get("unit_id"))
        for unit in state.get("unit_roster", []) or []
        if isinstance(unit, dict) and unit.get("role") == "fire" and unit.get("alive", True)
    }
    move_then_fire_committed = {
        unit_id
        for action in move_actions
        if str(action.get("Target_Id", "")).strip() in move_then_fire_targets
        for unit_id in action.get("UnitIds", []) or []
        if isinstance(unit_id, str) and unit_id in roster_trucks
    }
    if len(move_then_fire_committed) > _allowed_move_then_fire_commit(len(roster_trucks)):
        return ["over_committed_trucks"]
    return []


def _validate_actions(state: AgentState) -> Tuple[bool, str, str]:
    actions_json = state.get("actions_json")
    faction_name = state["faction_name"]
    if not isinstance(actions_json, list):
        return False, "Output must be a JSON array.", "operator"

    side = _normalize_faction(faction_name)
    if side == "RED":
        allocation_issues = _validate_red_allocation_plan(state)
        if allocation_issues:
            return False, f"Allocation plan violation: {', '.join(allocation_issues)}.", "allocator"

    if not actions_json:
        if side == "RED":
            output_issues = _validate_red_operator_output(state)
            if output_issues:
                return False, f"Operator output violation: {', '.join(output_issues)}.", "operator"
        return True, "", ""

    allowed = ALLOWED_TYPES[side]

    for idx, action in enumerate(actions_json):
        if not isinstance(action, dict):
            return False, f"Action#{idx} is not an object.", "operator"

        action_type = action.get("Type")
        if action_type not in allowed:
            return False, f"Action#{idx} has invalid Type={action_type}.", "operator"

        if side == "RED":
            if action_type == "ScoutArea" and (not isinstance(action.get("UnitIds"), list) or not _is_area(action.get("Area"))):
                return False, f"Action#{idx} ScoutArea missing UnitIds/Area.", "operator"
            if action_type == "MoveToEngage" and (not isinstance(action.get("UnitIds"), list) or not action.get("Target_Id")):
                return False, f"Action#{idx} MoveToEngage missing UnitIds/Target_Id.", "operator"
            if action_type == "GuideAttack" and (not isinstance(action.get("UnitIds"), list) or not action.get("Target_Id")):
                return False, f"Action#{idx} GuideAttack missing UnitIds/Target_Id.", "operator"
            if action_type == "FocusFire" and (not isinstance(action.get("UnitIds"), list) or not action.get("Target_Id")):
                return False, f"Action#{idx} FocusFire missing UnitIds/Target_Id.", "operator"
            if action_type == "ShootAndScoot" and (not isinstance(action.get("UnitIds"), list) or not action.get("Target_Id")):
                return False, f"Action#{idx} ShootAndScoot missing UnitIds/Target_Id.", "operator"
            semantic_issues = validate_red_action_semantics(action)
            if semantic_issues:
                return False, f"Action#{idx} semantic violation: {', '.join(semantic_issues)}.", "operator"
        else:
            if action_type == "ScoutArea" and (not isinstance(action.get("UnitIds"), list) or not _is_area(action.get("Area"))):
                return False, f"Action#{idx} ScoutArea missing UnitIds/Area.", "operator"
            if action_type == "SmartLaunchOnTarget" and (not action.get("UnitId") or not action.get("Target_Id")):
                return False, f"Action#{idx} SmartLaunchOnTarget missing UnitId/Target_Id.", "operator"
            if action_type == "AntiMissileDefense" and (
                not isinstance(action.get("UnitIds"), list) or not _is_area(action.get("Protected_Area"))
            ):
                return False, f"Action#{idx} AntiMissileDefense missing UnitIds/Protected_Area.", "operator"
            if action_type == "FocusFire":
                if not isinstance(action.get("UnitIds"), list):
                    return False, f"Action#{idx} FocusFire missing UnitIds.", "operator"
                if not action.get("Target_Id"):
                    return False, f"Action#{idx} FocusFire missing Target_Id.", "operator"
            if action_type == "EngageAndReposition" and (
                not isinstance(action.get("UnitIds"), list) or not _is_area(action.get("Move_Area"))
            ):
                return False, f"Action#{idx} EngageAndReposition missing UnitIds/Move_Area.", "operator"

    if side == "RED":
        output_issues = _validate_red_operator_output(state)
        if output_issues:
            return False, f"Operator output violation: {', '.join(output_issues)}.", "operator"

    return True, "", ""


def _timeline_text(events: List[dict]) -> str:
    return "\n".join(
        [
            f"{e.get('sim_time')}: {e.get('name')} unit={e.get('unit_id')} target={e.get('target_id')}"
            for e in (events or [])[-12:]
        ]
    )


def _node_event(
    node: str,
    duration_ms: int,
    output_summary: str,
    retry_count: int = 0,
    status: str = "ok",
    **extra,
) -> dict:
    event = {
        "node": node,
        "duration_ms": duration_ms,
        "output_summary": truncate_text(output_summary, 240),
        "retry_count": retry_count,
        "status": status,
    }
    event.update(extra)
    return event


async def analyst_node(state: AgentState, llm_manager):
    started = perf_counter()
    prompt_started = perf_counter()
    static_prefix, dynamic_payload = compile_analyst_prompt(state)
    prompt_build_ms = int((perf_counter() - prompt_started) * 1000)
    findings = await llm_manager.async_role_chat(
        static_prefix,
        dynamic_payload,
        state["faction_name"],
        role_profile="analyst",
        trace_ctx={
            "trace_id": state.get("trace_id"),
            "component": "analyst",
            "sim_time": state["sim_time"],
        },
    )
    duration_ms = int((perf_counter() - started) * 1000)
    return {
        "key_findings": findings or "",
        "logs": ["Analyst: key findings ready."],
        "trace_events": [
            _node_event(
                "analyst",
                duration_ms,
                findings or "",
                prompt_build_ms=prompt_build_ms,
            )
        ],
    }


async def commander_node(state: AgentState, llm_manager):
    started = perf_counter()
    prompt_started = perf_counter()
    static_prefix, dynamic_payload = compile_commander_prompt(state)
    prompt_build_ms = int((perf_counter() - prompt_started) * 1000)
    intent = await llm_manager.async_role_chat(
        static_prefix,
        dynamic_payload,
        state["faction_name"],
        role_profile="commander",
        trace_ctx={
            "trace_id": state.get("trace_id"),
            "component": "commander",
            "sim_time": state["sim_time"],
        },
    )
    duration_ms = int((perf_counter() - started) * 1000)
    return {
        "intent": intent or "",
        "logs": ["Commander: intent generated."],
        "trace_events": [
            _node_event(
                "commander",
                duration_ms,
                intent or "",
                prompt_build_ms=prompt_build_ms,
            )
        ],
    }


async def allocator_node(state: AgentState, llm_manager):
    started = perf_counter()
    critique = state.get("critique", "") if state.get("retry_stage") == "allocator" else ""
    prompt_started = perf_counter()
    static_prefix, dynamic_payload = compile_allocator_prompt(state, critique=critique)
    prompt_build_ms = int((perf_counter() - prompt_started) * 1000)
    plan = await llm_manager.async_structured_gen(
        schema=ALLOCATOR_OUTPUT_SCHEMA,
        static_prefix=static_prefix,
        dynamic_payload=dynamic_payload,
        faction_name=state["faction_name"],
        role_profile="allocator",
        trace_ctx={
            "trace_id": state.get("trace_id"),
            "component": "allocator",
            "sim_time": state["sim_time"],
        },
    )
    postprocess_started = perf_counter()
    plan = _hydrate_allocation_plan(plan, state)
    summary = _summarize_allocation_plan(plan)
    postprocess_ms = int((perf_counter() - postprocess_started) * 1000)
    duration_ms = int((perf_counter() - started) * 1000)
    return {
        "allocation_plan": plan,
        "allocation_summary": summary,
        "logs": ["Allocator: structured allocation plan ready."],
        "trace_events": [
            _node_event(
                "allocator",
                duration_ms,
                summary,
                state.get("retry_count", 0),
                prompt_build_ms=prompt_build_ms,
                postprocess_ms=postprocess_ms,
            )
        ],
        "retry_stage": "",
        "critique": "" if state.get("retry_stage") == "allocator" else state.get("critique", ""),
    }


async def operator_node(state: AgentState, llm_manager):
    started = perf_counter()
    critique = state.get("critique", "") if state.get("retry_stage") == "operator" else ""
    operator_mode = "deterministic"
    operator_llm_skipped = True
    prompt_build_ms = 0
    postprocess_ms = 0
    translate_started = perf_counter()
    actions = _translate_allocation_plan_to_actions(state)
    deterministic_translate_ms = int((perf_counter() - translate_started) * 1000)
    validation_state = dict(state)
    validation_state["actions_json"] = actions
    deterministic_issues = _validate_red_operator_output(validation_state)
    need_llm_fallback = bool(state.get("allocation_plan")) and (
        state.get("retry_stage") == "operator" or not actions or bool(deterministic_issues)
    )

    if need_llm_fallback:
        operator_mode = "llm"
        operator_llm_skipped = False
        prompt_started = perf_counter()
        static_prefix, dynamic_payload = compile_operator_prompt(state, critique=critique)
        prompt_build_ms = int((perf_counter() - prompt_started) * 1000)
        raw_actions = await llm_manager.async_structured_gen(
            schema=OPERATOR_OUTPUT_SCHEMA,
            static_prefix=static_prefix,
            dynamic_payload=dynamic_payload,
            faction_name=state["faction_name"],
            role_profile="operator",
            trace_ctx={
                "trace_id": state.get("trace_id"),
                "component": "operator",
                "sim_time": state["sim_time"],
            },
        )
        postprocess_started = perf_counter()
        actions = _normalize_operator_actions_output(raw_actions, state)
        postprocess_ms = int((perf_counter() - postprocess_started) * 1000)

    duration_ms = int((perf_counter() - started) * 1000)
    return {
        "actions_json": actions,
        "planned_at_sim_time": int(state.get("sim_time", 0)),
        "logs": [f"Operator({operator_mode}): generated {len(actions)} candidate actions."],
        "trace_events": [
            _node_event(
                "operator",
                duration_ms,
                summarize_json_actions(actions),
                state.get("retry_count", 0),
                operator_mode=operator_mode,
                operator_llm_skipped=operator_llm_skipped,
                prompt_build_ms=prompt_build_ms,
                postprocess_ms=postprocess_ms,
                deterministic_translate_ms=deterministic_translate_ms,
            )
        ],
        "retry_stage": "",
    }


async def critic_node(state: AgentState):
    started = perf_counter()
    rule_started = perf_counter()
    ok, critique, retry_stage = _validate_actions(state)
    rule_check_ms = int((perf_counter() - rule_started) * 1000)
    duration_ms = int((perf_counter() - started) * 1000)
    if ok:
        return {
            "is_valid": True,
            "critique": "",
            "retry_stage": "",
            "logs": ["Critic: validation passed with memory constraints considered."],
            "trace_events": [
                _node_event(
                    "critic",
                    duration_ms,
                    "validation passed",
                    state.get("retry_count", 0),
                    rule_check_ms=rule_check_ms,
                )
            ],
        }
    return {
        "is_valid": False,
        "critique": critique,
        "retry_count": state.get("retry_count", 0) + 1,
        "retry_stage": retry_stage or "operator",
        "logs": [f"Critic: validation failed - {critique}"],
        "trace_events": [
            _node_event(
                "critic",
                duration_ms,
                f"{retry_stage or 'operator'} retry: {critique}",
                state.get("retry_count", 0) + 1,
                status="retry",
                rule_check_ms=rule_check_ms,
            )
        ],
    }


def _route_after_critic(state: AgentState):
    if state.get("is_valid", False):
        return END
    if state.get("retry_count", 0) >= MAX_RETRY:
        logger.warning(
            "[%s] Critic failed and reached max retry, returning current candidate actions.",
            state.get("faction_name", "UNKNOWN"),
        )
        return END
    return "allocator" if state.get("retry_stage") == "allocator" else "operator"


def build_agent_graph(llm_manager, faction_name: str):
    workflow = StateGraph(AgentState)

    async def call_analyst(state):
        return await analyst_node(state, llm_manager)

    async def call_commander(state):
        return await commander_node(state, llm_manager)

    async def call_allocator(state):
        return await allocator_node(state, llm_manager)

    async def call_operator(state):
        return await operator_node(state, llm_manager)

    workflow.add_node("analyst", call_analyst)
    workflow.add_node("commander", call_commander)
    workflow.add_node("allocator", call_allocator)
    workflow.add_node("operator", call_operator)
    workflow.add_node("critic", critic_node)

    workflow.set_entry_point("analyst")
    workflow.add_edge("analyst", "commander")
    workflow.add_edge("commander", "allocator")
    workflow.add_edge("allocator", "operator")
    workflow.add_edge("operator", "critic")
    workflow.add_conditional_edges("critic", _route_after_critic)

    logger.info("[%s] LangGraph multi-agent workflow initialized.", faction_name)
    return workflow.compile()
