import json
import logging
import os
from typing import Any, Dict, List, Tuple

from red_trace_helper import truncate_text

logger = logging.getLogger(__name__)


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        return int(raw)
    except Exception:
        logger.warning("Invalid %s=%s, fallback to %s", name, raw, default)
        return default


RED_PROMPT_TOP_TARGETS = _read_int_env("RED_PROMPT_TOP_TARGETS", 6)
RED_PROMPT_TOP_UNITS = _read_int_env("RED_PROMPT_TOP_UNITS", 10)


def _attack_window_rank(window: str) -> int:
    return {"fire_now": 0, "move_then_fire": 1, "observe_only": 2}.get(str(window or ""), 3)


def _compute_track_staleness(sim_time: Any, last_seen_time: Any) -> int:
    try:
        now = int(sim_time)
        last_seen = int(last_seen_time)
        return max(0, now - last_seen)
    except Exception:
        return 0


def _compact_targets(target_table: List[dict], sim_time: int = 0, limit: int = RED_PROMPT_TOP_TARGETS) -> List[dict]:
    targets = []
    for target in sorted(
        target_table or [],
        key=lambda item: (
            _attack_window_rank(item.get("attack_window")),
            int(item.get("priority", 99)),
            -int(item.get("value", 0)),
        ),
    )[:limit]:
        if not isinstance(target, dict):
            continue
        target_type = str(target.get("target_type", ""))
        guide_count = len(target.get("guide_candidates", []) or [])
        targets.append(
            {
                "target_id": target.get("target_id"),
                "target_type": target_type,
                "priority": target.get("priority"),
                "value": target.get("value"),
                "attack_window": target.get("attack_window"),
                "last_seen_time": target.get("last_seen_time"),
                "track_staleness_sec": _compute_track_staleness(sim_time, target.get("last_seen_time")),
                "in_range_truck_count": len(target.get("in_range_trucks", []) or []),
                "move_then_fire_truck_count": len(target.get("move_then_fire_trucks", []) or []),
                "guide_count": guide_count,
                "guidance_bonus_pp": 20 if target_type.endswith("_Surface") else 0,
                "high_cost_ship_hit_rate": {"unguided": 0.7, "guided": 0.9},
                "low_cost_ship_hit_rate": {"unguided": 0.4, "guided": 0.6},
                "guidance_recommended": guide_count > 0 and str(target.get("attack_window", "")) in {"fire_now", "move_then_fire"},
                "min_low_cost_flight_time_sec": target.get("min_low_cost_flight_time_sec"),
                "min_high_cost_flight_time_sec": target.get("min_high_cost_flight_time_sec"),
                "surface_motion_risk": target.get("surface_motion_risk", ""),
                "low_cost_surface_policy": target.get("low_cost_surface_policy", ""),
                "low_cost_surface_recommended": target.get("low_cost_surface_recommended", False),
                "position": target.get("position"),
            }
        )
    return targets


def _compact_units(unit_roster: List[dict], role: str = "", limit: int = RED_PROMPT_TOP_UNITS) -> List[dict]:
    units = []
    filtered = []
    for unit in unit_roster or []:
        if not isinstance(unit, dict):
            continue
        if role and unit.get("role") != role:
            continue
        filtered.append(unit)

    filtered.sort(
        key=lambda item: (
            0 if item.get("available") else 1,
            0 if item.get("can_fire_now") else 1,
            -int(item.get("ammo_high_cost", 0)),
            -int(item.get("ammo_low_cost", 0)),
            str(item.get("unit_id", "")),
        )
    )
    for unit in filtered[:limit]:
        units.append(
            {
                "unit_id": unit.get("unit_id"),
                "unit_type": unit.get("unit_type"),
                "role": unit.get("role"),
                "available": unit.get("available"),
                "alive": unit.get("alive"),
                "ammo_high_cost": unit.get("ammo_high_cost"),
                "ammo_low_cost": unit.get("ammo_low_cost"),
                "can_fire_now": unit.get("can_fire_now"),
                "candidate_targets": list(unit.get("candidate_targets", [])[:3]),
                "last_task": unit.get("last_task"),
                "last_task_target": unit.get("last_task_target"),
            }
        )
    return units


def _compact_task_board(task_board: Dict[str, dict], limit: int = 14) -> List[dict]:
    items = []
    for unit_id, task in sorted((task_board or {}).items())[:limit]:
        if not isinstance(task, dict):
            continue
        items.append(
            {
                "unit_id": unit_id,
                "task_type": task.get("task_type"),
                "target_id": task.get("target_id"),
                "role": task.get("role"),
                "assigned_at": task.get("assigned_at"),
                "expires_at": task.get("expires_at"),
                "reassignable": task.get("reassignable"),
            }
        )
    return items


def _compact_events(event_timeline: List[dict], limit: int = 10) -> List[dict]:
    events = []
    for event in (event_timeline or [])[-limit:]:
        if not isinstance(event, dict):
            continue
        events.append(
            {
                "sim_time": event.get("sim_time"),
                "name": event.get("name"),
                "unit_id": event.get("unit_id"),
                "target_id": event.get("target_id"),
                "detail": truncate_text(event.get("detail", ""), 120),
            }
        )
    return events



def _compact_engagement_memory(engagement_memory: List[dict], limit: int = 6) -> List[dict]:
    items = []
    for entry in (engagement_memory or [])[:limit]:
        if not isinstance(entry, dict):
            continue
        items.append(
            {
                "target_id": entry.get("target_id"),
                "target_type": entry.get("target_type"),
                "last_engage_time": entry.get("last_engage_time"),
                "last_action_type": entry.get("last_action_type"),
                "recent_high_cost_launches": entry.get("recent_high_cost_launches", 0),
                "recent_low_cost_launches": entry.get("recent_low_cost_launches", 0),
                "recent_attack_cost": entry.get("recent_attack_cost", 0.0),
                "recent_score_gain": entry.get("recent_score_gain", 0.0),
                "guidance_used_recently": entry.get("guidance_used_recently", False),
                "target_still_visible": entry.get("target_still_visible", False),
                "pending_bda": entry.get("pending_bda", False),
                "repeat_high_cost_risk": entry.get("repeat_high_cost_risk", "none"),
                "engagement_recommendation": entry.get("engagement_recommendation", ""),
                "track_staleness_sec": entry.get("track_staleness_sec", 0),
                "attack_window": entry.get("attack_window", ""),
            }
        )
    return items


def _compact_ltm_lessons_structured(lessons: List[dict], limit: int = 6) -> List[dict]:
    items = []
    for entry in (lessons or [])[:limit]:
        if not isinstance(entry, dict):
            continue
        items.append(
            {
                "lesson_id": entry.get("lesson_id"),
                "type": entry.get("type", "general"),
                "phase": entry.get("phase", ""),
                "target_type": entry.get("target_type", ""),
                "symptom": entry.get("symptom", ""),
                "trigger": entry.get("trigger", ""),
                "lesson": entry.get("lesson", ""),
                "score_pattern": entry.get("score_pattern", ""),
                "cost_risk": entry.get("cost_risk", ""),
                "hybrid_score": entry.get("hybrid_score", 0.0),
            }
        )
    return items
def _friendly_overview(unit_roster: List[dict]) -> dict:
    overview = {
        "fire_total": 0,
        "fire_available": 0,
        "fire_can_fire_now": 0,
        "guide_total": 0,
        "guide_available": 0,
        "scout_total": 0,
        "scout_available": 0,
    }
    for unit in unit_roster or []:
        if not isinstance(unit, dict):
            continue
        role = str(unit.get("role", ""))
        if role == "fire":
            overview["fire_total"] += 1
            if unit.get("available"):
                overview["fire_available"] += 1
            if unit.get("can_fire_now"):
                overview["fire_can_fire_now"] += 1
        elif role == "guide":
            overview["guide_total"] += 1
            if unit.get("available"):
                overview["guide_available"] += 1
        elif role == "scout":
            overview["scout_total"] += 1
            if unit.get("available"):
                overview["scout_available"] += 1
    return overview


def _dump_payload(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


ANALYST_STATIC_PREFIX = """
你是 RED 方的 analyst。
任务：基于当前态势、最近窗口事件和目标表，提炼供 commander 使用的关键判断。
请重点分析：
- 哪些高价值舰船目标当前最值得优先处理。
- 每个目标的 attack_window 是 fire_now、move_then_fire 还是 observe_only，这对下一步意味着什么。
- 航迹是否陈旧；若 track_staleness_sec 较高，应优先补侦察刷新。
- 对舰船目标，GuideAttack 会显著提升命中率；如果 guide/scout 还没跟上，要明确指出。
- 近期是否存在重复投入、收益不足、需要重评估的迹象。
输出要求：
- 用简洁中文输出 3 到 6 条关键判断。
- 聚焦态势与机会窗口，不要直接生成动作 JSON。
""".strip()



COMMANDER_STATIC_PREFIX = """
你是 RED 方的 commander。
任务：综合 analyst 结论、近期事件、长期经验和近期交战记忆，形成本轮作战意图。
请重点判断：
- 本轮主攻目标是谁，为什么。
- 是否应该继续当前主攻，还是先重评估、补引导、补侦察或切换次目标。
- 对舰船目标，如果已经具备 fire_now 窗口，应明确考虑 GuideAttack；引导可提升命中率约 20 个百分点。
- 如果近期已经为同一目标投入较多高成本火力但尚未见到收益，应倾向于“先补支援/先重评估”，而不是机械重复集火。
输出要求：
- 先给出一句总体 intent。
- 再给出 2 到 4 条 allocation_notes，说明主攻、支援、侦察与保留兵力的原则。
- 不要直接输出动作 JSON。
""".strip()



ALLOCATOR_STATIC_PREFIX = """
?? RED ?? allocator?
???? commander ?????????? JSON?
?????
- main_attack ???? Truck_Ground-*?
- support_guidance ???? Guide_Ship_Surface-*?
- scout_tasks ???? Recon_UAV_FixWing-*?
- reserve_units / withheld_units ??? unit id?
?????
- ??????GuideAttack ???????????? 70% ??? 90%?????? 40% ??? 60%?
- ?????????????????????????????????????????????????????????????
- ?? target ? low_cost_surface_policy ?? refresh_and_guide_first?guide_then_consider_low_cost ? move_closer_then_consider_low_cost????? GuideAttack?ScoutArea ? move_then_fire??????????? main_attack?
- ?? engagement_memory ? ltm_lessons_structured ????????????????????????????????
- ? fire_now ?????????? Guide_Ship_Surface ???????? support_guidance??? allocation_notes ??????
?????
- ????? JSON????????? main_attack?support_guidance?scout_tasks?reserve_units?withheld_units?allocation_notes?
- ?????? "allocation_plan"?
- allocation_notes ??????????? ["????", "?????????"]?
- ??? JSON??? Markdown????????
""".strip()

ALLOCATOR_STATIC_PREFIX += "\n\nCanonical task_type contract:\n- main_attack: FocusFire | ShootAndScoot | MoveToEngage\n- support_guidance: GuideAttack\n- scout_tasks: ScoutArea\n- Never invent aliases like fire_high_cost, guide_attack, refresh_track."


OPERATOR_STATIC_PREFIX = """
你是 RED 方的 operator。
任务：把 allocation_plan 翻译成高层动作 JSON，不创造 allocator 没有授权的新目标。
转换规则：
- main_attack: fire_now 映射为 FocusFire 或 ShootAndScoot；move_then_fire 映射为 MoveToEngage。
- support_guidance 映射为 GuideAttack。
- scout_tasks 映射为 ScoutArea。
- reserve_units 不输出动作。
输出要求：
- 严格输出动作 JSON object。
- 不输出底层 Move / Launch / ChangeState。
""".strip()


OPERATOR_STATIC_PREFIX += "\n\nCanonical action Type only:\n- FocusFire | ShootAndScoot | MoveToEngage | GuideAttack | ScoutArea\n- Never emit aliases like fire_high_cost, guide_attack, refresh_track."

ALLOCATOR_OUTPUT_SCHEMA = {
    "title": "red_allocation_plan",
    "type": "object",
    "properties": {
        "main_attack": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task_type": {"type": "string"},
                    "unit_ids": {"type": "array", "items": {"type": "string"}},
                    "target_id": {"type": "string"},
                    "attack_window": {"type": "string"},
                    "notes": {"type": "string"},
                },
                "required": ["task_type", "unit_ids", "target_id"],
                "additionalProperties": True,
            },
        },
        "support_guidance": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task_type": {"type": "string"},
                    "unit_ids": {"type": "array", "items": {"type": "string"}},
                    "target_id": {"type": "string"},
                    "loiter_radius_km": {"type": "number"},
                    "notes": {"type": "string"},
                },
                "required": ["task_type", "unit_ids", "target_id"],
                "additionalProperties": True,
            },
        },
        "scout_tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task_type": {"type": "string"},
                    "unit_ids": {"type": "array", "items": {"type": "string"}},
                    "area": {"type": "object"},
                    "purpose": {"type": "string"},
                    "notes": {"type": "string"},
                },
                "required": ["task_type", "unit_ids", "area"],
                "additionalProperties": True,
            },
        },
        "reserve_units": {"type": "array", "items": {"type": "string"}},
        "withheld_units": {"type": "array", "items": {"type": "string"}},
        "allocation_notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["main_attack", "support_guidance", "scout_tasks", "reserve_units", "withheld_units", "allocation_notes"],
    "additionalProperties": False,
}


OPERATOR_OUTPUT_SCHEMA = {
    "title": "red_operator_actions",
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "Type": {"type": "string"},
                    "UnitIds": {"type": "array", "items": {"type": "string"}},
                    "Target_Id": {"type": "string"},
                    "Loiter_Radius_km": {"type": "number"},
                    "Area": {"type": "object"},
                    "Move_Area": {"type": "object"},
                },
                "required": ["Type"],
                "additionalProperties": True,
            },
        }
    },
    "required": ["actions"],
    "additionalProperties": False,
}


def compile_analyst_prompt(state: dict) -> Tuple[str, str]:
    sim_time = int(state.get("sim_time") or 0)
    payload = {
        "sim_time": sim_time,
        "memory_window": state.get("memory_window"),
        "window_summary": truncate_text(state.get("window_summary", ""), 1200),
        "recent_events": _compact_events(state.get("event_timeline", [])),
        "top_targets": _compact_targets(state.get("target_table", []), sim_time=sim_time),
        "friendly_overview": _friendly_overview(state.get("unit_roster", [])),
    }
    return ANALYST_STATIC_PREFIX, _dump_payload(payload)


def compile_commander_prompt(state: dict) -> Tuple[str, str]:
    sim_time = int(state.get("sim_time") or 0)
    payload = {
        "sim_time": sim_time,
        "window_summary": truncate_text(state.get("window_summary", ""), 900),
        "recent_events": _compact_events(state.get("event_timeline", []), limit=8),
        "key_findings": truncate_text(state.get("key_findings", ""), 1200),
        "top_targets": _compact_targets(state.get("target_table", []), sim_time=sim_time),
        "friendly_overview": _friendly_overview(state.get("unit_roster", [])),
        "engagement_summary": truncate_text(state.get("engagement_summary", ""), 1200),
        "ltm_lessons": truncate_text(state.get("ltm_lessons", ""), 1400),
    }
    return COMMANDER_STATIC_PREFIX, _dump_payload(payload)


def compile_allocator_prompt(state: dict, critique: str = "") -> Tuple[str, str]:
    sim_time = int(state.get("sim_time") or 0)
    fire_units = _compact_units(state.get("unit_roster", []), role="fire")
    guide_units = _compact_units(state.get("unit_roster", []), role="guide", limit=4)
    scout_units = _compact_units(state.get("unit_roster", []), role="scout", limit=4)
    payload = {
        "sim_time": sim_time,
        "window_summary": truncate_text(state.get("window_summary", ""), 900),
        "key_findings": truncate_text(state.get("key_findings", ""), 900),
        "intent": truncate_text(state.get("intent", ""), 700),
        "top_targets": _compact_targets(state.get("target_table", []), sim_time=sim_time),
        "available_fire_units": fire_units,
        "guide_units": guide_units,
        "scout_units": scout_units,
        "task_board": _compact_task_board(state.get("task_board", {})),
        "engagement_memory": _compact_engagement_memory(state.get("engagement_memory", [])),
        "ltm_lessons_structured": _compact_ltm_lessons_structured(state.get("ltm_lessons_structured", [])),
        "guidance_policy": {
            "guidance_bonus_pp": 20,
            "high_cost_ship_hit_rate": {"unguided": 0.7, "guided": 0.9},
            "low_cost_ship_hit_rate": {"unguided": 0.4, "guided": 0.6},
            "default_rule": "存在 fire_now 舰船目标时，应优先补齐 support_guidance。",
            "low_cost_moving_surface_rule": "对运动舰船，低成本弹仅在有引导、航迹新鲜、预计飞行时间可接受时才值得主攻；否则优先引导、补侦察或先机动。",
        },
        "engagement_policy": {
            "repeat_high_cost_rule": "同一目标最近已发生高成本攻击且未见收益时，不应直接重复高成本集火。",
            "preferred_alternatives": [
                "先补 GuideAttack",
                "先补 Recon_UAV_FixWing 刷新航迹",
                "先切次目标",
                "先保持 move_then_fire/observe_only",
            ],
            "recommit_condition": "只有当目标仍是主目标且收益预期显著提升时，才允许再次投入高成本火力。",
        },
        "task_type_contract": {
            "main_attack_allowed": ["FocusFire", "ShootAndScoot", "MoveToEngage"],
            "support_guidance_allowed": ["GuideAttack"],
            "scout_tasks_allowed": ["ScoutArea"],
            "do_not_invent_aliases": True,
            "alias_examples": {
                "fire_high_cost": "FocusFire",
                "guide_attack": "GuideAttack",
                "refresh_track": "ScoutArea",
                "move_then_fire": "MoveToEngage",
            },
        },
        "critic_feedback": truncate_text(critique, 600),
    }
    return ALLOCATOR_STATIC_PREFIX, _dump_payload(payload)

def compile_operator_prompt(state: dict, critique: str = "") -> Tuple[str, str]:
    payload = {
        "sim_time": state.get("sim_time"),
        "allocation_summary": truncate_text(state.get("allocation_summary", ""), 700),
        "allocation_plan": state.get("allocation_plan", {}),
        "task_board": _compact_task_board(state.get("task_board", {})),
        "action_contract": {
            "allowed_types": ["FocusFire", "ShootAndScoot", "MoveToEngage", "GuideAttack", "ScoutArea"],
            "alias_examples": {
                "fire_high_cost": "FocusFire",
                "guide_attack": "GuideAttack",
                "refresh_track": "ScoutArea",
                "move_then_fire": "MoveToEngage",
            },
            "do_not_emit_low_level_actions": True,
        },
        "critic_feedback": truncate_text(critique, 600),
    }
    return OPERATOR_STATIC_PREFIX, _dump_payload(payload)


