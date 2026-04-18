# prompt_library.py
# 鍔ㄤ綔绌洪棿涓庣瓥鐣?prompt 搴?
# 2025-11-07 绋冲畾淇鐗?鈥斺€?鍔犲叆鐧藉悕鍗?榛戝悕鍗曠害鏉熴€佸幓娉ㄩ噴銆佹棩蹇楀寮?

import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# 绾㈡柟鍔ㄤ綔绌洪棿
# ------------------------------------------------------------
RED_ACTION_SPACE = """
"RedMissionActions": [
    {
      "Type": "ShootAndScoot",
      "UnitIds": ["truck_1", "truck_2", "..."],
      "Target_Id": "Blue_Target_Id"
       // 浠呴檺绾㈡柟 Truck_Ground锛堟満鍔ㄥ彂灏勮溅锛?
    },
    {
      "Type": "FocusFire",
      "UnitIds": ["truck_1", "truck_3"],
      "Target_Id": "Blue_Target_Id"
       // 浠呴檺绾㈡柟 Truck_Ground锛堟満鍔ㄥ彂灏勮溅锛?
    },
    {
      "Type": "MoveToEngage",
      "UnitIds": ["truck_1", "truck_4"],
      "Target_Id": "Blue_Target_Id"
       // 浠呴檺绾㈡柟 Truck_Ground锛堟満鍔ㄥ彂灏勮溅锛?
    },
    {
      "Type": "ScoutArea",
      "UnitIds": ["uav_1", "uav_2", "guide_ship_1", "guide_ship_2", "..."],
      "Area": {
        "TopLeft": {"lon": float, "lat": float},
        "BottomRight": {"lon": float, "lat": float}
      }
    }
    {
      "Type": "GuideAttack",
      "UnitIds": ["guide_ship_1", "uav_1", "uav_2", "..."],
      "Target_Id": "Blue_Target_Id",
      "Loiter_Radius_km": 30
      // 寮曞蹇墖 / 鏃犱汉鏈?鍦ㄧ洰鏍囬檮杩戝崐寰勭害 Loiter_Radius_km 鍐呮満鍔ㄧ洏鏃嬶紝
      // 鎻愪緵楂樿川閲忕洰鏍囦綅缃俊鎭紝閰嶅悎鎴戞柟鍙戝皠杞﹀疄鏂界簿纭墦鍑汇€?
    }
]
"""



# ------------------------------------------------------------
# 钃濇柟鍔ㄤ綔绌洪棿
# ------------------------------------------------------------
BLUE_ACTION_SPACE = """
"Flagship_Surface": [
    {"Type": "Move", "Id": "unit_id", "lon": float, "lat": float, "alt": float},
    {"Type": "SmartLaunchOnTarget", "UnitId": "unit_id", "Target_Id": "red_target_id"}
    {"Type": "SetJammer", "Id": "unit_id", "Pattern": "1 (寮€) | 0 (鍏?"}
],
"Destroyer_Surface": [
    {"Type": "Move", "Id": "unit_id", "lon": float, "lat": float, "alt": float},
    {"Type": "SmartLaunchOnTarget", "UnitId": "unit_id", "Target_Id": "red_target_id"}
    {"Type": "SetRadar", "Id": "unit_id", "isHideOn": "1 (寮€) | 0 (鍏?"},
    {"Type": "SetJammer", "Id": "unit_id", "Pattern": "1 (寮€) | 0 (鍏?"}
],
"Cruiser_Surface": [
    {"Type": "Move", "Id": "unit_id", "lon": float, "lat": float, "alt": float},
    {"Type": "SetRadar", "Id": "unit_id", "isHideOn": "1 (寮€) | 0 (鍏?"},
    {"Type": "SetJammer", "Id": "unit_id", "Pattern": "1 (寮€) | 0 (鍏?"}
],
"Shipboard_Aircraft_FixWing": [
    {"Type": "Move", "Id": "unit_id", "lon": float, "lat": float, "alt": float},
    {"Type": "Launch", "Id": "unit_id", "weapon_type": "AIM | JDAM", "lon": float, "lat": float, "alt": float}
],


"SmartLaunchOnTarget": {"Type": "SmartLaunchOnTarget", "UnitId": "unit_id", "Target_Id": "target_unit_id"}
"FocusFire": {"Type": "FocusFire", "UnitIds": ["ship_id_1", "ship_id_2"], "Target_Id": "target_unit_id"}
"EngageAndReposition": {"Type": "EngageAndReposition", "UnitIds": ["ship_id_1", "ship_id_2", ...], "Target_Id": "target_unit_id", "Move_Area": {"TopLeft": {"lon": float, "lat": float}, "BottomRight": {"lon": float, "lat": float}}}
"""

# ------------------------------------------------------------
# 杈呭姪鍑芥暟

# "AircraftScout": {"Type": "AircraftScout", "UnitIds": ["aircraft_id_1", ...], "Area": {"TopLeft": {"lon": float, "lat": float}, "BottomRight": {"lon": float, "lat": float}}},鍒犻櫎鐨勮摑鏂瑰姩浣滅┖闂达紝鏈潵淇
# ------------------------------------------------------------
def _normalize_faction_name(name: Optional[str]) -> str:
    if not name:
        return "blue"
    n = str(name).strip().lower()
    red_aliases = {"red", "hongfang", "r", "redteam", "red_team"}
    blue_aliases = {"blue", "lanfang", "b", "blueteam", "blue_team"}
    if n in red_aliases:
        return "red"
    if n in blue_aliases:
        return "blue"
    if "red" in n:
        return "red"
    if "blue" in n:
        return "blue"
    return "blue"


def _get_action_space_by_side(side_norm: str) -> str:
    return RED_ACTION_SPACE if side_norm == "red" else BLUE_ACTION_SPACE


def _lessons_side(side_norm: str) -> str:
    return "red" if side_norm == "red" else "blue"

def _allowed_action_types(side_norm: str) -> list[str]:
    if side_norm == "red":
        return [
            "ShootAndScoot",
            "FocusFire",
            "MoveToEngage",
            "ScoutArea",
            "GuideAttack", 
        ]
    return [
        "Move", "Launch", "SetJammer", "SetRadar",
        "LaunchInterceptor",
        "FocusFire", "AntiMissileDefense", "AircraftScout",
        "EngageAndReposition","SmartLaunchOnTarget"
    ]



def _forbidden_action_types(side_norm: str) -> list[str]:
    if side_norm == "red":
        return ["EngageAndReposition", "LaunchInterceptor", "SetRadar", "AircraftScout"]
    return ["ShootAndScoot", "Hide", "GuideAttack", "Recon_UAV_FixWing"]

# ------------------------------------------------------------
# 鍔犺浇鍙嶆€濇枃浠?
# ------------------------------------------------------------
def load_past_lessons(side: str = "red", max_lessons: int = 5) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir, "test")
    paths = {
        "red": os.path.join(base_dir, "red_reflections.jsonl"),
        "blue": os.path.join(base_dir, "blue_reflections.jsonl"),
    }

    knowledge_base_path = paths.get(side.lower())
    logger.info("[REFLECTION] Loading %s reflections from: %s", side, knowledge_base_path)

    if not knowledge_base_path or not os.path.exists(knowledge_base_path):
        logger.warning("No reflection file found for %s side at: %s", side, knowledge_base_path)
        return f"{str(side).upper()} 方暂无历史 lessons。"

    try:
        with open(knowledge_base_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except Exception as e:
        logger.error("Failed to open %s: %s", knowledge_base_path, e)
        return f"{str(side).upper()} 方历史 lessons 加载失败。"

    if not content:
        return f"{str(side).upper()} 方暂无历史 lessons。"

    lessons = []
    for line in content.splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "lesson" in data:
            lessons.append(data)

    if not lessons:
        try:
            data = json.loads(content)
        except Exception as e:
            logger.error("Error parsing reflection file %s: %s", knowledge_base_path, e)
            return f"{str(side).upper()} 方历史 lessons 加载失败。"

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str):
                    lessons.append({"type": str(key), "lesson": value})
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            lessons.append({"type": str(key), "lesson": item})
                elif isinstance(value, dict):
                    for nested_value in value.values():
                        if isinstance(nested_value, str):
                            lessons.append({"type": str(key), "lesson": nested_value})
                        elif isinstance(nested_value, list):
                            for item in nested_value:
                                if isinstance(item, str):
                                    lessons.append({"type": str(key), "lesson": item})

    if not lessons:
        return f"{str(side).upper()} 方暂无历史 lessons。"

    formatted_lessons = []
    for lesson_data in lessons[-max_lessons:]:
        lesson_type = str(lesson_data.get("type", "General")).capitalize()
        lesson_text = str(lesson_data.get("lesson", "暂无细节。"))
        formatted_lessons.append(f"- [历史 {lesson_type} Lesson] {lesson_text}")

    logger.info("[REFLECTION] Loaded %s lesson items for %s.", len(lessons), side)
    return "\n".join(formatted_lessons)

# ------------------------------------------------------------
# 涓诲嚱鏁帮細鐢熸垚鎴樼暐 Prompt
# ------------------------------------------------------------
def load_past_lessons(side: str = "red", max_lessons: int = 5) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir, "test")

    def _infer_side(raw: str) -> str:
        s = str(raw or "").lower()
        if "blue" in s:
            return "blue"
        return "red"

    def _is_path_like(raw: str) -> bool:
        r = str(raw or "")
        return (os.sep in r) or ("/" in r) or r.endswith(".json") or r.endswith(".jsonl")

    def _norm(text: str) -> str:
        return " ".join(str(text or "").strip().lower().split())

    def _legacy_fallback_enabled(side_name: str) -> bool:
        env_name = f"{side_name.upper()}_LTM_ENABLE_LEGACY_FALLBACK"
        raw = str(os.getenv(env_name, "0" if side_name == "red" else "1")).strip().lower()
        return raw in {"1", "true", "yes", "on"}

    def _load_jsonl(path: str) -> list[dict]:
        out = []
        seen = set()
        if not path or not os.path.exists(path):
            return out
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(data, dict):
                        continue
                    lesson = data.get("lesson")
                    if not isinstance(lesson, str) or not lesson.strip():
                        continue
                    key = _norm(lesson)
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    out.append(
                        {
                            "type": data.get("type", "general"),
                            "lesson": lesson.strip(),
                            "observation": data.get("observation", ""),
                        }
                    )
        except Exception as e:
            logger.warning("Failed loading lessons from %s: %s", path, e)
        return out

    def _collect_strings(node, out: list[str]):
        if isinstance(node, dict):
            for v in node.values():
                _collect_strings(v, out)
        elif isinstance(node, list):
            for v in node:
                _collect_strings(v, out)
        elif isinstance(node, str):
            s = node.strip()
            if len(s) >= 12:
                out.append(s)

    def _load_legacy(path: str) -> list[dict]:
        if not path or not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
        except Exception as e:
            logger.warning("Failed reading legacy lessons %s: %s", path, e)
            return []

        rows = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if isinstance(item, dict) and isinstance(item.get("lesson"), str) and item["lesson"].strip():
                rows.append({"type": item.get("type", "legacy"), "lesson": item["lesson"].strip()})
        if rows:
            return rows

        try:
            blob = json.loads(content)
        except Exception:
            return []
        strings: list[str] = []
        _collect_strings(blob, strings)
        out = []
        seen = set()
        for s in strings:
            key = _norm(s)
            if key and key not in seen:
                seen.add(key)
                out.append({"type": "legacy", "lesson": s})
        return out

    raw = str(side or "red")
    side_norm = _infer_side(raw)
    default_legacy = {
        "red": os.path.join(base_dir, "red_reflections.jsonl"),
        "blue": os.path.join(base_dir, "blue_reflections.jsonl"),
    }

    # Backward compatibility: caller may pass side name or file path.
    if _is_path_like(raw):
        legacy_path = raw if os.path.isabs(raw) else os.path.join(current_dir, raw)
        side_norm = _infer_side(legacy_path)
    else:
        legacy_path = default_legacy.get(side_norm, default_legacy["red"])

    default_structured = os.path.join(base_dir, f"{side_norm}_lessons_structured.jsonl")
    env_store_name = f"{side_norm.upper()}_LTM_STORE_PATH"
    env_store = os.getenv(env_store_name, "").strip()
    if env_store:
        structured_path = env_store if os.path.isabs(env_store) else os.path.join(current_dir, env_store)
    else:
        structured_path = default_structured

    lessons = _load_jsonl(structured_path)
    source = f"structured:{structured_path}"
    if not lessons and _legacy_fallback_enabled(side_norm):
        lessons = _load_legacy(legacy_path)
        source = f"legacy:{legacy_path}"

    if not lessons:
        return f"{side_norm.upper()} 方暂无历史 lessons。"

    formatted = []
    for lesson_data in lessons[-max_lessons:]:
        ltype = str(lesson_data.get("type", "General")).capitalize()
        ltext = str(lesson_data.get("lesson", "暂无细节。"))
        formatted.append(f"- [历史 {ltype} Lesson] {ltext}")

    logger.info("[REFLECTION] Loaded %s lessons from %s (side=%s)", len(lessons), source, side_norm)
    return "\n".join(formatted)


def get_strategic_prompt(faction_name: str, sim_time: int, state_summary: str) -> str:
    side_norm = _normalize_faction_name(faction_name)
    action_space = _get_action_space_by_side(side_norm)
    side_to_load = _lessons_side(side_norm)
    past_lessons = load_past_lessons(side=side_to_load, max_lessons=5)

    def _strip_comments(txt: str) -> str:
        lines = []
        for line in txt.splitlines():
            if not line.strip().startswith("//"):
                lines.append(line)
        return "\n".join(lines)

    action_space = _strip_comments(action_space)
    allowed_types = _allowed_action_types(side_norm)
    forbidden_types = _forbidden_action_types(side_norm)
    allowed_list_str = ", ".join([f"\"{t}\"" for t in allowed_types])
    forbidden_list_str = ", ".join([f"\"{t}\"" for t in forbidden_types])

    if side_norm == "red":
        side_label = "RED"
        opponent_label = "BLUE"
        side_specific_guardrails = """
RED 方硬约束：
- MoveToEngage / FocusFire / ShootAndScoot 的 UnitIds 必须全部是 Truck_Ground-*。
- GuideAttack 的 UnitIds 必须全部是 Guide_Ship_Surface-* 或 Recon_UAV_FixWing-*。
- 不要把 HighCostAttackMissile-* 或 LowCostAttackMissile-* 当作 UnitIds。
- RED 方攻击目标只能是海上舰船：Flagship_Surface-* / Cruiser_Surface-* / Destroyer_Surface-*。
- 如果存在有效海上目标，至少包含一个 truck fire action，不要只返回 scout-only 或 guide-only 方案。
- 当目标尚未进入射程时，要先用 MoveToEngage，再考虑 fire。
"""
    else:
        side_label = "BLUE"
        opponent_label = "RED"
        side_specific_guardrails = """
BLUE 方硬约束：
- FocusFire 必须使用 Target_Id，不要输出 Target_Lon/Target_Lat。
- AntiMissileDefense 只在检测到 incoming missiles 时有效。
- 只为 BLUE 单位生成动作，不要提及 RED 专属动作类型。
"""

    return f"""
你是海上对抗仿真中的 {side_label} 方 commander。
只有 JSON 字段 "actions" 会被执行；"analysis" 字段仅用于推理说明。

仿真时间：{sim_time}
当前阵营：{side_label}
对手阵营：{opponent_label}

历史 lessons：
{past_lessons}

硬性约束：
- 允许的动作类型：[{allowed_list_str}]
- 禁止出现的动作类型或单位词：[{forbidden_list_str}]
- 只为 {side_label} 方规划，不要输出 {opponent_label} 方专属动作或单位类型。
{side_specific_guardrails}

当前状态摘要：
{state_summary}

动作空间参考：
{action_space}

请只返回严格 JSON：
{{
  "analysis": "引用上述约束的简短推理说明",
  "actions": []
}}
"""


