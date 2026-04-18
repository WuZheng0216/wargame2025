import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from battle_metrics import extract_run_battle_metrics
from runtime_paths import project_root, resolve_path


DEFAULT_PORTS = (30001, 40001, 50005)
RELEVANT_ENV_PREFIXES = (
    "AUTO_STOP_",
    "LANGGRAPH_",
    "LLM_",
    "MODEL_",
    "RED_",
    "BLUE_",
    "REFLECTION_",
    "APP_",
    "HTTPX_",
    "OPENAI_",
    "CHAT_MODEL_",
    "POST_BATTLE_",
    "COMMANDER_",
    "MAIN_",
    "BATCH_",
    "RUN_",
)
RELEVANT_ENV_KEYS = {
    "BASE_URL",
    "MISSION_SCENARIO",
    "SCENARIO_EDIT_MODEL_NAME",
    "SCENARIO_EDIT_ENDPOINT",
    "SIMULATION_CONTROL_RED_ENDPOINT",
    "SIMULATION_CONTROL_BLUE_ENDPOINT",
    "SIMULATION_PLATFORM_ENDPOINT",
    "LANGGRAPH_ENABLED_SIDES",
}
SENSITIVE_TOKENS = ("KEY", "TOKEN", "SECRET", "PASSWORD")
LESSON_TEXT_ALIAS_MAP = {
    "fire_high_cost": "FocusFire",
    "high_cost_fire": "FocusFire",
    "fire_low_cost": "FocusFire",
    "guide_attack": "GuideAttack",
    "refresh_track": "ScoutArea",
    "refresh_tracks": "ScoutArea",
    "move_then_fire": "MoveToEngage",
}
LESSON_RULE_BRANDING_MAP = {
    "CostGate": "cost check",
    "AmmoMixRule": "ammo mix guideline",
}


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        return float(raw)
    except Exception:
        return default


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        return int(raw)
    except Exception:
        return default


def _safe_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text or "").strip())
    return cleaned.strip("-") or "batch"


def _run_index_from_id(run_id: str) -> Optional[int]:
    match = re.fullmatch(r"run_(\d{4})", str(run_id or "").strip())
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, data: dict) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path, default: Optional[dict] = None) -> dict:
    if not path.exists():
        return dict(default or {})
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(default or {})


def _append_jsonl(path: Path, data: dict) -> None:
    _ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False) + "\n")


def _copy_or_touch(src: Path, dst: Path) -> None:
    _ensure_parent(dst)
    if src.exists():
        shutil.copyfile(src, dst)
    else:
        dst.write_text("", encoding="utf-8")


def _resolve_lessons_store(side: str) -> Path:
    env_name = f"{side.upper()}_LTM_STORE_PATH"
    raw = str(os.getenv(env_name, "")).strip()
    if raw:
        return Path(resolve_path(raw))
    return Path(project_root()) / "test" / f"{side.lower()}_lessons_structured.jsonl"


def _resolve_batch_base_store(side: str) -> Path:
    side = str(side or "").strip().lower()
    if side == "red":
        mode = str(os.getenv("RED_BATCH_BASE_MODE", "minimal")).strip().lower() or "minimal"
        if mode == "minimal":
            raw = str(os.getenv("RED_BATCH_BASE_STORE_PATH", "")).strip()
            if raw:
                return Path(resolve_path(raw))
            minimal_path = Path(project_root()) / "test" / "red_lessons_minimal.jsonl"
            if minimal_path.exists():
                return minimal_path
    return _resolve_lessons_store(side)


def _normalize_red_batch_learning_mode(raw: Optional[str]) -> str:
    mode = str(raw or os.getenv("RED_BATCH_LEARNING_MODE", "freeze")).strip().lower() or "freeze"
    if mode not in {"freeze", "online"}:
        return "freeze"
    return mode


def _normalize_red_online_review_pipeline(raw: Optional[str], *, for_online: bool, legacy_default: bool = False) -> str:
    if not for_online:
        return "disabled"
    default_value = "legacy" if legacy_default else str(os.getenv("RED_ONLINE_REVIEW_PIPELINE", "dual_stage_v1")).strip().lower() or "dual_stage_v1"
    mode = str(raw or default_value).strip().lower() or default_value
    if mode not in {"legacy", "dual_stage_v1", "disabled"}:
        return "legacy" if legacy_default else "dual_stage_v1"
    return mode


def _read_jsonl_rows(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        return rows
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue
            if isinstance(data, dict):
                rows.append(data)
    except Exception:
        return rows
    return rows


def _write_jsonl_rows(path: Path, rows: List[dict]) -> None:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_lesson_text(text: str) -> str:
    value = str(text or "").strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def _sanitize_lesson_text(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    for alias, canonical in LESSON_TEXT_ALIAS_MAP.items():
        value = re.sub(rf"\b{re.escape(alias)}\b", canonical, value, flags=re.IGNORECASE)
    for branded, generic in LESSON_RULE_BRANDING_MAP.items():
        value = re.sub(rf"\b{re.escape(branded)}\b", generic, value)
    value = value.replace("`", "")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _merge_sentence_like_text(existing: str, incoming: str, *, max_chars: int) -> str:
    parts: List[str] = []
    seen = set()
    for candidate in (existing, incoming):
        cleaned = _sanitize_lesson_text(candidate)
        if not cleaned:
            continue
        for chunk in re.split(r"[。！？；;]\s*|\n+", cleaned):
            piece = chunk.strip(" ，,。；;")
            if not piece:
                continue
            norm = _normalize_lesson_text(piece)
            if norm in seen:
                continue
            seen.add(norm)
            parts.append(piece)
    merged = "；".join(parts).strip()
    if len(merged) <= max_chars:
        return merged
    clipped = merged[:max_chars].rstrip("； ")
    return clipped


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _lesson_cluster_key(item: dict) -> str:
    parts = [
        str(item.get("phase", "")).strip().lower(),
        str(item.get("target_type", "")).strip().lower(),
        str(item.get("symptom", "")).strip().lower(),
        str(item.get("cost_risk", "")).strip().lower(),
    ]
    compact = "|".join(parts).strip("|")
    if compact:
        return compact
    return _normalize_lesson_text(item.get("lesson", ""))


def _lesson_quality_score(item: dict) -> float:
    structured_fields = [
        item.get("phase"),
        item.get("target_type"),
        item.get("symptom"),
        item.get("trigger"),
        item.get("score_pattern"),
        item.get("cost_risk"),
    ]
    structured_bonus = sum(1 for field in structured_fields if str(field or "").strip())
    tags_bonus = min(len(item.get("tags") or []), 6) * 0.25
    support_bonus = max(0.0, _safe_float(item.get("support_count", 1), 1.0) - 1.0) * 2.0
    outcome_bonus = max(0.0, _safe_float(item.get("quality_score", 0.0), 0.0))
    return structured_bonus + tags_bonus + support_bonus + outcome_bonus


def _sanitize_lesson_record(
    item: dict,
    *,
    summary: Optional[dict] = None,
    run_id: str = "",
    batch_id: str = "",
) -> Optional[dict]:
    if not isinstance(item, dict):
        return None
    lesson = _sanitize_lesson_text(item.get("lesson", ""))
    if not lesson:
        return None
    observation = _sanitize_lesson_text(item.get("observation", ""))
    trigger = _sanitize_lesson_text(item.get("trigger", ""))
    score_pattern = _sanitize_lesson_text(item.get("score_pattern", ""))
    symptom = _sanitize_lesson_text(item.get("symptom", ""))
    phase = str(item.get("phase", "")).strip().lower()
    target_type = _sanitize_lesson_text(item.get("target_type", ""))
    cost_risk = str(item.get("cost_risk", "")).strip().lower()

    tags = item.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    sanitized_tags: List[str] = []
    for tag in tags:
        cleaned = _sanitize_lesson_text(tag)
        if cleaned and cleaned not in sanitized_tags:
            sanitized_tags.append(cleaned)

    battle_meta = dict(item.get("battle_meta") or {})
    if summary:
        battle_meta["promotion_final_score_red"] = summary.get("final_score_red")
        battle_meta["promotion_final_score_blue"] = summary.get("final_score_blue")
        battle_meta["promotion_run_id"] = run_id or summary.get("run_id")
        battle_meta["promotion_batch_id"] = batch_id or summary.get("batch_id")

    normalized = _normalize_lesson_text(lesson)
    lesson_hash = str(item.get("lesson_hash", "")).strip()
    if not lesson_hash:
        lesson_hash = hashlib.sha1(normalized.encode("utf-8")).hexdigest()

    source_run_ids = item.get("source_run_ids")
    if not isinstance(source_run_ids, list):
        source_run_ids = []
    if run_id and run_id not in source_run_ids:
        source_run_ids.append(run_id)

    source_batch_ids = item.get("source_batch_ids")
    if not isinstance(source_batch_ids, list):
        source_batch_ids = []
    batch_id_value = batch_id or str((battle_meta or {}).get("source_batch_id") or "").strip()
    if batch_id_value and batch_id_value not in source_batch_ids:
        source_batch_ids.append(batch_id_value)

    outcome_margin = 0.0
    if summary:
        outcome_margin = _safe_float(summary.get("final_score_red")) - _safe_float(summary.get("final_score_blue"))
    elif battle_meta:
        outcome_margin = _safe_float(battle_meta.get("promotion_final_score_red")) - _safe_float(
            battle_meta.get("promotion_final_score_blue")
        )

    return {
        "lesson_id": str(item.get("lesson_id") or hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]),
        "type": str(item.get("type", "general")).strip() or "general",
        "observation": observation,
        "lesson": lesson,
        "tags": sanitized_tags,
        "phase": phase,
        "target_type": target_type,
        "symptom": symptom,
        "trigger": trigger,
        "score_pattern": score_pattern,
        "cost_risk": cost_risk,
        "battle_meta": battle_meta,
        "source": str(item.get("source", "reflection_agent")).strip() or "reflection_agent",
        "schema_version": max(3, int(_safe_float(item.get("schema_version"), 3))),
        "created_at": str(item.get("created_at") or datetime.utcnow().isoformat() + "Z"),
        "normalized_lesson": normalized,
        "lesson_hash": lesson_hash,
        "support_count": max(1, int(_safe_float(item.get("support_count"), 1))),
        "quality_score": max(_safe_float(item.get("quality_score"), 0.0), outcome_margin / 50.0),
        "source_run_ids": source_run_ids,
        "source_batch_ids": source_batch_ids,
        "protocol_safe": True,
    }


def _merge_lesson_records(existing: dict, incoming: dict) -> dict:
    merged = dict(existing)
    for field, max_chars in (("observation", 220), ("lesson", 280), ("trigger", 180), ("score_pattern", 180)):
        merged[field] = _merge_sentence_like_text(existing.get(field, ""), incoming.get(field, ""), max_chars=max_chars)

    for field in ("phase", "target_type", "symptom", "cost_risk"):
        if not str(merged.get(field, "")).strip() and str(incoming.get(field, "")).strip():
            merged[field] = incoming.get(field, "")

    tags = []
    for tag in (existing.get("tags") or []) + (incoming.get("tags") or []):
        cleaned = _sanitize_lesson_text(tag)
        if cleaned and cleaned not in tags:
            tags.append(cleaned)
    merged["tags"] = tags[:8]

    source_run_ids = []
    for run_id in (existing.get("source_run_ids") or []) + (incoming.get("source_run_ids") or []):
        if run_id and run_id not in source_run_ids:
            source_run_ids.append(run_id)
    merged["source_run_ids"] = source_run_ids

    source_batch_ids = []
    for batch_id in (existing.get("source_batch_ids") or []) + (incoming.get("source_batch_ids") or []):
        if batch_id and batch_id not in source_batch_ids:
            source_batch_ids.append(batch_id)
    merged["source_batch_ids"] = source_batch_ids

    merged["support_count"] = max(1, int(_safe_float(existing.get("support_count"), 1))) + max(
        1, int(_safe_float(incoming.get("support_count"), 1))
    )
    merged["quality_score"] = max(_safe_float(existing.get("quality_score"), 0.0), _safe_float(incoming.get("quality_score"), 0.0))
    merged["protocol_safe"] = bool(existing.get("protocol_safe", True)) and bool(incoming.get("protocol_safe", True))
    merged["normalized_lesson"] = _normalize_lesson_text(merged.get("lesson", ""))
    merged["lesson_hash"] = hashlib.sha1(merged["normalized_lesson"].encode("utf-8")).hexdigest()
    merged["lesson_id"] = hashlib.sha1(merged["normalized_lesson"].encode("utf-8")).hexdigest()[:16]
    return merged


def _consolidate_online_batch_base(
    *,
    existing_rows: List[dict],
    incoming_rows: List[dict],
    summary: dict,
    run_id: str,
    batch_id: str,
) -> dict:
    allow_new_clusters = _safe_float(summary.get("final_score_red")) >= _safe_float(summary.get("final_score_blue"))
    clusters: Dict[str, dict] = {}
    merged_count = 0
    added_count = 0
    dropped_count = 0

    for row in existing_rows:
        sanitized = _sanitize_lesson_record(row, run_id=run_id, batch_id=batch_id)
        if not sanitized:
            continue
        clusters[_lesson_cluster_key(sanitized)] = sanitized

    for row in incoming_rows:
        sanitized = _sanitize_lesson_record(row, summary=summary, run_id=run_id, batch_id=batch_id)
        if not sanitized:
            dropped_count += 1
            continue
        key = _lesson_cluster_key(sanitized)
        existing = clusters.get(key)
        if existing is not None:
            clusters[key] = _merge_lesson_records(existing, sanitized)
            merged_count += 1
            continue
        if not allow_new_clusters:
            dropped_count += 1
            continue
        clusters[key] = sanitized
        added_count += 1

    max_lessons = max(4, _read_int_env("RED_BATCH_BASE_MAX_LESSONS", 12))
    ordered = sorted(
        clusters.values(),
        key=lambda item: (
            _lesson_quality_score(item),
            int(_safe_float(item.get("support_count"), 1)),
            str(item.get("created_at", "")),
        ),
        reverse=True,
    )

    selected: List[dict] = []
    for item in ordered:
        selected.append(item)
        if len(selected) >= max_lessons:
            break

    if len(selected) < min(max_lessons, len(ordered)):
        for item in ordered:
            if item in selected:
                continue
            selected.append(item)
            if len(selected) >= max_lessons:
                break

    for item in selected:
        item["quality_score"] = round(_lesson_quality_score(item), 3)
        item["support_count"] = max(1, int(_safe_float(item.get("support_count"), 1)))
        item["battle_meta"] = dict(item.get("battle_meta") or {})
        item["battle_meta"]["last_promoted_run_id"] = run_id
        item["battle_meta"]["last_promoted_batch_id"] = batch_id

    return {
        "rows": selected,
        "allow_new_clusters": allow_new_clusters,
        "merged_count": merged_count,
        "added_count": added_count,
        "dropped_count": dropped_count,
        "retained_count": len(selected),
        "input_existing": len(existing_rows),
        "input_incoming": len(incoming_rows),
    }


def _promote_run_lessons_to_batch_base(
    *,
    red_batch_learning_mode: str,
    red_online_review_pipeline: str,
    red_batch_base_max_lessons: int,
    summary: dict,
    run_root: Path,
    batch_red_base: Path,
    batch_manifest_path: Path,
) -> None:
    if red_batch_learning_mode != "online":
        return
    if str(summary.get("exit_status", "")).strip().lower() != "completed":
        return
    lessons_after = run_root / "knowledge" / "lessons_after.jsonl"
    runtime_red_lessons = run_root / "knowledge" / "red_lessons_runtime.jsonl"
    source = lessons_after if lessons_after.exists() else runtime_red_lessons
    if not source.exists():
        return
    existing_rows = _read_jsonl_rows(batch_red_base)
    incoming_rows = _read_jsonl_rows(source)

    if red_online_review_pipeline == "dual_stage_v1":
        max_lessons = max(4, int(red_batch_base_max_lessons or 20))
        validated_map: Dict[str, dict] = {}
        for row in incoming_rows:
            sanitized = _sanitize_lesson_record(
                row,
                summary=summary,
                run_id=str(summary.get("run_id") or run_root.name),
                batch_id=str(summary.get("batch_id") or ""),
            )
            if not sanitized:
                continue
            key = _lesson_cluster_key(sanitized)
            existing = validated_map.get(key)
            validated_map[key] = _merge_lesson_records(existing, sanitized) if existing else sanitized

        validated_rows = sorted(
            validated_map.values(),
            key=lambda item: (
                _lesson_quality_score(item),
                int(_safe_float(item.get("support_count"), 1)),
                str(item.get("created_at", "")),
            ),
            reverse=True,
        )[:max_lessons]
        if not validated_rows:
            _append_jsonl(
                batch_manifest_path,
                {
                    "batch_id": summary.get("batch_id"),
                    "run_id": summary.get("run_id"),
                    "event": "red_batch_base_promotion_skipped",
                    "timestamp": datetime.now().isoformat(),
                    "mode": red_batch_learning_mode,
                    "review_pipeline": red_online_review_pipeline,
                    "reason": "no_valid_reviewed_rows",
                    "source": str(source),
                },
            )
            return

        allow_new_principles = _safe_float(summary.get("final_score_red")) >= _safe_float(summary.get("final_score_blue"))
        _write_jsonl_rows(batch_red_base, validated_rows)
        _append_jsonl(
            batch_manifest_path,
            {
                "batch_id": summary.get("batch_id"),
                "run_id": summary.get("run_id"),
                "event": "red_batch_base_promoted",
                "timestamp": datetime.now().isoformat(),
                "mode": red_batch_learning_mode,
                "review_pipeline": red_online_review_pipeline,
                "source": str(source),
                "target": str(batch_red_base),
                "allow_new_principles": allow_new_principles,
                "input_existing": len(existing_rows),
                "input_incoming": len(incoming_rows),
                "retained_count": len(validated_rows),
            },
        )
        return

    consolidated = _consolidate_online_batch_base(
        existing_rows=existing_rows,
        incoming_rows=incoming_rows,
        summary=summary,
        run_id=str(summary.get("run_id") or run_root.name),
        batch_id=str(summary.get("batch_id") or ""),
    )
    _write_jsonl_rows(batch_red_base, consolidated["rows"])
    _append_jsonl(
        batch_manifest_path,
        {
            "batch_id": summary.get("batch_id"),
            "run_id": summary.get("run_id"),
            "event": "red_batch_base_promoted",
            "timestamp": datetime.now().isoformat(),
            "mode": red_batch_learning_mode,
            "review_pipeline": red_online_review_pipeline,
            "source": str(source),
            "target": str(batch_red_base),
            "allow_new_clusters": consolidated["allow_new_clusters"],
            "merged_count": consolidated["merged_count"],
            "added_count": consolidated["added_count"],
            "dropped_count": consolidated["dropped_count"],
            "retained_count": consolidated["retained_count"],
            "input_existing": consolidated["input_existing"],
            "input_incoming": consolidated["input_incoming"],
        },
    )


def _mask_value(key: str, value: str) -> str:
    if any(token in key.upper() for token in SENSITIVE_TOKENS):
        if not value:
            return ""
        if len(value) <= 8:
            return "***"
        return value[:4] + "***" + value[-4:]
    return value


def _capture_env_snapshot(env: Dict[str, str]) -> dict:
    selected = {}
    for key, value in sorted(env.items()):
        if key in RELEVANT_ENV_KEYS or any(key.startswith(prefix) for prefix in RELEVANT_ENV_PREFIXES):
            selected[key] = _mask_value(key, value)
    selected["PYTHON_EXECUTABLE"] = sys.executable
    selected["PROJECT_ROOT"] = project_root()
    return selected


def _port_reachable(host: str, port: int, timeout_seconds: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def _check_platform_ready(host: str, timeout_seconds: float) -> dict:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        ports = {port: _port_reachable(host, port, timeout_seconds=1.0) for port in DEFAULT_PORTS}
        if all(ports.values()):
            return {"ready": True, "ports": ports}
        time.sleep(1.0)
    ports = {port: _port_reachable(host, port, timeout_seconds=1.0) for port in DEFAULT_PORTS}
    return {"ready": all(ports.values()), "ports": ports}


def _find_latest(directory: Path, pattern: str) -> Optional[Path]:
    matches = sorted(directory.glob(pattern), key=lambda item: item.stat().st_mtime)
    return matches[-1] if matches else None


def _parse_main_log(log_path: Optional[Path]) -> dict:
    result = {
        "stop_reason": None,
        "max_red_simtime": None,
        "max_blue_simtime": None,
        "reflection_status_red": "unknown",
        "reflection_status_blue": "unknown",
        "final_score_red": None,
        "final_score_blue": None,
    }
    if not log_path or not log_path.exists():
        return result

    simtime_regex = re.compile(r"red_simtime=(\d+)\s+blue_simtime=(\d+)")
    stop_regex = re.compile(r"main_wait_finished reason=([A-Za-z0-9_:-]+)")
    reflection_regex = re.compile(r"\[(RED|BLUE)\]\s+reflection_(done|timeout|error)")
    score_regex = re.compile(r"\[(RED|BLUE)\].*final_score=Final Score: ([^ ]+)")

    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if match := simtime_regex.search(line):
            red_simtime = int(match.group(1))
            blue_simtime = int(match.group(2))
            result["max_red_simtime"] = max(result["max_red_simtime"] or 0, red_simtime)
            result["max_blue_simtime"] = max(result["max_blue_simtime"] or 0, blue_simtime)
        if match := stop_regex.search(line):
            result["stop_reason"] = match.group(1)
        if match := reflection_regex.search(line):
            side = match.group(1).lower()
            result[f"reflection_status_{side}"] = match.group(2)
        if match := score_regex.search(line):
            side = match.group(1).lower()
            try:
                result[f"final_score_{side}"] = float(match.group(2))
            except Exception:
                result[f"final_score_{side}"] = match.group(2)

    return result


def _parse_battle_log_max_simtime(log_path: Optional[Path]) -> Optional[int]:
    if not log_path or not log_path.exists():
        return None
    max_simtime = None
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except Exception:
            continue
        sim_time = data.get("sim_time")
        if sim_time is None:
            continue
        try:
            numeric = int(float(sim_time))
        except Exception:
            continue
        max_simtime = max(max_simtime or numeric, numeric)
    return max_simtime


def _parse_red_trace(trace_jsonl_path: Optional[Path]) -> dict:
    result = {
        "red_trace_count": 0,
        "first_red_action_simtime": None,
        "red_submitted_action_count": 0,
        "red_llm_call_count": 0,
    }
    if not trace_jsonl_path or not trace_jsonl_path.exists():
        return result

    for line in trace_jsonl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        result["red_trace_count"] += 1
        sections = record.get("sections", {})
        for entry in sections.get("LLM Calls", []) or []:
            component = str(entry.get("component", ""))
            if not component.endswith(".json_parse"):
                result["red_llm_call_count"] += 1
        for entry in sections.get("Submit", []) or []:
            try:
                submitted = int(entry.get("submitted_action_count", 0) or 0)
            except Exception:
                submitted = 0
            result["red_submitted_action_count"] += submitted
            if submitted > 0 and result["first_red_action_simtime"] is None:
                try:
                    result["first_red_action_simtime"] = int(record.get("sim_time"))
                except Exception:
                    pass
    return result


def _merge_red_action_metrics(trace_metrics: dict, battle_metrics: dict) -> dict:
    merged = dict(trace_metrics or {})
    battle_first = battle_metrics.get("first_red_action_simtime")
    battle_submitted = int(battle_metrics.get("red_submitted_action_count") or 0)

    trace_first = merged.get("first_red_action_simtime")
    if battle_first is not None:
        if trace_first is None:
            merged["first_red_action_simtime"] = battle_first
        else:
            try:
                merged["first_red_action_simtime"] = min(int(trace_first), int(battle_first))
            except Exception:
                merged["first_red_action_simtime"] = battle_first

    trace_submitted = int(merged.get("red_submitted_action_count") or 0)
    if battle_submitted > trace_submitted:
        merged["red_submitted_action_count"] = battle_submitted
    return merged


def _build_artifact_paths(run_root: Path) -> dict:
    logs_dir = run_root / "logs"
    battle_logs_dir = run_root / "battle_logs"
    llm_outputs_dir = run_root / "llm_outputs"
    knowledge_dir = run_root / "knowledge"
    return {
        "run_root": str(run_root),
        "main_log": str(_find_latest(logs_dir, "wargame_*.log")) if logs_dir.exists() else None,
        "red_trace_log": str(_find_latest(logs_dir, "red_trace_*.log")) if logs_dir.exists() else None,
        "red_trace_jsonl": str(_find_latest(logs_dir, "red_trace_*.jsonl")) if logs_dir.exists() else None,
        "battle_logs_dir": str(battle_logs_dir) if battle_logs_dir.exists() else None,
        "llm_outputs_dir": str(llm_outputs_dir) if llm_outputs_dir.exists() else None,
        "stdout_log": str(run_root / "stdout.log"),
        "stderr_log": str(run_root / "stderr.log"),
        "env_snapshot": str(run_root / "env_snapshot.json"),
        "lessons_before": str(knowledge_dir / "lessons_before.jsonl"),
        "lessons_after": str(knowledge_dir / "lessons_after.jsonl"),
    }


def _write_batch_summary_csv(path: Path, rows: List[dict]) -> None:
    fieldnames = [
        "batch_id",
        "run_id",
        "exit_status",
        "timeout",
        "stop_reason",
        "start_time",
        "end_time",
        "wall_clock_seconds",
        "max_red_simtime",
        "max_blue_simtime",
        "reflection_status_red",
        "reflection_status_blue",
        "first_red_action_simtime",
        "red_submitted_action_count",
        "red_trace_count",
        "red_llm_call_count",
        "red_unique_targets_detected",
        "blue_unique_targets_detected",
        "red_high_value_targets_detected",
        "blue_high_value_targets_detected",
        "red_units_lost",
        "blue_units_lost",
        "red_targets_destroyed",
        "blue_targets_destroyed",
        "red_high_value_targets_destroyed",
        "blue_high_value_targets_destroyed",
        "red_intercept_launch_count",
        "blue_intercept_launch_count",
        "red_attack_launch_count",
        "blue_attack_launch_count",
        "red_score_gain_total",
        "blue_score_gain_total",
        "red_positive_score_events",
        "blue_positive_score_events",
        "final_score_red",
        "final_score_blue",
        "run_root",
        "main_log",
        "red_trace_log",
        "red_trace_jsonl",
        "battle_logs_dir",
        "llm_outputs_dir",
        "stdout_log",
        "stderr_log",
        "lessons_before",
        "lessons_after",
    ]
    _ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            for key, value in (row.get("artifact_paths") or {}).items():
                flat[key] = value
            writer.writerow({name: flat.get(name) for name in fieldnames})


def _resolve_batch_root(batch_ref: str) -> Path:
    raw = str(batch_ref or "").strip()
    if not raw:
        raise ValueError("resume batch reference is required")
    candidate = Path(raw)
    if candidate.exists():
        return candidate.resolve()
    named = Path(project_root()) / "test" / "batches" / raw
    if named.exists():
        return named.resolve()
    raise FileNotFoundError(f"Batch not found: {batch_ref}")


def _load_existing_summaries(batch_root: Path) -> List[dict]:
    summaries: List[dict] = []
    for run_dir in sorted(batch_root.glob("run_*"), key=lambda p: _run_index_from_id(p.name) or 0):
        summary_path = run_dir / "run_summary.json"
        if not summary_path.exists():
            continue
        summary = _read_json(summary_path)
        if summary:
            summaries.append(summary)
    return summaries


def _status_runs_from_summaries(summaries: List[dict]) -> List[dict]:
    runs = []
    for summary in summaries:
        runs.append(
            {
                "run_id": summary.get("run_id"),
                "exit_status": summary.get("exit_status"),
                "run_root": str((summary.get("artifact_paths") or {}).get("run_root") or ""),
                "started_at": summary.get("start_time"),
                "ended_at": summary.get("end_time"),
            }
        )
    return runs


def _build_child_env(
    run_root: Path,
    end_simtime: int,
    timeout_seconds: int,
    runtime_red_lessons: Path,
    runtime_blue_lessons: Path,
    red_batch_learning_mode: str,
    red_online_review_pipeline: str,
    red_batch_base_max_lessons: int,
) -> Dict[str, str]:
    env = os.environ.copy()
    env["RUN_OUTPUT_ROOT"] = str(run_root)
    env["APP_ENABLE_CONSOLE_LOG"] = "0"
    env["APP_CONSOLE_LOG_LEVEL"] = "CRITICAL"
    env["BATCH_MODE"] = "1"
    env["AUTO_STOP_SIMTIME_SECONDS"] = str(end_simtime)
    env.setdefault("MAIN_POLL_INTERVAL_SECONDS", "1.0")
    env.setdefault("BATCH_RUN_TIMEOUT_SECONDS", str(timeout_seconds))
    env.setdefault(
        "POST_BATTLE_REFLECTION_TIMEOUT_SECONDS",
        str(_read_float_env("POST_BATTLE_REFLECTION_TIMEOUT_SECONDS", 0.0)),
    )
    env.setdefault(
        "COMMANDER_JOIN_TIMEOUT_SECONDS",
        str(_read_float_env("COMMANDER_JOIN_TIMEOUT_SECONDS", 600.0)),
    )
    if str(env.get("LLM_TRUST_ENV_PROXY", "0")).strip().lower() in {"0", "false", "no", "off", ""}:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            env.pop(key, None)
    env["RED_LTM_STORE_PATH"] = str(runtime_red_lessons)
    env["BLUE_LTM_STORE_PATH"] = str(runtime_blue_lessons)
    env["RED_BATCH_LEARNING_MODE"] = red_batch_learning_mode
    env["RED_ONLINE_REVIEW_PIPELINE"] = red_online_review_pipeline
    env["RED_BATCH_BASE_MAX_LESSONS"] = str(red_batch_base_max_lessons)
    return env


def _prepare_run_knowledge(batch_red_base: Path, batch_blue_base: Path, run_root: Path) -> dict:
    knowledge_dir = run_root / "knowledge"
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    lessons_before = knowledge_dir / "lessons_before.jsonl"
    runtime_red_lessons = knowledge_dir / "red_lessons_runtime.jsonl"
    runtime_blue_lessons = knowledge_dir / "blue_lessons_runtime.jsonl"
    _copy_or_touch(batch_red_base, lessons_before)
    _copy_or_touch(batch_red_base, runtime_red_lessons)
    _copy_or_touch(batch_blue_base, runtime_blue_lessons)
    return {
        "lessons_before": lessons_before,
        "runtime_red_lessons": runtime_red_lessons,
        "runtime_blue_lessons": runtime_blue_lessons,
        "lessons_after": knowledge_dir / "lessons_after.jsonl",
    }


def _finalize_run_knowledge(runtime_red_lessons: Path, lessons_after: Path) -> None:
    _copy_or_touch(runtime_red_lessons, lessons_after)


def _run_single(
    *,
    batch_id: str,
    run_index: int,
    run_root: Path,
    end_simtime: int,
    engine_host: str,
    step_interval: float,
    timeout_seconds: int,
    batch_red_base: Path,
    batch_blue_base: Path,
    batch_manifest_path: Path,
    status: dict,
    red_batch_learning_mode: str,
    red_online_review_pipeline: str,
    red_batch_base_max_lessons: int,
) -> dict:
    run_id = f"run_{run_index:04d}"
    run_root.mkdir(parents=True, exist_ok=True)

    knowledge_paths = _prepare_run_knowledge(batch_red_base, batch_blue_base, run_root)
    child_env = _build_child_env(
        run_root=run_root,
        end_simtime=end_simtime,
        timeout_seconds=timeout_seconds,
        runtime_red_lessons=knowledge_paths["runtime_red_lessons"],
        runtime_blue_lessons=knowledge_paths["runtime_blue_lessons"],
        red_batch_learning_mode=red_batch_learning_mode,
        red_online_review_pipeline=red_online_review_pipeline,
        red_batch_base_max_lessons=red_batch_base_max_lessons,
    )
    _write_json(run_root / "env_snapshot.json", _capture_env_snapshot(child_env))

    started_at = datetime.now().isoformat()
    manifest_base = {
        "batch_id": batch_id,
        "run_id": run_id,
        "run_root": str(run_root),
        "timestamp": started_at,
    }
    _append_jsonl(batch_manifest_path, {**manifest_base, "event": "run_started"})

    stdout_handle = (run_root / "stdout.log").open("w", encoding="utf-8")
    stderr_handle = (run_root / "stderr.log").open("w", encoding="utf-8")
    command = [
        sys.executable,
        "main.py",
        "--end-simtime",
        str(end_simtime),
        "--engine-host",
        engine_host,
        "--step-interval",
        str(step_interval),
    ]

    timeout = False
    return_code = None
    process = None
    try:
        process = subprocess.Popen(
            command,
            cwd=project_root(),
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=child_env,
        )
        return_code = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timeout = True
        if process is not None:
            process.kill()
            return_code = process.wait(timeout=15)
    finally:
        stdout_handle.close()
        stderr_handle.close()

    finished_at = datetime.now().isoformat()
    wall_clock_seconds = round(
        max(0.0, datetime.fromisoformat(finished_at).timestamp() - datetime.fromisoformat(started_at).timestamp()),
        3,
    )

    artifact_paths = _build_artifact_paths(run_root)
    main_log_path = Path(artifact_paths["main_log"]) if artifact_paths.get("main_log") else None
    red_trace_jsonl_path = Path(artifact_paths["red_trace_jsonl"]) if artifact_paths.get("red_trace_jsonl") else None

    main_log_summary = _parse_main_log(main_log_path)
    red_trace_metrics = _parse_red_trace(red_trace_jsonl_path)
    red_battle_log = _find_latest(run_root / "battle_logs", "battle_log_RED_*.jsonl")
    blue_battle_log = _find_latest(run_root / "battle_logs", "battle_log_BLUE_*.jsonl")
    battle_metrics = extract_run_battle_metrics(red_battle_log, blue_battle_log)
    red_metrics = _merge_red_action_metrics(red_trace_metrics, battle_metrics)

    max_red_simtime = main_log_summary["max_red_simtime"] or _parse_battle_log_max_simtime(red_battle_log)
    max_blue_simtime = main_log_summary["max_blue_simtime"] or _parse_battle_log_max_simtime(blue_battle_log)

    _finalize_run_knowledge(knowledge_paths["runtime_red_lessons"], knowledge_paths["lessons_after"])

    if timeout:
        exit_status = "timeout"
    elif return_code == 0:
        exit_status = "completed"
    else:
        exit_status = "process_error"

    summary = {
        "batch_id": batch_id,
        "run_id": run_id,
        "start_time": started_at,
        "end_time": finished_at,
        "wall_clock_seconds": wall_clock_seconds,
        "exit_status": exit_status,
        "timeout": timeout,
        "stop_reason": main_log_summary["stop_reason"],
        "max_red_simtime": max_red_simtime,
        "max_blue_simtime": max_blue_simtime,
        "reflection_status_red": main_log_summary["reflection_status_red"],
        "reflection_status_blue": main_log_summary["reflection_status_blue"],
        "artifact_paths": artifact_paths,
        "first_red_action_simtime": red_metrics["first_red_action_simtime"],
        "red_submitted_action_count": red_metrics["red_submitted_action_count"],
        "red_trace_count": red_metrics["red_trace_count"],
        "red_llm_call_count": red_metrics["red_llm_call_count"],
        "red_unique_targets_detected": battle_metrics["red_unique_targets_detected"],
        "blue_unique_targets_detected": battle_metrics["blue_unique_targets_detected"],
        "red_high_value_targets_detected": battle_metrics["red_high_value_targets_detected"],
        "blue_high_value_targets_detected": battle_metrics["blue_high_value_targets_detected"],
        "red_units_lost": battle_metrics["red_units_lost"],
        "blue_units_lost": battle_metrics["blue_units_lost"],
        "red_targets_destroyed": battle_metrics["red_targets_destroyed"],
        "blue_targets_destroyed": battle_metrics["blue_targets_destroyed"],
        "red_high_value_targets_destroyed": battle_metrics["red_high_value_targets_destroyed"],
        "blue_high_value_targets_destroyed": battle_metrics["blue_high_value_targets_destroyed"],
        "red_intercept_launch_count": battle_metrics["red_intercept_launch_count"],
        "blue_intercept_launch_count": battle_metrics["blue_intercept_launch_count"],
        "red_attack_launch_count": battle_metrics["red_attack_launch_count"],
        "blue_attack_launch_count": battle_metrics["blue_attack_launch_count"],
        "red_score_gain_total": battle_metrics["red_score_gain_total"],
        "blue_score_gain_total": battle_metrics["blue_score_gain_total"],
        "red_positive_score_events": battle_metrics["red_positive_score_events"],
        "blue_positive_score_events": battle_metrics["blue_positive_score_events"],
        "final_score_red": battle_metrics["final_score_red"] if battle_metrics["final_score_red"] is not None else main_log_summary["final_score_red"],
        "final_score_blue": battle_metrics["final_score_blue"] if battle_metrics["final_score_blue"] is not None else main_log_summary["final_score_blue"],
        "return_code": return_code,
    }

    _write_json(run_root / "run_summary.json", summary)
    _append_jsonl(
        batch_manifest_path,
        {
            **manifest_base,
            "event": "run_finished",
            "timestamp": finished_at,
            "exit_status": exit_status,
            "timeout": timeout,
            "return_code": return_code,
        },
    )
    status["current_run_id"] = None
    return summary


def _parse_args():
    parser = argparse.ArgumentParser(description="Batch experiment supervisor for unattended simulation runs.")
    parser.add_argument("--runs", type=int, required=True)
    parser.add_argument("--end-simtime", type=int, default=None, dest="end_simtime")
    parser.add_argument("--engine-host", type=str, default=None, dest="engine_host")
    parser.add_argument("--step-interval", type=float, default=None, dest="step_interval")
    parser.add_argument("--batch-name", type=str, default="batch", dest="batch_name")
    parser.add_argument("--resume-batch", type=str, default=None, dest="resume_batch")
    parser.add_argument("--red-batch-learning-mode", type=str, default=None, dest="red_batch_learning_mode")
    return parser.parse_args()


def main():
    args = _parse_args()
    batch_timeout_seconds = _read_int_env("BATCH_RUN_TIMEOUT_SECONDS", 5400)
    batch_cooldown_seconds = _read_float_env("BATCH_COOLDOWN_SECONDS", 5.0)
    port_check_timeout_seconds = _read_float_env("BATCH_PORT_CHECK_TIMEOUT_SECONDS", 15.0)
    requested_red_batch_learning_mode = _normalize_red_batch_learning_mode(args.red_batch_learning_mode)

    is_resume = bool(args.resume_batch)
    if is_resume:
        batch_root = _resolve_batch_root(args.resume_batch)
        batch_config_path = batch_root / "batch_config.json"
        batch_status_path = batch_root / "batch_status.json"
        batch_manifest_path = batch_root / "run_manifest.jsonl"
        batch_summary_path = batch_root / "batch_summary.csv"
        batch_config = _read_json(batch_config_path)
        if not batch_config:
            raise FileNotFoundError(f"batch_config.json not found or unreadable under {batch_root}")

        existing_batch_id = str(batch_config.get("batch_id") or batch_root.name)
        existing_batch_name = str(batch_config.get("batch_name") or batch_root.name)
        configured_end_simtime = int(batch_config.get("end_simtime") or 1800)
        configured_engine_host = str(batch_config.get("engine_host") or "127.0.0.1")
        configured_step_interval = float(batch_config.get("step_interval") or 0.1)
        red_batch_learning_mode = _normalize_red_batch_learning_mode(
            batch_config.get("red_batch_learning_mode") or batch_config.get("red_batch_base_mode")
        )
        red_online_review_pipeline = _normalize_red_online_review_pipeline(
            batch_config.get("red_online_review_pipeline"),
            for_online=(red_batch_learning_mode == "online"),
            legacy_default=("red_online_review_pipeline" not in batch_config),
        )
        red_batch_base_max_lessons = int(batch_config.get("red_batch_base_max_lessons") or _read_int_env("RED_BATCH_BASE_MAX_LESSONS", 20))

        if args.end_simtime is not None and int(args.end_simtime) != configured_end_simtime:
            raise ValueError(
                f"Resumed batch must keep end_simtime={configured_end_simtime}, got {args.end_simtime}"
            )
        if args.engine_host is not None and str(args.engine_host) != configured_engine_host:
            raise ValueError(
                f"Resumed batch must keep engine_host={configured_engine_host}, got {args.engine_host}"
            )
        if args.step_interval is not None and float(args.step_interval) != configured_step_interval:
            raise ValueError(
                f"Resumed batch must keep step_interval={configured_step_interval}, got {args.step_interval}"
            )
        if args.red_batch_learning_mode is not None and requested_red_batch_learning_mode != red_batch_learning_mode:
            raise ValueError(
                f"Resumed batch must keep red_batch_learning_mode={red_batch_learning_mode}, got {requested_red_batch_learning_mode}"
            )

        batch_id = existing_batch_id
        args.batch_name = existing_batch_name
        args.end_simtime = configured_end_simtime
        args.engine_host = configured_engine_host
        args.step_interval = configured_step_interval

        batch_knowledge_dir = batch_root / "knowledge"
        batch_knowledge_dir.mkdir(parents=True, exist_ok=True)
        batch_red_base = Path(
            batch_config.get("batch_red_base") or (batch_knowledge_dir / "red_lessons_batch_base.jsonl")
        )
        batch_blue_base = Path(
            batch_config.get("batch_blue_base") or (batch_knowledge_dir / "blue_lessons_batch_base.jsonl")
        )

        summaries = _load_existing_summaries(batch_root)
        status = _read_json(batch_status_path)
        if not status:
            status = {
                "batch_id": batch_id,
                "batch_name": args.batch_name,
                "started_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "runs_total": args.runs,
                "runs_completed": 0,
                "runs_succeeded": 0,
                "runs_failed": 0,
                "current_run_id": None,
                "runs": [],
            }
        status["batch_id"] = batch_id
        status["batch_name"] = args.batch_name
        status["runs_total"] = args.runs
        status["runs"] = _status_runs_from_summaries(summaries)
        status["runs_completed"] = len(summaries)
        status["runs_succeeded"] = sum(1 for summary in summaries if summary.get("exit_status") == "completed")
        status["runs_failed"] = len(summaries) - status["runs_succeeded"]
        status["current_run_id"] = None
        status["updated_at"] = datetime.now().isoformat()
        status.pop("finished_at", None)

        batch_config["runs"] = args.runs
        batch_config["resumed_at"] = datetime.now().isoformat()
        batch_config["resume_batch_root"] = str(batch_root)
        batch_config["red_batch_learning_mode"] = red_batch_learning_mode
        batch_config["red_online_review_pipeline"] = red_online_review_pipeline
        batch_config["red_batch_base_max_lessons"] = red_batch_base_max_lessons
        _append_jsonl(
            batch_manifest_path,
            {
                "batch_id": batch_id,
                "event": "batch_resumed",
                "timestamp": datetime.now().isoformat(),
                "runs_completed": len(summaries),
                "runs_target": args.runs,
                "red_batch_learning_mode": red_batch_learning_mode,
                "red_online_review_pipeline": red_online_review_pipeline,
            },
        )
        _write_json(batch_config_path, batch_config)
        _write_json(batch_status_path, status)
        _write_batch_summary_csv(batch_summary_path, summaries)

        next_run_index = max([_run_index_from_id(summary.get("run_id")) or 0 for summary in summaries] + [0]) + 1
        if next_run_index > args.runs:
            status["finished_at"] = datetime.now().isoformat()
            status["updated_at"] = status["finished_at"]
            _write_json(batch_status_path, status)
            try:
                from analyze_batch import generate_batch_analysis

                generate_batch_analysis(batch_root)
            except Exception as exc:
                _append_jsonl(
                    batch_manifest_path,
                    {
                        "batch_id": batch_id,
                        "event": "batch_analysis_failed",
                        "timestamp": datetime.now().isoformat(),
                        "error": str(exc),
                    },
                )
            return
    else:
        red_batch_learning_mode = requested_red_batch_learning_mode
        red_online_review_pipeline = _normalize_red_online_review_pipeline(
            None,
            for_online=(red_batch_learning_mode == "online"),
            legacy_default=False,
        )
        red_batch_base_max_lessons = _read_int_env("RED_BATCH_BASE_MAX_LESSONS", 20)
        batch_id = f"{_safe_name(args.batch_name)}_{_timestamp()}"
        batch_root = Path(project_root()) / "test" / "batches" / batch_id
        batch_root.mkdir(parents=True, exist_ok=True)
        batch_manifest_path = batch_root / "run_manifest.jsonl"
        batch_status_path = batch_root / "batch_status.json"
        batch_summary_path = batch_root / "batch_summary.csv"
        batch_knowledge_dir = batch_root / "knowledge"
        batch_knowledge_dir.mkdir(parents=True, exist_ok=True)

        batch_red_base = batch_knowledge_dir / "red_lessons_batch_base.jsonl"
        batch_blue_base = batch_knowledge_dir / "blue_lessons_batch_base.jsonl"
        _copy_or_touch(_resolve_batch_base_store("red"), batch_red_base)
        _copy_or_touch(_resolve_lessons_store("blue"), batch_blue_base)

        args.end_simtime = int(args.end_simtime or 1800)
        args.engine_host = str(args.engine_host or "127.0.0.1")
        args.step_interval = float(args.step_interval or 0.1)

        status = {
            "batch_id": batch_id,
            "batch_name": args.batch_name,
            "started_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "runs_total": args.runs,
            "runs_completed": 0,
            "runs_succeeded": 0,
            "runs_failed": 0,
            "current_run_id": None,
            "runs": [],
        }
        batch_config = {
            "batch_id": batch_id,
            "batch_name": args.batch_name,
            "runs": args.runs,
            "end_simtime": args.end_simtime,
            "engine_host": args.engine_host,
            "step_interval": args.step_interval,
            "run_timeout_seconds": batch_timeout_seconds,
            "cooldown_seconds": batch_cooldown_seconds,
            "port_check_timeout_seconds": port_check_timeout_seconds,
            "reflection_timeout_seconds": _read_float_env("POST_BATTLE_REFLECTION_TIMEOUT_SECONDS", 0.0),
            "batch_red_base": str(batch_red_base),
            "batch_blue_base": str(batch_blue_base),
            "red_batch_base_mode": str(os.getenv("RED_BATCH_BASE_MODE", "minimal")).strip().lower() or "minimal",
            "red_batch_learning_mode": red_batch_learning_mode,
            "red_online_review_pipeline": red_online_review_pipeline,
            "red_batch_base_max_lessons": red_batch_base_max_lessons,
            "env_snapshot": _capture_env_snapshot(os.environ),
        }
        _write_json(batch_root / "batch_config.json", batch_config)
        _write_json(batch_status_path, status)
        summaries = []
        next_run_index = 1

    for run_index in range(next_run_index, args.runs + 1):
        run_id = f"run_{run_index:04d}"
        run_root = batch_root / run_id
        status["current_run_id"] = run_id
        status["updated_at"] = datetime.now().isoformat()
        _write_json(batch_status_path, status)

        readiness = _check_platform_ready(args.engine_host, timeout_seconds=port_check_timeout_seconds)
        if not readiness["ready"]:
            run_root.mkdir(parents=True, exist_ok=True)
            knowledge_paths = _prepare_run_knowledge(batch_red_base, batch_blue_base, run_root)
            child_env = _build_child_env(
                run_root=run_root,
                end_simtime=args.end_simtime,
                timeout_seconds=batch_timeout_seconds,
                runtime_red_lessons=knowledge_paths["runtime_red_lessons"],
                runtime_blue_lessons=knowledge_paths["runtime_blue_lessons"],
                red_batch_learning_mode=red_batch_learning_mode,
                red_online_review_pipeline=red_online_review_pipeline,
                red_batch_base_max_lessons=red_batch_base_max_lessons,
            )
            _write_json(run_root / "env_snapshot.json", _capture_env_snapshot(child_env))
            (run_root / "stdout.log").write_text("", encoding="utf-8")
            (run_root / "stderr.log").write_text("", encoding="utf-8")
            _finalize_run_knowledge(knowledge_paths["runtime_red_lessons"], knowledge_paths["lessons_after"])
            summary = {
                "batch_id": batch_id,
                "run_id": run_id,
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "wall_clock_seconds": 0.0,
                "exit_status": "platform_unreachable",
                "timeout": False,
                "stop_reason": "platform_unreachable",
                "max_red_simtime": None,
                "max_blue_simtime": None,
                "reflection_status_red": "unknown",
                "reflection_status_blue": "unknown",
                "artifact_paths": _build_artifact_paths(run_root),
                "first_red_action_simtime": None,
                "red_submitted_action_count": 0,
                "red_trace_count": 0,
                "red_llm_call_count": 0,
                "red_unique_targets_detected": 0,
                "blue_unique_targets_detected": 0,
                "red_high_value_targets_detected": 0,
                "blue_high_value_targets_detected": 0,
                "red_units_lost": 0,
                "blue_units_lost": 0,
                "red_targets_destroyed": 0,
                "blue_targets_destroyed": 0,
                "red_high_value_targets_destroyed": 0,
                "blue_high_value_targets_destroyed": 0,
                "red_intercept_launch_count": 0,
                "blue_intercept_launch_count": 0,
                "red_attack_launch_count": 0,
                "blue_attack_launch_count": 0,
                "red_score_gain_total": 0.0,
                "blue_score_gain_total": 0.0,
                "red_positive_score_events": 0,
                "blue_positive_score_events": 0,
                "final_score_red": None,
                "final_score_blue": None,
                "return_code": None,
                "ports": readiness["ports"],
            }
            run_root.mkdir(parents=True, exist_ok=True)
            _write_json(run_root / "run_summary.json", summary)
            _append_jsonl(
                batch_manifest_path,
                {
                    "batch_id": batch_id,
                    "run_id": run_id,
                    "event": "platform_unreachable",
                    "timestamp": datetime.now().isoformat(),
                    "ports": readiness["ports"],
                },
            )
        else:
            try:
                summary = _run_single(
                    batch_id=batch_id,
                    run_index=run_index,
                    run_root=run_root,
                    end_simtime=args.end_simtime,
                    engine_host=args.engine_host,
                    step_interval=args.step_interval,
                    timeout_seconds=batch_timeout_seconds,
                    batch_red_base=batch_red_base,
                    batch_blue_base=batch_blue_base,
                    batch_manifest_path=batch_manifest_path,
                    status=status,
                    red_batch_learning_mode=red_batch_learning_mode,
                    red_online_review_pipeline=red_online_review_pipeline,
                    red_batch_base_max_lessons=red_batch_base_max_lessons,
                )
            except Exception as exc:
                run_root.mkdir(parents=True, exist_ok=True)
                summary = {
                    "batch_id": batch_id,
                    "run_id": run_id,
                    "start_time": datetime.now().isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "wall_clock_seconds": 0.0,
                    "exit_status": "runner_error",
                    "timeout": False,
                    "stop_reason": "runner_error",
                    "max_red_simtime": None,
                    "max_blue_simtime": None,
                    "reflection_status_red": "unknown",
                    "reflection_status_blue": "unknown",
                    "artifact_paths": _build_artifact_paths(run_root),
                    "first_red_action_simtime": None,
                    "red_submitted_action_count": 0,
                    "red_trace_count": 0,
                    "red_llm_call_count": 0,
                    "red_unique_targets_detected": 0,
                    "blue_unique_targets_detected": 0,
                    "red_high_value_targets_detected": 0,
                    "blue_high_value_targets_detected": 0,
                    "red_units_lost": 0,
                    "blue_units_lost": 0,
                    "red_targets_destroyed": 0,
                    "blue_targets_destroyed": 0,
                    "red_high_value_targets_destroyed": 0,
                    "blue_high_value_targets_destroyed": 0,
                    "red_intercept_launch_count": 0,
                    "blue_intercept_launch_count": 0,
                    "red_attack_launch_count": 0,
                    "blue_attack_launch_count": 0,
                    "red_score_gain_total": 0.0,
                    "blue_score_gain_total": 0.0,
                    "red_positive_score_events": 0,
                    "blue_positive_score_events": 0,
                    "final_score_red": None,
                    "final_score_blue": None,
                    "return_code": None,
                    "runner_error": str(exc),
                }
                _write_json(run_root / "run_summary.json", summary)
                _append_jsonl(
                    batch_manifest_path,
                    {
                        "batch_id": batch_id,
                        "run_id": run_id,
                        "event": "runner_error",
                        "timestamp": datetime.now().isoformat(),
                        "error": str(exc),
                    },
                )

        summaries.append(summary)
        status["runs"].append(
            {
                "run_id": run_id,
                "exit_status": summary["exit_status"],
                "run_root": str(run_root),
                "started_at": summary["start_time"],
                "ended_at": summary["end_time"],
            }
        )
        status["runs_completed"] = len(summaries)
        if summary["exit_status"] == "completed":
            status["runs_succeeded"] += 1
        else:
            status["runs_failed"] += 1
        status["updated_at"] = datetime.now().isoformat()

        _write_json(batch_status_path, status)
        _write_batch_summary_csv(batch_summary_path, summaries)
        _promote_run_lessons_to_batch_base(
            red_batch_learning_mode=red_batch_learning_mode,
            red_online_review_pipeline=red_online_review_pipeline,
            red_batch_base_max_lessons=red_batch_base_max_lessons,
            summary=summary,
            run_root=run_root,
            batch_red_base=batch_red_base,
            batch_manifest_path=batch_manifest_path,
        )

        if run_index < args.runs:
            time.sleep(max(0.0, batch_cooldown_seconds))

    status["current_run_id"] = None
    status["finished_at"] = datetime.now().isoformat()
    status["updated_at"] = status["finished_at"]
    _write_json(batch_status_path, status)

    try:
        from analyze_batch import generate_batch_analysis

        generate_batch_analysis(batch_root)
    except Exception as exc:
        _append_jsonl(
            batch_manifest_path,
            {
                "batch_id": batch_id,
                "event": "batch_analysis_failed",
                "timestamp": datetime.now().isoformat(),
                "error": str(exc),
            },
        )


if __name__ == "__main__":
    main()
