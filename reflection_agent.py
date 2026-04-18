import hashlib
import json
import logging
import os
import re
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

try:
    from jsqlsim.llm.model.chat_model import Base as ChatModelBase
except Exception:  # pragma: no cover - allows offline/reconstruction utilities to import safely
    class ChatModelBase:  # type: ignore[override]
        pass

from ltm_retriever import build_structured_lesson
from prompt_library import load_past_lessons
from runtime_paths import ensure_output_dir, project_root

logger = logging.getLogger(__name__)


WEAPON_COSTS = {
    "HighCostAttackMissile": 8.0,
    "LowCostAttackMissile": 3.0,
    "ShipToGround_CruiseMissile": 5.0,
    "ShipToLandCruiseMissile": 5.0,
    "Long_Range_InterceptMissile": 3.0,
    "LongInterceptMissile": 3.0,
    "Short_Range_InterceptMissile": 1.5,
    "ShortInterceptMissile": 1.5,
    "AIM": 4.0,
    "AIM_AirMissile": 4.0,
    "AirToAirMissile": 4.0,
    "JDAM": 4.0,
    "AirToSurfaceMissile": 4.0,
}

TYPE_LABELS = {
    "Flagship_Surface": "\u65d7\u8230",
    "Cruiser_Surface": "\u5de1\u6d0b\u8230",
    "Destroyer_Surface": "\u9a71\u9010\u8230",
    "Guide_Ship_Surface": "\u5f15\u5bfc\u5feb\u8247",
    "Truck_Ground": "\u53d1\u5c04\u8f66",
    "Recon_UAV_FixWing": "\u4fa6\u5bdf\u65e0\u4eba\u673a",
    "Shipboard_Aircraft_FixWing": "\u8230\u8f7d\u673a",
    "Merchant_Ship_Surface": "\u5546\u8239",
    "HighCostAttackMissile": "\u9ad8\u6210\u672c\u653b\u51fb\u5f39",
    "LowCostAttackMissile": "\u4f4e\u6210\u672c\u653b\u51fb\u5f39",
    "ShipToGround_CruiseMissile": "\u5bf9\u5730\u5de1\u822a\u5bfc\u5f39",
    "ShipToLandCruiseMissile": "\u5bf9\u5730\u5de1\u822a\u5bfc\u5f39",
    "Long_Range_InterceptMissile": "\u8fdc\u7a0b\u62e6\u622a\u5f39",
    "LongInterceptMissile": "\u8fdc\u7a0b\u62e6\u622a\u5f39",
    "Short_Range_InterceptMissile": "\u8fd1\u7a0b\u62e6\u622a\u5f39",
    "ShortInterceptMissile": "\u8fd1\u7a0b\u62e6\u622a\u5f39",
    "AIM": "\u7a7a\u7a7a\u5bfc\u5f39",
    "AIM_AirMissile": "\u7a7a\u7a7a\u5bfc\u5f39",
    "AirToAirMissile": "\u7a7a\u7a7a\u5bfc\u5f39",
    "JDAM": "\u5bf9\u5730\u5f39\u836f",
    "AirToSurfaceMissile": "\u5bf9\u5730\u5f39\u836f",
}
LESSON_RULE_BRANDING_MAP = {
    "CostGate": "\u6210\u672c\u68c0\u67e5",
    "AmmoMixRule": "\u5f39\u79cd\u642d\u914d\u51c6\u5219",
}
LESSON_TEXT_ALIAS_MAP = {
    "fire_high_cost": "FocusFire",
    "guide_attack": "GuideAttack",
    "refresh_track": "ScoutArea",
    "move_then_fire": "MoveToEngage",
}
REVIEW_DECISIONS = {"keep", "rewrite", "drop", "merge_into_existing"}

def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        return int(raw)
    except Exception:
        logger.warning("Invalid %s=%s, fallback to %s", name, raw, default)
        return default


def _safe_sim_time(value, default: int = 10**9) -> int:
    if value is None:
        return default
    try:
        return int(float(value))
    except Exception:
        return default


class PostBattleReflectionAgent:
    def __init__(self, llm_model: ChatModelBase, knowledge_base_path: str):
        self.llm_model = llm_model
        self.knowledge_base_path = knowledge_base_path
        self.side = self._infer_side(knowledge_base_path)
        self.auto_writeback = (self.side == "red") and (_read_int_env("RED_LTM_AUTO_WRITEBACK", 1) == 1)
        self.structured_store_path = self._resolve_structured_store_path()
        logger.info(
            "ReflectionAgent initialized. side=%s legacy_store=%s structured_store=%s auto_writeback=%s",
            self.side,
            self.knowledge_base_path,
            self.structured_store_path,
            self.auto_writeback,
        )

    def _review_pipeline_mode(self) -> str:
        if self.side != "red" or not self.auto_writeback:
            return "disabled"
        if str(os.getenv("BATCH_MODE", "0")).strip().lower() not in {"1", "true", "yes", "on"}:
            return "legacy"
        raw = str(os.getenv("RED_ONLINE_REVIEW_PIPELINE", "legacy")).strip().lower() or "legacy"
        if raw not in {"legacy", "dual_stage_v1", "disabled"}:
            return "legacy"
        return raw

    def _review_capacity(self) -> int:
        return max(4, _read_int_env("RED_BATCH_BASE_MAX_LESSONS", 20))

    def _reflection_artifact_dir(self, reconstructed: bool = False) -> str:
        return ensure_output_dir("reflection_reconstructed" if reconstructed else "reflection")

    def _write_text(self, path: str, content: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(str(content or ""))

    def _write_json(self, path: str, payload) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def _read_jsonl_rows(self, path: str) -> List[dict]:
        rows: List[dict] = []
        if not os.path.exists(path):
            return rows
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(data, dict):
                        rows.append(data)
        except Exception as exc:
            logger.warning("Failed to read jsonl %s: %s", path, exc)
        return rows

    def _write_jsonl_rows(self, path: str, rows: List[dict]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _extract_json_payload(self, raw_text: str):
        text = str(raw_text or "").strip()
        if not text:
            raise RuntimeError("Review agent returned empty response")
        if text.startswith("**ERROR**"):
            raise RuntimeError(f"Review agent returned error sentinel: {text[:240]}")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        start_obj = text.find("{")
        end_obj = text.rfind("}")
        if start_obj >= 0 and end_obj > start_obj:
            candidate = text[start_obj : end_obj + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        start_arr = text.find("[")
        end_arr = text.rfind("]")
        if start_arr >= 0 and end_arr > start_arr:
            candidate = text[start_arr : end_arr + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        raise RuntimeError(f"Failed to parse review JSON: {text[:240]}")

    def _call_json_agent(self, prompt: str, *, stage_name: str) -> Tuple[dict, str]:
        self.llm_model.history = [{"role": "system", "content": prompt}]
        response_text = str(self.llm_model.chat() or "").strip()
        parsed = self._extract_json_payload(response_text)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"{stage_name} returned non-dict JSON: {type(parsed).__name__}")
        return parsed, response_text

    def _infer_side(self, path: str) -> str:
        p = str(path or "").lower()
        if "blue" in p:
            return "blue"
        if "red" in p:
            return "red"
        return "red"

    def _resolve_structured_store_path(self) -> str:
        env_name = f"{self.side.upper()}_LTM_STORE_PATH"
        env_path = os.getenv(env_name, "").strip()
        if env_path:
            if os.path.isabs(env_path):
                return env_path
            return os.path.join(project_root(), env_path)
        return os.path.join(project_root(), "test", f"{self.side}_lessons_structured.jsonl")

    def _resolve_rules_context_path(self) -> str:
        raw = os.getenv("REFLECTION_RULES_CONTEXT_PATH", "").strip()
        if raw:
            if os.path.isabs(raw):
                return raw
            return os.path.join(project_root(), raw)
        return os.path.join(project_root(), "summary", "reflection_rules_context.md")

    def _load_rules_context(self) -> str:
        path = self._resolve_rules_context_path()
        if not os.path.exists(path):
            logger.info("Reflection rules context not found at %s", path)
            return "\u672a\u63d0\u4f9b\u5916\u90e8\u89c4\u5219\u6458\u8981\u3002"

        try:
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read().strip()
        except Exception as e:
            logger.warning("Failed to load reflection rules context %s: %s", path, e)
            return "\u89c4\u5219\u6458\u8981\u8bfb\u53d6\u5931\u8d25\u3002"

        max_chars = _read_int_env("REFLECTION_RULES_CONTEXT_MAX_CHARS", 5000)
        return self._truncate_text(text, max_chars)

    def _classify_identifier(self, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return "\u672a\u77e5"
        for prefix, label in TYPE_LABELS.items():
            if text.startswith(prefix):
                return label
        return text.split("-", 1)[0]

    def _format_counter(self, counter: Counter, limit: int = 8) -> str:
        if not counter:
            return "\u65e0"
        items = [f"{key}:{value}" for key, value in counter.most_common(limit)]
        return "\uff0c".join(items)

    def _extract_action_counters(self, actions: List[dict]):
        action_types = Counter()
        actor_types = Counter()
        weapon_types = Counter()
        target_types = Counter()
        estimated_cost = 0.0

        for action in actions or []:
            if not isinstance(action, dict):
                continue

            action_type = str(action.get("Type") or action.get("type") or "unknown").strip()
            if action_type:
                action_types[action_type] += 1

            unit_ids = action.get("UnitIds") or action.get("unit_ids") or []
            if not unit_ids:
                single_unit = action.get("Id") or action.get("UnitId") or action.get("unit_id")
                if single_unit:
                    unit_ids = [single_unit]
            for actor_id in unit_ids:
                actor_types[self._classify_identifier(str(actor_id))] += 1

            target_id = action.get("Target_Id") or action.get("TargetId") or action.get("target_id")
            if target_id:
                target_types[self._classify_identifier(str(target_id))] += 1

            weapon_type = str(action.get("WeaponType") or action.get("weapon_type") or "").strip()
            if weapon_type:
                weapon_types[self._classify_identifier(weapon_type)] += 1
                estimated_cost += WEAPON_COSTS.get(weapon_type, 0.0)

        return action_types, actor_types, weapon_types, target_types, estimated_cost

    def _truncate_text(self, text: str, max_chars: int) -> str:
        text = str(text or "").strip()
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        clipped = text[:max_chars].rstrip()
        if "\n" in clipped:
            clipped = clipped.rsplit("\n", 1)[0].rstrip()
        return clipped + "\n..."

    def _score_key_map(self) -> Dict[str, str]:
        if self.side == "blue":
            return {
                "side_score": "blueScore",
                "side_destroy": "blueDestroyScore",
                "side_cost": "blueCost",
                "enemy_score": "redScore",
                "enemy_destroy": "redDestroyScore",
                "enemy_cost": "redCost",
            }
        return {
            "side_score": "redScore",
            "side_destroy": "redDestroyScore",
            "side_cost": "redCost",
            "enemy_score": "blueScore",
            "enemy_destroy": "blueDestroyScore",
            "enemy_cost": "blueCost",
        }

    def _safe_float(self, value, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _cluster_key(self, item: dict) -> str:
        parts = [
            str(item.get("phase", "")).strip().lower(),
            str(item.get("target_type", "")).strip().lower(),
            str(item.get("symptom", "")).strip().lower(),
            str(item.get("cost_risk", "")).strip().lower(),
        ]
        compact = "|".join(parts).strip("|")
        if compact:
            return compact
        return self._normalize_lesson(item.get("lesson", ""))

    def _lesson_quality_score(self, item: dict) -> float:
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
        support_bonus = max(0.0, self._safe_float(item.get("support_count"), 1.0) - 1.0) * 2.0
        existing_bonus = max(0.0, self._safe_float(item.get("quality_score"), 0.0))
        return structured_bonus + tags_bonus + support_bonus + existing_bonus

    def _merge_sentence_text(self, left: str, right: str, max_chars: int = 280) -> str:
        parts: List[str] = []
        seen = set()
        for raw in (left, right):
            cleaned = self._sanitize_lesson_text(raw)
            if not cleaned:
                continue
            for chunk in re.split(r"[;；。]\s*|\n+", cleaned):
                piece = chunk.strip(" ;；。")
                if not piece:
                    continue
                normalized = self._normalize_lesson(piece)
                if normalized in seen:
                    continue
                seen.add(normalized)
                parts.append(piece)
        merged = "；".join(parts).strip()
        if len(merged) <= max_chars:
            return merged
        return merged[:max_chars].rstrip("； ")

    def _sanitize_review_lesson_record(
        self,
        item: dict,
        *,
        fallback_type: str,
        battle_meta: dict,
        source: str,
    ) -> Optional[dict]:
        if not isinstance(item, dict):
            return None

        lesson = self._sanitize_lesson_text(item.get("lesson", ""))
        if not lesson:
            return None
        observation = self._sanitize_lesson_text(item.get("observation", ""))
        trigger = self._sanitize_lesson_text(item.get("trigger", ""))
        score_pattern = self._sanitize_lesson_text(item.get("score_pattern", ""))
        symptom = self._sanitize_lesson_text(item.get("symptom", ""))
        target_type = self._sanitize_lesson_text(item.get("target_type", ""))
        phase = str(item.get("phase", "")).strip().lower()
        cost_risk = str(item.get("cost_risk", "")).strip().lower()

        tags = item.get("tags") or []
        if not isinstance(tags, list):
            tags = []
        sanitized_tags: List[str] = []
        for tag in tags:
            cleaned = self._sanitize_lesson_text(tag)
            if cleaned and cleaned not in sanitized_tags:
                sanitized_tags.append(cleaned)

        merged_meta = dict(battle_meta or {})
        incoming_meta = item.get("battle_meta") or {}
        if isinstance(incoming_meta, dict):
            merged_meta.update(incoming_meta)

        record = build_structured_lesson(
            lesson_type=str(item.get("type", fallback_type)).strip() or fallback_type,
            observation=observation,
            lesson=lesson,
            tags=sanitized_tags,
            battle_meta=merged_meta,
            source=source,
            phase=phase,
            target_type=target_type,
            symptom=symptom,
            trigger=trigger,
            score_pattern=score_pattern,
            cost_risk=cost_risk,
        )
        normalized = self._normalize_lesson(lesson)
        record["normalized_lesson"] = normalized
        record["lesson_hash"] = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
        record["support_count"] = max(1, int(self._safe_float(item.get("support_count"), 1)))
        record["quality_score"] = max(0.0, self._safe_float(item.get("quality_score"), 0.0))
        record["protocol_safe"] = True
        source_run_ids = item.get("source_run_ids") or []
        if not isinstance(source_run_ids, list):
            source_run_ids = []
        source_batch_ids = item.get("source_batch_ids") or []
        if not isinstance(source_batch_ids, list):
            source_batch_ids = []
        record["source_run_ids"] = source_run_ids
        record["source_batch_ids"] = source_batch_ids
        return record

    def _merge_review_lesson_records(self, existing: dict, incoming: dict) -> dict:
        merged = dict(existing)
        for field, max_chars in (("observation", 220), ("lesson", 280), ("trigger", 180), ("score_pattern", 180)):
            merged[field] = self._merge_sentence_text(existing.get(field, ""), incoming.get(field, ""), max_chars=max_chars)
        for field in ("phase", "target_type", "symptom", "cost_risk"):
            if not str(merged.get(field, "")).strip() and str(incoming.get(field, "")).strip():
                merged[field] = incoming.get(field, "")

        tags: List[str] = []
        for tag in (existing.get("tags") or []) + (incoming.get("tags") or []):
            cleaned = self._sanitize_lesson_text(tag)
            if cleaned and cleaned not in tags:
                tags.append(cleaned)
        merged["tags"] = tags[:8]
        merged["support_count"] = max(1, int(self._safe_float(existing.get("support_count"), 1))) + max(
            1, int(self._safe_float(incoming.get("support_count"), 1))
        )
        merged["quality_score"] = max(self._safe_float(existing.get("quality_score"), 0.0), self._safe_float(incoming.get("quality_score"), 0.0))
        merged["normalized_lesson"] = self._normalize_lesson(merged.get("lesson", ""))
        merged["lesson_hash"] = hashlib.sha1(merged["normalized_lesson"].encode("utf-8")).hexdigest()
        merged["lesson_id"] = hashlib.sha1(merged["normalized_lesson"].encode("utf-8")).hexdigest()[:16]

        source_run_ids: List[str] = []
        for run_id in (existing.get("source_run_ids") or []) + (incoming.get("source_run_ids") or []):
            if run_id and run_id not in source_run_ids:
                source_run_ids.append(run_id)
        source_batch_ids: List[str] = []
        for batch_id in (existing.get("source_batch_ids") or []) + (incoming.get("source_batch_ids") or []):
            if batch_id and batch_id not in source_batch_ids:
                source_batch_ids.append(batch_id)
        merged["source_run_ids"] = source_run_ids
        merged["source_batch_ids"] = source_batch_ids
        return merged

    def _format_score_breakdown(self, score_breakdown: Dict) -> str:
        if not isinstance(score_breakdown, dict) or not score_breakdown:
            return "none"
        ordered = [
            "redScore",
            "redDestroyScore",
            "redCost",
            "blueScore",
            "blueDestroyScore",
            "blueCost",
        ]
        parts = []
        seen = set()
        for key in ordered:
            if key in score_breakdown:
                parts.append(f"{key}={score_breakdown[key]}")
                seen.add(key)
        for key in sorted(score_breakdown.keys()):
            if key not in seen:
                parts.append(f"{key}={score_breakdown[key]}")
        return ", ".join(parts) if parts else "none"

    def _build_reflection_digest(self, events: List[dict], final_score: str) -> str:
        if not events:
            return f"\u6700\u7ec8\u5f97\u5206\uff1a{final_score}\n\u672c\u5c40\u672a\u91c7\u96c6\u5230\u53ef\u7528\u6218\u6597\u4e8b\u4ef6\u3002"

        max_key_events = _read_int_env("REFLECTION_DIGEST_MAX_KEY_EVENTS", 12)
        max_chars = _read_int_env("REFLECTION_BATTLE_SUMMARY_MAX_CHARS", 6000)
        score_key_map = self._score_key_map()

        event_type_counts = Counter()
        detected_target_types = Counter()
        first_detection_time: Dict[str, int] = {}
        lost_unit_types = Counter()
        decision_source_counts = Counter()
        action_type_counts = Counter()
        actor_type_counts = Counter()
        weapon_type_counts = Counter()
        target_type_counts = Counter()
        score_deltas: List[float] = []
        selected_timeline: List[str] = []
        estimated_weapon_cost = 0.0
        score_points: List[dict] = []
        latest_score_breakdown: Dict = {}
        first_score_breakdown: Dict = {}

        sorted_events = sorted(events, key=lambda x: (_safe_sim_time(x.get("sim_time")), str(x.get("type", ""))))

        for event in sorted_events:
            event_type = str(event.get("type", "UNKNOWN"))
            sim_time = event.get("sim_time", "?")
            extra = event.get("extra") or {}
            event_type_counts[event_type] += 1

            if event_type == "NEW_TARGET_DETECTED":
                target_id = str(extra.get("target_id") or "")
                target_label = self._classify_identifier(target_id)
                detected_target_types[target_label] += 1
                if target_id and target_id not in first_detection_time:
                    first_detection_time[target_id] = sim_time
                    if len(selected_timeline) < max_key_events and target_label in {"\u65d7\u8230", "\u5de1\u6d0b\u8230", "\u9a71\u9010\u8230", "\u5546\u8239"}:
                        selected_timeline.append(f"[t={sim_time}] \u9996\u6b21\u53d1\u73b0\u9ad8\u4ef7\u503c\u76ee\u6807 {target_id}")

            elif event_type == "UNIT_LOST":
                unit_id = str(extra.get("unit_id") or "")
                unit_label = self._classify_identifier(unit_id)
                lost_unit_types[unit_label] += 1
                if len(selected_timeline) < max_key_events:
                    selected_timeline.append(f"[t={sim_time}] \u5355\u4f4d\u635f\u5931 {unit_id}")

            elif event_type in {"SCORE_CHANGED", "SCORE_SNAPSHOT"}:
                score_breakdown = extra.get("score_breakdown") or {}
                old_score = extra.get("old_score")
                new_score = extra.get("new_score")
                side_score = extra.get("side_score")
                delta = self._safe_float(extra.get("delta"), 0.0)

                if event_type == "SCORE_CHANGED":
                    if "delta" not in extra:
                        delta = self._safe_float(new_score) - self._safe_float(old_score)
                    score_deltas.append(delta)

                if isinstance(score_breakdown, dict) and score_breakdown:
                    latest_score_breakdown = dict(score_breakdown)
                    if not first_score_breakdown:
                        first_score_breakdown = dict(score_breakdown)
                    side_score_value = self._safe_float(score_breakdown.get(score_key_map["side_score"]), self._safe_float(side_score))
                    score_points.append(
                        {
                            "sim_time": _safe_sim_time(sim_time),
                            "side_score": side_score_value,
                            "score_breakdown": dict(score_breakdown),
                            "delta": delta,
                        }
                    )

                if len(selected_timeline) < max_key_events:
                    if event_type == "SCORE_CHANGED":
                        selected_timeline.append(
                            f"[t={sim_time}] \u5206\u6570\u53d8\u5316 {old_score} -> {new_score}\uff0c\u6784\u6210 {self._format_score_breakdown(score_breakdown)}"
                        )
                    else:
                        selected_timeline.append(
                            f"[t={sim_time}] \u5206\u6570\u5feb\u7167\uff0c\u6784\u6210 {self._format_score_breakdown(score_breakdown)}"
                        )

            elif event_type == "DECISION":
                source = str(event.get("source", "unknown"))
                decision_source_counts[source] += 1
                actions = event.get("actions") or []
                if isinstance(actions, list):
                    a_counts, actor_counts, weapon_counts, target_counts, est_cost = self._extract_action_counters(actions)
                    action_type_counts.update(a_counts)
                    actor_type_counts.update(actor_counts)
                    weapon_type_counts.update(weapon_counts)
                    target_type_counts.update(target_counts)
                    estimated_weapon_cost += est_cost
                    if len(selected_timeline) < max_key_events and (a_counts.get("Launch", 0) > 0 or source == "system2"):
                        selected_timeline.append(f"[t={sim_time}] \u51b3\u7b56[{source}] \u52a8\u4f5c={self._format_counter(a_counts, limit=4)}")

        high_value_detections = []
        for target_id, detected_time in sorted(first_detection_time.items(), key=lambda item: _safe_sim_time(item[1])):
            label = self._classify_identifier(target_id)
            if label in {"\u65d7\u8230", "\u5de1\u6d0b\u8230", "\u9a71\u9010\u8230"}:
                high_value_detections.append(f"{target_id}@t={detected_time}")

        score_turning_points: List[str] = []
        score_points.sort(key=lambda item: item["sim_time"])
        if score_points:
            best_point = max(score_points, key=lambda item: item["side_score"])
            worst_point = min(score_points, key=lambda item: item["side_score"])
            score_turning_points.append(f"\u6700\u9ad8\u5206\u65f6\u523b\uff1at={best_point['sim_time']}\uff0cscore={best_point['side_score']:.1f}")
            score_turning_points.append(f"\u6700\u4f4e\u5206\u65f6\u523b\uff1at={worst_point['sim_time']}\uff0cscore={worst_point['side_score']:.1f}")

        side_destroy_gain = 0.0
        side_cost_gain = 0.0
        enemy_destroy_gain = 0.0
        enemy_cost_gain = 0.0
        if latest_score_breakdown and first_score_breakdown:
            side_destroy_gain = self._safe_float(latest_score_breakdown.get(score_key_map["side_destroy"])) - self._safe_float(
                first_score_breakdown.get(score_key_map["side_destroy"])
            )
            side_cost_gain = self._safe_float(latest_score_breakdown.get(score_key_map["side_cost"])) - self._safe_float(
                first_score_breakdown.get(score_key_map["side_cost"])
            )
            enemy_destroy_gain = self._safe_float(latest_score_breakdown.get(score_key_map["enemy_destroy"])) - self._safe_float(
                first_score_breakdown.get(score_key_map["enemy_destroy"])
            )
            enemy_cost_gain = self._safe_float(latest_score_breakdown.get(score_key_map["enemy_cost"])) - self._safe_float(
                first_score_breakdown.get(score_key_map["enemy_cost"])
            )

        score_reason_hints = []
        if latest_score_breakdown:
            score_reason_hints.append("\u6700\u7ec8\u5206\u6570\u5b57\u6bb5\uff1a" + self._format_score_breakdown(latest_score_breakdown))
            score_reason_hints.append(
                f"\u672c\u65b9\u6bc1\u4f24\u5f97\u5206\u53d8\u5316={side_destroy_gain:+.1f}\uff0c\u672c\u65b9\u6210\u672c\u53d8\u5316={side_cost_gain:+.1f}\uff1b"
                f"\u5bf9\u624b\u6bc1\u4f24\u5f97\u5206\u53d8\u5316={enemy_destroy_gain:+.1f}\uff0c\u5bf9\u624b\u6210\u672c\u53d8\u5316={enemy_cost_gain:+.1f}"
            )
            if side_cost_gain > max(0.0, side_destroy_gain):
                score_reason_hints.append("\u672c\u65b9\u6210\u672c\u4e0a\u6da8\u5feb\u4e8e\u6bc1\u4f24\u6536\u76ca\uff0c\u8bf4\u660e\u5b58\u5728\u8fc7\u5ea6\u53d1\u5c04\u3001\u6536\u76ca\u4e0d\u8db3\u6216\u6210\u672c\u63a7\u5236\u504f\u5f31\u7684\u95ee\u9898\u3002")
            if enemy_destroy_gain > side_destroy_gain:
                score_reason_hints.append("\u5bf9\u624b\u6bc1\u4f24\u6536\u76ca\u9ad8\u4e8e\u672c\u65b9\uff0c\u8bf4\u660e\u672c\u5c40\u5728\u4ea4\u6362\u6bd4\u6216\u5173\u952e\u5355\u4f4d\u4fdd\u62a4\u4e0a\u5904\u4e8e\u52a3\u52bf\u3002")
            if side_destroy_gain > side_cost_gain and side_destroy_gain > 0:
                score_reason_hints.append("\u672c\u65b9\u6bc1\u4f24\u6536\u76ca\u9ad8\u4e8e\u6210\u672c\u589e\u957f\uff0c\u8bf4\u660e\u5f53\u524d\u6253\u6cd5\u5728\u5f97\u5206\u6548\u7387\u4e0a\u6709\u4e00\u5b9a\u4f18\u52bf\u3002")
        else:
            score_reason_hints.append("battle log \u4e2d\u7f3a\u5c11\u6709\u6548\u7684 score breakdown\uff0c\u53cd\u601d\u53ea\u80fd\u57fa\u4e8e\u4e8b\u4ef6\u8fd1\u4f3c\u5224\u65ad\u5f97\u5206\u539f\u56e0\u3002")

        lines = [
            f"\u6700\u7ec8\u5f97\u5206\uff1a{final_score}",
            "",
            "\u4e00\u3001\u6218\u5c40\u6982\u89c8",
            f"- \u4e8b\u4ef6\u5206\u5e03\uff1a{self._format_counter(event_type_counts, limit=10)}",
            f"- \u51b3\u7b56\u6765\u6e90\u5206\u5e03\uff1a{self._format_counter(decision_source_counts, limit=6)}",
            f"- \u63a2\u6d4b\u5230\u7684\u76ee\u6807\u7c7b\u578b\uff1a{self._format_counter(detected_target_types, limit=10)}",
            f"- \u9ad8\u4ef7\u503c\u76ee\u6807\u9996\u6b21\u53d1\u73b0\uff1a{', '.join(high_value_detections[:8]) if high_value_detections else '\\u65e0'}",
            f"- \u5df2\u65b9\u635f\u5931\u7c7b\u578b\uff1a{self._format_counter(lost_unit_types, limit=8)}",
            f"- \u52a8\u4f5c\u7c7b\u578b\u7edf\u8ba1\uff1a{self._format_counter(action_type_counts, limit=10)}",
            f"- \u6267\u884c\u52a8\u4f5c\u5e73\u53f0\uff1a{self._format_counter(actor_type_counts, limit=10)}",
            f"- \u6b66\u5668\u4f7f\u7528\u7edf\u8ba1\uff1a{self._format_counter(weapon_type_counts, limit=10)}",
            f"- \u4e3b\u8981\u6253\u51fb\u76ee\u6807\u7c7b\u578b\uff1a{self._format_counter(target_type_counts, limit=8)}",
            f"- \u8fd1\u4f3c\u6b66\u5668\u6210\u672c\uff1a{estimated_weapon_cost:.1f}",
            f"- \u5206\u6570\u53d8\u5316\u5e8f\u5217\uff1a{', '.join(f'{delta:+.1f}' for delta in score_deltas[:10]) if score_deltas else '\\u65e0'}",
            "",
            "\u4e8c\u3001\u5f97\u5206\u6784\u6210\u4e0e\u8f6c\u6298",
        ]
        lines.extend(f"- {item}" for item in score_reason_hints)
        if score_turning_points:
            lines.extend(f"- {item}" for item in score_turning_points)

        lines.extend(["", "\u4e09\u3001\u5173\u952e\u65f6\u95f4\u7ebf"])
        if selected_timeline:
            lines.extend(f"- {item}" for item in selected_timeline[:max_key_events])
        else:
            lines.append("- \u65e0")

        return self._truncate_text("\n".join(lines), max_chars)

    def _generate_timeline(self, event_log_path: str, final_score: str) -> str:
        try:
            events = self._load_event_log(event_log_path)
        except Exception as e:
            logger.error("Failed to load event log %s: %s", event_log_path, e)
            return f"\u6700\u7ec8\u5f97\u5206\uff1a{final_score}\n\n\u4e8b\u4ef6\u65e5\u5fd7\u8bfb\u53d6\u5931\u8d25\u3002"
        return self._build_reflection_digest(events, final_score)

    def _build_outcome_summary(self, events: List[dict], final_score: str) -> dict:
        score_breakdown = {}
        key_map = self._score_key_map()
        for event in sorted(events or [], key=lambda item: _safe_sim_time(item.get("sim_time"))):
            extra = event.get("extra") or {}
            if not isinstance(extra, dict):
                continue
            breakdown = extra.get("score_breakdown") or {}
            if isinstance(breakdown, dict) and breakdown:
                score_breakdown = dict(breakdown)

        summary = {
            "final_score_text": final_score,
            "final_score_red": self._safe_float(score_breakdown.get("redScore")),
            "final_score_blue": self._safe_float(score_breakdown.get("blueScore")),
            "destroy_score_red": self._safe_float(score_breakdown.get("redDestroyScore")),
            "destroy_score_blue": self._safe_float(score_breakdown.get("blueDestroyScore")),
            "cost_red": self._safe_float(score_breakdown.get("redCost")),
            "cost_blue": self._safe_float(score_breakdown.get("blueCost")),
        }
        summary["score_margin"] = summary["final_score_red"] - summary["final_score_blue"]
        summary["destroy_margin"] = summary["destroy_score_red"] - summary["destroy_score_blue"]
        red_efficiency = summary["destroy_score_red"] - summary["cost_red"]
        blue_efficiency = summary["destroy_score_blue"] - summary["cost_blue"]
        summary["cost_efficiency_margin"] = red_efficiency - blue_efficiency
        summary["allow_new_principles"] = (
            summary["score_margin"] >= 0
            or (
                summary["score_margin"] >= -10
                and summary["destroy_margin"] >= 0
                and summary["cost_efficiency_margin"] >= -10
            )
        )
        return summary

    def _collect_reflection_candidates(self, reflection: Dict, battle_meta: dict) -> List[dict]:
        candidates: List[dict] = []
        existing_hashes = set()
        for lesson_type, key in (("success", "successes"), ("failure", "failures")):
            for item in reflection.get(key, []) or []:
                if not isinstance(item, dict):
                    continue
                lesson_record = self._sanitize_review_lesson_record(
                    item,
                    fallback_type=lesson_type,
                    battle_meta=battle_meta,
                    source="reflection_agent",
                )
                if not lesson_record:
                    continue
                lesson_hash = str(lesson_record.get("lesson_hash", "")).strip()
                if lesson_hash and lesson_hash in existing_hashes:
                    continue
                existing_hashes.add(lesson_hash)
                lesson_record["candidate_index"] = len(candidates)
                candidates.append(lesson_record)
        return candidates

    def _review_rows_for_prompt(self, rows: List[dict]) -> List[dict]:
        compact_rows: List[dict] = []
        for item in rows or []:
            if not isinstance(item, dict):
                continue
            compact_rows.append(
                {
                    "candidate_index": item.get("candidate_index"),
                    "lesson_id": item.get("lesson_id"),
                    "type": item.get("type"),
                    "phase": item.get("phase", ""),
                    "target_type": item.get("target_type", ""),
                    "symptom": item.get("symptom", ""),
                    "trigger": item.get("trigger", ""),
                    "score_pattern": item.get("score_pattern", ""),
                    "cost_risk": item.get("cost_risk", ""),
                    "observation": item.get("observation", ""),
                    "lesson": item.get("lesson", ""),
                    "tags": item.get("tags") or [],
                    "support_count": item.get("support_count", 1),
                    "quality_score": item.get("quality_score", 0.0),
                }
            )
        return compact_rows

    def _build_review_stage1_prompt(
        self,
        *,
        current_base_rows: List[dict],
        candidate_rows: List[dict],
        outcome_summary: dict,
        timeline: str,
    ) -> str:
        return f"""
\u4f60\u662f RED lessons \u5ba1\u6838 Agent\uff08\u7b2c 1 \u9636\u6bb5\uff09\u3002\u4f60\u53ea\u8d1f\u8d23\u5ba1\u6838\u672c\u5c40\u65b0\u5019\u9009 lessons\uff0c\u4e0d\u91cd\u5199\u6574\u4e2a\u57fa\u7ebf\u3002

\u76ee\u6807\uff1a\u5224\u65ad\u6bcf\u6761\u5019\u9009 lesson \u662f\u5426\u503c\u5f97\u4fdd\u7559\uff0c\u662f\u5426\u9700\u8981\u6539\u5199\uff0c\u662f\u5426\u53ea\u80fd\u5408\u5e76\u5230\u65e7 lesson\uff0c\u6216\u8005\u5e94\u8be5\u4e22\u5f03\u3002

\u786c\u6027\u7ea6\u675f\uff1a
1. \u7981\u6b62\u51fa\u73b0\u4efb\u4f55\u81ea\u521b\u51fd\u6570\u540d\u3001\u8c03\u7528\u540d\u3001task \u540d\u3001\u89c4\u5219\u54c1\u724c\u540d\u3002
2. \u5019\u9009 lessons \u53ea\u80fd\u8868\u8fbe\u201c\u53ef\u590d\u7528\u7684\u539f\u5219\u201d\uff0c\u4e0d\u8981\u5199\u534f\u8bae\u7ea7\u52a8\u4f5c\u5b57\u6bb5\u3002
3. \u5982\u679c\u5019\u9009 lesson \u53ea\u662f\u5355\u5c40\u5076\u7136\u73b0\u8c61\uff0c\u5fc5\u987b `drop`\u3002
4. \u5982\u679c\u5b83\u4e0e\u65e7 lessons \u5b9e\u8d28\u4e0a\u662f\u540c\u4e00\u4e3b\u9898\uff0c\u4f18\u5148 `merge_into_existing` \u6216 `rewrite`\uff0c\u4e0d\u8981\u65b0\u5efa\u51e0\u4e4e\u91cd\u590d\u7684\u539f\u5219\u3002
5. \u672c\u5c40\u662f{"\u975e\u9000\u5316/\u5141\u8bb8\u5f15\u5165\u65b0\u539f\u5219" if outcome_summary.get("allow_new_principles") else "\u9000\u5316\u5c40/\u4e0d\u5141\u8bb8\u5f15\u5165\u65b0\u539f\u5219"}\u3002\u9000\u5316\u5c40\u53ea\u80fd `merge_into_existing`\u3001`rewrite` \u5df2\u6709\u539f\u5219\u6216 `drop`\uff0c\u4e0d\u8981 `keep` \u4e3a\u5168\u65b0\u4e3b\u9898\u3002

\u672c\u5c40\u7ed3\u679c\u6458\u8981\uff1a
{json.dumps(outcome_summary, ensure_ascii=False, indent=2)}

\u672c\u5c40\u6218\u6597\u6458\u8981\uff1a
{timeline}

\u5f53\u524d batch base\uff1a
{json.dumps(self._review_rows_for_prompt(current_base_rows), ensure_ascii=False, indent=2)}

\u672c\u5c40\u65b0\u5019\u9009 lessons\uff1a
{json.dumps(self._review_rows_for_prompt(candidate_rows), ensure_ascii=False, indent=2)}

\u8bf7\u53ea\u8f93\u51fa\u4e25\u683c JSON\uff1a
{{
  "decisions": [
    {{
      "candidate_index": 0,
      "decision": "keep|rewrite|drop|merge_into_existing",
      "merge_lesson_id": "",
      "reason": "",
      "rewritten_lesson": {{
        "type": "",
        "observation": "",
        "lesson": "",
        "tags": [],
        "phase": "",
        "target_type": "",
        "symptom": "",
        "trigger": "",
        "score_pattern": "",
        "cost_risk": ""
      }}
    }}
  ]
}}
""".strip()

    def _build_review_stage2_prompt(
        self,
        *,
        current_base_rows: List[dict],
        reviewed_candidates: List[dict],
        outcome_summary: dict,
        max_lessons: int,
    ) -> str:
        return f"""
\u4f60\u662f RED lessons \u5ba1\u6838 Agent\uff08\u7b2c 2 \u9636\u6bb5\uff09\u3002\u4f60\u8981\u57fa\u4e8e\u201c\u5f53\u524d batch base + \u5df2\u901a\u8fc7\u5ba1\u6838\u7684\u65b0\u5019\u9009\u201d\uff0c\u91cd\u5199\u51fa\u4e0b\u4e00\u5c40\u8981\u4f7f\u7528\u7684 RED \u57fa\u7ebf lessons\u3002

\u76ee\u6807\uff1a
- \u4fdd\u7559\u603b\u91cf\u7a33\u5b9a\uff0c\u4ee5 {max_lessons} \u6761\u4e3a\u4e0a\u9650\u548c\u76ee\u6807
- \u878d\u5408\u65e7\u7ecf\u9a8c\u548c\u65b0\u7ecf\u9a8c
- \u53bb\u91cd\u3001\u53bb\u51b2\u7a81\u3001\u538b\u7f29\u91cd\u590d\u4e3b\u9898
- \u7edd\u5bf9\u4fdd\u6301\u534f\u8bae\u5b89\u5168\uff0c\u4e0d\u5f97\u5f15\u5165\u4efb\u4f55\u65b0\u6307\u4ee4\u540d\u3001\u65b0\u51fd\u6570\u540d\u3001\u65b0\u89c4\u5219\u54c1\u724c\u540d

\u786c\u6027\u7ea6\u675f\uff1a
1. \u7981\u6b62\u8f93\u51fa\u4efb\u4f55\u534f\u8bae\u7ea7\u52a8\u4f5c\u5b57\u6bb5\uff0c\u53ea\u4fdd\u7559 lessons \u6240\u9700\u7684\u6218\u672f\u4e0a\u4e0b\u6587\u3001\u73b0\u8c61\u3001\u539f\u5219\u3002
2. \u4f18\u5148\u8f93\u51fa\u201c\u53ef\u91cd\u590d\u4f7f\u7528\u7684\u7b56\u7565\u539f\u5219\u201d\uff0c\u800c\u4e0d\u662f\u5355\u5c40\u53e3\u53f7\u3002
3. \u5982\u679c\u672c\u5c40\u4e0d\u5141\u8bb8\u65b0\u589e\u539f\u5219\uff0c\u90a3\u4e48\u53ea\u80fd\u4fee\u6539\u6216\u5f3a\u5316\u73b0\u6709\u4e3b\u9898\uff0c\u4e0d\u8981\u65b0\u5efa\u4e3b\u9898\u3002
4. \u5982\u679c\u9ad8\u8d28\u91cf lessons \u4e0d\u8db3 {max_lessons} \u6761\uff0c\u4f18\u5148\u4fdd\u771f\uff0c\u4e0d\u8981\u4e3a\u4e86\u51d1\u6570\u91cf\u7f16\u9020 new lesson\u3002

\u672c\u5c40\u7ed3\u679c\u6458\u8981\uff1a
{json.dumps(outcome_summary, ensure_ascii=False, indent=2)}

\u5f53\u524d batch base\uff1a
{json.dumps(self._review_rows_for_prompt(current_base_rows), ensure_ascii=False, indent=2)}

\u7ecf\u5ba1\u6838\u7684\u65b0\u5019\u9009\uff1a
{json.dumps(self._review_rows_for_prompt(reviewed_candidates), ensure_ascii=False, indent=2)}

\u8bf7\u53ea\u8f93\u51fa\u4e25\u683c JSON\uff1a
{{
  "lessons": [
    {{
      "type": "",
      "observation": "",
      "lesson": "",
      "tags": [],
      "phase": "",
      "target_type": "",
      "symptom": "",
      "trigger": "",
      "score_pattern": "",
      "cost_risk": "",
      "support_count": 1,
      "quality_score": 0.0
    }}
  ]
}}
""".strip()

    def _save_review_artifacts(self, *, stage_name: str, prompt: str, response_text: str = "", parsed=None, extra=None) -> None:
        try:
            artifact_dir = self._reflection_artifact_dir(reconstructed=False)
            prefix = f"{self.side}_{stage_name}"
            self._write_text(os.path.join(artifact_dir, f"{prefix}_prompt.txt"), prompt)
            if response_text:
                self._write_text(os.path.join(artifact_dir, f"{prefix}_response.txt"), response_text)
            if parsed is not None:
                self._write_json(os.path.join(artifact_dir, f"{prefix}_parsed.json"), parsed)
            if extra is not None:
                self._write_json(os.path.join(artifact_dir, f"{prefix}_extra.json"), extra)
        except Exception:
            logger.debug("Failed to save review artifacts for %s", stage_name, exc_info=True)

    def _apply_stage1_decisions(
        self,
        *,
        current_base_rows: List[dict],
        candidate_rows: List[dict],
        stage1_decisions: dict,
        battle_meta: dict,
    ) -> List[dict]:
        decisions = stage1_decisions.get("decisions") or []
        if not isinstance(decisions, list):
            raise RuntimeError("Review stage1 decisions must be a list")

        base_ids = {str(item.get("lesson_id", "")).strip() for item in current_base_rows if isinstance(item, dict)}
        candidates_by_index = {int(item.get("candidate_index")): item for item in candidate_rows}
        reviewed_rows: List[dict] = []
        seen_hashes = set()

        for decision in decisions:
            if not isinstance(decision, dict):
                continue
            try:
                candidate_index = int(decision.get("candidate_index"))
            except Exception:
                continue
            candidate = candidates_by_index.get(candidate_index)
            if not candidate:
                continue
            action = str(decision.get("decision", "")).strip().lower()
            if action not in REVIEW_DECISIONS:
                continue
            if action == "drop":
                continue

            candidate_payload = decision.get("rewritten_lesson") if action == "rewrite" else candidate
            sanitized = self._sanitize_review_lesson_record(
                candidate_payload if isinstance(candidate_payload, dict) and candidate_payload else candidate,
                fallback_type=str(candidate.get("type", "general")).strip() or "general",
                battle_meta=battle_meta,
                source="reflection_review_agent",
            )
            if not sanitized:
                continue

            if action == "merge_into_existing":
                merge_lesson_id = str(decision.get("merge_lesson_id", "")).strip()
                if merge_lesson_id and merge_lesson_id not in base_ids:
                    continue
                sanitized["merge_target_lesson_id"] = merge_lesson_id

            lesson_hash = str(sanitized.get("lesson_hash", "")).strip()
            if lesson_hash and lesson_hash in seen_hashes:
                continue
            seen_hashes.add(lesson_hash)
            reviewed_rows.append(sanitized)
        return reviewed_rows

    def _refine_reviewed_base(
        self,
        *,
        current_base_rows: List[dict],
        reviewed_candidates: List[dict],
        stage2_payload: dict,
        battle_meta: dict,
        max_lessons: int,
    ) -> List[dict]:
        lessons = stage2_payload.get("lessons") or []
        if not isinstance(lessons, list):
            raise RuntimeError("Review stage2 lessons must be a list")

        keyed_rows: Dict[str, dict] = {}
        for row in lessons:
            sanitized = self._sanitize_review_lesson_record(
                row,
                fallback_type=str(row.get("type", "general")).strip() or "general",
                battle_meta=battle_meta,
                source="reflection_review_agent",
            )
            if not sanitized:
                continue
            key = self._cluster_key(sanitized)
            existing = keyed_rows.get(key)
            keyed_rows[key] = self._merge_review_lesson_records(existing, sanitized) if existing else sanitized

        ordered = sorted(
            keyed_rows.values(),
            key=lambda item: (
                self._lesson_quality_score(item),
                int(self._safe_float(item.get("support_count"), 1)),
                str(item.get("created_at", "")),
            ),
            reverse=True,
        )
        return ordered[:max_lessons]

    def _run_review_pipeline(
        self,
        *,
        reflection: Dict,
        battle_meta: dict,
        outcome_summary: dict,
        timeline: str,
    ) -> List[dict]:
        current_base_rows = self._read_jsonl_rows(self.structured_store_path)
        candidate_rows = self._collect_reflection_candidates(reflection, battle_meta)
        if not candidate_rows:
            return current_base_rows

        stage1_prompt = self._build_review_stage1_prompt(
            current_base_rows=current_base_rows,
            candidate_rows=candidate_rows,
            outcome_summary=outcome_summary,
            timeline=timeline,
        )
        stage1_payload, stage1_response = self._call_json_agent(stage1_prompt, stage_name="review_stage1")
        self._save_review_artifacts(
            stage_name="review_stage1",
            prompt=stage1_prompt,
            response_text=stage1_response,
            parsed=stage1_payload,
            extra={
                "current_base": self._review_rows_for_prompt(current_base_rows),
                "candidate_rows": self._review_rows_for_prompt(candidate_rows),
                "outcome_summary": outcome_summary,
            },
        )
        reviewed_candidates = self._apply_stage1_decisions(
            current_base_rows=current_base_rows,
            candidate_rows=candidate_rows,
            stage1_decisions=stage1_payload,
            battle_meta=battle_meta,
        )

        stage2_prompt = self._build_review_stage2_prompt(
            current_base_rows=current_base_rows,
            reviewed_candidates=reviewed_candidates,
            outcome_summary=outcome_summary,
            max_lessons=self._review_capacity(),
        )
        stage2_payload, stage2_response = self._call_json_agent(stage2_prompt, stage_name="review_stage2")
        self._save_review_artifacts(
            stage_name="review_stage2",
            prompt=stage2_prompt,
            response_text=stage2_response,
            parsed=stage2_payload,
            extra={
                "reviewed_candidates": self._review_rows_for_prompt(reviewed_candidates),
                "outcome_summary": outcome_summary,
            },
        )
        refined_rows = self._refine_reviewed_base(
            current_base_rows=current_base_rows,
            reviewed_candidates=reviewed_candidates,
            stage2_payload=stage2_payload,
            battle_meta=battle_meta,
            max_lessons=self._review_capacity(),
        )
        if not refined_rows:
            refined_rows = current_base_rows
        self._save_review_artifacts(
            stage_name="review_stage2_diff",
            prompt="",
            parsed={
                "before_count": len(current_base_rows),
                "after_count": len(refined_rows),
                "before_lesson_ids": [str(item.get("lesson_id", "")) for item in current_base_rows],
                "after_lesson_ids": [str(item.get("lesson_id", "")) for item in refined_rows],
            },
        )
        return refined_rows

    def _build_reflection_prompt(self, rules_context: str, old_lessons: str, timeline: str) -> str:
        return f"""
\u4f60\u662f\u4e00\u540d\u8d1f\u8d23\u590d\u76d8 RED \u65b9\u6218\u5c40\u7684\u519b\u4e8b\u667a\u80fd\u4f53\u3002\u4f60\u7684\u4efb\u52a1\u4e0d\u662f\u590d\u8ff0\u65e5\u5fd7\uff0c\u800c\u662f\u63d0\u70bc\u53ef\u590d\u7528\u3001\u53ef\u68c0\u7d22\u3001\u53ef\u6307\u5bfc\u540e\u7eed\u51b3\u7b56\u7684\u9ad8\u8d28\u91cf lessons\u3002
\u9635\u8425\uff1a{self.side.upper()}

\u89c4\u5219\u6458\u8981\uff1a
{rules_context}

\u5386\u53f2 lessons\uff08\u907f\u514d\u91cd\u590d\uff09\uff1a
{old_lessons}

\u672c\u5c40\u6218\u6597\u6458\u8981\uff1a
{timeline}

\u8bf7\u4e25\u683c\u5b8c\u6210\u4ee5\u4e0b\u4efb\u52a1\uff1a
1. \u660e\u786e\u603b\u7ed3\u672c\u5c40\u201c\u4e3a\u4ec0\u4e48\u6700\u540e\u5f97\u8fd9\u4e2a\u5206\u201d\uff0c\u5c24\u5176\u8981\u533a\u5206\u6bc1\u4f24\u6536\u76ca\u3001\u6210\u672c\u60e9\u7f5a\u3001\u5173\u952e\u5355\u4f4d\u635f\u5931\u3001\u4fa6\u5bdf/\u5f15\u5bfc\u662f\u5426\u5230\u4f4d\u3002
2. \u751f\u6210 2 \u5230 3 \u6761 successes \u548c 2 \u5230 3 \u6761 failures\u3002
3. \u6bcf\u6761 lesson \u90fd\u5fc5\u987b\u53ef\u6267\u884c\uff0c\u4e0d\u8981\u5199\u7a7a\u6cdb\u8868\u8ff0\u3002
4. `observation` \u8d1f\u8d23\u63cf\u8ff0\u73b0\u8c61\uff0c`lesson` \u8d1f\u8d23\u5199\u7ed3\u8bba\uff0c`trigger/score_pattern/cost_risk` \u8d1f\u8d23\u8865\u8db3\u4ec0\u4e48\u60c5\u5f62\u4e0b\u8be5\u5f15\u7528\u8fd9\u6761\u7ecf\u9a8c\u3002
5. \u5c3d\u91cf\u56f4\u7ed5\u8fd9\u4e9b\u4e3b\u9898\u63d0\u70bc\uff1a
   - \u91cd\u590d\u9ad8\u6210\u672c\u6253\u51fb\u4f46\u6536\u76ca\u4e0d\u8db3
   - GuideAttack / \u4fa6\u5bdf\u65e0\u4eba\u673a\u4f7f\u7528\u4e0d\u8db3
   - \u9ad8\u4ef7\u503c\u76ee\u6807\u6253\u51fb\u65f6\u673a
   - score gain \u4e0e cost \u589e\u957f\u7684\u5173\u7cfb
   - \u4e3b\u76ee\u6807\u5df2\u8fbe\u6210\u540e\u662f\u5426\u53ca\u65f6\u6536\u624b
6. \u907f\u514d\u4e0e\u5386\u53f2 lessons \u91cd\u590d\u3002
7. \u53ea\u8f93\u51fa\u4e25\u683c JSON\uff0c\u4e0d\u8981\u9644\u52a0\u89e3\u91ca\u6587\u5b57\u3002
8. \u4e0d\u8981\u8f93\u51fa `recommended_action` \u5b57\u6bb5\uff0c\u4e5f\u4e0d\u8981\u81ea\u521b\u534f\u8bae\u7ea7\u6307\u4ee4\u540d\u3002\u82e5 lesson \u6587\u672c\u91cc\u9700\u8981\u63d0\u5230\u52a8\u4f5c\uff0c\u53ea\u80fd\u63d0\u5230 canonical \u540d\u79f0\uff1aFocusFire\u3001ShootAndScoot\u3001MoveToEngage\u3001GuideAttack\u3001ScoutArea\u3002
9. \u4e0d\u8981\u81ea\u521b\u89c4\u5219\u540d\u3001\u68c0\u67e5\u5668\u540d\u6216\u7f29\u7565\u8bcd\uff08\u4f8b\u5982 CostGate\uff0cAmmoMixRule\uff09\uff1b\u8bf7\u76f4\u63a5\u7528\u666e\u901a\u7684\u7b56\u7565\u8868\u8ff0\u3002
10. \u4f18\u5148\u201c\u4fee\u6b63/\u5f3a\u5316\u65e7 lessons\u201d\uff0c\u4e0d\u8981\u4e3a\u5355\u5c40\u5076\u7136\u73b0\u8c61\u521b\u5efa\u8fc7\u591a\u65b0\u7684\u7ec6\u788e\u89c4\u5219\u3002

JSON \u6a21\u5f0f\uff1a
{{
  "successes": [
    {{
      "observation": "...",
      "lesson": "...",
      "tags": ["..."],
      "phase": "opening|midgame|endgame",
      "target_type": "...",
      "symptom": "...",
      "trigger": "...",
      "score_pattern": "...",
      "cost_risk": "low|medium|high"
    }}
  ],
  "failures": [
    {{
      "observation": "...",
      "lesson": "...",
      "tags": ["..."],
      "phase": "opening|midgame|endgame",
      "target_type": "...",
      "symptom": "...",
      "trigger": "...",
      "score_pattern": "...",
      "cost_risk": "low|medium|high"
    }}
  ]
}}
""".strip()

    def _save_reflection_artifacts(
        self,
        *,
        event_log_path: str,
        final_score: str,
        rules_context: str,
        old_lessons: str,
        timeline: str,
        prompt: str,
        response_text: str = "",
        parsed_reflection=None,
        error: str = "",
        reconstructed: bool = False,
        output_source: str = "",
    ) -> None:
        try:
            artifact_dir = self._reflection_artifact_dir(reconstructed=reconstructed)
            prefix = f"{self.side}_{'reconstructed' if reconstructed else 'live'}"
            self._write_text(os.path.join(artifact_dir, f"{prefix}_rules_context.md"), rules_context)
            self._write_text(os.path.join(artifact_dir, f"{prefix}_historical_lessons.txt"), old_lessons)
            self._write_text(os.path.join(artifact_dir, f"{prefix}_battle_digest.txt"), timeline)
            self._write_text(os.path.join(artifact_dir, f"{prefix}_prompt.txt"), prompt)
            if response_text:
                self._write_text(os.path.join(artifact_dir, f"{prefix}_response.txt"), response_text)
            if parsed_reflection is not None:
                self._write_json(os.path.join(artifact_dir, f"{prefix}_parsed_reflection.json"), parsed_reflection)
            meta = {
                "side": self.side,
                "event_log_path": event_log_path,
                "final_score": final_score,
                "rules_context_path": self._resolve_rules_context_path(),
                "structured_store_path": self.structured_store_path,
                "knowledge_base_path": self.knowledge_base_path,
                "reconstructed": reconstructed,
                "output_source": output_source,
                "response_available": bool(response_text),
                "parsed_reflection_available": parsed_reflection is not None,
                "error": error,
            }
            self._write_json(os.path.join(artifact_dir, f"{prefix}_meta.json"), meta)
        except Exception:
            logger.debug("Failed to save reflection artifacts", exc_info=True)

    def _load_event_log(self, event_log_path: str) -> List[dict]:
        if not os.path.exists(event_log_path):
            return []
        with open(event_log_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            logger.warning("Event log is empty: %s", event_log_path)
            return []

        try:
            data = json.loads(content)
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]
            if isinstance(data, dict):
                return [data]
        except Exception:
            pass

        events = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if isinstance(item, dict):
                events.append(item)
        return events

    def reflect(self, event_log_path: str, final_score: str):
        logger.info("Running post-battle reflection. log=%s", event_log_path)
        events = self._load_event_log(event_log_path)
        timeline = self._build_reflection_digest(events, final_score)
        outcome_summary = self._build_outcome_summary(events, final_score)
        old_lessons = load_past_lessons(self.side, max_lessons=10)
        rules_context = self._load_rules_context()
        prompt = self._build_reflection_prompt(rules_context, old_lessons, timeline)
        self._save_reflection_artifacts(
            event_log_path=event_log_path,
            final_score=final_score,
            rules_context=rules_context,
            old_lessons=old_lessons,
            timeline=timeline,
            prompt=prompt,
            output_source="live_reflection",
        )

        self.llm_model.history = [{"role": "system", "content": prompt}]
        response_str = ""
        response_str = self.llm_model.chat()
        response_text = str(response_str or "").strip()
        if not response_text:
            self._save_reflection_artifacts(
                event_log_path=event_log_path,
                final_score=final_score,
                rules_context=rules_context,
                old_lessons=old_lessons,
                timeline=timeline,
                prompt=prompt,
                response_text=response_text,
                error="Reflection model returned empty response",
                output_source="live_reflection",
            )
            raise RuntimeError("Reflection model returned empty response")
        if response_text.startswith("**ERROR**"):
            self._save_reflection_artifacts(
                event_log_path=event_log_path,
                final_score=final_score,
                rules_context=rules_context,
                old_lessons=old_lessons,
                timeline=timeline,
                prompt=prompt,
                response_text=response_text,
                error=response_text[:240],
                output_source="live_reflection",
            )
            raise RuntimeError(f"Reflection model returned error sentinel: {response_text[:240]}")
        try:
            reflection = json.loads(response_text)
        except json.JSONDecodeError as exc:
            logger.error("Reflection JSON parse failed: %s", response_text)
            self._save_reflection_artifacts(
                event_log_path=event_log_path,
                final_score=final_score,
                rules_context=rules_context,
                old_lessons=old_lessons,
                timeline=timeline,
                prompt=prompt,
                response_text=response_text,
                error=f"Reflection JSON parse failed: {response_text[:240]}",
                output_source="live_reflection",
            )
            raise RuntimeError(f"Reflection JSON parse failed: {response_text[:240]}") from exc
        if not isinstance(reflection, dict):
            self._save_reflection_artifacts(
                event_log_path=event_log_path,
                final_score=final_score,
                rules_context=rules_context,
                old_lessons=old_lessons,
                timeline=timeline,
                prompt=prompt,
                response_text=response_text,
                error=f"Reflection returned non-dict JSON: {type(reflection).__name__}",
                output_source="live_reflection",
            )
            raise RuntimeError(f"Reflection returned non-dict JSON: {type(reflection).__name__}")
        self._save_reflection_artifacts(
            event_log_path=event_log_path,
            final_score=final_score,
            rules_context=rules_context,
            old_lessons=old_lessons,
            timeline=timeline,
            prompt=prompt,
            response_text=response_text,
            parsed_reflection=reflection,
            output_source="live_reflection",
        )
        self._save_lessons(
            reflection,
            event_log_path,
            final_score,
            timeline=timeline,
            outcome_summary=outcome_summary,
        )

    def _save_lessons(
        self,
        reflection: Dict,
        event_log_path: str,
        final_score: str,
        *,
        timeline: str,
        outcome_summary: dict,
    ):
        if not self.auto_writeback:
            logger.info("Structured writeback disabled by RED_LTM_AUTO_WRITEBACK=0")
            return

        os.makedirs(os.path.dirname(self.structured_store_path), exist_ok=True)

        battle_meta = {
            "event_log_path": event_log_path,
            "final_score": final_score,
            "side": self.side,
            "written_at": datetime.utcnow().isoformat() + "Z",
            "source_run_id": str(os.path.basename(str(os.getenv("RUN_OUTPUT_ROOT", ""))).strip() or ""),
            "source_batch_id": str(os.path.basename(os.path.dirname(str(os.getenv("RUN_OUTPUT_ROOT", "")).strip())) or ""),
            "allow_new_principles": bool(outcome_summary.get("allow_new_principles", True)),
        }
        review_mode = self._review_pipeline_mode()
        if review_mode == "dual_stage_v1":
            try:
                refined_rows = self._run_review_pipeline(
                    reflection=reflection,
                    battle_meta=battle_meta,
                    outcome_summary=outcome_summary,
                    timeline=timeline,
                )
                self._write_jsonl_rows(self.structured_store_path, refined_rows)
                logger.info(
                    "Saved reviewed RED batch base to %s. review_mode=%s retained=%s allow_new=%s",
                    self.structured_store_path,
                    review_mode,
                    len(refined_rows),
                    outcome_summary.get("allow_new_principles", True),
                )
                return
            except Exception as exc:
                logger.error("Review pipeline failed: %s", exc, exc_info=True)
                raise

        existing_hashes = self._load_existing_lesson_hashes(self.structured_store_path)
        to_write: List[dict] = []
        for lesson_type, key in (("success", "successes"), ("failure", "failures")):
            for item in reflection.get(key, []) or []:
                self._append_reflection_item(
                    item=item,
                    lesson_type=lesson_type,
                    battle_meta=battle_meta,
                    existing_hashes=existing_hashes,
                    out=to_write,
                )

        if not to_write:
            logger.info("No new structured lessons to append after dedupe.")
            return

        try:
            with open(self.structured_store_path, "a", encoding="utf-8") as f:
                for row in to_write:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            logger.info("Saved %s structured lessons to %s", len(to_write), self.structured_store_path)
        except Exception as e:
            logger.error("Failed to save structured lessons: %s", e, exc_info=True)
            raise

    def _append_reflection_item(
        self,
        item: dict,
        lesson_type: str,
        battle_meta: dict,
        existing_hashes: set,
        out: List[dict],
    ):
        if not isinstance(item, dict):
            return
        observation = self._sanitize_lesson_text(item.get("observation", ""))
        lesson = self._sanitize_lesson_text(item.get("lesson", ""))
        tags = item.get("tags", [])
        if not lesson:
            return
        if not isinstance(tags, list):
            tags = []
        tags = [self._sanitize_lesson_text(t) for t in tags if self._sanitize_lesson_text(t)]
        phase = str(item.get("phase", "")).strip()
        target_type = self._sanitize_lesson_text(item.get("target_type", ""))
        symptom = self._sanitize_lesson_text(item.get("symptom", ""))
        trigger = self._sanitize_lesson_text(item.get("trigger", ""))
        score_pattern = self._sanitize_lesson_text(item.get("score_pattern", ""))
        cost_risk = str(item.get("cost_risk", "")).strip()

        normalized = self._normalize_lesson(lesson)
        lesson_hash = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
        if lesson_hash in existing_hashes:
            return
        existing_hashes.add(lesson_hash)

        lesson_record = build_structured_lesson(
            lesson_type=lesson_type,
            observation=observation,
            lesson=lesson,
            tags=tags,
            battle_meta=battle_meta,
            source="reflection_agent",
            phase=phase,
            target_type=target_type,
            symptom=symptom,
            trigger=trigger,
            score_pattern=score_pattern,
            cost_risk=cost_risk,
        )
        lesson_record["normalized_lesson"] = normalized
        lesson_record["lesson_hash"] = lesson_hash
        lesson_record["support_count"] = max(1, int(self._safe_float(item.get("support_count"), 1)))
        lesson_record["quality_score"] = max(0.0, self._safe_float(item.get("quality_score"), 0.0))
        lesson_record["protocol_safe"] = True
        lesson_record["source_run_ids"] = [str(battle_meta.get("source_run_id", "")).strip()] if str(battle_meta.get("source_run_id", "")).strip() else []
        lesson_record["source_batch_ids"] = [str(battle_meta.get("source_batch_id", "")).strip()] if str(battle_meta.get("source_batch_id", "")).strip() else []
        out.append(lesson_record)

    def _load_existing_lesson_hashes(self, path: str) -> set:
        hashes = set()
        if not os.path.exists(path):
            return hashes
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
                    lesson = str(data.get("lesson", "")).strip()
                    normalized = str(data.get("normalized_lesson", "")).strip() or self._normalize_lesson(lesson)
                    lesson_hash = str(data.get("lesson_hash", "")).strip()
                    if not lesson_hash and normalized:
                        lesson_hash = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
                    if lesson_hash:
                        hashes.add(lesson_hash)
        except Exception as e:
            logger.warning("Failed loading existing hashes from %s: %s", path, e)
        return hashes

    def _normalize_lesson(self, text: str) -> str:
        t = str(text or "").lower().strip()
        t = re.sub(r"\s+", " ", t)
        return t

    def _sanitize_lesson_text(self, value) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        for alias, canonical in LESSON_TEXT_ALIAS_MAP.items():
            text = re.sub(rf"\b{re.escape(alias)}\b", canonical, text, flags=re.IGNORECASE)
        for branded, generic in LESSON_RULE_BRANDING_MAP.items():
            text = re.sub(rf"\b{re.escape(branded)}\b", generic, text)
        text = text.replace("`", "")
        text = re.sub(r"\s+", " ", text).strip()
        return text
