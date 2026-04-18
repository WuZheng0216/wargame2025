import logging
import os
from typing import Dict, List, Optional

from state_access import get_detected_target_ids

logger = logging.getLogger(__name__)

ATTACK_COST_MAP = {
    "HighCostAttackMissile": 8.0,
    "LowCostAttackMissile": 3.0,
    "ShipToGround_CruiseMissile": 10.0,
}


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        return float(raw)
    except Exception:
        logger.warning("Invalid %s=%s, fallback to %s", name, raw, default)
        return default


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        return int(raw)
    except Exception:
        logger.warning("Invalid %s=%s, fallback to %s", name, raw, default)
        return default


def _read_bool_env(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return raw not in {"0", "false", "no", "off", ""}


class EngagementMemoryManager:
    def __init__(self):
        self.window_seconds = _read_int_env("RED_ENGAGEMENT_MEMORY_WINDOW_SECONDS", 240)
        self.summary_max_targets = _read_int_env("RED_ENGAGEMENT_SUMMARY_MAX_TARGETS", 4)
        self.memory_max_targets = _read_int_env("RED_ENGAGEMENT_MEMORY_MAX_TARGETS", 6)
        self.pending_bda_margin_seconds = _read_float_env("RED_PENDING_BDA_MARGIN_SECONDS", 30.0)
        self.repeat_high_cost_lookback_seconds = _read_float_env("RED_REPEAT_HIGH_COST_LOOKBACK_SECONDS", 120.0)
        self.repeat_high_cost_min_launches = _read_int_env("RED_REPEAT_HIGH_COST_MIN_LAUNCHES", 3)
        self.repeat_high_cost_min_estimated_cost = _read_float_env("RED_REPEAT_HIGH_COST_MIN_ESTIMATED_COST", 24.0)
        self.repeat_high_cost_require_score_gain = _read_bool_env("RED_REPEAT_HIGH_COST_REQUIRE_SCORE_GAIN", True)
        self.soft_constraint_enabled = _read_bool_env("RED_ENGAGEMENT_SOFT_CONSTRAINT", True)
        self.track_stale_threshold_seconds = _read_int_env("RED_TRACK_STALE_THRESHOLD_SECONDS", 90)
        self.guidance_recent_window_seconds = _read_float_env("RED_GUIDANCE_RECENT_WINDOW_SECONDS", 120.0)
        self.records: Dict[str, dict] = {}
        self._last_score: Optional[float] = None

    def update(self, state, sim_time: int, side_score: float, score_breakdown: Optional[dict] = None) -> None:
        detected_targets = set(get_detected_target_ids(state))
        self._last_score = float(side_score)
        expired = []
        for target_id, record in self.records.items():
            if sim_time - float(record.get("last_touch_time", sim_time)) > float(self.window_seconds):
                expired.append(target_id)
                continue
            score_at_last_engage = float(record.get("score_at_last_engage", side_score))
            record["score_gain_since_engage"] = max(0.0, float(side_score) - score_at_last_engage)
            record["target_still_visible"] = target_id in detected_targets
            if not record["target_still_visible"] and float(record.get("last_engage_time", -1)) >= 0:
                record["target_disappeared_after_engage"] = True
        for target_id in expired:
            self.records.pop(target_id, None)

    def record_submitted_engagements(
        self,
        engagement_events: List[dict],
        sim_time: int,
        side_score: float,
        score_breakdown: Optional[dict] = None,
    ) -> None:
        if not engagement_events:
            return
        for event in engagement_events:
            if not isinstance(event, dict):
                continue
            target_id = str(event.get("target_id") or "").strip()
            if not target_id:
                continue
            target_type = str(event.get("target_type") or "").strip()
            action_type = str(event.get("action_type") or "").strip()
            unit_ids = sorted({str(unit_id) for unit_id in event.get("unit_ids", []) if str(unit_id).strip()})
            guide_unit_ids = sorted({str(unit_id) for unit_id in event.get("guide_unit_ids", []) if str(unit_id).strip()})
            launch_actions = [
                launch
                for launch in event.get("launch_actions", []) or []
                if isinstance(launch, dict)
            ]
            high_cost_launch_count = int(event.get("high_cost_launch_count") or 0)
            low_cost_launch_count = int(event.get("low_cost_launch_count") or 0)
            estimated_attack_cost = float(event.get("estimated_attack_cost") or 0.0)
            estimated_impact_until = float(event.get("estimated_impact_until") or 0.0)
            pending_bda_until = (
                estimated_impact_until + float(self.pending_bda_margin_seconds)
                if estimated_impact_until > 0
                else 0.0
            )
            record = self.records.setdefault(
                target_id,
                {
                    "target_id": target_id,
                    "target_type": target_type,
                    "events": [],
                    "guide_history": [],
                    "unit_ids": [],
                    "guide_unit_ids": [],
                    "last_touch_time": float(sim_time),
                    "last_engage_time": -1.0,
                    "last_action_type": "",
                    "score_at_last_engage": float(side_score),
                    "score_gain_since_engage": 0.0,
                    "estimated_impact_until": 0.0,
                    "pending_bda_until": 0.0,
                    "target_still_visible": True,
                    "target_disappeared_after_engage": False,
                    "guidance_used": False,
                    "scout_support_recommended": False,
                    "last_track_staleness_sec": 0,
                    "last_attack_window": "",
                },
            )
            record["target_type"] = target_type or record.get("target_type", "")
            record["last_touch_time"] = float(sim_time)
            record["unit_ids"] = sorted(set(record.get("unit_ids", [])) | set(unit_ids))
            record["guide_unit_ids"] = sorted(set(record.get("guide_unit_ids", [])) | set(guide_unit_ids))
            record["last_track_staleness_sec"] = int(event.get("track_staleness_sec") or 0)
            record["last_attack_window"] = str(event.get("attack_window") or record.get("last_attack_window") or "")
            if action_type == "GuideAttack":
                record["guidance_used"] = True
                record["last_guide_time"] = float(sim_time)
                record["guide_history"].append(
                    {
                        "sim_time": float(sim_time),
                        "unit_ids": guide_unit_ids or unit_ids,
                    }
                )
                continue

            record["last_engage_time"] = float(sim_time)
            record["last_action_type"] = action_type
            record["score_at_last_engage"] = float(side_score)
            record["score_gain_since_engage"] = 0.0
            record["estimated_impact_until"] = max(float(record.get("estimated_impact_until") or 0.0), estimated_impact_until)
            record["pending_bda_until"] = max(float(record.get("pending_bda_until") or 0.0), pending_bda_until)
            record["target_disappeared_after_engage"] = False
            if event.get("guidance_used"):
                record["guidance_used"] = True
                record["last_guide_time"] = float(sim_time)
            record["events"].append(
                {
                    "sim_time": float(sim_time),
                    "action_type": action_type,
                    "unit_ids": unit_ids,
                    "guide_unit_ids": guide_unit_ids,
                    "high_cost_launch_count": high_cost_launch_count,
                    "low_cost_launch_count": low_cost_launch_count,
                    "estimated_attack_cost": estimated_attack_cost,
                    "estimated_impact_until": estimated_impact_until,
                    "pending_bda_until": pending_bda_until,
                    "guidance_used": bool(event.get("guidance_used")),
                    "track_staleness_sec": int(event.get("track_staleness_sec") or 0),
                    "attack_window": str(event.get("attack_window") or ""),
                    "launch_actions": launch_actions,
                }
            )
        self._prune(sim_time)

    def build_prompt_payload(self, sim_time: int, target_table: Optional[List[dict]] = None) -> dict:
        target_index = {
            str(item.get("target_id")): item
            for item in (target_table or [])
            if isinstance(item, dict) and str(item.get("target_id", "")).strip()
        }
        entries = []
        total_recent_attack_cost = 0.0
        for target_id, record in list(self.records.items()):
            entry = self._summarize_record(record, sim_time, target_index.get(target_id))
            if entry is None:
                continue
            entries.append(entry)
            total_recent_attack_cost += float(entry.get("recent_attack_cost") or 0.0)
        entries.sort(
            key=lambda item: (
                0 if item.get("repeat_high_cost_risk") == "high" else 1 if item.get("repeat_high_cost_risk") == "watch" else 2,
                0 if item.get("pending_bda") else 1,
                -float(item.get("recent_attack_cost") or 0.0),
                -float(item.get("last_engage_time") or -1.0),
            )
        )
        memory = entries[: max(0, self.memory_max_targets)]
        summary_lines = []
        for item in entries[: max(0, self.summary_max_targets)]:
            summary_lines.append(
                "- {target_id}: high={recent_high_cost_launches} low={recent_low_cost_launches} "
                "cost={recent_attack_cost:.1f} gain={recent_score_gain:.1f} "
                "visible={visible} pending_bda={pending_bda} risk={risk} rec={recommendation}".format(
                    target_id=item.get("target_id"),
                    recent_high_cost_launches=int(item.get("recent_high_cost_launches") or 0),
                    recent_low_cost_launches=int(item.get("recent_low_cost_launches") or 0),
                    recent_attack_cost=float(item.get("recent_attack_cost") or 0.0),
                    recent_score_gain=float(item.get("recent_score_gain") or 0.0),
                    visible="yes" if item.get("target_still_visible") else "no",
                    pending_bda="yes" if item.get("pending_bda") else "no",
                    risk=item.get("repeat_high_cost_risk"),
                    recommendation=item.get("engagement_recommendation"),
                )
            )
        summary = "\n".join(summary_lines) if summary_lines else "No recent engagement memory."
        return {
            "engagement_summary": summary,
            "engagement_memory": memory,
            "repeat_high_cost_targets": [item["target_id"] for item in entries if item.get("repeat_high_cost_risk") == "high"],
            "pending_bda_targets": [item["target_id"] for item in entries if item.get("pending_bda")],
            "recent_attack_cost": round(total_recent_attack_cost, 2),
        }

    def _summarize_record(self, record: dict, sim_time: int, target_meta: Optional[dict]) -> Optional[dict]:
        recent_events = [
            event
            for event in record.get("events", [])
            if sim_time - float(event.get("sim_time") or 0.0) <= float(self.repeat_high_cost_lookback_seconds)
        ]
        recent_guides = [
            event
            for event in record.get("guide_history", [])
            if sim_time - float(event.get("sim_time") or 0.0) <= float(self.guidance_recent_window_seconds)
        ]
        recent_high_cost_launches = sum(int(event.get("high_cost_launch_count") or 0) for event in recent_events)
        recent_low_cost_launches = sum(int(event.get("low_cost_launch_count") or 0) for event in recent_events)
        recent_attack_cost = sum(float(event.get("estimated_attack_cost") or 0.0) for event in recent_events)
        recent_salvo_count = sum(
            1
            for event in recent_events
            if int(event.get("high_cost_launch_count") or 0) + int(event.get("low_cost_launch_count") or 0) > 0
        )
        recent_engage_count = len(recent_events)
        pending_bda_until = float(record.get("pending_bda_until") or 0.0)
        pending_bda = pending_bda_until > 0 and float(sim_time) <= pending_bda_until
        target_still_visible = bool(record.get("target_still_visible"))
        last_track_staleness_sec = int(record.get("last_track_staleness_sec") or 0)
        attack_window = str(record.get("last_attack_window") or "")
        target_type = str(record.get("target_type") or "")
        if isinstance(target_meta, dict):
            target_type = str(target_meta.get("target_type") or target_type)
            attack_window = str(target_meta.get("attack_window") or attack_window)
            try:
                last_track_staleness_sec = int(target_meta.get("track_staleness_sec") or last_track_staleness_sec)
            except Exception:
                pass
            target_still_visible = True
        guidance_used_recently = bool(recent_guides) or (
            "last_guide_time" in record
            and float(sim_time) - float(record.get("last_guide_time") or 0.0) <= float(self.guidance_recent_window_seconds)
        )
        scout_support_recommended = last_track_staleness_sec >= self.track_stale_threshold_seconds or attack_window == "observe_only"
        recent_score_gain = float(record.get("score_gain_since_engage") or 0.0)
        repeat_high_cost_risk = "none"
        if self.soft_constraint_enabled:
            risk_threshold_hit = (
                recent_high_cost_launches >= self.repeat_high_cost_min_launches
                or recent_attack_cost >= self.repeat_high_cost_min_estimated_cost
            )
            gain_ok = recent_score_gain > 0.0 if self.repeat_high_cost_require_score_gain else True
            just_past_bda = pending_bda_until > 0 and float(sim_time) > pending_bda_until
            if risk_threshold_hit and not gain_ok and (target_still_visible or just_past_bda):
                repeat_high_cost_risk = "high"
            elif risk_threshold_hit or (recent_high_cost_launches > 0 and recent_score_gain <= 0):
                repeat_high_cost_risk = "watch"

        engagement_recommendation = "continue_if_priority_high"
        if pending_bda and recent_score_gain <= 0:
            engagement_recommendation = "hold_bda"
        elif repeat_high_cost_risk == "high" and not guidance_used_recently and target_type.endswith("_Surface"):
            engagement_recommendation = "guide_then_reassess"
        elif repeat_high_cost_risk == "high" and scout_support_recommended:
            engagement_recommendation = "refresh_track_before_salvo"
        elif repeat_high_cost_risk == "high":
            engagement_recommendation = "switch_target_or_reassess"
        elif attack_window == "observe_only":
            engagement_recommendation = "move_or_scout_before_attack"
        elif not guidance_used_recently and target_type.endswith("_Surface") and attack_window in {"fire_now", "move_then_fire"}:
            engagement_recommendation = "consider_guidance"

        if recent_engage_count <= 0 and not recent_guides and sim_time - float(record.get("last_touch_time") or sim_time) > self.window_seconds:
            return None
        return {
            "target_id": record.get("target_id"),
            "target_type": target_type,
            "last_engage_time": record.get("last_engage_time"),
            "last_action_type": record.get("last_action_type"),
            "unit_ids": list(record.get("unit_ids", [])),
            "guide_unit_ids": list(record.get("guide_unit_ids", [])),
            "recent_high_cost_launches": recent_high_cost_launches,
            "recent_low_cost_launches": recent_low_cost_launches,
            "recent_attack_cost": round(recent_attack_cost, 2),
            "recent_score_gain": round(recent_score_gain, 2),
            "guidance_used_recently": guidance_used_recently,
            "target_still_visible": target_still_visible,
            "target_disappeared_after_engage": bool(record.get("target_disappeared_after_engage")),
            "pending_bda": pending_bda,
            "pending_bda_until": pending_bda_until,
            "estimated_impact_until": float(record.get("estimated_impact_until") or 0.0),
            "recent_engage_count": recent_engage_count,
            "recent_salvo_count": recent_salvo_count,
            "repeat_high_cost_risk": repeat_high_cost_risk,
            "engagement_recommendation": engagement_recommendation,
            "scout_support_recommended": scout_support_recommended,
            "track_staleness_sec": last_track_staleness_sec,
            "attack_window": attack_window,
        }

    def _prune(self, sim_time: int) -> None:
        expired = []
        for target_id, record in self.records.items():
            recent_events = [
                event
                for event in record.get("events", [])
                if sim_time - float(event.get("sim_time") or 0.0) <= float(self.window_seconds)
            ]
            recent_guides = [
                event
                for event in record.get("guide_history", [])
                if sim_time - float(event.get("sim_time") or 0.0) <= float(self.window_seconds)
            ]
            record["events"] = recent_events
            record["guide_history"] = recent_guides
            if not recent_events and not recent_guides and sim_time - float(record.get("last_touch_time", sim_time)) > float(self.window_seconds):
                expired.append(target_id)
        for target_id in expired:
            self.records.pop(target_id, None)
