import logging
import os
from collections import Counter
from typing import List, Tuple

from state_access import get_detected_target_ids, get_target_position

logger = logging.getLogger(__name__)


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        return int(raw)
    except Exception:
        logger.warning("Invalid %s=%s, fallback to %s", name, raw, default)
        return default


class ActionFreshnessGuard:
    def __init__(self):
        self.max_staleness = _read_int_env("RED_ACTION_MAX_STALENESS", 90)
        self.force_replan_enabled = _read_int_env("RED_STALE_FORCE_REPLAN", 1) == 1

    def filter_actions(self, actions_json: List[dict], state, planned_at: int) -> Tuple[List[dict], dict]:
        now = self._safe_simtime(state)
        reason_counter = Counter()

        if now is None:
            return [], {"planned_at": planned_at, "now": None, "drop_count": len(actions_json), "drop_reason_dist": {"invalid_state": len(actions_json)}, "force_replan": False}

        if planned_at is None:
            planned_at = now

        if now - planned_at > self.max_staleness:
            reason_counter["time_stale"] += len(actions_json)
            report = {
                "planned_at": planned_at,
                "now": now,
                "drop_count": len(actions_json),
                "drop_reason_dist": dict(reason_counter),
                "force_replan": self.force_replan_enabled and len(actions_json) > 0,
            }
            return [], report

        kept = []
        for action in actions_json or []:
            reasons = self._validate_action(action, state)
            if reasons:
                for r in reasons:
                    reason_counter[r] += 1
                continue
            kept.append(action)

        drop_count = len(actions_json or []) - len(kept)
        force_replan = self.force_replan_enabled and drop_count > 0 and len(kept) == 0
        report = {
            "planned_at": planned_at,
            "now": now,
            "drop_count": drop_count,
            "drop_reason_dist": dict(reason_counter),
            "force_replan": force_replan,
        }
        return kept, report

    # ------------------------------
    # Internal helpers
    # ------------------------------
    def _safe_simtime(self, state):
        try:
            return int(state.simtime())
        except Exception:
            return None

    def _validate_action(self, action: dict, state) -> List[str]:
        if not isinstance(action, dict):
            return ["invalid_action_object"]

        reasons = []
        unit_ids = self._extract_unit_ids(action)
        if unit_ids:
            if not any(self._unit_exists(state, uid) for uid in unit_ids):
                reasons.append("unit_invalid")

        target_id = self._extract_target_id(action)
        if target_id:
            if not self._target_exists(state, target_id):
                reasons.append("target_invalid")

        # Lightweight context drift: action requires target lock but no target picture at all.
        action_type = str(action.get("Type", "")).lower()
        if action_type in {"focusfire", "movetoengage", "guideattack", "shootandscoot"}:
            if self._detected_target_count(state) == 0:
                reasons.append("context_drift")

        return reasons

    def _extract_unit_ids(self, action: dict) -> List[str]:
        out = []
        if isinstance(action.get("UnitIds"), list):
            out.extend([str(x) for x in action.get("UnitIds") if x])
        for key in ("UnitId", "Id"):
            v = action.get(key)
            if isinstance(v, str) and v:
                out.append(v)
        return out

    def _extract_target_id(self, action: dict):
        for key in ("Target_Id", "target_id", "target_ID", "TargetID"):
            v = action.get(key)
            if isinstance(v, str) and v:
                return v
        return None

    def _unit_exists(self, state, unit_id: str) -> bool:
        if not unit_id:
            return False
        try:
            if state.get_unit_state(unit_id) is not None:
                return True
        except Exception:
            pass
        try:
            found = state.find_units(id_contain_str=unit_id)
            if found:
                return True
        except Exception:
            pass
        return False

    def _target_exists(self, state, target_id: str) -> bool:
        if not target_id:
            return False
        if get_target_position(state, target_id) is not None:
            return True
        targets = get_detected_target_ids(state)
        if target_id in targets:
            return True
        try:
            if state.get_unit_state(target_id) is not None:
                return True
        except Exception:
            pass
        return False

    def _detected_target_count(self, state) -> int:
        return len(get_detected_target_ids(state))
