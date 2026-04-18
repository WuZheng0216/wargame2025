import logging
import os
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from state_access import friendly_unit_count, get_detected_target_ids, incoming_threat_count

logger = logging.getLogger(__name__)


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        return int(raw)
    except Exception:
        logger.warning("Invalid %s=%s, fallback to %s", name, raw, default)
        return default


@dataclass
class MemoryFrame:
    sim_time: int
    friendly_count: int = 0
    detected_target_ids: List[str] = field(default_factory=list)
    detected_target_count: int = 0
    threat_count: int = 0
    key_events: List[dict] = field(default_factory=list)
    derived_events: List[dict] = field(default_factory=list)


class ShortTermMemoryManager:
    def __init__(self):
        self.window_seconds = _read_int_env("RED_STM_WINDOW_SECONDS", 180)
        self.max_events = _read_int_env("RED_STM_MAX_EVENTS", 50)
        self.frames: Deque[MemoryFrame] = deque()

    def record(self, state, faction_name: str):
        sim_time = self._safe_simtime(state)
        if sim_time is None:
            return

        detected_target_ids = self._safe_find_all_detect_targets(state)
        frame = MemoryFrame(
            sim_time=sim_time,
            friendly_count=self._count_friendlies(state, faction_name),
            detected_target_ids=detected_target_ids,
            detected_target_count=len(detected_target_ids),
            threat_count=self._count_threats(state),
            key_events=self._extract_key_events(state, sim_time),
        )
        previous = self.frames[-1] if self.frames else None
        frame.derived_events = self._derive_frame_events(previous, frame)
        self.frames.append(frame)
        self._evict_old(sim_time)

    def build_memory_packet(self, now: int, current_snapshot: str) -> dict:
        relevant = [f for f in self.frames if f.sim_time >= now - self.window_seconds]
        if not relevant:
            return {
                "current_snapshot": current_snapshot,
                "window_summary": "No short-term memory frames yet.",
                "event_timeline": [],
                "recent_failures": [],
                "memory_window": {"start_sim_time": now, "end_sim_time": now},
            }

        first = relevant[0]
        last = relevant[-1]

        first_targets = set(first.detected_target_ids)
        last_targets = set(last.detected_target_ids)
        new_targets = sorted(last_targets - first_targets)
        lost_targets = sorted(first_targets - last_targets)

        all_events: List[dict] = []
        for frame in relevant:
            all_events.extend(frame.key_events)
            all_events.extend(frame.derived_events)
        all_events = sorted(all_events, key=lambda e: int(e.get("sim_time", 0)))
        if len(all_events) > self.max_events:
            all_events = all_events[-self.max_events :]

        event_counter = Counter()
        for e in all_events:
            event_counter[e.get("name", "UNKNOWN")] += 1

        recent_failures = [
            e
            for e in all_events
            if "FAIL" in str(e.get("name", "")).upper() or "ERROR" in str(e.get("name", "")).upper()
        ][-10:]
        target_churn = len(new_targets) + len(lost_targets)
        threat_spike_count = sum(1 for e in all_events if str(e.get("name", "")).upper() == "THREAT_COUNT_SPIKE")

        summary_lines = [
            f"Window: [{first.sim_time}, {last.sim_time}] ({last.sim_time - first.sim_time}s)",
            f"Friendly trend: {first.friendly_count} -> {last.friendly_count}",
            f"Threat trend: {first.threat_count} -> {last.threat_count}",
            f"Targets trend: {len(first_targets)} -> {len(last_targets)}",
            f"New targets: {', '.join(new_targets[:8]) if new_targets else 'None'}",
            f"Lost targets: {', '.join(lost_targets[:8]) if lost_targets else 'None'}",
            f"Target churn: {target_churn}",
            f"Threat spike count: {threat_spike_count}",
        ]
        if event_counter:
            top_events = ", ".join([f"{name}:{count}" for name, count in event_counter.most_common(8)])
            summary_lines.append(f"Key event counts: {top_events}")
        else:
            summary_lines.append("Key event counts: None")

        return {
            "current_snapshot": current_snapshot,
            "window_summary": "\n".join(summary_lines),
            "event_timeline": all_events,
            "recent_failures": recent_failures,
            "memory_window": {"start_sim_time": first.sim_time, "end_sim_time": last.sim_time},
        }

    # ------------------------------
    # Internal helpers
    # ------------------------------
    def _safe_simtime(self, state) -> Optional[int]:
        try:
            return int(state.simtime())
        except Exception:
            return None

    def _evict_old(self, now: int):
        min_keep = now - self.window_seconds
        while self.frames and self.frames[0].sim_time < min_keep:
            self.frames.popleft()

    def _count_friendlies(self, state, faction_name: str) -> int:
        return friendly_unit_count(state)

    def _safe_find_all_detect_targets(self, state) -> List[str]:
        return get_detected_target_ids(state)

    def _count_threats(self, state) -> int:
        return incoming_threat_count(state)

    def _derive_frame_events(self, previous: Optional[MemoryFrame], current: MemoryFrame) -> List[dict]:
        if previous is None:
            return []

        events: List[dict] = []
        previous_targets = set(previous.detected_target_ids)
        current_targets = set(current.detected_target_ids)

        for target_id in sorted(current_targets - previous_targets)[:6]:
            events.append(
                {
                    "sim_time": current.sim_time,
                    "name": "TARGET_APPEARED",
                    "unit_id": None,
                    "target_id": target_id,
                    "raw": {"target_id": target_id},
                }
            )

        for target_id in sorted(previous_targets - current_targets)[:6]:
            events.append(
                {
                    "sim_time": current.sim_time,
                    "name": "TARGET_DISAPPEARED",
                    "unit_id": None,
                    "target_id": target_id,
                    "raw": {"target_id": target_id},
                }
            )

        if current.friendly_count < previous.friendly_count:
            events.append(
                {
                    "sim_time": current.sim_time,
                    "name": "FRIENDLY_COUNT_DROP",
                    "unit_id": None,
                    "target_id": None,
                    "raw": {
                        "previous_friendly_count": previous.friendly_count,
                        "current_friendly_count": current.friendly_count,
                    },
                }
            )

        if current.threat_count > previous.threat_count:
            events.append(
                {
                    "sim_time": current.sim_time,
                    "name": "THREAT_COUNT_SPIKE",
                    "unit_id": None,
                    "target_id": None,
                    "raw": {
                        "previous_threat_count": previous.threat_count,
                        "current_threat_count": current.threat_count,
                    },
                }
            )

        return events

    def _extract_key_events(self, state, sim_time: int) -> List[dict]:
        events = []
        if hasattr(state, "recent_events"):
            try:
                raw = state.recent_events() or []
                for e in raw[-20:]:
                    if not isinstance(e, dict):
                        continue
                    name = str(e.get("name", "UNKNOWN"))
                    if self._is_key_event(name):
                        events.append(
                            {
                                "sim_time": sim_time,
                                "name": name,
                                "unit_id": e.get("unit_id"),
                                "target_id": e.get("target_id"),
                                "raw": e,
                            }
                        )
            except Exception:
                pass
        return events

    def _is_key_event(self, name: str) -> bool:
        n = name.upper()
        keys = [
            "DETECTED",
            "LOST",
            "HIT",
            "DESTROY",
            "FIRE_FAILED",
            "MISSILE",
            "INTERCEPT",
            "JAM",
            "FAIL",
            "ERROR",
        ]
        return any(k in n for k in keys)
