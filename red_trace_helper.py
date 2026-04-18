import json
import os
import threading
from collections import Counter
from datetime import datetime
from time import perf_counter
from typing import Any, Dict, List, Optional

from runtime_paths import ensure_output_dir, trace_live_enabled

_GLOBAL_TRACE_LOGGER = None


def set_global_trace_logger(logger) -> None:
    global _GLOBAL_TRACE_LOGGER
    _GLOBAL_TRACE_LOGGER = logger


def get_global_trace_logger():
    return _GLOBAL_TRACE_LOGGER


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").split())


def truncate_text(text: Any, limit: int = 240) -> str:
    normalized = normalize_text(text)
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(limit - 3, 0)] + "..."


def summarize_timeline(events: List[dict], max_items: int = 5) -> str:
    if not events:
        return "none"
    lines = []
    for event in events[-max_items:]:
        lines.append(
            f"t={event.get('sim_time')} {event.get('name')} unit={event.get('unit_id')} target={event.get('target_id')}"
        )
    return " | ".join(lines)


def summarize_lessons(lessons: List[dict], max_items: int = 5) -> str:
    if not lessons:
        return "none"
    lines = []
    for item in lessons[:max_items]:
        score = float(item.get("hybrid_score", 0.0))
        lesson_type = item.get("type", "general")
        lesson_id = str(item.get("lesson_id", ""))[:8] or "n/a"
        action = truncate_text(item.get("recommended_action", ""), 48) or "-"
        lesson = truncate_text(item.get("lesson", ""), 120)
        lines.append(f"id={lesson_id} [{lesson_type}] score={score:.3f} action={action} {lesson}")
    return " | ".join(lines)


def summarize_json_actions(actions_json: List[dict], max_items: int = 5) -> str:
    if not actions_json:
        return "none"
    counter = Counter()
    examples = []
    for action in actions_json:
        if not isinstance(action, dict):
            counter["invalid"] += 1
            continue
        action_type = str(action.get("Type", "UNKNOWN"))
        counter[action_type] += 1
        if len(examples) < max_items:
            target_id = action.get("Target_Id") or action.get("target_id")
            unit_ids = action.get("UnitIds") or action.get("UnitId") or action.get("Id")
            examples.append(f"{action_type}(target={target_id}, units={unit_ids})")
    counts = ", ".join(f"{k}:{v}" for k, v in sorted(counter.items()))
    return f"counts=[{counts}] examples=[{' | '.join(examples)}]"


def summarize_engine_actions(actions: List[Any], max_items: int = 5) -> str:
    if not actions:
        return "none"
    counter = Counter()
    examples = []
    for action in actions:
        action_type = type(action).__name__
        counter[action_type] += 1
        if len(examples) < max_items:
            try:
                payload = action.to_cmd_dict() if hasattr(action, "to_cmd_dict") else str(action)
            except Exception:
                payload = str(action)
            examples.append(truncate_text(payload, 140))
    counts = ", ".join(f"{k}:{v}" for k, v in sorted(counter.items()))
    return f"counts=[{counts}] examples=[{' | '.join(examples)}]"


class RedTraceLogger:
    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = log_dir or ensure_output_dir("logs")
        os.makedirs(self.log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = os.path.join(self.log_dir, f"red_trace_{ts}.log")
        self.jsonl_path = os.path.join(self.log_dir, f"red_trace_{ts}.jsonl")
        self.live_path = os.path.join(self.log_dir, "red_trace_live.json")
        self.live_snapshot_enabled = trace_live_enabled()
        self._lock = threading.Lock()
        self._traces: Dict[str, dict] = {}
        if self.live_snapshot_enabled:
            self._write_live_snapshot()

    def start_trace(self, trace_id: str, sim_time: int, reason: str, memory_packet: Optional[dict] = None) -> None:
        memory_packet = memory_packet or {}
        now = datetime.now()
        record = {
            "trace_id": trace_id,
            "sim_time": sim_time,
            "reason": reason,
            "started_at": now.isoformat(),
            "started_perf": perf_counter(),
            "status": "running",
            "sections": {
                "STM": [
                    {
                        "memory_window": memory_packet.get("memory_window", {}),
                        "current_snapshot": truncate_text(memory_packet.get("current_snapshot", ""), 300),
                        "window_summary": truncate_text(memory_packet.get("window_summary", ""), 500),
                        "recent_events": summarize_timeline(memory_packet.get("event_timeline", []) or []),
                    }
                ],
                "LTM": [],
                "Graph Nodes": [],
                "LLM Calls": [],
                "Guard": [],
                "Parsed Actions": [],
                "Submit": [],
            },
            "errors": [],
        }
        with self._lock:
            self._traces[trace_id] = record
            self._write_live_snapshot_locked()

    def log_step(self, trace_id: Optional[str], section: str, **data) -> None:
        if not trace_id:
            return
        with self._lock:
            record = self._traces.get(trace_id)
            if record is None:
                return
            record["sections"].setdefault(section, []).append(data)
            self._write_live_snapshot_locked()

    def log_error(self, trace_id: Optional[str], message: str) -> None:
        if not trace_id:
            return
        with self._lock:
            record = self._traces.get(trace_id)
            if record is None:
                return
            record["errors"].append(truncate_text(message, 400))
            self._write_live_snapshot_locked()

    def finish_trace(self, trace_id: Optional[str], status: str = "completed", **extra) -> None:
        if not trace_id:
            return
        with self._lock:
            record = self._traces.pop(trace_id, None)
        if record is None:
            return

        record["status"] = status
        record.update(extra)
        record["total_duration_ms"] = int((perf_counter() - record["started_perf"]) * 1000)
        block = self._format_block(record)
        with self._lock:
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(block)
                f.write("\n")
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(self._json_record(record), ensure_ascii=False))
                f.write("\n")
            self._write_live_snapshot_locked()

    def finalize_all(self, status: str = "aborted") -> None:
        with self._lock:
            trace_ids = list(self._traces.keys())
        for trace_id in trace_ids:
            self.finish_trace(trace_id, status=status)

    def _json_record(self, record: dict) -> dict:
        data = dict(record)
        data.pop("started_perf", None)
        return data

    def _write_live_snapshot(self) -> None:
        if not self.live_snapshot_enabled:
            return
        with self._lock:
            self._write_live_snapshot_locked()

    def _write_live_snapshot_locked(self) -> None:
        if not self.live_snapshot_enabled:
            return
        payload = {
            "updated_at": datetime.now().isoformat(),
            "active_count": len(self._traces),
            "traces": [self._json_record(record) for record in self._traces.values()],
        }
        with open(self.live_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _format_block(self, record: dict) -> str:
        lines = []
        trace_id = record["trace_id"]
        lines.append("=" * 120)
        lines.append(
            "TRACE "
            f"trace_id={trace_id} sim_time={record.get('sim_time')} reason={record.get('reason')} "
            f"status={record.get('status')} total_ms={record.get('total_duration_ms')}"
        )
        lines.append(f"started_at={record.get('started_at')}")

        for section_name in ["STM", "LTM", "Graph Nodes", "LLM Calls", "Guard", "Parsed Actions", "Submit"]:
            lines.append(f"{section_name}:")
            entries = record["sections"].get(section_name, [])
            if not entries:
                lines.append("  - none")
                continue
            for entry in entries:
                formatted = self._format_entry(section_name, entry)
                for item in formatted:
                    lines.append(f"  - {item}")

        if record.get("errors"):
            lines.append("Errors:")
            for error in record["errors"]:
                lines.append(f"  - {error}")

        lines.append(f"END TRACE trace_id={trace_id}")
        return "\n".join(lines)

    def _format_entry(self, section_name: str, entry: dict) -> List[str]:
        if section_name == "STM":
            return [
                f"window={entry.get('memory_window')}",
                f"snapshot={entry.get('current_snapshot')}",
                f"summary={entry.get('window_summary')}",
                f"recent_events={entry.get('recent_events')}",
            ]
        if section_name == "LTM":
            return [
                f"query={entry.get('query_summary', 'none')}",
                f"hit_count={entry.get('hit_count', 0)}",
                f"lessons={entry.get('lessons_summary', 'none')}",
            ]
        if section_name == "Graph Nodes":
            extras = {
                k: v
                for k, v in entry.items()
                if k not in {"node", "duration_ms", "status", "retry_count", "output_summary"}
            }
            extras_text = ""
            if extras:
                extras_text = " " + " ".join(f"{k}={v}" for k, v in extras.items())
            return [
                (
                    f"node={entry.get('node')} duration_ms={entry.get('duration_ms')} "
                    f"status={entry.get('status', 'ok')} retry={entry.get('retry_count', 0)} "
                    f"output={entry.get('output_summary', 'none')}{extras_text}"
                )
            ]
        if section_name == "LLM Calls":
            return [
                (
                    f"component={entry.get('component')} sim_time={entry.get('sim_time')} "
                    f"duration_ms={entry.get('duration_ms')} success={entry.get('success')} "
                    f"prompt_chars={entry.get('prompt_chars')} response_chars={entry.get('response_chars')} "
                    f"path={entry.get('file_path')} summary={entry.get('summary')}"
                )
            ]
        if section_name == "Guard":
            return [
                (
                    f"planned_at={entry.get('planned_at')} now={entry.get('now')} "
                    f"drop_count={entry.get('drop_count')} reasons={entry.get('drop_reason_dist')} "
                    f"force_replan={entry.get('force_replan')}"
                )
            ]
        if section_name == "Parsed Actions":
            return [
                f"json_action_count={entry.get('json_action_count')} json_summary={entry.get('json_summary')}",
                f"parsed_action_count={entry.get('parsed_action_count')} parsed_summary={entry.get('parsed_summary')}",
                f"ignored_summary={entry.get('ignored_summary')}",
            ]
        if section_name == "Submit":
            return [
                (
                    f"submitted_action_count={entry.get('submitted_action_count')} "
                    f"submitted_summary={entry.get('submitted_summary')} note={entry.get('note', '')}"
                )
            ]
        return [truncate_text(json.dumps(entry, ensure_ascii=False), 500)]
