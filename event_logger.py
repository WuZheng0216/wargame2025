import json
import logging
import os
import queue
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from state_access import extract_score_dict, get_detected_target_ids, get_unit_hp, infer_unit_type, iter_platform_units

logger = logging.getLogger(__name__)


class EventLogger:
    """Collect battle events and decisions into a durable JSONL file."""

    def __init__(self, faction_name: str, log_dir: str):
        self.faction_name = faction_name
        self.log_dir = log_dir
        self.filename = os.path.join(
            log_dir,
            f"battle_log_{faction_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
        )
        self.filepath = self.filename
        self.previous_state = None
        self._seen_targets = set()
        self._lost_units = set()
        self._write_lock = threading.Lock()
        self._queue: "queue.Queue[Optional[dict]]" = queue.Queue()
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name=f"{self.faction_name}-BattleLogWriter",
            daemon=True,
        )
        self._file_handle = None
        self._finalized = False
        self._writer_closed = False
        self._write_ok_logged = False
        self.event_count = 0

        os.makedirs(log_dir, exist_ok=True)
        self._file_handle = open(self.filename, "a", encoding="utf-8")
        self._writer_thread.start()
        logger.info("[%s] EventLogger initialized. path=%s", faction_name, self.filename)

    def _infer_unit_type(self, unit_id: str) -> Optional[str]:
        return infer_unit_type(unit_id)

    def _tracked_unit_snapshot(self, state) -> Dict[str, Dict[str, Any]]:
        snapshots: Dict[str, Dict[str, Any]] = {}
        if state is None:
            return snapshots
        for unit in iter_platform_units(state):
            unit_id = str(getattr(unit, "id", "") or "")
            unit_type = self._infer_unit_type(unit_id)
            if not unit_id or unit_type is None:
                continue
            snapshots[unit_id] = {
                "unit_type": unit_type,
                "hp": get_unit_hp(state=state, unit_id=unit_id, unit=unit),
            }
        return snapshots

    def log_event(self, event_type: str, message: str, extra: Optional[dict] = None, sim_time: Optional[float] = None):
        entry = {
            "time": datetime.now().isoformat(),
            "type": event_type,
            "sim_time": sim_time,
            "message": message,
            "extra": extra or {},
        }
        self._append_entry(entry)
        logger.debug("[%s] EVENT[%s]: %s", self.faction_name, event_type, message)

    def log_decision(
        self,
        sim_time: float,
        analysis: str,
        actions: List[Any],
        source: str = "unknown",
        trace: Optional[dict] = None,
    ):
        entry = {
            "time": datetime.now().isoformat(),
            "type": "DECISION",
            "sim_time": sim_time,
            "source": source,
            "analysis": str(analysis or ""),
            "actions": self._safe_serialize(actions),
            "trace": self._safe_serialize(trace or {}),
        }
        self._append_entry(entry)
        logger.debug(
            "[%s] DECISION sim_time=%.1f source=%s action_count=%d",
            self.faction_name,
            sim_time,
            source,
            len(actions),
        )

    def _safe_serialize(self, obj):
        if callable(getattr(obj, "to_cmd_dict", None)):
            try:
                return self._safe_serialize(obj.to_cmd_dict())
            except Exception:
                return str(obj)
        if isinstance(obj, list):
            return [self._safe_serialize(x) for x in obj]
        if isinstance(obj, tuple):
            return [self._safe_serialize(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): self._safe_serialize(v) for k, v in obj.items()}
        if hasattr(obj, "__dict__"):
            try:
                return {k: self._safe_serialize(v) for k, v in obj.__dict__.items()}
            except Exception:
                return str(obj)
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        return str(obj)

    def _writer_loop(self):
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break

            batch = [item]
            try:
                while len(batch) < 64:
                    batch.append(self._queue.get_nowait())
            except queue.Empty:
                pass

            try:
                with self._write_lock:
                    for entry in batch:
                        if entry is None:
                            continue
                        self._file_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        self.event_count += 1
                    self._file_handle.flush()
                    if batch and not self._write_ok_logged:
                        logger.info("[%s] battle_log_write_ok path=%s", self.faction_name, self.filename)
                        self._write_ok_logged = True
            except Exception as e:
                logger.error("[%s] failed to append battle log entry: %s", self.faction_name, e, exc_info=True)
            finally:
                for _ in batch:
                    self._queue.task_done()

        with self._write_lock:
            if self._file_handle and not self._file_handle.closed:
                self._file_handle.flush()
                self._file_handle.close()
        self._writer_closed = True

    def _append_entry(self, entry: Dict[str, Any]) -> bool:
        safe_entry = self._safe_serialize(entry)
        if self._finalized:
            logger.warning("[%s] battle_log_write_skipped finalized path=%s", self.faction_name, self.filename)
            return False
        self._queue.put(safe_entry)
        return True

    def save_log(self):
        try:
            if self._finalized:
                logger.info(
                    "[%s] battle_log_finalize_ok path=%s event_count=%d (already finalized)",
                    self.faction_name,
                    self.filename,
                    self.event_count,
                )
                return

            self._finalized = True
            finalize_entry = {
                "time": datetime.now().isoformat(),
                "type": "LOG_FINALIZED",
                "sim_time": None,
                "message": "Battle log finalized",
                "extra": {"event_count": self.event_count, "faction": self.faction_name},
            }
            self._queue.put(self._safe_serialize(finalize_entry))
            self._queue.join()
            self._queue.put(None)
            self._writer_thread.join(timeout=5.0)
            if not self._writer_closed:
                logger.warning("[%s] battle_log_writer_join_timeout path=%s", self.faction_name, self.filename)

            logger.info(
                "[%s] battle_log_finalize_ok path=%s event_count=%d",
                self.faction_name,
                self.filename,
                self.event_count,
            )
        except Exception as e:
            logger.error("[%s] failed to finalize battle log: %s", self.faction_name, e, exc_info=True)
            raise

    def check_events(self, state, new_score, old_score):
        try:
            sim_time = state.simtime()
            score_breakdown = extract_score_dict(state)

            if new_score != old_score:
                self.log_event(
                    "SCORE_CHANGED",
                    f"Score changed {old_score} -> {new_score} (delta={new_score - old_score:+.1f})",
                    extra={
                        "old_score": old_score,
                        "new_score": new_score,
                        "delta": new_score - old_score,
                        "score_breakdown": score_breakdown,
                    },
                    sim_time=sim_time,
                )

            try:
                current_units = self._tracked_unit_snapshot(state)
                previous_units = self._tracked_unit_snapshot(self.previous_state)

                for unit_id, snapshot in previous_units.items():
                    if unit_id in current_units or unit_id in self._lost_units:
                        continue
                    self._lost_units.add(unit_id)
                    self.log_event(
                        "UNIT_LOST",
                        f"{unit_id} lost at sim_time={sim_time}",
                        extra={
                            "unit_id": unit_id,
                            "unit_type": snapshot.get("unit_type"),
                            "reason": "missing_from_state",
                            "last_hp": snapshot.get("hp"),
                        },
                        sim_time=sim_time,
                    )

                for unit_id, snapshot in current_units.items():
                    hp = snapshot.get("hp")
                    if hp is None or unit_id in self._lost_units:
                        continue
                    try:
                        hp_value = float(hp)
                    except Exception:
                        continue
                    if hp_value <= 0:
                        self._lost_units.add(unit_id)
                        self.log_event(
                            "UNIT_LOST",
                            f"{unit_id} lost at sim_time={sim_time}",
                            extra={
                                "unit_id": unit_id,
                                "unit_type": snapshot.get("unit_type"),
                                "reason": "hp_depleted",
                                "last_hp": hp_value,
                            },
                            sim_time=sim_time,
                        )
            except Exception as e:
                logger.debug("[%s] unit loss check failed: %s", self.faction_name, e)

            try:
                detected_targets = get_detected_target_ids(state)
                for target_id in detected_targets:
                    if target_id not in self._seen_targets:
                        self._seen_targets.add(target_id)
                        self.log_event(
                            "NEW_TARGET_DETECTED",
                            f"Detected new target {target_id} at sim_time={sim_time}",
                            extra={"target_id": target_id},
                            sim_time=sim_time,
                        )
            except Exception as e:
                logger.debug("[%s] target detection check failed: %s", self.faction_name, e)

        except Exception as e:
            logger.warning("[%s] check_events() failed: %s", self.faction_name, e)
