import json
import logging
import os
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from runtime_paths import ensure_output_dir
from state_access import (
    extract_score_dict,
    get_side_score,
    get_unit_hp,
    get_unit_position,
    get_unit_velocity,
    infer_unit_type,
    iter_missile_units,
    iter_platform_units,
    safe_simtime,
    serialize_position,
)


logger = logging.getLogger(__name__)

_BLUE_TRACKED_TYPES = {
    "Cruiser_Surface",
    "Destroyer_Surface",
    "Flagship_Surface",
    "Merchant_Ship_Surface",
    "Shipboard_Aircraft_FixWing",
}

_RED_TRACKED_MISSILE_TYPES = {"LowCostAttackMissile"}


def _read_bool_env(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return raw not in {"0", "false", "no", "off", ""}


def _read_float_env(name: str, default: float) -> float:
    raw = str(os.getenv(name, str(default))).strip()
    try:
        return float(raw)
    except Exception:
        return default


class TrajectoryDiagnosticsRecorder:
    def __init__(self, faction_name: str, output_dir: Optional[str] = None):
        self.faction_name = str(faction_name or "").upper()
        self.output_dir = output_dir or ensure_output_dir("diagnostics")
        self.interval_seconds = max(0.0, _read_float_env("TRAJECTORY_DIAGNOSTICS_INTERVAL_SECONDS", 1.0))
        self.record_detected_targets = _read_bool_env("TRAJECTORY_DIAGNOSTICS_RECORD_DETECTED_TARGETS", True)
        self.filename = os.path.join(
            self.output_dir,
            f"trajectory_{self.faction_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
        )
        self._file = open(self.filename, "a", encoding="utf-8")
        self._lock = threading.Lock()
        self._last_recorded_sim_time: Optional[float] = None
        self.frame_count = 0
        logger.info(
            "[%s] trajectory_diagnostics_enabled path=%s interval=%.1fs detected_targets=%s",
            self.faction_name,
            self.filename,
            self.interval_seconds,
            self.record_detected_targets,
        )

    @property
    def filepath(self) -> str:
        return self.filename

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.flush()
                self._file.close()
        logger.info(
            "[%s] trajectory_diagnostics_closed path=%s frames=%s",
            self.faction_name,
            self.filename,
            self.frame_count,
        )

    def record_state(self, state) -> None:
        sim_time = safe_simtime(state)
        if sim_time is None:
            return
        if (
            self._last_recorded_sim_time is not None
            and float(sim_time) < float(self._last_recorded_sim_time) + self.interval_seconds
        ):
            return

        entry = {
            "time": datetime.now().isoformat(),
            "schema_version": 1,
            "faction": self.faction_name,
            "sim_time": sim_time,
            "side_score": get_side_score(state, self.faction_name),
            "score_breakdown": extract_score_dict(state),
        }

        if self.faction_name == "RED":
            entry["lowcost_missiles"] = self._capture_lowcost_missiles(state)
            entry["detected_targets"] = self._capture_detected_targets(state) if self.record_detected_targets else []
        elif self.faction_name == "BLUE":
            entry["surface_units"] = self._capture_blue_units(state)
        else:
            return

        with self._lock:
            self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._file.flush()
        self._last_recorded_sim_time = float(sim_time)
        self.frame_count += 1

    def _capture_lowcost_missiles(self, state) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        for unit in iter_missile_units(state):
            unit_id = str(getattr(unit, "id", "") or "")
            unit_type = infer_unit_type(unit_id)
            if unit_type not in _RED_TRACKED_MISSILE_TYPES:
                continue
            pos = serialize_position(get_unit_position(unit=unit))
            if pos is None:
                continue
            samples.append(
                {
                    "unit_id": unit_id,
                    "unit_type": unit_type,
                    "position": pos,
                    "velocity_mps": get_unit_velocity(unit),
                    "hp": get_unit_hp(unit=unit),
                }
            )
        return samples

    def _capture_blue_units(self, state) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        for unit in iter_platform_units(state):
            unit_id = str(getattr(unit, "id", "") or "")
            unit_type = infer_unit_type(unit_id)
            if unit_type not in _BLUE_TRACKED_TYPES:
                continue
            pos = serialize_position(get_unit_position(unit=unit))
            if pos is None:
                continue
            samples.append(
                {
                    "unit_id": unit_id,
                    "unit_type": unit_type,
                    "position": pos,
                    "velocity_mps": get_unit_velocity(unit),
                    "hp": get_unit_hp(unit=unit),
                }
            )
        return samples

    def _capture_detected_targets(self, state) -> List[Dict[str, Any]]:
        chosen: Dict[str, Dict[str, Any]] = {}
        try:
            detector_states = state.find_detector_states() or []
        except Exception:
            detector_states = []

        for ds in detector_states:
            target_id = str(getattr(ds, "target_id", "") or "")
            target_type = infer_unit_type(target_id) or str(getattr(ds, "_target_type", "") or "")
            if target_type not in _BLUE_TRACKED_TYPES:
                continue
            pos = serialize_position(getattr(ds, "target_position", None))
            if pos is None:
                continue

            sample = {
                "target_id": target_id,
                "target_type": target_type,
                "position": pos,
                "state_change": bool(getattr(ds, "state_change", False)),
                "detector_id": str(getattr(ds, "id", "") or ""),
                "real_id": str(getattr(ds, "real_id", "") or ""),
            }
            existing = chosen.get(target_id)
            if existing is None:
                chosen[target_id] = sample
                continue
            if sample["state_change"] and not existing.get("state_change"):
                chosen[target_id] = sample

        return list(chosen.values())
