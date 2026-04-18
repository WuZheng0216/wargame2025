from __future__ import annotations

import argparse
import json
import py_compile
import re
import shutil
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_ONESHOT_TARGET = Path(
    r"C:\Users\Tk\.conda\envs\scene\Lib\site-packages\jsqlsim\world\policy\oneshot.py"
)
DEFAULT_GAME_STATE_TARGET = Path(
    r"C:\Users\Tk\.conda\envs\scene\Lib\site-packages\jsqlsim\world\game_state.py"
)
ARCHIVE_ROOT = REPO_ROOT / "test" / "archive" / "scene_patches"


HELPER_BLOCK = """
    def _collapse_history_samples(self, raw):
        if not raw:
            return []
        buckets = collections.OrderedDict()
        for simtime, lon, lat in raw:
            key = float(simtime)
            buckets.setdefault(key, []).append((float(lon), float(lat)))

        collapsed = []
        for simtime, points in buckets.items():
            avg_lon = sum(point[0] for point in points) / len(points)
            avg_lat = sum(point[1] for point in points) / len(points)
            collapsed.append((simtime, avg_lon, avg_lat))
        return collapsed

    def _select_target_detector(self, ds_list):
        if not ds_list:
            return None
        fresh = [ds for ds in ds_list if getattr(ds, "state_change", False)]
        candidates = fresh or ds_list
        if len(candidates) == 1:
            return candidates[0]
        return max(
            candidates,
            key=lambda ds: (
                1 if getattr(ds, "state_change", False) else 0,
                1 if getattr(ds, "target_position", None) is not None else 0,
            ),
        )

    def _meters_per_degree_lon(self, lat_deg: float) -> float:
        return max(1.0, 111320.0 * np.cos(np.deg2rad(float(lat_deg))))

    def _estimate_recent_velocity(self, raw, max_segments: int = 3):
        samples = self._collapse_history_samples(raw)
        if len(samples) < 2:
            return None

        usable = samples[-(max_segments + 1):]
        ref_lat = float(usable[-1][2])
        meters_per_deg_lon = self._meters_per_degree_lon(ref_lat)
        meters_per_deg_lat = 111320.0

        vx_sum = 0.0
        vy_sum = 0.0
        weight_sum = 0.0
        segment_index = 1
        for idx in range(len(usable) - 1, 0, -1):
            t1, lon1, lat1 = usable[idx - 1]
            t2, lon2, lat2 = usable[idx]
            dt = float(t2) - float(t1)
            if dt <= 1e-6:
                continue
            dx = (float(lon2) - float(lon1)) * meters_per_deg_lon
            dy = (float(lat2) - float(lat1)) * meters_per_deg_lat
            weight = float(max_segments - segment_index + 1)
            vx_sum += weight * (dx / dt)
            vy_sum += weight * (dy / dt)
            weight_sum += weight
            segment_index += 1

        if weight_sum <= 0.0:
            return None

        return {
            "vx_mps": vx_sum / weight_sum,
            "vy_mps": vy_sum / weight_sum,
            "ref_lat": ref_lat,
            "ref_lon": float(usable[-1][1]),
            "last_obs_time": float(usable[-1][0]),
            "last_obs_pos": Position(float(usable[-1][1]), float(usable[-1][2]), 0),
        }

    def _project_linear_position(
        self,
        origin: Position,
        ref_lat: float,
        vx_mps: float,
        vy_mps: float,
        dt: float,
    ) -> Position:
        meters_per_deg_lon = self._meters_per_degree_lon(ref_lat)
        meters_per_deg_lat = 111320.0
        lon = float(origin.lon) + (float(vx_mps) * float(dt)) / meters_per_deg_lon
        lat = float(origin.lat) + (float(vy_mps) * float(dt)) / meters_per_deg_lat
        return Position(lon, lat, float(getattr(origin, "alt", 0.0) or 0.0))

    def _solve_linear_intercept_time(
        self,
        attacker_pos: Position,
        target_now: Position,
        ref_lat: float,
        vx_mps: float,
        vy_mps: float,
        weapon_speed: float,
    ):
        meters_per_deg_lon = self._meters_per_degree_lon(ref_lat)
        meters_per_deg_lat = 111320.0
        rx = (float(target_now.lon) - float(attacker_pos.lon)) * meters_per_deg_lon
        ry = (float(target_now.lat) - float(attacker_pos.lat)) * meters_per_deg_lat
        speed = max(float(weapon_speed), 1.0)
        a = float(vx_mps) * float(vx_mps) + float(vy_mps) * float(vy_mps) - speed * speed
        b = 2.0 * (rx * float(vx_mps) + ry * float(vy_mps))
        c = rx * rx + ry * ry

        roots = []
        if abs(a) < 1e-6:
            if abs(b) < 1e-6:
                return None
            t = -c / b
            if t >= 0.0:
                roots.append(t)
        else:
            disc = b * b - 4.0 * a * c
            if disc < 0.0:
                return None
            sqrt_disc = np.sqrt(disc)
            for t in ((-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)):
                if t >= 0.0:
                    roots.append(t)
        if not roots:
            return None
        return min(float(t) for t in roots)

    def _blend_positions(self, p1: Position, p2: Position, weight_p1: float = 0.7) -> Position:
        w1 = min(max(float(weight_p1), 0.0), 1.0)
        w2 = 1.0 - w1
        return Position(
            float(p1.lon) * w1 + float(p2.lon) * w2,
            float(p1.lat) * w1 + float(p2.lat) * w2,
            float(getattr(p1, "alt", 0.0) or 0.0) * w1 + float(getattr(p2, "alt", 0.0) or 0.0) * w2,
        )
"""


INTERCEPT_BLOCK_V3 = """
    def _predict_intercept_position(
        self,
        attacker_pos: Position,
        target_id: str,
        target_position: Position,
        weapon_speed: float,
        delta_t: float,
        history_points: int = 6,
        current_simtime: float = 0.0,
    ) -> tuple[Position, float, int, float]:
        self._ensure_history_capacity(max(history_points, 6))
        raw = GLOBAL_HISTORY.last_n(target_id, max(history_points, 6))
        raw = self._collapse_history_samples(raw)
        history_count = len(raw)
        track_age = 0.0
        if history_count >= 1:
            try:
                track_age = max(float(current_simtime) - float(raw[-1][0]), 0.0)
            except Exception:
                track_age = 0.0
        if history_count < 2:
            return target_position, min(max(float(delta_t), 0.0), 720.0), history_count, track_age

        predicted = target_position
        intercept_t = min(max(float(delta_t), 0.0), 720.0)

        velocity = self._estimate_recent_velocity(raw, max_segments=3)
        linear_prediction = None
        if velocity is not None:
            target_now = self._project_linear_position(
                origin=velocity["last_obs_pos"],
                ref_lat=velocity["ref_lat"],
                vx_mps=velocity["vx_mps"],
                vy_mps=velocity["vy_mps"],
                dt=track_age,
            )
            linear_flight_t = self._solve_linear_intercept_time(
                attacker_pos=attacker_pos,
                target_now=target_now,
                ref_lat=velocity["ref_lat"],
                vx_mps=velocity["vx_mps"],
                vy_mps=velocity["vy_mps"],
                weapon_speed=weapon_speed,
            )
            if linear_flight_t is not None:
                intercept_t = min(max(float(linear_flight_t), 0.0), 720.0)
                linear_prediction = self._project_linear_position(
                    origin=target_now,
                    ref_lat=velocity["ref_lat"],
                    vx_mps=velocity["vx_mps"],
                    vy_mps=velocity["vy_mps"],
                    dt=intercept_t,
                )

        lsq_prediction = None
        current_t = intercept_t
        for _ in range(4):
            candidate = self._predict_lsq(
                target_id,
                current_t,
                min(max(history_points, 4), 6),
                current_simtime=current_simtime,
            )
            if candidate is None:
                break
            lsq_prediction = candidate
            try:
                next_t = attacker_pos.distance(candidate) / max(float(weapon_speed), 1.0)
            except Exception:
                break
            next_t = min(max(float(next_t), 0.0), 720.0)
            if abs(next_t - current_t) < 0.75:
                current_t = next_t
                break
            current_t = 0.35 * current_t + 0.65 * next_t

        if linear_prediction is not None and lsq_prediction is not None:
            try:
                gap_m = linear_prediction.distance(lsq_prediction)
            except Exception:
                gap_m = 0.0
            predicted = self._blend_positions(
                linear_prediction,
                lsq_prediction,
                0.75 if gap_m > 3000.0 else 0.6,
            )
            intercept_t = current_t
        elif linear_prediction is not None:
            predicted = linear_prediction
        elif lsq_prediction is not None:
            predicted = lsq_prediction
            intercept_t = current_t

        return predicted, intercept_t, history_count, track_age
"""


GAME_STATE_FUNCTION = """
    def _record_detector_to_global_history(self):
        simtime = float(self._simtime)

        try:
            grouped = {}
            for unit in self.unit_states:
                for ds in unit.detector_states:
                    pos = ds.target_position
                    if pos is None:
                        continue
                    grouped.setdefault(ds.target_id, []).append(ds)

            for target_id, detector_states in grouped.items():
                fresh = [ds for ds in detector_states if getattr(ds, "state_change", False)]
                candidates = fresh or detector_states
                if not candidates:
                    continue

                avg_lon = sum(float(ds.target_position.lon) for ds in candidates) / len(candidates)
                avg_lat = sum(float(ds.target_position.lat) for ds in candidates) / len(candidates)
                avg_alt = sum(
                    float(getattr(ds.target_position, "alt", 0.0) or 0.0)
                    for ds in candidates
                ) / len(candidates)
                pos = Position(lon=avg_lon, lat=avg_lat, alt=avg_alt)

                last = GLOBAL_HISTORY.last_n(target_id, 1)
                if last:
                    last_time, last_lon, last_lat = last[-1]
                    if abs(float(last_time) - simtime) < 1e-6:
                        continue
                    if not fresh:
                        last_pos = Position(lon=float(last_lon), lat=float(last_lat), alt=avg_alt)
                        try:
                            if last_pos.distance(pos) < 50.0:
                                continue
                        except Exception:
                            continue

                GLOBAL_HISTORY.record(target_id, simtime, pos)
        except Exception as e:
            logging.warning(f"[GameState] GLOBAL_HISTORY record failed: {e}")
"""


def replace_once(text: str, old: str, new: str, label: str) -> tuple[str, bool]:
    if old not in text:
        return text, False
    return text.replace(old, new, 1), True


def patch_oneshot_text(text: str) -> tuple[str, bool]:
    changed = False

    if "def _predict_intercept_position(" not in text:
        raise RuntimeError(
            "oneshot.py does not look like the first-stage low-cost intercept patch. "
            "Please restore/apply v1 first."
        )

    if "def _collapse_history_samples(" not in text:
        anchor = "    def _predict_intercept_position(\n"
        if anchor not in text:
            raise RuntimeError("Could not find insertion anchor in oneshot.py")
        text = text.replace(anchor, HELPER_BLOCK.strip("\n") + "\n\n" + anchor, 1)
        changed = True
    elif "def _estimate_recent_velocity(" not in text:
        anchor = "    def _predict_intercept_position(\n"
        if anchor not in text:
            raise RuntimeError("Could not find v3 insertion anchor in oneshot.py")
        extra_helpers = """
    def _meters_per_degree_lon(self, lat_deg: float) -> float:
        return max(1.0, 111320.0 * np.cos(np.deg2rad(float(lat_deg))))

    def _estimate_recent_velocity(self, raw, max_segments: int = 3):
        samples = self._collapse_history_samples(raw)
        if len(samples) < 2:
            return None

        usable = samples[-(max_segments + 1):]
        ref_lat = float(usable[-1][2])
        meters_per_deg_lon = self._meters_per_degree_lon(ref_lat)
        meters_per_deg_lat = 111320.0

        vx_sum = 0.0
        vy_sum = 0.0
        weight_sum = 0.0
        segment_index = 1
        for idx in range(len(usable) - 1, 0, -1):
            t1, lon1, lat1 = usable[idx - 1]
            t2, lon2, lat2 = usable[idx]
            dt = float(t2) - float(t1)
            if dt <= 1e-6:
                continue
            dx = (float(lon2) - float(lon1)) * meters_per_deg_lon
            dy = (float(lat2) - float(lat1)) * meters_per_deg_lat
            weight = float(max_segments - segment_index + 1)
            vx_sum += weight * (dx / dt)
            vy_sum += weight * (dy / dt)
            weight_sum += weight
            segment_index += 1

        if weight_sum <= 0.0:
            return None

        return {
            "vx_mps": vx_sum / weight_sum,
            "vy_mps": vy_sum / weight_sum,
            "ref_lat": ref_lat,
            "ref_lon": float(usable[-1][1]),
            "last_obs_time": float(usable[-1][0]),
            "last_obs_pos": Position(float(usable[-1][1]), float(usable[-1][2]), 0),
        }

    def _project_linear_position(
        self,
        origin: Position,
        ref_lat: float,
        vx_mps: float,
        vy_mps: float,
        dt: float,
    ) -> Position:
        meters_per_deg_lon = self._meters_per_degree_lon(ref_lat)
        meters_per_deg_lat = 111320.0
        lon = float(origin.lon) + (float(vx_mps) * float(dt)) / meters_per_deg_lon
        lat = float(origin.lat) + (float(vy_mps) * float(dt)) / meters_per_deg_lat
        return Position(lon, lat, float(getattr(origin, "alt", 0.0) or 0.0))

    def _solve_linear_intercept_time(
        self,
        attacker_pos: Position,
        target_now: Position,
        ref_lat: float,
        vx_mps: float,
        vy_mps: float,
        weapon_speed: float,
    ):
        meters_per_deg_lon = self._meters_per_degree_lon(ref_lat)
        meters_per_deg_lat = 111320.0
        rx = (float(target_now.lon) - float(attacker_pos.lon)) * meters_per_deg_lon
        ry = (float(target_now.lat) - float(attacker_pos.lat)) * meters_per_deg_lat
        speed = max(float(weapon_speed), 1.0)
        a = float(vx_mps) * float(vx_mps) + float(vy_mps) * float(vy_mps) - speed * speed
        b = 2.0 * (rx * float(vx_mps) + ry * float(vy_mps))
        c = rx * rx + ry * ry

        roots = []
        if abs(a) < 1e-6:
            if abs(b) < 1e-6:
                return None
            t = -c / b
            if t >= 0.0:
                roots.append(t)
        else:
            disc = b * b - 4.0 * a * c
            if disc < 0.0:
                return None
            sqrt_disc = np.sqrt(disc)
            for t in ((-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)):
                if t >= 0.0:
                    roots.append(t)
        if not roots:
            return None
        return min(float(t) for t in roots)

    def _blend_positions(self, p1: Position, p2: Position, weight_p1: float = 0.7) -> Position:
        w1 = min(max(float(weight_p1), 0.0), 1.0)
        w2 = 1.0 - w1
        return Position(
            float(p1.lon) * w1 + float(p2.lon) * w2,
            float(p1.lat) * w1 + float(p2.lat) * w2,
            float(getattr(p1, "alt", 0.0) or 0.0) * w1 + float(getattr(p2, "alt", 0.0) or 0.0) * w2,
        )
"""
        text = text.replace(anchor, extra_helpers.strip("\n") + "\n\n" + anchor, 1)
        changed = True

    old_sig = "    def _predict_lsq(self, target_id: str, delta_t: float, history_points: int = 4):"
    new_sig = (
        "    def _predict_lsq(self, target_id: str, delta_t: float, history_points: int = 4, "
        "current_simtime: Optional[float] = None):"
    )
    text, updated = replace_once(text, old_sig, new_sig, "_predict_lsq signature")
    changed |= updated

    old_raw = "        raw = GLOBAL_HISTORY.last_n(target_id, history_points)\n"
    new_raw = (
        "        raw = GLOBAL_HISTORY.last_n(target_id, history_points)\n"
        "        raw = self._collapse_history_samples(raw)\n"
    )
    if "raw = self._collapse_history_samples(raw)" not in text:
        text, updated = replace_once(text, old_raw, new_raw, "history collapse")
        changed |= updated

    old_dt = "        dt = delta_t\n"
    new_dt = (
        "        last_obs_simtime = raw[-1][0]\n"
        "        track_age = 0.0\n"
        "        if current_simtime is not None:\n"
        "            try:\n"
        "                track_age = max(float(current_simtime) - float(last_obs_simtime), 0.0)\n"
        "            except Exception:\n"
        "                track_age = 0.0\n"
        "        dt = max(float(delta_t), 0.0) + track_age\n"
    )
    if "track_age = max(float(current_simtime) - float(last_obs_simtime), 0.0)" not in text:
        text, updated = replace_once(text, old_dt, new_dt, "track age compensation")
        changed |= updated

    intercept_pattern = re.compile(
        r"    def _predict_intercept_position\(\n.*?(?=\n    def _maybe_log_prediction_debug\()",
        re.DOTALL,
    )
    updated_text, count = intercept_pattern.subn(INTERCEPT_BLOCK_V3.strip("\n") + "\n", text, count=1)
    if count != 1:
        raise RuntimeError("Could not replace _predict_intercept_position in oneshot.py")
    text = updated_text
    changed = True

    old_debug_sig = (
        "        intercept_t: float,\n"
        "        history_points: int,\n"
        "        raw_target_pos: Position,\n"
        "        predicted_pos: Position,\n"
        "    ):"
    )
    new_debug_sig = (
        "        intercept_t: float,\n"
        "        history_points: int,\n"
        "        track_age: float,\n"
        "        raw_target_pos: Position,\n"
        "        predicted_pos: Position,\n"
        "    ):"
    )
    text, updated = replace_once(text, old_debug_sig, new_debug_sig, "debug signature")
    changed |= updated

    old_debug_msg = '            f"history_points={history_points} "\n'
    new_debug_msg = (
        '            f"history_points={history_points} "\n'
        '            f"track_age={track_age:.2f} "\n'
    )
    if 'f"track_age={track_age:.2f} "' not in text:
        text, updated = replace_once(text, old_debug_msg, new_debug_msg, "debug track age")
        changed |= updated

    old_ds = "            ds = ds_list[0]\n"
    new_ds = "            ds = self._select_target_detector(ds_list)\n"
    if "ds = self._select_target_detector(ds_list)" not in text:
        text, updated = replace_once(text, old_ds, new_ds, "detector selection")
        changed |= updated

    old_track_default = "            delta_t = distance_m / weapon_speed\n\n"
    new_track_default = "            delta_t = distance_m / weapon_speed\n            track_age = 0.0\n\n"
    if "track_age = 0.0" not in text.split("delta_t = distance_m / weapon_speed", 1)[1]:
        text, updated = replace_once(text, old_track_default, new_track_default, "track age default")
        changed |= updated

    old_low_cost = (
        "            elif weapon_name == WeaponNameEnum.LOW_COST_ATTACK_MISSLE:\n"
        "                predicted_pos, intercept_t, history_count = self._predict_intercept_position(\n"
        "                    attacker_pos=attacker_pos,\n"
        "                    target_id=target_id,\n"
        "                    target_position=target_pos,\n"
        "                    weapon_speed=weapon_speed,\n"
        "                    delta_t=delta_t,\n"
        "                    history_points=6,\n"
        "                )\n"
    )
    new_low_cost = (
        "            elif weapon_name == WeaponNameEnum.LOW_COST_ATTACK_MISSLE:\n"
        "                predicted_pos, intercept_t, history_count, track_age = self._predict_intercept_position(\n"
        "                    attacker_pos=attacker_pos,\n"
        "                    target_id=target_id,\n"
        "                    target_position=target_pos,\n"
        "                    weapon_speed=weapon_speed,\n"
        "                    delta_t=delta_t,\n"
        "                    history_points=6,\n"
        "                    current_simtime=state.simtime(),\n"
        "                )\n"
    )
    if "current_simtime=state.simtime()" not in text:
        text, updated = replace_once(text, old_low_cost, new_low_cost, "low-cost intercept call")
        changed |= updated

    old_debug_call = (
        "                intercept_t=intercept_t,\n"
        "                history_points=history_count,\n"
        "                raw_target_pos=target_pos,\n"
        "                predicted_pos=predicted_pos,\n"
        "            )\n"
    )
    new_debug_call = (
        "                intercept_t=intercept_t,\n"
        "                history_points=history_count,\n"
        "                track_age=track_age,\n"
        "                raw_target_pos=target_pos,\n"
        "                predicted_pos=predicted_pos,\n"
        "            )\n"
    )
    if "track_age=track_age" not in text:
        text, updated = replace_once(text, old_debug_call, new_debug_call, "debug call track age")
        changed |= updated

    return text, changed


def patch_game_state_text(text: str) -> tuple[str, bool]:
    if "avg_lon = sum(float(ds.target_position.lon) for ds in candidates) / len(candidates)" in text:
        return text, False

    pattern = re.compile(
        r"    def _record_detector_to_global_history\(self\):\n.*?(?=\n    def update_state_temp\(self\):)",
        re.DOTALL,
    )
    replacement = GAME_STATE_FUNCTION.strip("\n") + "\n"
    updated, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError("Could not find _record_detector_to_global_history in game_state.py")
    return updated, True


def write_metadata(entries: list[dict], metadata_path: Path) -> None:
    metadata = {
        "patched_at": datetime.now().isoformat(timespec="seconds"),
        "patch": "scene_low_cost_tracking_v3",
        "files": entries,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def patch_file(path: Path, patch_fn) -> tuple[bool, Path | None]:
    original = path.read_text(encoding="utf-8")
    patched, changed = patch_fn(original)
    if not changed:
        return False, None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.name}.bak.{timestamp}")
    shutil.copy2(path, backup)
    path.write_text(patched, encoding="utf-8")
    py_compile.compile(str(path), doraise=True)
    return True, backup


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Patch scene oneshot/game_state for low-cost missile tracking freshness."
    )
    parser.add_argument("--oneshot-target", default=str(DEFAULT_ONESHOT_TARGET))
    parser.add_argument("--game-state-target", default=str(DEFAULT_GAME_STATE_TARGET))
    args = parser.parse_args()

    oneshot_target = Path(args.oneshot_target).expanduser().resolve()
    game_state_target = Path(args.game_state_target).expanduser().resolve()
    for target in [oneshot_target, game_state_target]:
        if not target.exists():
            raise FileNotFoundError(f"Target file not found: {target}")

    metadata_entries = []
    oneshot_changed, oneshot_backup = patch_file(oneshot_target, patch_oneshot_text)
    metadata_entries.append(
        {
            "target_path": str(oneshot_target),
            "backup_path": str(oneshot_backup) if oneshot_backup else None,
            "changed": oneshot_changed,
        }
    )

    game_state_changed, game_state_backup = patch_file(game_state_target, patch_game_state_text)
    metadata_entries.append(
        {
            "target_path": str(game_state_target),
            "backup_path": str(game_state_backup) if game_state_backup else None,
            "changed": game_state_changed,
        }
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata_path = ARCHIVE_ROOT / f"scene_oneshot_patch_{timestamp}.json"
    write_metadata(metadata_entries, metadata_path)

    print(f"Oneshot target: {oneshot_target}")
    print(f"Oneshot backup: {oneshot_backup}")
    print(f"GameState target: {game_state_target}")
    print(f"GameState backup: {game_state_backup}")
    print(f"Metadata: {metadata_path}")
    if not oneshot_changed and not game_state_changed:
        print("No changes were required. Targets already look patched.")
    else:
        print("Patch applied successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
