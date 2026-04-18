import argparse
import glob
import json
import math
import os
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional


TERMINAL_WINDOW_LOW_COST_METERS = 5000.0
TERMINAL_WINDOW_HIGH_COST_METERS = 7000.0


def _latest(path_pattern: str) -> Optional[str]:
    matches = glob.glob(path_pattern)
    if not matches:
        return None
    matches.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return matches[0]


def _load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _haversine_m(a: Dict[str, float], b: Dict[str, float]) -> float:
    lat1 = math.radians(float(a["lat"]))
    lon1 = math.radians(float(a["lon"]))
    lat2 = math.radians(float(b["lat"]))
    lon2 = math.radians(float(b["lon"]))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return 6371000.0 * 2.0 * math.asin(min(1.0, math.sqrt(h)))


def _position_of(entity: Dict[str, Any]) -> Optional[Dict[str, float]]:
    pos = entity.get("position")
    if not isinstance(pos, dict):
        return None
    if "lat" not in pos or "lon" not in pos:
        return None
    return pos


def _nearest_frame(sim_time: int, frames_by_time: Dict[int, dict], tolerance: int = 1) -> Optional[dict]:
    if sim_time in frames_by_time:
        return frames_by_time[sim_time]
    for delta in range(1, tolerance + 1):
        if sim_time - delta in frames_by_time:
            return frames_by_time[sim_time - delta]
        if sim_time + delta in frames_by_time:
            return frames_by_time[sim_time + delta]
    return None


def _latest_detect_before(history: Dict[str, List[dict]], target_id: str, sim_time: int) -> Optional[dict]:
    samples = history.get(target_id, [])
    candidate = None
    for sample in samples:
        if int(sample["sim_time"]) <= int(sim_time):
            candidate = sample
        else:
            break
    return candidate


def _classify_miss(min_distance_m: Optional[float], observation_error_m: Optional[float]) -> str:
    if min_distance_m is None:
        return "no_truth_overlap"
    if min_distance_m <= TERMINAL_WINDOW_LOW_COST_METERS:
        return "entered_terminal_window"
    if min_distance_m <= 15000.0:
        return "near_miss_predictor_tunable"
    if observation_error_m is not None and observation_error_m >= 8000.0:
        return "track_error_dominant"
    return "large_miss_or_intercept"


def _assessment_text(total: int, near_miss_count: int, terminal_count: int, no_truth_overlap: int) -> str:
    if total == 0:
        return "本次诊断没有捕获到低成本攻击弹轨迹，样本不足，暂时无法判断预测函数是否还有优化空间。"
    if near_miss_count / total >= 0.3:
        return "有相当一部分导弹属于差一点进入末制导窗口的近失误，说明预测函数仍有继续优化的价值。"
    if terminal_count / total >= 0.2:
        return "已经有一部分低成本弹进入末制导窗口，问题不只是观测误差，蓝方防御或目标机动也可能在显著影响最终命中。"
    if no_truth_overlap / total >= 0.3:
        return "大量导弹没有和蓝方真实轨迹形成有效重叠，优先应检查侦察链、目标选择和发射时机，而不是只微调预测函数。"
    return "低成本弹轨迹表现一般，既有一定接近目标的能力，也存在较大离散性。下一步适合继续做多策略对照，而不是只凭单局下结论。"


def analyze(red_file: str, blue_file: str) -> Dict[str, Any]:
    red_rows = _load_jsonl(red_file)
    blue_rows = _load_jsonl(blue_file)

    red_frames = {int(row["sim_time"]): row for row in red_rows if "sim_time" in row}
    blue_frames = {int(row["sim_time"]): row for row in blue_rows if "sim_time" in row}

    missile_tracks: Dict[str, List[dict]] = defaultdict(list)
    detected_history: Dict[str, List[dict]] = defaultdict(list)

    for row in red_rows:
        sim_time = int(row.get("sim_time", -1))
        for missile in row.get("lowcost_missiles", []) or []:
            pos = _position_of(missile)
            if pos is None:
                continue
            missile_tracks[str(missile.get("unit_id"))].append(
                {
                    "sim_time": sim_time,
                    "unit_id": str(missile.get("unit_id")),
                    "position": pos,
                    "velocity_mps": missile.get("velocity_mps"),
                }
            )
        for target in row.get("detected_targets", []) or []:
            pos = _position_of(target)
            if pos is None:
                continue
            detected_history[str(target.get("target_id"))].append(
                {
                    "sim_time": sim_time,
                    "target_id": str(target.get("target_id")),
                    "target_type": str(target.get("target_type")),
                    "position": pos,
                    "state_change": bool(target.get("state_change", False)),
                }
            )

    for samples in detected_history.values():
        samples.sort(key=lambda item: int(item["sim_time"]))

    missiles_summary: List[Dict[str, Any]] = []
    min_distances: List[float] = []
    observation_errors: List[float] = []
    class_counts: Dict[str, int] = defaultdict(int)

    for missile_id, track in sorted(missile_tracks.items()):
        track.sort(key=lambda item: int(item["sim_time"]))
        launch = track[0]
        candidate_target_id = None
        candidate_target_type = None
        min_distance_m = None
        min_distance_sim_time = None

        for point in track:
            blue_frame = _nearest_frame(int(point["sim_time"]), blue_frames, tolerance=1)
            if not blue_frame:
                continue
            for target in blue_frame.get("surface_units", []) or []:
                pos = _position_of(target)
                if pos is None:
                    continue
                distance_m = _haversine_m(point["position"], pos)
                if min_distance_m is None or distance_m < min_distance_m:
                    min_distance_m = distance_m
                    min_distance_sim_time = int(point["sim_time"])
                    candidate_target_id = str(target.get("unit_id"))
                    candidate_target_type = str(target.get("unit_type"))

        truth_at_launch = None
        launch_frame = _nearest_frame(int(launch["sim_time"]), blue_frames, tolerance=1)
        if candidate_target_id and launch_frame:
            for target in launch_frame.get("surface_units", []) or []:
                if str(target.get("unit_id")) == candidate_target_id:
                    truth_at_launch = _position_of(target)
                    break

        latest_detect = _latest_detect_before(detected_history, candidate_target_id or "", int(launch["sim_time"]))
        observation_error_m = None
        if latest_detect and truth_at_launch:
            observation_error_m = _haversine_m(latest_detect["position"], truth_at_launch)

        classification = _classify_miss(min_distance_m, observation_error_m)
        class_counts[classification] += 1
        if min_distance_m is not None:
            min_distances.append(min_distance_m)
        if observation_error_m is not None:
            observation_errors.append(observation_error_m)

        missiles_summary.append(
            {
                "missile_id": missile_id,
                "launch_time": int(launch["sim_time"]),
                "end_time": int(track[-1]["sim_time"]),
                "track_points": len(track),
                "candidate_target_id": candidate_target_id,
                "candidate_target_type": candidate_target_type,
                "closest_approach_m": round(min_distance_m, 2) if min_distance_m is not None else None,
                "closest_approach_sim_time": min_distance_sim_time,
                "entered_lowcost_terminal_window": bool(
                    min_distance_m is not None and min_distance_m <= TERMINAL_WINDOW_LOW_COST_METERS
                ),
                "entered_highcost_terminal_window": bool(
                    min_distance_m is not None and min_distance_m <= TERMINAL_WINDOW_HIGH_COST_METERS
                ),
                "launch_observation_error_m": round(observation_error_m, 2) if observation_error_m is not None else None,
                "detected_track_age_at_launch": (
                    int(launch["sim_time"]) - int(latest_detect["sim_time"]) if latest_detect is not None else None
                ),
                "classification": classification,
            }
        )

    total = len(missiles_summary)
    near_miss_count = sum(1 for item in missiles_summary if item["classification"] == "near_miss_predictor_tunable")
    terminal_count = sum(1 for item in missiles_summary if item["entered_lowcost_terminal_window"])
    no_truth_overlap = sum(1 for item in missiles_summary if item["classification"] == "no_truth_overlap")
    assessment = _assessment_text(total, near_miss_count, terminal_count, no_truth_overlap)

    return {
        "red_diagnostics_file": os.path.abspath(red_file),
        "blue_diagnostics_file": os.path.abspath(blue_file),
        "summary": {
            "missile_count": total,
            "entered_lowcost_terminal_window_count": terminal_count,
            "entered_highcost_terminal_window_count": sum(
                1 for item in missiles_summary if item["entered_highcost_terminal_window"]
            ),
            "near_miss_predictor_tunable_count": near_miss_count,
            "no_truth_overlap_count": no_truth_overlap,
            "median_closest_approach_m": round(statistics.median(min_distances), 2) if min_distances else None,
            "mean_closest_approach_m": round(statistics.fmean(min_distances), 2) if min_distances else None,
            "median_launch_observation_error_m": (
                round(statistics.median(observation_errors), 2) if observation_errors else None
            ),
            "mean_launch_observation_error_m": (
                round(statistics.fmean(observation_errors), 2) if observation_errors else None
            ),
            "classification_counts": dict(class_counts),
        },
        "assessment": assessment,
        "missiles": missiles_summary,
    }


def _default_output_path(red_file: str) -> str:
    root = os.path.dirname(os.path.abspath(red_file))
    return os.path.join(root, "analysis_lowcost_trajectory.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze recorded RED low-cost missile and BLUE target trajectories.")
    parser.add_argument("--diagnostics-root", type=str, default=os.path.join("test", "diagnostics"))
    parser.add_argument("--red-file", type=str, default=None)
    parser.add_argument("--blue-file", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    red_file = args.red_file or _latest(os.path.join(args.diagnostics_root, "trajectory_RED_*.jsonl"))
    blue_file = args.blue_file or _latest(os.path.join(args.diagnostics_root, "trajectory_BLUE_*.jsonl"))
    if not red_file or not blue_file:
        raise SystemExit("Missing diagnostics files. Provide --red-file and --blue-file or ensure diagnostics exist.")

    report = analyze(red_file, blue_file)
    output_path = args.output or _default_output_path(red_file)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(report["assessment"])
    print(f"saved={os.path.abspath(output_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
