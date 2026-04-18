import argparse
import csv
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from analyze_lowcost_trajectory_diagnostics import analyze as analyze_trajectory


@dataclass(frozen=True)
class StrategyPreset:
    name: str
    strategy: str
    salvo_size: int
    salvo_interval: float
    move_interval: float
    guide_interval: float
    guide_units: int
    guide_hold_seconds: float
    scout_interval: float
    refresh_threshold: float
    note: str


GUIDANCE_PRESETS: List[StrategyPreset] = [
    StrategyPreset(
        name="no_guidance_moving_blue",
        strategy="direct_salvo",
        salvo_size=12,
        salvo_interval=20.0,
        move_interval=30.0,
        guide_interval=30.0,
        guide_units=1,
        guide_hold_seconds=0.0,
        scout_interval=25.0,
        refresh_threshold=25.0,
        note="蓝方持续机动，无引导条件下直接低成本齐射，用来观察纯预测效果。",
    ),
    StrategyPreset(
        name="with_guidance_moving_blue",
        strategy="guide_then_salvo",
        salvo_size=12,
        salvo_interval=20.0,
        move_interval=30.0,
        guide_interval=15.0,
        guide_units=4,
        guide_hold_seconds=12.0,
        scout_interval=25.0,
        refresh_threshold=25.0,
        note="蓝方持续机动，四艘引导船并行建立引导后再低成本齐射。",
    ),
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _scene_python() -> str:
    return str(Path(r"C:\Users\Tk\.conda\envs\scene\python.exe"))


def _latest_file(root: Path, pattern: str) -> Optional[Path]:
    matches = list(root.glob(pattern))
    if not matches:
        return None
    matches.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return matches[0]


def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _parse_red_battle_metrics(red_battle_log: Path) -> Dict[str, object]:
    rows = _load_jsonl(red_battle_log)
    lowcost_launch_count = 0
    highcost_launch_count = 0
    score_events = []
    final_breakdown = {}
    for row in rows:
        row_type = str(row.get("type", ""))
        if row_type == "DECISION":
            for action in row.get("actions", []) or []:
                if not isinstance(action, dict):
                    continue
                if str(action.get("Type", "")) != "Launch":
                    continue
                weapon_type = str(action.get("WeaponType", ""))
                if weapon_type == "LowCostAttackMissile":
                    lowcost_launch_count += 1
                elif weapon_type == "HighCostAttackMissile":
                    highcost_launch_count += 1
        elif row_type == "SCORE_CHANGED":
            extra = row.get("extra", {}) or {}
            score_events.append(
                {
                    "sim_time": row.get("sim_time"),
                    "delta": extra.get("delta"),
                    "new_score": extra.get("new_score"),
                }
            )
            final_breakdown = extra.get("score_breakdown", final_breakdown) or final_breakdown
        elif row_type == "SCORE_SNAPSHOT":
            extra = row.get("extra", {}) or {}
            if extra.get("stage") == "battle_end":
                final_breakdown = extra.get("score_breakdown", final_breakdown) or final_breakdown
    return {
        "red_lowcost_launch_count": lowcost_launch_count,
        "red_highcost_launch_count": highcost_launch_count,
        "score_event_count": len(score_events),
        "red_final_score": final_breakdown.get("redScore"),
        "red_destroy_score": final_breakdown.get("redDestroyScore"),
        "red_cost": final_breakdown.get("redCost"),
        "blue_final_score": final_breakdown.get("blueScore"),
        "blue_destroy_score": final_breakdown.get("blueDestroyScore"),
        "blue_cost": final_breakdown.get("blueCost"),
    }


def _parse_runner_result(stdout_text: str) -> Dict[str, object]:
    for line in reversed((stdout_text or "").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return {}


def _run_preset(
    preset: StrategyPreset,
    end_simtime: int,
    engine_host: str,
    step_interval: float,
    matrix_root: Path,
    oneshot_debug: bool,
    diagnostic_interval_seconds: float,
    desired_launches: int,
    post_launch_tail_seconds: float,
    blue_move_interval: float,
    blue_move_offset_lon: float,
    blue_move_offset_lat: float,
) -> Dict[str, object]:
    repo_root = _repo_root()
    run_root = matrix_root / preset.name
    run_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["RUN_OUTPUT_ROOT"] = str(run_root)
    env["APP_ENABLE_CONSOLE_LOG"] = "0"
    if oneshot_debug:
        env["JSQLSIM_ONESHOT_DEBUG"] = "1"
    else:
        env.pop("JSQLSIM_ONESHOT_DEBUG", None)

    cmd = [
        _scene_python(),
        str(repo_root / "run_lowcost_rule_diagnostic.py"),
        "--end-simtime",
        str(end_simtime),
        "--engine-host",
        engine_host,
        "--step-interval",
        str(step_interval),
        "--strategy",
        preset.strategy,
        "--salvo-size",
        str(preset.salvo_size),
        "--salvo-interval",
        str(preset.salvo_interval),
        "--move-interval",
        str(preset.move_interval),
        "--guide-interval",
        str(preset.guide_interval),
        "--guide-units",
        str(preset.guide_units),
        "--guide-hold-seconds",
        str(preset.guide_hold_seconds),
        "--scout-interval",
        str(preset.scout_interval),
        "--refresh-threshold",
        str(preset.refresh_threshold),
        "--desired-launches",
        str(desired_launches),
        "--post-launch-tail-seconds",
        str(post_launch_tail_seconds),
        "--diagnostic-interval-seconds",
        str(diagnostic_interval_seconds),
        "--blue-move-interval",
        str(blue_move_interval),
        "--blue-move-offset-lon",
        str(blue_move_offset_lon),
        "--blue-move-offset-lat",
        str(blue_move_offset_lat),
    ]
    if oneshot_debug:
        cmd.append("--oneshot-debug")

    print(
        f"[matrix] running {preset.name}: strategy={preset.strategy} guide_units={preset.guide_units} "
        f"desired_launches={desired_launches} diag_interval={diagnostic_interval_seconds}s blue_move={blue_move_interval}s"
    )
    result = subprocess.run(cmd, cwd=str(repo_root), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (run_root / "stdout.log").write_text(result.stdout or "", encoding="utf-8")
    (run_root / "stderr.log").write_text(result.stderr or "", encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"preset {preset.name} failed with exit code {result.returncode}")

    runner_result = _parse_runner_result(result.stdout)
    red_diag = _latest_file(run_root / "diagnostics", "trajectory_RED_*.jsonl")
    blue_diag = _latest_file(run_root / "diagnostics", "trajectory_BLUE_*.jsonl")
    red_battle_log = _latest_file(run_root / "battle_logs", "battle_log_RED_*.jsonl")
    if red_diag is None or blue_diag is None or red_battle_log is None:
        raise RuntimeError(f"preset {preset.name} missing diagnostics or battle logs")

    trajectory_report = analyze_trajectory(str(red_diag), str(blue_diag))
    trajectory_output = run_root / "diagnostics" / "analysis_lowcost_trajectory.json"
    trajectory_output.write_text(json.dumps(trajectory_report, ensure_ascii=False, indent=2), encoding="utf-8")
    battle_metrics = _parse_red_battle_metrics(red_battle_log)

    summary = trajectory_report.get("summary", {}) or {}
    terminal = int(summary.get("entered_lowcost_terminal_window_count") or 0)
    missiles = int(summary.get("missile_count") or 0)
    row = {
        "status": "success",
        "preset": preset.name,
        "strategy": preset.strategy,
        "note": preset.note,
        "guide_units": preset.guide_units,
        "guide_hold_seconds": preset.guide_hold_seconds,
        "salvo_size": preset.salvo_size,
        "salvo_interval": preset.salvo_interval,
        "move_interval": preset.move_interval,
        "guide_interval": preset.guide_interval,
        "scout_interval": preset.scout_interval,
        "refresh_threshold": preset.refresh_threshold,
        "desired_launches": desired_launches,
        "post_launch_tail_seconds": post_launch_tail_seconds,
        "diagnostic_interval_seconds": diagnostic_interval_seconds,
        "blue_move_interval": blue_move_interval,
        "blue_move_offset_lon": blue_move_offset_lon,
        "blue_move_offset_lat": blue_move_offset_lat,
        "stop_reason": runner_result.get("stop_reason"),
        "diagnostic_lowcost_launches": runner_result.get("lowcost_launches"),
        "last_lowcost_launch_time": runner_result.get("last_lowcost_launch_time"),
        "missile_count": missiles,
        "entered_lowcost_terminal_window_count": terminal,
        "entered_lowcost_terminal_window_rate": round(terminal / missiles, 4) if missiles else 0.0,
        "near_miss_predictor_tunable_count": int(summary.get("near_miss_predictor_tunable_count") or 0),
        "median_closest_approach_m": summary.get("median_closest_approach_m"),
        "mean_closest_approach_m": summary.get("mean_closest_approach_m"),
        "median_launch_observation_error_m": summary.get("median_launch_observation_error_m"),
        "assessment": trajectory_report.get("assessment", ""),
        "run_root": str(run_root),
        "red_diag": str(red_diag),
        "blue_diag": str(blue_diag),
        "red_battle_log": str(red_battle_log),
    }
    row.update(battle_metrics)
    return row


def _write_outputs(matrix_root: Path, rows: List[Dict[str, object]]) -> None:
    summary_json = matrix_root / "matrix_summary.json"
    summary_csv = matrix_root / "matrix_summary.csv"
    summary_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_rank(rows: List[Dict[str, object]]) -> None:
    successful_rows = [row for row in rows if str(row.get("status", "")) == "success"]
    if not successful_rows:
        print("\n[matrix] no successful runs to rank")
        return
    ranked = sorted(
        successful_rows,
        key=lambda row: (
            -float(row.get("entered_lowcost_terminal_window_rate") or 0.0),
            -float(row.get("red_final_score") or 0.0),
            float(row.get("median_closest_approach_m") or 1e12),
        ),
    )
    print("\n[matrix] ranking")
    for idx, row in enumerate(ranked, start=1):
        print(
            f"  {idx}. {row['preset']}: term_rate={row['entered_lowcost_terminal_window_rate']:.2%} "
            f"missiles={row['missile_count']} median_close={row['median_closest_approach_m']} "
            f"red_score={row['red_final_score']} red_destroy={row['red_destroy_score']} blue_cost={row['blue_cost']}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pure-rule low-cost diagnostics in two scenarios: no guidance vs guidance.")
    parser.add_argument("--end-simtime", type=int, default=1200, dest="end_simtime")
    parser.add_argument("--engine-host", type=str, default="127.0.0.1", dest="engine_host")
    parser.add_argument("--step-interval", type=float, default=0.1, dest="step_interval")
    parser.add_argument("--presets", nargs="*", default=None, help="Optional subset of preset names.")
    parser.add_argument("--diagnostic-interval-seconds", type=float, default=3.0, dest="diagnostic_interval_seconds")
    parser.add_argument("--desired-launches", type=int, default=18, dest="desired_launches")
    parser.add_argument("--post-launch-tail-seconds", type=float, default=120.0, dest="post_launch_tail_seconds")
    parser.add_argument("--blue-move-interval", type=float, default=20.0, dest="blue_move_interval")
    parser.add_argument("--blue-move-offset-lon", type=float, default=0.05, dest="blue_move_offset_lon")
    parser.add_argument("--blue-move-offset-lat", type=float, default=0.035, dest="blue_move_offset_lat")
    parser.add_argument("--oneshot-debug", action="store_true", dest="oneshot_debug")
    return parser


def main() -> int:
    args = _parser().parse_args()
    presets = GUIDANCE_PRESETS
    if args.presets:
        wanted = {name.strip() for name in args.presets if str(name).strip()}
        presets = [preset for preset in GUIDANCE_PRESETS if preset.name in wanted]
        if not presets:
            raise SystemExit(f"No valid presets selected. Available: {[preset.name for preset in GUIDANCE_PRESETS]}")

    matrix_root = _repo_root() / "test" / "lowcost_guidance_matrix" / datetime.now().strftime("%Y%m%d_%H%M%S")
    matrix_root.mkdir(parents=True, exist_ok=True)
    print(f"[matrix] output_root={matrix_root}")

    rows: List[Dict[str, object]] = []
    for preset in presets:
        try:
            row = _run_preset(
                preset=preset,
                end_simtime=args.end_simtime,
                engine_host=args.engine_host,
                step_interval=args.step_interval,
                matrix_root=matrix_root,
                oneshot_debug=args.oneshot_debug,
                diagnostic_interval_seconds=args.diagnostic_interval_seconds,
                desired_launches=args.desired_launches,
                post_launch_tail_seconds=args.post_launch_tail_seconds,
                blue_move_interval=args.blue_move_interval,
                blue_move_offset_lon=args.blue_move_offset_lon,
                blue_move_offset_lat=args.blue_move_offset_lat,
            )
            rows.append(row)
            print(
                f"[matrix] done {preset.name}: term_rate={row['entered_lowcost_terminal_window_rate']:.2%} "
                f"red_score={row['red_final_score']} launches={row['red_lowcost_launch_count']} stop={row['stop_reason']}"
            )
        except Exception as exc:
            rows.append(
                {
                    "status": "failed",
                    "preset": preset.name,
                    "strategy": preset.strategy,
                    "note": preset.note,
                    "error": str(exc),
                }
            )
            print(f"[matrix] failed {preset.name}: {exc}")

    _write_outputs(matrix_root, rows)
    _print_rank(rows)
    print(f"\n[matrix] saved={matrix_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
