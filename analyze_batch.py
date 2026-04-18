import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from battle_metrics import extract_run_battle_metrics
from runtime_paths import project_root


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    try:
        return int(parsed)
    except Exception:
        return None


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_rows_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _latest_batch_root() -> Optional[Path]:
    batches_dir = Path(project_root()) / "test" / "batches"
    if not batches_dir.exists():
        return None
    candidates = [item for item in batches_dir.iterdir() if item.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _read_batch_rows(batch_root: Path) -> List[dict]:
    summary_csv = batch_root / "batch_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"batch_summary.csv not found under {batch_root}")

    rows: List[dict] = []
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row = dict(raw)
            battle_logs_dir_raw = str(row.get("battle_logs_dir") or "").strip()
            battle_logs_dir = Path(battle_logs_dir_raw) if battle_logs_dir_raw else None
            backfilled_metrics = {}
            if battle_logs_dir and battle_logs_dir.exists():
                red_log = next(iter(sorted(battle_logs_dir.glob("battle_log_RED_*.jsonl"))), None)
                blue_log = next(iter(sorted(battle_logs_dir.glob("battle_log_BLUE_*.jsonl"))), None)
                backfilled_metrics = extract_run_battle_metrics(red_log, blue_log)
            row["wall_clock_seconds"] = _safe_float(row.get("wall_clock_seconds"))
            row["max_red_simtime"] = _safe_int(row.get("max_red_simtime"))
            row["max_blue_simtime"] = _safe_int(row.get("max_blue_simtime"))
            row["first_red_action_simtime"] = _safe_int(row.get("first_red_action_simtime"))
            row["red_submitted_action_count"] = _safe_int(row.get("red_submitted_action_count")) or 0
            row["red_trace_count"] = _safe_int(row.get("red_trace_count")) or 0
            row["red_llm_call_count"] = _safe_int(row.get("red_llm_call_count")) or 0
            row["red_unique_targets_detected"] = _safe_int(row.get("red_unique_targets_detected"))
            row["blue_unique_targets_detected"] = _safe_int(row.get("blue_unique_targets_detected"))
            row["red_high_value_targets_detected"] = _safe_int(row.get("red_high_value_targets_detected"))
            row["blue_high_value_targets_detected"] = _safe_int(row.get("blue_high_value_targets_detected"))
            row["red_units_lost"] = _safe_int(row.get("red_units_lost"))
            row["blue_units_lost"] = _safe_int(row.get("blue_units_lost"))
            row["red_targets_destroyed"] = _safe_int(row.get("red_targets_destroyed"))
            row["blue_targets_destroyed"] = _safe_int(row.get("blue_targets_destroyed"))
            row["red_high_value_targets_destroyed"] = _safe_int(row.get("red_high_value_targets_destroyed"))
            row["blue_high_value_targets_destroyed"] = _safe_int(row.get("blue_high_value_targets_destroyed"))
            row["red_intercept_launch_count"] = _safe_int(row.get("red_intercept_launch_count"))
            row["blue_intercept_launch_count"] = _safe_int(row.get("blue_intercept_launch_count"))
            row["red_attack_launch_count"] = _safe_int(row.get("red_attack_launch_count"))
            row["blue_attack_launch_count"] = _safe_int(row.get("blue_attack_launch_count"))
            row["red_score_gain_total"] = _safe_float(row.get("red_score_gain_total"))
            row["blue_score_gain_total"] = _safe_float(row.get("blue_score_gain_total"))
            row["red_positive_score_events"] = _safe_int(row.get("red_positive_score_events"))
            row["blue_positive_score_events"] = _safe_int(row.get("blue_positive_score_events"))
            row["final_score_red"] = _safe_float(row.get("final_score_red"))
            row["final_score_blue"] = _safe_float(row.get("final_score_blue"))
            for key, default in (
                ("first_red_action_simtime", None),
                ("red_submitted_action_count", 0),
                ("red_unique_targets_detected", 0),
                ("blue_unique_targets_detected", 0),
                ("red_high_value_targets_detected", 0),
                ("blue_high_value_targets_detected", 0),
                ("red_units_lost", 0),
                ("blue_units_lost", 0),
                ("red_targets_destroyed", 0),
                ("blue_targets_destroyed", 0),
                ("red_high_value_targets_destroyed", 0),
                ("blue_high_value_targets_destroyed", 0),
                ("red_intercept_launch_count", 0),
                ("blue_intercept_launch_count", 0),
                ("red_attack_launch_count", 0),
                ("blue_attack_launch_count", 0),
                ("red_score_gain_total", 0.0),
                ("blue_score_gain_total", 0.0),
                ("red_positive_score_events", 0),
                ("blue_positive_score_events", 0),
            ):
                if row.get(key) is None:
                    row[key] = backfilled_metrics.get(key, default)
            if row.get("first_red_action_simtime") is None:
                row["first_red_action_simtime"] = backfilled_metrics.get("first_red_action_simtime")
            backfilled_submitted = int(backfilled_metrics.get("red_submitted_action_count") or 0)
            if backfilled_submitted > int(row.get("red_submitted_action_count") or 0):
                row["red_submitted_action_count"] = backfilled_submitted
            if row.get("final_score_red") is None:
                row["final_score_red"] = backfilled_metrics.get("final_score_red")
            if row.get("final_score_blue") is None:
                row["final_score_blue"] = backfilled_metrics.get("final_score_blue")
            row["timeout"] = str(row.get("timeout", "")).strip().lower() == "true"
            row["main_log"] = row.get("main_log") or ""
            rows.append(row)
    return rows


def _parse_reflection_timeout_from_log(log_path: Path) -> dict:
    result = {
        "configured_timeout_red": None,
        "configured_timeout_blue": None,
        "timeouts_red": 0,
        "timeouts_blue": 0,
        "done_red": 0,
        "done_blue": 0,
        "errors_red": 0,
        "errors_blue": 0,
    }
    if not log_path.exists():
        return result

    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if "reflection_start" in line and "timeout=" in line:
            if "[RED]" in line:
                result["configured_timeout_red"] = _safe_float(line.rsplit("timeout=", 1)[-1].rstrip("s"))
            elif "[BLUE]" in line:
                result["configured_timeout_blue"] = _safe_float(line.rsplit("timeout=", 1)[-1].rstrip("s"))
        if "[RED]" in line and "reflection_timeout" in line:
            result["timeouts_red"] += 1
        elif "[BLUE]" in line and "reflection_timeout" in line:
            result["timeouts_blue"] += 1
        elif "[RED]" in line and "reflection_done" in line:
            result["done_red"] += 1
        elif "[BLUE]" in line and "reflection_done" in line:
            result["done_blue"] += 1
        elif "[RED]" in line and "reflection_error" in line:
            result["errors_red"] += 1
        elif "[BLUE]" in line and "reflection_error" in line:
            result["errors_blue"] += 1
    return result


def _mean(values: List[float]) -> Optional[float]:
    cleaned = [value for value in values if value is not None]
    if not cleaned:
        return None
    return statistics.mean(cleaned)


def _median(values: List[float]) -> Optional[float]:
    cleaned = [value for value in values if value is not None]
    if not cleaned:
        return None
    return statistics.median(cleaned)


def _min(values: List[float]) -> Optional[float]:
    cleaned = [value for value in values if value is not None]
    if not cleaned:
        return None
    return min(cleaned)


def _max(values: List[float]) -> Optional[float]:
    cleaned = [value for value in values if value is not None]
    if not cleaned:
        return None
    return max(cleaned)


def _rank_rows(rows: List[dict], key: str, reverse: bool = True, top_k: int = 3) -> List[dict]:
    sortable = [row for row in rows if row.get(key) is not None]
    ordered = sorted(sortable, key=lambda item: item.get(key), reverse=reverse)
    return ordered[:top_k]


def _build_summary(batch_root: Path, rows: List[dict]) -> dict:
    wall_clock = [row.get("wall_clock_seconds") for row in rows]
    first_action = [row.get("first_red_action_simtime") for row in rows]
    submitted = [row.get("red_submitted_action_count") for row in rows]
    traces = [row.get("red_trace_count") for row in rows]
    llm_calls = [row.get("red_llm_call_count") for row in rows]
    red_detected = [row.get("red_unique_targets_detected") for row in rows]
    blue_detected = [row.get("blue_unique_targets_detected") for row in rows]
    red_destroyed = [row.get("red_targets_destroyed") for row in rows]
    blue_destroyed = [row.get("blue_targets_destroyed") for row in rows]
    red_hv_destroyed = [row.get("red_high_value_targets_destroyed") for row in rows]
    blue_hv_destroyed = [row.get("blue_high_value_targets_destroyed") for row in rows]
    red_intercepts = [row.get("red_intercept_launch_count") for row in rows]
    blue_intercepts = [row.get("blue_intercept_launch_count") for row in rows]
    red_score = [row.get("final_score_red") for row in rows]
    blue_score = [row.get("final_score_blue") for row in rows]

    reflection_diagnostics = {
        "configured_timeout_red": [],
        "configured_timeout_blue": [],
        "timeouts_red": 0,
        "timeouts_blue": 0,
        "done_red": 0,
        "done_blue": 0,
        "errors_red": 0,
        "errors_blue": 0,
    }
    for row in rows:
        main_log = Path(row["main_log"]) if row.get("main_log") else None
        if not main_log or not main_log.exists():
            continue
        parsed = _parse_reflection_timeout_from_log(main_log)
        for side in ("red", "blue"):
            configured = parsed.get(f"configured_timeout_{side}")
            if configured is not None:
                reflection_diagnostics[f"configured_timeout_{side}"].append(configured)
            reflection_diagnostics[f"timeouts_{side}"] += parsed.get(f"timeouts_{side}", 0)
            reflection_diagnostics[f"done_{side}"] += parsed.get(f"done_{side}", 0)
            reflection_diagnostics[f"errors_{side}"] += parsed.get(f"errors_{side}", 0)

    stop_reason_counts: Dict[str, int] = {}
    exit_status_counts: Dict[str, int] = {}
    for row in rows:
        stop_reason = str(row.get("stop_reason") or "unknown")
        exit_status = str(row.get("exit_status") or "unknown")
        stop_reason_counts[stop_reason] = stop_reason_counts.get(stop_reason, 0) + 1
        exit_status_counts[exit_status] = exit_status_counts.get(exit_status, 0) + 1

    return {
        "batch_root": str(batch_root),
        "batch_id": batch_root.name,
        "runs_total": len(rows),
        "exit_status_counts": exit_status_counts,
        "stop_reason_counts": stop_reason_counts,
        "wall_clock_seconds": {
            "mean": _mean(wall_clock),
            "median": _median(wall_clock),
            "min": _min(wall_clock),
            "max": _max(wall_clock),
        },
        "first_red_action_simtime": {
            "mean": _mean(first_action),
            "median": _median(first_action),
            "min": _min(first_action),
            "max": _max(first_action),
        },
        "red_submitted_action_count": {
            "mean": _mean(submitted),
            "median": _median(submitted),
            "min": _min(submitted),
            "max": _max(submitted),
        },
        "red_trace_count": {
            "mean": _mean(traces),
            "median": _median(traces),
            "min": _min(traces),
            "max": _max(traces),
        },
        "red_llm_call_count": {
            "mean": _mean(llm_calls),
            "median": _median(llm_calls),
            "min": _min(llm_calls),
            "max": _max(llm_calls),
        },
        "red_unique_targets_detected": {
            "mean": _mean(red_detected),
            "median": _median(red_detected),
            "min": _min(red_detected),
            "max": _max(red_detected),
        },
        "blue_unique_targets_detected": {
            "mean": _mean(blue_detected),
            "median": _median(blue_detected),
            "min": _min(blue_detected),
            "max": _max(blue_detected),
        },
        "red_targets_destroyed": {
            "mean": _mean(red_destroyed),
            "median": _median(red_destroyed),
            "min": _min(red_destroyed),
            "max": _max(red_destroyed),
        },
        "blue_targets_destroyed": {
            "mean": _mean(blue_destroyed),
            "median": _median(blue_destroyed),
            "min": _min(blue_destroyed),
            "max": _max(blue_destroyed),
        },
        "red_high_value_targets_destroyed": {
            "mean": _mean(red_hv_destroyed),
            "median": _median(red_hv_destroyed),
            "min": _min(red_hv_destroyed),
            "max": _max(red_hv_destroyed),
        },
        "blue_high_value_targets_destroyed": {
            "mean": _mean(blue_hv_destroyed),
            "median": _median(blue_hv_destroyed),
            "min": _min(blue_hv_destroyed),
            "max": _max(blue_hv_destroyed),
        },
        "red_intercept_launch_count": {
            "mean": _mean(red_intercepts),
            "median": _median(red_intercepts),
            "min": _min(red_intercepts),
            "max": _max(red_intercepts),
        },
        "blue_intercept_launch_count": {
            "mean": _mean(blue_intercepts),
            "median": _median(blue_intercepts),
            "min": _min(blue_intercepts),
            "max": _max(blue_intercepts),
        },
        "final_score_red": {
            "mean": _mean(red_score),
            "median": _median(red_score),
            "min": _min(red_score),
            "max": _max(red_score),
        },
        "final_score_blue": {
            "mean": _mean(blue_score),
            "median": _median(blue_score),
            "min": _min(blue_score),
            "max": _max(blue_score),
        },
        "top_submitted_runs": [
            {
                "run_id": row.get("run_id"),
                "red_submitted_action_count": row.get("red_submitted_action_count"),
                "first_red_action_simtime": row.get("first_red_action_simtime"),
                "red_llm_call_count": row.get("red_llm_call_count"),
                "wall_clock_seconds": row.get("wall_clock_seconds"),
                "red_targets_destroyed": row.get("red_targets_destroyed"),
                "blue_targets_destroyed": row.get("blue_targets_destroyed"),
            }
            for row in _rank_rows(rows, "red_submitted_action_count", reverse=True)
        ],
        "lowest_submitted_runs": [
            {
                "run_id": row.get("run_id"),
                "red_submitted_action_count": row.get("red_submitted_action_count"),
                "first_red_action_simtime": row.get("first_red_action_simtime"),
                "red_llm_call_count": row.get("red_llm_call_count"),
                "wall_clock_seconds": row.get("wall_clock_seconds"),
                "red_targets_destroyed": row.get("red_targets_destroyed"),
                "blue_targets_destroyed": row.get("blue_targets_destroyed"),
            }
            for row in _rank_rows(rows, "red_submitted_action_count", reverse=False)
        ],
        "reflection": {
            "configured_timeout_red_mean": _mean(reflection_diagnostics["configured_timeout_red"]),
            "configured_timeout_blue_mean": _mean(reflection_diagnostics["configured_timeout_blue"]),
            "timeouts_red": reflection_diagnostics["timeouts_red"],
            "timeouts_blue": reflection_diagnostics["timeouts_blue"],
            "done_red": reflection_diagnostics["done_red"],
            "done_blue": reflection_diagnostics["done_blue"],
            "errors_red": reflection_diagnostics["errors_red"],
            "errors_blue": reflection_diagnostics["errors_blue"],
        },
    }


def _style_axes(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)


def _save_bar_chart(rows: List[dict], key: str, output_path: Path, title: str, ylabel: str, color: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_ids = [row["run_id"] for row in rows]
    values = [row.get(key) if row.get(key) is not None else 0 for row in rows]
    colors = [color if row.get(key) is not None else "#cfcfcf" for row in rows]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(run_ids, values, color=colors)
    _style_axes(ax, title, ylabel)
    ax.tick_params(axis="x", rotation=45)
    upper = max(values) if values else 1
    ax.set_ylim(0, upper * 1.18 if upper > 0 else 1)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + upper * 0.02, f"{value}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_scatter(rows: List[dict], x_key: str, y_key: str, output_path: Path, title: str, xlabel: str, ylabel: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    points = [(row.get(x_key), row.get(y_key), row["run_id"]) for row in rows if row.get(x_key) is not None and row.get(y_key) is not None]
    fig, ax = plt.subplots(figsize=(8, 6))
    if points:
        xs = [item[0] for item in points]
        ys = [item[1] for item in points]
        ax.scatter(xs, ys, color="#0f766e", s=60, alpha=0.85)
        for x_value, y_value, run_id in points:
            ax.annotate(run_id, (x_value, y_value), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_status_chart(rows: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "RED done": sum(1 for row in rows if row.get("reflection_status_red") == "done"),
        "RED timeout": sum(1 for row in rows if row.get("reflection_status_red") == "timeout"),
        "RED error": sum(1 for row in rows if row.get("reflection_status_red") == "error"),
        "BLUE done": sum(1 for row in rows if row.get("reflection_status_blue") == "done"),
        "BLUE timeout": sum(1 for row in rows if row.get("reflection_status_blue") == "timeout"),
        "BLUE error": sum(1 for row in rows if row.get("reflection_status_blue") == "error"),
    }
    labels = list(metrics.keys())
    values = list(metrics.values())
    colors = ["#2563eb", "#dc2626", "#f59e0b", "#0f766e", "#b91c1c", "#d97706"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colors)
    _style_axes(ax, "Reflection outcomes by side", "Runs")
    ax.tick_params(axis="x", rotation=30)
    upper = max(values) if values else 1
    ax.set_ylim(0, upper * 1.2 if upper > 0 else 1)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(upper * 0.02, 0.05), f"{value}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _format_metric(value: Optional[float], digits: int = 1) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int) or (isinstance(value, float) and math.isclose(value, round(value))):
        return str(int(round(value)))
    return f"{value:.{digits}f}"


def _build_html(summary: dict, charts: Dict[str, str], rows: List[dict]) -> str:
    reflection = summary["reflection"]
    configured_red = reflection.get("configured_timeout_red_mean")
    configured_blue = reflection.get("configured_timeout_blue_mean")
    if reflection.get("timeouts_red") == summary["runs_total"] and reflection.get("timeouts_blue") == summary["runs_total"]:
        if configured_red is not None and configured_blue is not None:
            reflection_note = (
                f"All runs timed out during reflection. Parsed logs show a configured timeout of about RED {configured_red:.1f}s / BLUE {configured_blue:.1f}s, so the likely bottleneck is timeout budget rather than a broken reflection pipeline."
            )
        else:
            reflection_note = (
                "All runs timed out during reflection. The reflection threads did start, so the more likely bottleneck is timeout budget rather than startup failure."
            )
    else:
        reflection_note = (
            "Reflection results are mixed across runs. Check each run's main log for reflection_start and reflection_done/timeout lines if you need exact attribution."
        )

    def _rows_to_html(items: List[dict]) -> str:
        lines = []
        for item in items:
            lines.append(
                "<tr>"
                f"<td>{item.get('run_id')}</td>"
                f"<td>{item.get('red_submitted_action_count')}</td>"
                f"<td>{item.get('first_red_action_simtime')}</td>"
                f"<td>{item.get('red_llm_call_count')}</td>"
                f"<td>{_format_metric(item.get('wall_clock_seconds'), 1)}</td>"
                f"<td>{item.get('red_targets_destroyed')}</td>"
                "</tr>"
            )
        return "\n".join(lines)

    top_rows = _rows_to_html(summary["top_submitted_runs"])
    low_rows = _rows_to_html(summary["lowest_submitted_runs"])
    per_run_rows = []
    for row in rows:
        per_run_rows.append(
            "<tr>"
            f"<td>{row.get('run_id')}</td>"
            f"<td>{row.get('exit_status')}</td>"
            f"<td>{row.get('stop_reason')}</td>"
            f"<td>{row.get('first_red_action_simtime')}</td>"
            f"<td>{row.get('red_submitted_action_count')}</td>"
            f"<td>{row.get('red_llm_call_count')}</td>"
            f"<td>{row.get('red_unique_targets_detected')}</td>"
            f"<td>{row.get('blue_intercept_launch_count')}</td>"
            f"<td>{row.get('red_targets_destroyed')}</td>"
            f"<td>{row.get('red_high_value_targets_destroyed')}</td>"
            f"<td>{_format_metric(row.get('final_score_red'), 1)}</td>"
            f"<td>{_format_metric(row.get('final_score_blue'), 1)}</td>"
            f"<td>{_format_metric(row.get('wall_clock_seconds'), 1)}</td>"
            f"<td>{row.get('reflection_status_red')}</td>"
            f"<td>{row.get('reflection_status_blue')}</td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Batch Analysis - {summary['batch_id']}</title>
  <style>
    body {{ font-family: 'Segoe UI', 'PingFang SC', sans-serif; margin: 24px; color: #1f2937; background: #f8fafc; }}
    h1, h2 {{ margin-bottom: 12px; }}
    .meta {{ color: #475569; margin-bottom: 20px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }}
    .card {{ background: white; border-radius: 12px; padding: 16px; box-shadow: 0 6px 18px rgba(15,23,42,0.08); }}
    .metric {{ font-size: 28px; font-weight: 700; margin: 6px 0; }}
    .label {{ color: #64748b; font-size: 14px; }}
    .note {{ background: #fff7ed; border-left: 4px solid #f97316; padding: 14px 16px; border-radius: 8px; margin-bottom: 24px; }}
    .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 20px; }}
    .chart-card {{ background: white; border-radius: 12px; padding: 16px; box-shadow: 0 6px 18px rgba(15,23,42,0.08); }}
    img {{ width: 100%; border-radius: 8px; background: #fff; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 6px 18px rgba(15,23,42,0.08); }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #e2e8f0; text-align: left; font-size: 14px; }}
    th {{ background: #e2e8f0; }}
    .two-col {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 20px; margin: 24px 0; }}
    .section {{ margin-top: 28px; }}
  </style>
</head>
<body>
  <h1>Batch Analysis: {summary['batch_id']}</h1>
  <div class="meta">Batch directory: {summary['batch_root']}</div>

  <div class="grid">
    <div class="card"><div class="label">Runs Completed</div><div class="metric">{summary['runs_total']}</div><div>exit={summary['exit_status_counts']}</div></div>
    <div class="card"><div class="label">Avg First RED Action</div><div class="metric">{_format_metric(summary['first_red_action_simtime']['mean'], 1)}</div><div>sim_time seconds</div></div>
    <div class="card"><div class="label">Avg RED Submitted Actions</div><div class="metric">{_format_metric(summary['red_submitted_action_count']['mean'], 1)}</div><div>per run</div></div>
    <div class="card"><div class="label">Avg RED LLM Calls</div><div class="metric">{_format_metric(summary['red_llm_call_count']['mean'], 1)}</div><div>per run</div></div>
    <div class="card"><div class="label">Avg RED Targets Detected</div><div class="metric">{_format_metric(summary['red_unique_targets_detected']['mean'], 1)}</div><div>unique targets</div></div>
    <div class="card"><div class="label">Avg BLUE Intercept Launches</div><div class="metric">{_format_metric(summary['blue_intercept_launch_count']['mean'], 1)}</div><div>from battle_log</div></div>
    <div class="card"><div class="label">Avg RED Targets Destroyed</div><div class="metric">{_format_metric(summary['red_targets_destroyed']['mean'], 1)}</div><div>enemy unit losses</div></div>
    <div class="card"><div class="label">Avg RED HV Targets Destroyed</div><div class="metric">{_format_metric(summary['red_high_value_targets_destroyed']['mean'], 1)}</div><div>flagship/cruiser/destroyer</div></div>
    <div class="card"><div class="label">Avg Wall Clock</div><div class="metric">{_format_metric(summary['wall_clock_seconds']['mean'], 1)}</div><div>seconds</div></div>
    <div class="card"><div class="label">RED Reflection Timeout Runs</div><div class="metric">{reflection['timeouts_red']}/{summary['runs_total']}</div><div>current batch</div></div>
  </div>

  <div class="note">{reflection_note}</div>

  <div class="charts">
    <div class="chart-card"><h2>First RED Action by Run</h2><img src="{charts['first_action']}" alt="first action chart" /></div>
    <div class="chart-card"><h2>Submitted RED Actions by Run</h2><img src="{charts['submitted_actions']}" alt="submitted actions chart" /></div>
    <div class="chart-card"><h2>RED LLM Calls by Run</h2><img src="{charts['llm_calls']}" alt="llm calls chart" /></div>
    <div class="chart-card"><h2>Wall Clock by Run</h2><img src="{charts['wall_clock']}" alt="wall clock chart" /></div>
    <div class="chart-card"><h2>RED Unique Targets Detected</h2><img src="{charts['red_detected']}" alt="red detected chart" /></div>
    <div class="chart-card"><h2>BLUE Intercept Launches</h2><img src="{charts['blue_intercepts']}" alt="blue intercept chart" /></div>
    <div class="chart-card"><h2>RED Targets Destroyed</h2><img src="{charts['red_destroyed']}" alt="red destroyed chart" /></div>
    <div class="chart-card"><h2>RED Final Score</h2><img src="{charts['red_score']}" alt="red score chart" /></div>
    <div class="chart-card"><h2>Submitted Actions vs LLM Calls</h2><img src="{charts['scatter']}" alt="scatter chart" /></div>
    <div class="chart-card"><h2>Reflection Outcomes</h2><img src="{charts['reflection_status']}" alt="reflection status chart" /></div>
  </div>

  <div class="section two-col">
    <div>
      <h2>Top 3 Active Runs</h2>
      <table>
        <thead><tr><th>Run</th><th>Submitted</th><th>First Action</th><th>LLM Calls</th><th>Wall Clock</th><th>Destroyed</th></tr></thead>
        <tbody>{top_rows}</tbody>
      </table>
    </div>
    <div>
      <h2>Bottom 3 Active Runs</h2>
      <table>
        <thead><tr><th>Run</th><th>Submitted</th><th>First Action</th><th>LLM Calls</th><th>Wall Clock</th><th>Destroyed</th></tr></thead>
        <tbody>{low_rows}</tbody>
      </table>
    </div>
  </div>

  <div class="section">
    <h2>Per-run Metrics</h2>
    <table>
      <thead>
        <tr>
          <th>Run</th><th>Exit</th><th>Stop Reason</th><th>First RED Action</th><th>Submitted RED Actions</th><th>RED LLM Calls</th><th>RED Detected</th><th>BLUE Intercepts</th><th>RED Destroyed</th><th>RED HV Destroyed</th><th>RED Score</th><th>BLUE Score</th><th>Wall Clock</th><th>RED Reflection</th><th>BLUE Reflection</th>
        </tr>
      </thead>
      <tbody>
        {''.join(per_run_rows)}
      </tbody>
    </table>
  </div>
</body>
</html>"""


def generate_batch_analysis(batch_root: Path) -> dict:
    batch_root = batch_root.resolve()
    rows = _read_batch_rows(batch_root)
    summary = _build_summary(batch_root, rows)

    analysis_dir = batch_root / "analysis"
    plots_dir = analysis_dir / "plots"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    _save_bar_chart(rows, "first_red_action_simtime", plots_dir / "first_red_action_simtime.png", "First RED action sim_time by run", "sim_time", "#2563eb")
    _save_bar_chart(rows, "red_submitted_action_count", plots_dir / "red_submitted_action_count.png", "RED submitted action count by run", "actions", "#0f766e")
    _save_bar_chart(rows, "red_llm_call_count", plots_dir / "red_llm_call_count.png", "RED LLM calls by run", "calls", "#9333ea")
    _save_bar_chart(rows, "wall_clock_seconds", plots_dir / "wall_clock_seconds.png", "Wall clock seconds by run", "seconds", "#ea580c")
    _save_bar_chart(rows, "red_unique_targets_detected", plots_dir / "red_unique_targets_detected.png", "RED unique targets detected by run", "targets", "#0f766e")
    _save_bar_chart(rows, "blue_intercept_launch_count", plots_dir / "blue_intercept_launch_count.png", "BLUE intercept launches by run", "launches", "#2563eb")
    _save_bar_chart(rows, "red_targets_destroyed", plots_dir / "red_targets_destroyed.png", "RED targets destroyed by run", "destroyed", "#dc2626")
    _save_bar_chart(rows, "final_score_red", plots_dir / "final_score_red.png", "RED final score by run", "score", "#7c3aed")
    _save_scatter(rows, "red_llm_call_count", "red_submitted_action_count", plots_dir / "submitted_vs_llm_calls.png", "Submitted RED actions vs RED LLM calls", "RED LLM calls", "Submitted RED actions")
    _save_status_chart(rows, plots_dir / "reflection_status.png")

    charts = {
        "first_action": "plots/first_red_action_simtime.png",
        "submitted_actions": "plots/red_submitted_action_count.png",
        "llm_calls": "plots/red_llm_call_count.png",
        "wall_clock": "plots/wall_clock_seconds.png",
        "red_detected": "plots/red_unique_targets_detected.png",
        "blue_intercepts": "plots/blue_intercept_launch_count.png",
        "red_destroyed": "plots/red_targets_destroyed.png",
        "red_score": "plots/final_score_red.png",
        "scatter": "plots/submitted_vs_llm_calls.png",
        "reflection_status": "plots/reflection_status.png",
    }
    html = _build_html(summary, charts, rows)
    (analysis_dir / "index.html").write_text(html, encoding="utf-8")
    _write_json(analysis_dir / "summary.json", summary)
    _write_rows_csv(analysis_dir / "batch_summary_backfilled.csv", rows)
    return {
        "batch_root": str(batch_root),
        "analysis_dir": str(analysis_dir),
        "html_path": str(analysis_dir / "index.html"),
        "summary_path": str(analysis_dir / "summary.json"),
    }


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate batch visualizations and summary report.")
    parser.add_argument("--batch-root", type=str, default="", dest="batch_root")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    batch_root = Path(args.batch_root).resolve() if args.batch_root else _latest_batch_root()
    if batch_root is None:
        raise FileNotFoundError("No batch directory found under test/batches")
    result = generate_batch_analysis(batch_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
