import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


ATTACK_WEAPONS = {
    "HighCostAttackMissile",
    "LowCostAttackMissile",
    "ShipToGround_CruiseMissile",
    "JDAM",
    "AIM_AirMissile",
}


INTERCEPT_LINE_RE = re.compile(
    r"^(?P<wall>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+\s+-INFO\s+.*?"
    r"rule_type=intercept\s+target_id=(?P<target>[A-Za-z0-9_\-]+)\s+reason=(?P<reason>\S+)"
)


@dataclass
class BattleEvent:
    wall_time: Optional[datetime]
    sim_time: Optional[int]
    event_type: str
    record: Dict[str, Any]


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


def _parse_sim(value: Any) -> Optional[int]:
    try:
        return int(float(value))
    except Exception:
        return None


def load_battle_events(log_path: Path) -> List[BattleEvent]:
    events: List[BattleEvent] = []
    for raw_line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        events.append(
            BattleEvent(
                wall_time=_parse_iso(record.get("time")),
                sim_time=_parse_sim(record.get("sim_time")),
                event_type=str(record.get("type") or ""),
                record=record,
            )
        )
    return events


def summarize_red_log(log_path: Path) -> Dict[str, Any]:
    events = load_battle_events(log_path)
    weapon_counts = Counter()
    launch_waves: List[Dict[str, Any]] = []
    score_changes: List[Dict[str, Any]] = []
    decision_count = 0

    for event in events:
        if event.event_type == "DECISION":
            decision_count += 1
            actions = event.record.get("actions") or []
            launches: List[Dict[str, Any]] = []
            for action in actions:
                if not isinstance(action, dict):
                    continue
                if str(action.get("Type") or "") != "Launch":
                    continue
                weapon = str(action.get("WeaponType") or "")
                if weapon in ATTACK_WEAPONS:
                    weapon_counts[weapon] += 1
                    launches.append(
                        {
                            "weapon": weapon,
                            "unit_id": action.get("Id"),
                            "lon": action.get("Lon"),
                            "lat": action.get("Lat"),
                            "alt": action.get("Alt"),
                        }
                    )
            if launches:
                wave_counter = Counter(item["weapon"] for item in launches)
                launch_waves.append(
                    {
                        "sim_time": event.sim_time,
                        "source": event.record.get("source"),
                        "launch_count": len(launches),
                        "by_weapon": dict(wave_counter),
                        "examples": launches[:4],
                    }
                )

        elif event.event_type == "SCORE_CHANGED":
            extra = event.record.get("extra") or {}
            try:
                old_score = float(extra.get("old_score"))
                new_score = float(extra.get("new_score"))
                delta = new_score - old_score
            except Exception:
                continue
            score_changes.append(
                {
                    "sim_time": event.sim_time,
                    "delta": delta,
                    "old_score": old_score,
                    "new_score": new_score,
                    "score_breakdown": extra.get("score_breakdown") or {},
                }
            )

    battle_start = events[0].wall_time if events else None
    battle_end = events[-1].wall_time if events else None
    final_snapshot = next(
        (
            ev.record.get("extra") or {}
            for ev in reversed(events)
            if ev.event_type == "SCORE_SNAPSHOT"
        ),
        {},
    )
    final_score_breakdown = final_snapshot.get("score_breakdown") or {}

    positive_scores = [item for item in score_changes if item["delta"] > 0]
    negative_scores = [item for item in score_changes if item["delta"] < 0]

    return {
        "battle_start": battle_start,
        "battle_end": battle_end,
        "decision_count": decision_count,
        "weapon_counts": dict(weapon_counts),
        "launch_waves": launch_waves,
        "score_changes": score_changes,
        "positive_score_events": positive_scores,
        "negative_score_events": negative_scores,
        "final_score_breakdown": final_score_breakdown,
    }


def parse_anti_missile_log(
    log_path: Path,
    start_time: Optional[datetime],
    end_time: Optional[datetime],
) -> Dict[str, Any]:
    if not log_path.exists():
        return {"events": [], "by_target_prefix": {}, "count": 0}

    start_filter = start_time - timedelta(seconds=5) if start_time else None
    end_filter = end_time + timedelta(seconds=60) if end_time else None
    events: List[Dict[str, Any]] = []

    for raw_line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = INTERCEPT_LINE_RE.search(raw_line)
        if not match:
            continue
        wall_time = datetime.strptime(match.group("wall"), "%Y-%m-%d %H:%M:%S")
        if start_filter and wall_time < start_filter:
            continue
        if end_filter and wall_time > end_filter:
            continue
        target_id = match.group("target")
        target_prefix = target_id.split("-")[0]
        events.append(
            {
                "wall_time": wall_time.isoformat(sep=" "),
                "sim_time": None,
                "target_id": target_id,
                "target_prefix": target_prefix,
                "reason": match.group("reason"),
            }
        )

    by_target_prefix = Counter(item["target_prefix"] for item in events)
    return {
        "events": events,
        "by_target_prefix": dict(by_target_prefix),
        "count": len(events),
    }


def build_assessment(red_summary: Dict[str, Any], anti_summary: Dict[str, Any]) -> List[str]:
    notes: List[str] = []
    low_count = red_summary["weapon_counts"].get("LowCostAttackMissile", 0)
    high_count = red_summary["weapon_counts"].get("HighCostAttackMissile", 0)
    positive_events = red_summary["positive_score_events"]
    final_breakdown = red_summary["final_score_breakdown"]
    low_intercepts = anti_summary["by_target_prefix"].get("LowCostAttackMissile", 0)
    high_intercepts = anti_summary["by_target_prefix"].get("HighCostAttackMissile", 0)

    if low_count:
        notes.append(
            f"本局红方发射低成本攻击弹 {low_count} 枚，高成本攻击弹 {high_count} 枚。"
        )
    if low_intercepts:
        notes.append(
            f"蓝方在本局窗口内记录到 {low_intercepts} 次针对低成本攻击弹的拦截动作，高成本弹拦截记录为 {high_intercepts} 次。"
        )
    if positive_events:
        total_gain = sum(item["delta"] for item in positive_events)
        notes.append(
            f"本局仅出现 {len(positive_events)} 次正向得分事件，累计正向得分 {total_gain:.1f}。"
        )
    if final_breakdown:
        red_destroy = final_breakdown.get("redDestroyScore")
        red_cost = final_breakdown.get("redCost")
        if red_destroy is not None and red_cost is not None:
            notes.append(
                f"终局得分构成为 redDestroyScore={red_destroy}、redCost={red_cost}，说明成本压力已经显著影响净得分。"
            )

    if low_count >= 20 and low_intercepts >= max(5, low_count // 4):
        notes.append(
            "低成本弹并非完全不可能命中机动目标，但在当前“机动目标 + 蓝方持续拦截/规避”条件下，它更像压迫和消耗武器，而不是稳定主杀伤手段。"
        )
    elif low_count >= 20 and len(positive_events) <= 1:
        notes.append(
            "低成本弹的投入与收益明显不成比例，当前更像是预测、时机和目标选择共同失配，而不是单次偶发失手。"
        )
    else:
        notes.append(
            "现有数据还不足以断言低成本弹完全无效，但已经足以说明它不适合在当前条件下作为远距主攻武器大量使用。"
        )
    return notes


def latest_file(directory: Path, pattern: str) -> Path:
    candidates = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No files matching {pattern} in {directory}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze low-cost missile effectiveness for a single run.")
    parser.add_argument("--red-log", type=Path, help="Path to battle_log_RED_*.jsonl")
    parser.add_argument("--anti-log", type=Path, default=Path(r"C:\Users\Tk\.conda\envs\scene\Lib\site-packages\logs\anti_missile.log"))
    parser.add_argument("--output", type=Path, help="Optional output JSON path")
    args = parser.parse_args()

    root = Path.cwd()
    red_log = args.red_log or latest_file(root / "test" / "battle_logs", "battle_log_RED_*.jsonl")
    red_summary = summarize_red_log(red_log)
    anti_summary = parse_anti_missile_log(
        args.anti_log,
        red_summary["battle_start"],
        red_summary["battle_end"],
    )
    report = {
        "red_log": str(red_log),
        "battle_window": {
            "start": red_summary["battle_start"].isoformat(sep=" ") if red_summary["battle_start"] else None,
            "end": red_summary["battle_end"].isoformat(sep=" ") if red_summary["battle_end"] else None,
        },
        "weapon_counts": red_summary["weapon_counts"],
        "score_changes": red_summary["score_changes"],
        "positive_score_events": red_summary["positive_score_events"],
        "negative_score_events": red_summary["negative_score_events"],
        "final_score_breakdown": red_summary["final_score_breakdown"],
        "launch_waves_preview": red_summary["launch_waves"][:12],
        "blue_intercept_summary": {
            "count": anti_summary["count"],
            "by_target_prefix": anti_summary["by_target_prefix"],
            "events_preview": anti_summary["events"][:20],
        },
        "assessment": build_assessment(red_summary, anti_summary),
    }

    if args.output:
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
