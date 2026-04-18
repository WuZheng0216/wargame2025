import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from reflection_agent import PostBattleReflectionAgent


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows: List[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _format_lessons_for_prompt(path: Path, side: str, max_lessons: int = 10) -> str:
    lessons = _load_jsonl(path)
    if not lessons:
        return f"{side.upper()} 方暂无历史 lessons。"
    formatted = []
    for lesson_data in lessons[-max_lessons:]:
        ltype = str(lesson_data.get("type", "General")).capitalize()
        ltext = str(lesson_data.get("lesson", "暂无细节。")).strip() or "暂无细节。"
        formatted.append(f"- [历史 {ltype} Lesson] {ltext}")
    return "\n".join(formatted)


def _reconstruct_output(before_path: Path, after_path: Path) -> Dict[str, List[dict]]:
    before_rows = _load_jsonl(before_path)
    after_rows = _load_jsonl(after_path)
    before_hashes = {
        str(row.get("lesson_hash") or row.get("lesson_id") or "")
        for row in before_rows
        if isinstance(row, dict)
    }
    added_rows = []
    for row in after_rows:
        if not isinstance(row, dict):
            continue
        row_key = str(row.get("lesson_hash") or row.get("lesson_id") or "")
        if row_key and row_key not in before_hashes:
            added_rows.append(row)
    parsed = {"successes": [], "failures": []}
    for row in added_rows:
        item = {
            "observation": row.get("observation", ""),
            "lesson": row.get("lesson", ""),
            "tags": row.get("tags", []),
        }
        if str(row.get("type", "")).lower() == "failure":
            parsed["failures"].append(item)
        else:
            parsed["successes"].append(item)
    return parsed


def _find_battle_log(run_root: Path, side: str) -> Path:
    battle_dir = run_root / "battle_logs"
    candidates = sorted(battle_dir.glob(f"battle_log_{side.upper()}_*.jsonl"))
    return candidates[0] if candidates else Path()


def _run_score(run_summary: dict, side: str) -> str:
    key = f"final_score_{side.lower()}"
    value = run_summary.get(key)
    return f"Final Score: {value}" if value is not None else "Final Score: unknown"


def _runtime_store_path(run_root: Path, side: str) -> Path:
    knowledge_dir = run_root / "knowledge"
    if side.lower() == "blue":
        return knowledge_dir / "blue_lessons_runtime.jsonl"
    return knowledge_dir / "red_lessons_runtime.jsonl"


def reconstruct_run(run_root: Path, side: str) -> bool:
    run_summary = _load_json(run_root / "run_summary.json")
    if not run_summary:
        return False

    battle_log_path = _find_battle_log(run_root, side)
    if not battle_log_path.exists():
        return False

    side_lower = side.lower()
    final_score = _run_score(run_summary, side_lower)
    before_path = run_root / "knowledge" / "lessons_before.jsonl"
    after_path = run_root / "knowledge" / "lessons_after.jsonl"
    old_lessons = _format_lessons_for_prompt(before_path, side_lower, max_lessons=10)
    reconstructed_output = _reconstruct_output(before_path, after_path) if before_path.exists() and after_path.exists() else None

    os.environ["RUN_OUTPUT_ROOT"] = str(run_root)
    if side_lower == "red":
        os.environ["RED_LTM_STORE_PATH"] = str(_runtime_store_path(run_root, side_lower))
    else:
        os.environ["BLUE_LTM_STORE_PATH"] = str(_runtime_store_path(run_root, side_lower))

    agent = PostBattleReflectionAgent(llm_model=None, knowledge_base_path=str(_runtime_store_path(run_root, side_lower)))
    rules_context = agent._load_rules_context()
    battle_digest = agent._generate_timeline(str(battle_log_path), final_score)
    prompt = agent._build_reflection_prompt(rules_context, old_lessons, battle_digest)
    agent._save_reflection_artifacts(
        event_log_path=str(battle_log_path),
        final_score=final_score,
        rules_context=rules_context,
        old_lessons=old_lessons,
        timeline=battle_digest,
        prompt=prompt,
        parsed_reflection=reconstructed_output,
        reconstructed=True,
        output_source="reconstructed_from_lessons_diff",
    )
    return True


def iter_run_roots(batch_root: Path) -> List[Path]:
    return sorted(
        [path for path in batch_root.iterdir() if path.is_dir() and path.name.startswith("run_")],
        key=lambda item: item.name,
    )


def main():
    parser = argparse.ArgumentParser(description="Reconstruct reflection prompt/output artifacts from historical runs.")
    parser.add_argument("--batch-root", type=str, default="", help="Batch root directory containing run_xxxx folders.")
    parser.add_argument("--run-root", type=str, default="", help="Single run root directory.")
    parser.add_argument("--side", type=str, default="red", choices=["red", "blue"], help="Which side to reconstruct.")
    args = parser.parse_args()

    targets: List[Path] = []
    if args.run_root:
        targets.append(Path(args.run_root).resolve())
    elif args.batch_root:
        targets.extend(iter_run_roots(Path(args.batch_root).resolve()))
    else:
        raise SystemExit("Please provide --run-root or --batch-root")

    success_count = 0
    for run_root in targets:
        if reconstruct_run(run_root, args.side):
            success_count += 1
            print(f"reconstructed {args.side} reflection artifacts for {run_root}")
        else:
            print(f"skipped {run_root}")
    print(f"done: {success_count}/{len(targets)}")


if __name__ == "__main__":
    main()
