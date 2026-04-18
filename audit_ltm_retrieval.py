import argparse
import json
from pathlib import Path
from typing import Iterable, List

from runtime_paths import project_root


def _iter_run_roots(batch_root: Path) -> Iterable[Path]:
    for run_dir in sorted(batch_root.glob("run_*")):
        if run_dir.is_dir():
            yield run_dir


def _latest_red_trace_jsonl(run_root: Path) -> Path | None:
    logs_dir = run_root / "logs"
    matches = sorted(logs_dir.glob("red_trace_*.jsonl"), key=lambda item: item.stat().st_mtime)
    return matches[-1] if matches else None


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except Exception:
            continue
        if isinstance(data, dict):
            rows.append(data)
    return rows


def _extract_ltm_rows(run_root: Path) -> List[dict]:
    trace_path = _latest_red_trace_jsonl(run_root)
    if trace_path is None:
        return []
    rows: List[dict] = []
    for record in _read_jsonl(trace_path):
        trace_id = record.get("trace_id")
        sim_time = record.get("sim_time")
        for entry in (record.get("sections", {}) or {}).get("LTM", []) or []:
            rows.append(
                {
                    "run_id": run_root.name,
                    "trace_id": trace_id,
                    "sim_time": sim_time,
                    "ltm_query_preview": entry.get("ltm_query_preview", ""),
                    "retrieved_lesson_ids": entry.get("retrieved_lesson_ids", []),
                    "retrieved_lesson_scores": entry.get("retrieved_lesson_scores", []),
                    "retrieved_lesson_tags": entry.get("retrieved_lesson_tags", []),
                    "retrieved_lesson_actions": entry.get("retrieved_lesson_actions", []),
                    "retrieved_lessons": entry.get("retrieved_lessons", []),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit RED LTM retrieval records from red trace JSONL files.")
    parser.add_argument("--batch-root", required=True, help="Batch root or run root")
    args = parser.parse_args()

    root = Path(args.batch_root)
    if not root.is_absolute():
        root = (Path(project_root()) / args.batch_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root}")

    if root.name.startswith("run_"):
        run_roots = [root]
        output_dir = root / "analysis"
    else:
        run_roots = list(_iter_run_roots(root))
        output_dir = root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    for run_root in run_roots:
        rows.extend(_extract_ltm_rows(run_root))

    output_path = output_dir / "ltm_audit.json"
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(output_path))


if __name__ == "__main__":
    main()
