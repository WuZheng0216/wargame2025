import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from runtime_paths import project_root


DEFAULT_BATCH_ID = "paper_v2_20260403_104358"


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        return rows
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


def _normalize_lesson(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _lesson_key(row: dict) -> str:
    lesson_hash = str(row.get("lesson_hash", "")).strip()
    if lesson_hash:
        return lesson_hash
    normalized = str(row.get("normalized_lesson", "")).strip() or _normalize_lesson(row.get("lesson", ""))
    return normalized


def _extract_new_lessons(run_root: Path) -> List[dict]:
    knowledge_dir = run_root / "knowledge"
    before_rows = _read_jsonl(knowledge_dir / "lessons_before.jsonl")
    after_rows = _read_jsonl(knowledge_dir / "lessons_after.jsonl")
    before_keys = {_lesson_key(row) for row in before_rows if _lesson_key(row)}
    new_rows = []
    for row in after_rows:
        key = _lesson_key(row)
        if not key or key in before_keys:
            continue
        new_rows.append(row)
    return new_rows


def _archive_existing_files(output_path: Path, legacy_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = output_path.parent / "archive" / f"red_lessons_legacy_{timestamp}"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for path in (output_path, legacy_path):
        if path.exists():
            shutil.move(str(path), str(archive_dir / path.name))
    return archive_dir


def _enrich_row(row: dict, batch_id: str, run_id: str, run_written_at: str) -> dict:
    data = dict(row)
    battle_meta = dict(data.get("battle_meta") or {})
    battle_meta.setdefault("source_batch_id", batch_id)
    battle_meta.setdefault("source_run_id", run_id)
    battle_meta.setdefault("source_written_at", run_written_at)
    battle_meta.setdefault("source_mode", "latest_successful_batch_rebuild")
    data["battle_meta"] = battle_meta
    data["source_batch_id"] = batch_id
    data["source_run_id"] = run_id
    data["source_written_at"] = run_written_at
    data["source_mode"] = "latest_successful_batch_rebuild"
    return data


def _collect_clean_rows(batch_root: Path) -> List[dict]:
    rows: List[dict] = []
    seen = set()
    for run_dir in sorted(batch_root.glob("run_*")):
        run_summary = _read_json(run_dir / "run_summary.json")
        run_id = str(run_summary.get("run_id") or run_dir.name)
        run_written_at = str(run_summary.get("end_time") or run_summary.get("finished_at") or "")
        for row in _extract_new_lessons(run_dir):
            key = _lesson_key(row)
            if not key or key in seen:
                continue
            seen.add(key)
            rows.append(_enrich_row(row, batch_root.name, run_id, run_written_at))
    return rows


def rebuild_clean_store(batch_root: Path, output_path: Path, legacy_path: Path) -> Dict[str, object]:
    archive_dir = _archive_existing_files(output_path, legacy_path)
    clean_rows = _collect_clean_rows(batch_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in clean_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "batch_root": str(batch_root),
        "output_path": str(output_path),
        "archived_to": str(archive_dir),
        "row_count": len(clean_rows),
        "rebuilt_at": datetime.now().isoformat(),
    }
    (archive_dir / "rebuild_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild a clean RED lessons store from the latest successful batch.")
    parser.add_argument("--batch-root", default=f"test/batches/{DEFAULT_BATCH_ID}")
    args = parser.parse_args()

    root = Path(project_root())
    batch_root = (root / args.batch_root).resolve() if not Path(args.batch_root).is_absolute() else Path(args.batch_root)
    if not batch_root.exists():
        raise FileNotFoundError(f"Batch root not found: {batch_root}")

    output_path = root / "test" / "red_lessons_structured.jsonl"
    legacy_path = root / "test" / "red_reflections.jsonl"
    manifest = rebuild_clean_store(batch_root, output_path, legacy_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
