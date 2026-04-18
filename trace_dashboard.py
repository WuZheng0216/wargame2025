import argparse
import json
import mimetypes
import re
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse


BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "test" / "logs"
LLM_DIR = BASE_DIR / "test" / "llm_outputs"
STATIC_DIR = BASE_DIR / "trace_dashboard"
ROOT_STATIC_FILES = {
    "index.html": BASE_DIR / "dashboard_index.html",
    "styles.css": BASE_DIR / "styles.css",
    "app.js": BASE_DIR / "app.js",
}
SECTION_NAMES = ["STM", "LTM", "Graph Nodes", "LLM Calls", "Guard", "Parsed Actions", "Submit"]
NODE_ORDER = ["analyst", "commander", "allocator", "operator", "critic"]
FILE_RECORD_CACHE: Dict[str, dict] = {}
TIMING_FIELDS = [
    "total_ms",
    "llm_ms",
    "parse_ms",
    "prompt_build_ms",
    "postprocess_ms",
    "deterministic_translate_ms",
    "rule_check_ms",
]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "ok"}


def _parse_iso_epoch(value: Any) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except Exception:
        return 0.0


def _epoch_to_iso(value: Any) -> str:
    try:
        epoch = float(value)
    except Exception:
        return ""
    if epoch <= 0:
        return ""
    try:
        return datetime.fromtimestamp(epoch).isoformat()
    except Exception:
        return ""


def _strip_graph_node_suffix_extras(text: str) -> tuple[str, dict]:
    remaining = text
    extras: Dict[str, Any] = {}
    parsers = {
        "operator_mode": str,
        "operator_llm_skipped": _safe_bool,
        "prompt_build_ms": _safe_int,
        "postprocess_ms": _safe_int,
        "deterministic_translate_ms": _safe_int,
        "rule_check_ms": _safe_int,
    }
    matched = True
    while matched:
        matched = False
        for key, parser in parsers.items():
            suffix_match = re.search(rf"\s{key}=(?P<value>[^\s]+)$", remaining)
            if suffix_match:
                extras[key] = parser(suffix_match.group("value"))
                remaining = remaining[: suffix_match.start()].rstrip()
                matched = True
    return remaining, extras


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_key_value_entry(line: str) -> dict:
    key, sep, value = line.partition("=")
    if not sep:
        return {"raw": line}
    return {"key": key.strip(), "value": value.strip(), "raw": line}


def _parse_graph_node_entry(line: str) -> dict:
    match = re.match(
        r"node=(?P<node>\w+)\s+duration_ms=(?P<duration_ms>\d+)\s+status=(?P<status>\w+)\s+retry=(?P<retry>\d+)\s+output=(?P<output>.*)",
        line,
    )
    if not match:
        return {"raw": line}

    output_summary, extra = _strip_graph_node_suffix_extras(match.group("output"))

    return {
        "node": match.group("node"),
        "duration_ms": _safe_int(match.group("duration_ms")),
        "status": match.group("status"),
        "retry_count": _safe_int(match.group("retry")),
        "output_summary": output_summary,
        "raw": line,
        **extra,
    }


def _parse_llm_call_entry(line: str) -> dict:
    match = re.match(
        r"component=(?P<component>\S+)\s+sim_time=(?P<sim_time>\S+)\s+duration_ms=(?P<duration_ms>\S+)\s+"
        r"success=(?P<success>\S+)\s+prompt_chars=(?P<prompt_chars>\S+)\s+response_chars=(?P<response_chars>\S+)\s+"
        r"path=(?P<file_path>\S+)\s+summary=(?P<summary>.*)",
        line,
    )
    if not match:
        return {"raw": line}
    return {
        "component": match.group("component"),
        "sim_time": _safe_int(match.group("sim_time")),
        "duration_ms": _safe_int(match.group("duration_ms")),
        "success": _safe_bool(match.group("success")),
        "prompt_chars": _safe_int(match.group("prompt_chars")),
        "response_chars": _safe_int(match.group("response_chars")),
        "file_path": match.group("file_path"),
        "summary": match.group("summary"),
        "raw": line,
    }


def _parse_submit_entry(line: str) -> dict:
    match = re.match(
        r"submitted_action_count=(?P<count>\S+)\s+submitted_summary=(?P<summary>.*?)\s+note=(?P<note>.*)",
        line,
    )
    if not match:
        return {"raw": line}
    return {
        "submitted_action_count": _safe_int(match.group("count")),
        "submitted_summary": match.group("summary"),
        "note": match.group("note"),
        "raw": line,
    }


def _parse_guard_entry(line: str) -> dict:
    match = re.match(
        r"planned_at=(?P<planned_at>\S+)\s+now=(?P<now>\S+)\s+drop_count=(?P<drop_count>\S+)\s+"
        r"reasons=(?P<reasons>.*?)\s+force_replan=(?P<force_replan>\S+)$",
        line,
    )
    if not match:
        return {"raw": line}
    return {
        "planned_at": _safe_int(match.group("planned_at")),
        "now": _safe_int(match.group("now")),
        "drop_count": _safe_int(match.group("drop_count")),
        "drop_reason_dist": match.group("reasons"),
        "force_replan": _safe_bool(match.group("force_replan")),
        "raw": line,
    }


def _materialize_sections_from_raw(raw_sections: Dict[str, List[str]]) -> Dict[str, List[dict]]:
    sections: Dict[str, List[dict]] = {}
    for section_name in SECTION_NAMES:
        raw_entries = raw_sections.get(section_name, [])
        parsed_entries = []
        for line in raw_entries:
            if section_name == "Graph Nodes":
                parsed_entries.append(_parse_graph_node_entry(line))
            elif section_name == "LLM Calls":
                parsed_entries.append(_parse_llm_call_entry(line))
            elif section_name == "Submit":
                parsed_entries.append(_parse_submit_entry(line))
            elif section_name == "Guard":
                parsed_entries.append(_parse_guard_entry(line))
            else:
                parsed_entries.append(_parse_key_value_entry(line))
        sections[section_name] = parsed_entries
    return sections


def _parse_trace_text_file(path: Path) -> List[dict]:
    records: List[dict] = []
    current: Optional[dict] = None
    current_section: Optional[str] = None

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if line.startswith("TRACE "):
                match = re.match(
                    r"TRACE trace_id=(?P<trace_id>\S+)\s+sim_time=(?P<sim_time>\S+)\s+reason=(?P<reason>\S+)\s+"
                    r"status=(?P<status>\S+)\s+total_ms=(?P<total_ms>\S+)",
                    line,
                )
                if not match:
                    current = None
                    current_section = None
                    continue
                current = {
                    "trace_id": match.group("trace_id"),
                    "sim_time": _safe_int(match.group("sim_time")),
                    "reason": match.group("reason"),
                    "status": match.group("status"),
                    "total_duration_ms": _safe_int(match.group("total_ms")),
                    "sections_raw": {name: [] for name in SECTION_NAMES},
                    "errors": [],
                    "source_file": str(path),
                    "source_kind": "text_log",
                }
                current_section = None
                continue

            if current is None:
                continue

            if line.startswith("started_at="):
                current["started_at"] = line.split("=", 1)[1]
                continue

            if line == "Errors:":
                current_section = "Errors"
                continue

            if line.endswith(":") and line[:-1] in SECTION_NAMES:
                current_section = line[:-1]
                continue

            if line.startswith("END TRACE"):
                current["sections"] = _materialize_sections_from_raw(current["sections_raw"])
                records.append(current)
                current = None
                current_section = None
                continue

            if line.startswith("  - ") and current_section:
                payload = line[4:]
                if current_section == "Errors":
                    current["errors"].append(payload)
                else:
                    current["sections_raw"][current_section].append(payload)

    return records


def _load_jsonl_records(path: Path) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record["source_file"] = str(path)
            record["source_kind"] = "jsonl"
            records.append(record)
    return records


def _load_live_records(path: Path) -> List[dict]:
    if not path.exists():
        return []
    try:
        payload = _load_json(path)
    except Exception:
        return []
    traces = []
    for record in payload.get("traces", []):
        item = dict(record)
        item["source_file"] = str(path)
        item["source_kind"] = "live"
        item["live_updated_at"] = payload.get("updated_at", "")
        traces.append(item)
    return traces


def _load_cached_records(path: Path, loader) -> List[dict]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return []
    cache_key = str(path)
    signature = (stat.st_mtime_ns, stat.st_size)
    cached = FILE_RECORD_CACHE.get(cache_key)
    if cached and cached.get("signature") == signature:
        return cached.get("records", [])

    records = loader(path)
    source_mtime_epoch = float(stat.st_mtime)
    source_mtime_iso = _epoch_to_iso(source_mtime_epoch)
    for record in records:
        record.setdefault("source_mtime_epoch", source_mtime_epoch)
        record.setdefault("source_mtime_iso", source_mtime_iso)
    FILE_RECORD_CACHE[cache_key] = {
        "signature": signature,
        "records": records,
    }
    return records


def _load_llm_io(file_path: str, cache: Dict[str, dict]) -> Optional[dict]:
    if not file_path:
        return None
    if file_path in cache:
        return cache[file_path]

    path = Path(file_path)
    if not path.exists():
        candidate = LLM_DIR / path.name
        if candidate.exists():
            path = candidate
        else:
            cache[file_path] = None
            return None

    try:
        payload = _load_json(path)
    except Exception:
        cache[file_path] = None
        return None

    result = {
        "path": str(path),
        "model": payload.get("model"),
        "timestamp": payload.get("timestamp"),
        "success": payload.get("success"),
        "prompt": payload.get("prompt", ""),
        "raw_response": payload.get("raw_response", ""),
    }
    cache[file_path] = result
    return result


def _ensure_sections(record: dict) -> Dict[str, List[dict]]:
    sections = record.get("sections")
    if isinstance(sections, dict):
        return {name: list(sections.get(name, [])) for name in SECTION_NAMES}
    raw_sections = record.get("sections_raw", {})
    return _materialize_sections_from_raw(raw_sections)


def _accumulate_optional(current: Optional[int], value: Any) -> Optional[int]:
    if value is None:
        return current
    amount = _safe_int(value)
    if current is None:
        return amount
    return current + amount


def _resolve_started_at_epoch(record: dict) -> float:
    started_at_epoch = _parse_iso_epoch(record.get("started_at"))
    if started_at_epoch > 0:
        return started_at_epoch
    live_updated_epoch = _parse_iso_epoch(record.get("live_updated_at"))
    if live_updated_epoch > 0:
        return live_updated_epoch
    try:
        source_epoch = float(record.get("source_mtime_epoch", 0.0))
    except Exception:
        source_epoch = 0.0
    return source_epoch


def _resolve_display_started_at(record: dict) -> str:
    started_at = str(record.get("started_at") or "").strip()
    if started_at:
        return started_at
    live_updated_at = str(record.get("live_updated_at") or "").strip()
    if live_updated_at:
        return live_updated_at
    return str(record.get("source_mtime_iso") or "")


def _build_node_views(record: dict) -> List[dict]:
    sections = _ensure_sections(record)
    llm_cache: Dict[str, Optional[dict]] = {}
    node_map = {
        node: {
            "node": node,
            "status": "pending",
            "retry_count": 0,
            "duration_ms": None,
            "output_summary": "",
            "operator_mode": None,
            "operator_llm_skipped": None,
            "llm_calls": [],
            "prompt": "",
            "raw_response": "",
            "model": "",
            "prompt_file": "",
            "has_llm_io": False,
            "timing": {
                "total_ms": None,
                "llm_ms": 0,
                "parse_ms": 0,
                "prompt_build_ms": None,
                "postprocess_ms": None,
                "deterministic_translate_ms": None,
                "rule_check_ms": None,
            },
        }
        for node in NODE_ORDER
    }

    for entry in sections.get("Graph Nodes", []):
        node = entry.get("node")
        if node not in node_map:
            continue
        timing = node_map[node]["timing"]
        timing["total_ms"] = _accumulate_optional(timing.get("total_ms"), entry.get("duration_ms"))
        timing["prompt_build_ms"] = _accumulate_optional(timing.get("prompt_build_ms"), entry.get("prompt_build_ms"))
        timing["postprocess_ms"] = _accumulate_optional(timing.get("postprocess_ms"), entry.get("postprocess_ms"))
        timing["deterministic_translate_ms"] = _accumulate_optional(
            timing.get("deterministic_translate_ms"), entry.get("deterministic_translate_ms")
        )
        timing["rule_check_ms"] = _accumulate_optional(timing.get("rule_check_ms"), entry.get("rule_check_ms"))
        node_map[node].update(
            {
                "status": entry.get("status", "ok"),
                "retry_count": max(node_map[node].get("retry_count", 0), entry.get("retry_count", 0)),
                "duration_ms": timing.get("total_ms"),
                "output_summary": entry.get("output_summary", ""),
                "operator_mode": entry.get("operator_mode") or node_map[node].get("operator_mode"),
                "operator_llm_skipped": (
                    entry.get("operator_llm_skipped")
                    if entry.get("operator_llm_skipped") is not None
                    else node_map[node].get("operator_llm_skipped")
                ),
            }
        )

    for entry in sections.get("LLM Calls", []):
        component = entry.get("component", "")
        base_component = component.split(".", 1)[0]
        if base_component not in node_map:
            continue
        llm_io = _load_llm_io(entry.get("file_path", ""), llm_cache)
        llm_call = dict(entry)
        if llm_io is not None:
            llm_call["llm_io"] = llm_io
        node_map[base_component]["llm_calls"].append(llm_call)
        if component.endswith(".json_parse"):
            node_map[base_component]["timing"]["parse_ms"] += _safe_int(entry.get("duration_ms"))
        else:
            node_map[base_component]["timing"]["llm_ms"] += _safe_int(entry.get("duration_ms"))
        if component.endswith(".json_parse"):
            continue
        if llm_io is not None:
            node_map[base_component]["prompt"] = llm_io.get("prompt", "")
            node_map[base_component]["raw_response"] = llm_io.get("raw_response", "")
            node_map[base_component]["model"] = llm_io.get("model", "")
            node_map[base_component]["prompt_file"] = llm_io.get("path", "")
            node_map[base_component]["has_llm_io"] = True

    if node_map["critic"]["status"] == "pending":
        node_map["critic"]["status"] = "rule_only"
    if node_map["operator"]["status"] != "pending" and node_map["operator"]["operator_mode"] is None:
        node_map["operator"]["operator_mode"] = "deterministic"

    return [node_map[name] for name in NODE_ORDER]


def _sections_to_text_map(record: dict) -> Dict[str, List[str]]:
    text_map: Dict[str, List[str]] = {name: [] for name in SECTION_NAMES}
    raw_sections = record.get("sections_raw")
    if isinstance(raw_sections, dict):
        for name in SECTION_NAMES:
            text_map[name] = list(raw_sections.get(name, []))
        return text_map

    sections = _ensure_sections(record)
    for name in SECTION_NAMES:
        rendered = []
        for entry in sections.get(name, []):
            if "raw" in entry:
                rendered.append(entry["raw"])
            else:
                rendered.append(json.dumps(entry, ensure_ascii=False))
        text_map[name] = rendered
    return text_map


def _prepare_record(record: dict) -> dict:
    prepared = dict(record)
    prepared["started_at_epoch"] = _resolve_started_at_epoch(prepared)
    prepared["started_at"] = _resolve_display_started_at(prepared)
    prepared["sections"] = _ensure_sections(record)
    prepared["sections_text"] = _sections_to_text_map(record)
    prepared["node_views"] = _build_node_views(prepared)
    return prepared


def _sort_key(record: dict) -> tuple:
    return (
        float(record.get("started_at_epoch", 0.0)),
        _safe_int(record.get("sim_time")),
        str(record.get("trace_id", "")),
    )


def collect_trace_records() -> List[dict]:
    records_by_id: Dict[str, dict] = {}

    live_path = LOG_DIR / "red_trace_live.json"
    for record in _load_live_records(live_path):
        trace_id = record.get("trace_id")
        if trace_id:
            records_by_id[trace_id] = _prepare_record(record)

    for path in sorted(LOG_DIR.glob("red_trace_*.jsonl")):
        for record in _load_cached_records(path, _load_jsonl_records):
            trace_id = record.get("trace_id")
            if trace_id:
                records_by_id[trace_id] = _prepare_record(record)

    for path in sorted(LOG_DIR.glob("red_trace_*.log")):
        for record in _load_cached_records(path, _parse_trace_text_file):
            trace_id = record.get("trace_id")
            if not trace_id or trace_id in records_by_id:
                continue
            records_by_id[trace_id] = _prepare_record(record)

    return sorted(records_by_id.values(), key=_sort_key, reverse=True)


def collect_trace_overview() -> List[dict]:
    items = []
    for record in collect_trace_records():
        items.append(
            {
                "trace_id": record.get("trace_id"),
                "sim_time": record.get("sim_time"),
                "reason": record.get("reason"),
                "status": record.get("status"),
                "started_at": record.get("started_at", ""),
                "started_at_epoch": record.get("started_at_epoch", 0.0),
                "total_duration_ms": record.get("total_duration_ms"),
                "source_kind": record.get("source_kind", ""),
                "nodes": [
                    {
                        "node": view.get("node"),
                        "status": view.get("status"),
                        "duration_ms": view.get("duration_ms"),
                    }
                    for view in record.get("node_views", [])
                ],
            }
        )
    return items


def get_trace_detail(trace_id: str) -> Optional[dict]:
    for record in collect_trace_records():
        if record.get("trace_id") == trace_id:
            return record
    return None


class TraceDashboardHandler(BaseHTTPRequestHandler):
    server_version = "TraceDashboard/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/health":
            self._send_json({"ok": True})
            return
        if path == "/api/traces":
            items = collect_trace_overview()
            self._send_json({"items": items, "count": len(items)})
            return
        if path.startswith("/api/trace/"):
            trace_id = unquote(path.split("/api/trace/", 1)[1])
            detail = get_trace_detail(trace_id)
            if detail is None:
                self._send_json({"error": "trace_not_found", "trace_id": trace_id}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(detail)
            return
        self._serve_static(path)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _serve_static(self, request_path: str) -> None:
        relative_path = request_path.lstrip("/") or "index.html"
        file_path = ROOT_STATIC_FILES.get(relative_path)
        if file_path is None:
            file_path = (STATIC_DIR / relative_path).resolve()
            if STATIC_DIR.resolve() not in file_path.parents and file_path != STATIC_DIR.resolve():
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            if file_path.is_dir():
                file_path = file_path / "index.html"
            if not file_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
        content_type, _ = mimetypes.guess_type(str(file_path))
        with open(file_path, "rb") as f:
            data = f.read()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def run_server(host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), TraceDashboardHandler)
    print(f"Trace dashboard running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="View RED system2 trace data in a lightweight dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
