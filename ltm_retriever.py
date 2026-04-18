import hashlib
import json
import logging
import math
import os
import re
from datetime import datetime
from typing import Dict, List, Optional

try:
    import jieba  # type: ignore
except Exception:
    jieba = None

try:
    import httpx  # type: ignore
except Exception:
    httpx = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
except Exception:
    TfidfVectorizer = None

from runtime_paths import project_root

logger = logging.getLogger(__name__)


def canonicalize_recommended_action(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    key = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    alias_map = {
        "focusfire": "FocusFire",
        "focus_fire": "FocusFire",
        "fire": "FocusFire",
        "fire_high_cost": "FocusFire",
        "high_cost_fire": "FocusFire",
        "fire_low_cost": "FocusFire",
        "low_cost_fire": "FocusFire",
        "shootandscoot": "ShootAndScoot",
        "shoot_and_scoot": "ShootAndScoot",
        "movetoengage": "MoveToEngage",
        "move_to_engage": "MoveToEngage",
        "move_then_fire": "MoveToEngage",
        "guideattack": "GuideAttack",
        "guide_attack": "GuideAttack",
        "guide": "GuideAttack",
        "guidance": "GuideAttack",
        "scoutarea": "ScoutArea",
        "scout_area": "ScoutArea",
        "scout": "ScoutArea",
        "recon": "ScoutArea",
        "refresh_track": "ScoutArea",
        "refresh_tracks": "ScoutArea",
        "track_refresh": "ScoutArea",
    }
    return alias_map.get(key) or alias_map.get(key.replace("_", "")) or raw


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        return int(raw)
    except Exception:
        logger.warning("Invalid %s=%s, fallback to %s", name, raw, default)
        return default


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        return float(raw)
    except Exception:
        logger.warning("Invalid %s=%s, fallback to %s", name, raw, default)
        return default


def _read_bool_env(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    logger.warning("Invalid %s=%s, fallback to %s", name, raw, default)
    return default


def _contains_cjk(text: str) -> bool:
    for ch in str(text or ""):
        if "\u4e00" <= ch <= "\u9fff":
            return True
    return False


class LongTermMemoryRetriever:
    def __init__(self, side: str):
        self.side = side.lower()
        self.top_k = _read_int_env("RED_LTM_TOPK", 5)
        self.alpha = _read_float_env("RED_LTM_HYBRID_ALPHA", 0.6)
        self.tokenizer_mode = str(os.getenv("RED_LTM_TOKENIZER", "hybrid")).strip().lower() or "hybrid"
        self.vector_backend = str(os.getenv("RED_LTM_VECTOR_BACKEND", "tfidf")).strip().lower() or "tfidf"
        self.cjk_ngram_min = max(1, _read_int_env("RED_LTM_CJK_NGRAM_MIN", 2))
        self.cjk_ngram_max = max(self.cjk_ngram_min, _read_int_env("RED_LTM_CJK_NGRAM_MAX", 3))
        self.embedding_endpoint = str(
            os.getenv("RED_LTM_EMBEDDING_ENDPOINT", os.getenv("TEI_MODEL_ENDPOINT", ""))
        ).strip()
        self.embedding_timeout_seconds = max(1.0, _read_float_env("RED_LTM_EMBEDDING_TIMEOUT_SECONDS", 20.0))
        legacy_default = False if self.side == "red" else True
        self.enable_legacy_fallback = _read_bool_env(
            f"{self.side.upper()}_LTM_ENABLE_LEGACY_FALLBACK",
            legacy_default,
        )
        self.filter_mojibake = _read_bool_env(f"{self.side.upper()}_LTM_FILTER_MOJIBAKE", True)

        current_dir = project_root()
        default_store = os.path.join(current_dir, "test", f"{self.side}_lessons_structured.jsonl")
        env_name = f"{self.side.upper()}_LTM_STORE_PATH"
        self.structured_store_path = os.getenv(env_name, default_store)
        if not os.path.isabs(self.structured_store_path):
            self.structured_store_path = os.path.join(current_dir, self.structured_store_path)
        self.legacy_reflection_path = os.path.join(current_dir, "test", f"{self.side}_reflections.jsonl")

        logger.info(
            "[LTM-%s] store=%s top_k=%s alpha=%.2f tokenizer=%s vector=%s jieba=%s tfidf=%s tei=%s legacy_fallback=%s",
            self.side.upper(),
            self.structured_store_path,
            self.top_k,
            self.alpha,
            self.tokenizer_mode,
            self.vector_backend,
            jieba is not None,
            TfidfVectorizer is not None,
            bool(self.embedding_endpoint),
            self.enable_legacy_fallback,
        )

    def retrieve_for_context(self, memory_packet: dict) -> List[dict]:
        query = self.build_query_for_context(memory_packet)
        if not query.strip():
            return []

        docs = self._load_documents()
        if not docs:
            return []

        query_vec = self._term_freq(self._tokenize(query))
        doc_texts = [self._doc_text(doc) for doc in docs]
        backend_scores = self._vector_scores(query, doc_texts)

        scored = []
        for idx, doc in enumerate(docs):
            text = doc_texts[idx]
            tokens = self._tokenize(text)
            if not tokens:
                continue

            keyword_score = self._keyword_score(query, doc, tokens)
            if backend_scores is not None:
                vector_score = backend_scores[idx]
            else:
                vector_score = self._cosine(query_vec, self._term_freq(tokens))

            hybrid = self.alpha * keyword_score + (1.0 - self.alpha) * vector_score
            item = dict(doc)
            item["hybrid_score"] = hybrid
            item["keyword_score"] = keyword_score
            item["vector_score"] = vector_score
            scored.append(item)

        scored.sort(key=lambda x: float(x.get("hybrid_score", 0.0)), reverse=True)
        return scored[: self.top_k]

    def build_query_for_context(self, memory_packet: dict) -> str:
        return self._build_query(memory_packet)

    def format_lessons_block(self, lessons: List[dict]) -> str:
        if not lessons:
            return "No relevant long-term lessons retrieved."

        lines = []
        for idx, item in enumerate(lessons, 1):
            lesson = str(item.get("lesson", "")).strip()
            typ = str(item.get("type", "general")).strip()
            score = float(item.get("hybrid_score", 0.0))
            lesson_id = str(item.get("lesson_id", ""))[:12] or "n/a"
            phase = str(item.get("phase", "")).strip() or "-"
            target_type = str(item.get("target_type", "")).strip() or "-"
            symptom = str(item.get("symptom", "")).strip() or "-"
            trigger = str(item.get("trigger", "")).strip() or "-"
            score_pattern = str(item.get("score_pattern", "")).strip() or "-"
            cost_risk = str(item.get("cost_risk", "")).strip() or "-"
            lines.append(
                f"{idx}. id={lesson_id} [{typ}] score={score:.3f} | phase={phase} | target={target_type} "
                f"| symptom={symptom} | trigger={trigger} | score_pattern={score_pattern} "
                f"| cost_risk={cost_risk} | lesson={lesson}"
            )
        return "\n".join(lines)

    def format_lessons_structured(self, lessons: List[dict]) -> List[dict]:
        formatted: List[dict] = []
        for item in lessons or []:
            if not isinstance(item, dict):
                continue
            formatted.append(
                {
                    "lesson_id": item.get("lesson_id"),
                    "type": item.get("type", "general"),
                    "phase": item.get("phase", ""),
                    "target_type": item.get("target_type", ""),
                    "symptom": item.get("symptom", ""),
                    "trigger": item.get("trigger", ""),
                    "lesson": item.get("lesson", ""),
                    "score_pattern": item.get("score_pattern", ""),
                    "cost_risk": item.get("cost_risk", ""),
                    "hybrid_score": float(item.get("hybrid_score", 0.0)),
                }
            )
        return formatted

    def _load_documents(self) -> List[dict]:
        docs = self._load_structured_lessons()
        if docs:
            return docs
        if self.enable_legacy_fallback:
            return self._load_legacy_reflections()
        return []

    def _load_structured_lessons(self) -> List[dict]:
        path = self.structured_store_path
        if not os.path.exists(path):
            return []

        out = []
        seen = set()
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except Exception:
                        continue
                    if self.filter_mojibake and not self._is_usable_structured_lesson(data):
                        continue
                    norm = self._normalize_lesson_text(data.get("lesson", ""))
                    if not norm or norm in seen:
                        continue
                    seen.add(norm)
                    out.append(
                        {
                            "lesson_id": data.get("lesson_id", self._make_lesson_id(norm)),
                            "type": data.get("type", "general"),
                            "observation": data.get("observation", ""),
                            "lesson": data.get("lesson", ""),
                            "tags": data.get("tags", []),
                            "phase": data.get("phase", ""),
                            "target_type": data.get("target_type", ""),
                            "symptom": data.get("symptom", ""),
                            "trigger": data.get("trigger", ""),
                            "score_pattern": data.get("score_pattern", ""),
                            "cost_risk": data.get("cost_risk", ""),
                            "battle_meta": data.get("battle_meta", {}),
                            "source": data.get("source", "structured"),
                            "created_at": data.get("created_at", ""),
                        }
                    )
        except Exception as e:
            logger.warning("Failed loading structured lessons: %s", e)
            return []
        return out

    def _is_usable_structured_lesson(self, data: dict) -> bool:
        lesson = str(data.get("lesson", "") or "")
        observation = str(data.get("observation", "") or "")
        combined = f"{lesson}\n{observation}".strip()
        if not combined:
            return False
        if "�" in combined:
            return False
        suspicious_chars = "æåçéèïâ€œ”€™¢£¥©®±¼½¾"
        suspicious_count = sum(combined.count(ch) for ch in suspicious_chars)
        if suspicious_count >= 4 and not _contains_cjk(combined):
            return False
        if suspicious_count > 0 and suspicious_count / max(1, len(combined)) > 0.06 and not _contains_cjk(combined):
            return False
        return True

    def _load_legacy_reflections(self) -> List[dict]:
        path = self.legacy_reflection_path
        if not os.path.exists(path):
            return []

        try:
            raw_text = open(path, "r", encoding="utf-8").read().strip()
        except Exception as e:
            logger.warning("Failed reading legacy reflections: %s", e)
            return []

        docs: List[dict] = []
        seen = set()

        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            lesson = data.get("lesson")
            if isinstance(lesson, str) and lesson.strip():
                norm = self._normalize_lesson_text(lesson)
                if norm in seen:
                    continue
                seen.add(norm)
                docs.append(
                    {
                        "lesson_id": self._make_lesson_id(norm),
                        "type": data.get("type", "legacy"),
                        "observation": data.get("observation", ""),
                        "lesson": lesson,
                        "tags": data.get("tags", []),
                        "phase": data.get("phase", ""),
                        "target_type": data.get("target_type", ""),
                        "symptom": data.get("symptom", ""),
                        "trigger": data.get("trigger", ""),
                        "score_pattern": data.get("score_pattern", ""),
                        "cost_risk": data.get("cost_risk", ""),
                        "battle_meta": data.get("battle_meta", {}),
                        "source": "legacy-line",
                        "created_at": data.get("created_at", ""),
                    }
                )

        if docs:
            return docs

        try:
            blob = json.loads(raw_text)
            strings = self._collect_long_strings(blob)
            for text in strings:
                norm = self._normalize_lesson_text(text)
                if norm in seen:
                    continue
                seen.add(norm)
                docs.append(
                    {
                        "lesson_id": self._make_lesson_id(norm),
                        "type": "legacy",
                        "observation": "",
                        "lesson": text,
                        "tags": [],
                        "battle_meta": {},
                        "source": "legacy-blob",
                        "created_at": "",
                    }
                )
        except Exception:
            pass

        return docs

    def _build_query(self, memory_packet: dict) -> str:
        current_snapshot = str(memory_packet.get("current_snapshot", ""))
        window_summary = str(memory_packet.get("window_summary", ""))
        event_timeline = memory_packet.get("event_timeline", []) or []
        recent_failures = memory_packet.get("recent_failures", []) or []
        score_summary = str(memory_packet.get("score_summary", ""))
        engagement_summary = str(memory_packet.get("engagement_summary", ""))
        battle_phase = str(memory_packet.get("battle_phase", ""))
        repeat_high_cost_targets = memory_packet.get("repeat_high_cost_targets", []) or []
        pending_bda_targets = memory_packet.get("pending_bda_targets", []) or []
        top_targets = memory_packet.get("top_targets", []) or []
        recent_attack_cost = memory_packet.get("recent_attack_cost", 0.0)
        snapshot_focus = self._compact_snapshot_for_query(current_snapshot)
        score_focus = self._compact_score_for_query(score_summary, current_snapshot)
        target_focus = self._compact_targets_for_query(top_targets)
        repeat_focus = ",".join([str(item) for item in repeat_high_cost_targets[:6]]) or "none"
        pending_bda_focus = ",".join([str(item) for item in pending_bda_targets[:6]]) or "none"

        event_text = "\n".join(
            [
                f"{event.get('sim_time')}: {event.get('name')} unit={event.get('unit_id')} target={event.get('target_id')}"
                for event in event_timeline[-20:]
            ]
        )
        fail_text = "\n".join(
            [
                f"{event.get('sim_time')}: {event.get('name')} unit={event.get('unit_id')} target={event.get('target_id')}"
                for event in recent_failures[-10:]
            ]
        )

        return (
            f"BattlePhase:\n{battle_phase}\n"
            f"ScoreSummary:\n{score_focus}\n"
            f"WindowSummary:\n{window_summary}\n"
            f"SnapshotFocus:\n{snapshot_focus}\n"
            f"TopTargets:\n{target_focus}\n"
            f"EngagementSummary:\n{engagement_summary}\n"
            f"RepeatHighCostTargets:\n{repeat_focus}\n"
            f"PendingBdaTargets:\n{pending_bda_focus}\n"
            f"RecentAttackCost:\n{recent_attack_cost}\n"
            f"RecentEvents:\n{event_text}\n"
            f"RecentFailures:\n{fail_text}"
        )

    def _doc_text(self, doc: dict) -> str:
        return (
            f"{doc.get('type', '')} "
            f"{doc.get('phase', '')} "
            f"{doc.get('target_type', '')} "
            f"{doc.get('symptom', '')} "
            f"{doc.get('trigger', '')} "
            f"{doc.get('score_pattern', '')} "
            f"{doc.get('cost_risk', '')} "
            f"{doc.get('lesson', '')} "
            f"{' '.join([str(tag) for tag in (doc.get('tags', []) or []) if str(tag).strip()])}"
        )

    def _keyword_score(self, query: str, doc: dict, doc_tokens: List[str]) -> float:
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return 0.0

        q_set = set(q_tokens)
        d_set = set(doc_tokens)
        overlap = len(q_set.intersection(d_set))
        base = overlap / max(len(q_set), 1)

        query_upper = query.upper()
        doc_type = str(doc.get("type", "")).upper()
        if ("FAIL" in query_upper or "失败" in query or "风险" in query) and (
            "FAIL" in doc_type or "FAILURE" in doc_type
        ):
            base += 0.1

        tags = doc.get("tags", [])
        if isinstance(tags, list):
            tag_tokens = set(self._tokenize(" ".join([str(tag) for tag in tags])))
            if tag_tokens:
                base += 0.1 * (len(q_set.intersection(tag_tokens)) / max(len(q_set), 1))

        return min(base, 1.0)

    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []

        lower_text = str(text).lower()
        ascii_tokens = re.findall(r"[a-z0-9_]+", lower_text)
        cjk_spans = re.findall(r"[\u4e00-\u9fff]+", lower_text)
        cjk_tokens: List[str] = []
        use_jieba = self.tokenizer_mode in {"jieba", "hybrid"} and jieba is not None

        if use_jieba:
            for span in cjk_spans:
                try:
                    cjk_tokens.extend([token.strip() for token in jieba.cut(span) if token and token.strip()])
                except Exception:
                    pass

        if self.tokenizer_mode in {"hybrid", "char", "jieba"}:
            for span in cjk_spans:
                if not span:
                    continue
                for n in range(self.cjk_ngram_min, self.cjk_ngram_max + 1):
                    if len(span) < n:
                        continue
                    for idx in range(0, len(span) - n + 1):
                        cjk_tokens.append(span[idx : idx + n])
                if not use_jieba:
                    cjk_tokens.extend(list(span))

        return [token for token in (ascii_tokens + cjk_tokens) if token and token.strip()]

    def _vector_scores(self, query: str, doc_texts: List[str]) -> Optional[List[float]]:
        if self.vector_backend == "tei":
            return self._tei_embedding_scores(query, doc_texts)
        if self.vector_backend != "tfidf" or TfidfVectorizer is None:
            return None

        tokenized_docs = [self._tokenize(text) for text in doc_texts]
        tokenized_query = self._tokenize(query)
        if not tokenized_query or not any(tokenized_docs):
            return None

        corpus = [" ".join(tokens) for tokens in tokenized_docs] + [" ".join(tokenized_query)]
        try:
            vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\S+\b")
            matrix = vectorizer.fit_transform(corpus)
            query_vec = matrix[-1]
            doc_matrix = matrix[:-1]
            return [float(score) for score in (doc_matrix @ query_vec.T).toarray().ravel().tolist()]
        except Exception as e:
            logger.warning("TF-IDF scoring failed, fallback to BOW cosine: %s", e)
            return None

    def _tei_embedding_scores(self, query: str, doc_texts: List[str]) -> Optional[List[float]]:
        if not self.embedding_endpoint or httpx is None:
            return None

        texts = [query] + doc_texts
        try:
            with httpx.Client(timeout=self.embedding_timeout_seconds, trust_env=False) as client:
                response = client.post(
                    self.embedding_endpoint.rstrip("/") + "/embed",
                    json={"inputs": texts},
                )
                response.raise_for_status()
                payload = response.json()
        except Exception as e:
            logger.warning("TEI embedding scoring failed, fallback to local retriever: %s", e)
            return None

        if isinstance(payload, dict):
            embeddings = payload.get("embeddings") or payload.get("data") or payload.get("vectors")
        else:
            embeddings = payload

        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            logger.warning("Unexpected TEI embedding payload shape, fallback to local retriever")
            return None

        query_vec = embeddings[0]
        doc_vecs = embeddings[1:]
        if not isinstance(query_vec, list):
            return None

        query_dense = self._dense_normalize(query_vec)
        if not query_dense:
            return None

        scores: List[float] = []
        for doc_vec in doc_vecs:
            doc_dense = self._dense_normalize(doc_vec if isinstance(doc_vec, list) else [])
            if not doc_dense:
                scores.append(0.0)
                continue
            scores.append(self._dense_cosine(query_dense, doc_dense))
        return scores

    def _term_freq(self, tokens: List[str]) -> Dict[str, float]:
        if not tokens:
            return {}
        freq: Dict[str, float] = {}
        for token in tokens:
            freq[token] = freq.get(token, 0.0) + 1.0
        norm = float(len(tokens))
        for key in list(freq.keys()):
            freq[key] /= norm
        return freq

    def _cosine(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        keys = set(a.keys()).intersection(set(b.keys()))
        dot = sum(a[key] * b[key] for key in keys)
        norm_a = math.sqrt(sum(value * value for value in a.values()))
        norm_b = math.sqrt(sum(value * value for value in b.values()))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _dense_normalize(self, values: List[float]) -> List[float]:
        try:
            dense = [float(value) for value in values]
        except Exception:
            return []
        norm = math.sqrt(sum(value * value for value in dense))
        if norm == 0.0:
            return []
        return [value / norm for value in dense]

    def _dense_cosine(self, a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        return sum(x * y for x, y in zip(a, b))

    def _collect_long_strings(self, data) -> List[str]:
        out: List[str] = []

        def walk(node):
            if isinstance(node, dict):
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)
            elif isinstance(node, str):
                text = node.strip()
                if len(text) >= 12:
                    out.append(text)

        walk(data)
        return out[:300]

    def _normalize_lesson_text(self, text: str) -> str:
        normalized = str(text or "").strip().lower()
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized

    def _make_lesson_id(self, normalized_lesson: str) -> str:
        digest = hashlib.sha1(normalized_lesson.encode("utf-8")).hexdigest()[:16]
        return f"{self.side}_{digest}"

    def _compact_snapshot_for_query(self, snapshot: str) -> str:
        text = str(snapshot or "").strip()
        if not text:
            return ""

        keywords = (
            "score",
            "destroyscore",
            "cost",
            "得分",
            "毁伤",
            "成本",
            "高成本弹",
            "低成本弹",
            "flagship",
            "cruiser",
            "destroyer",
            "旗舰",
            "巡洋舰",
            "驱逐舰",
            "highcostattackmissile",
            "lowcostattackmissile",
        )

        selected = []
        for line in text.splitlines():
            stripped = line.strip()
            lower_line = stripped.lower()
            if any(keyword in lower_line for keyword in keywords):
                selected.append(stripped)
            if len(selected) >= 16:
                break

        if selected:
            return "\n".join(selected)
        return text[:800]

    def _compact_score_for_query(self, score_summary: str, snapshot: str) -> str:
        score_text = str(score_summary or "").strip()
        if score_text:
            return score_text
        return self._compact_snapshot_for_query(snapshot)

    def _compact_targets_for_query(self, top_targets: List[dict]) -> str:
        if not isinstance(top_targets, list) or not top_targets:
            return ""
        lines = []
        for item in top_targets[:6]:
            if not isinstance(item, dict):
                continue
            lines.append(
                " | ".join(
                    [
                        f"target_id={item.get('target_id')}",
                        f"type={item.get('target_type')}",
                        f"window={item.get('attack_window')}",
                        f"priority={item.get('priority')}",
                        f"value={item.get('value')}",
                        f"risk={item.get('repeat_high_cost_risk', 'none')}",
                        f"recommendation={item.get('engagement_recommendation', '')}",
                    ]
                )
            )
        return "\n".join(lines)


def build_structured_lesson(
    lesson_type: str,
    observation: str,
    lesson: str,
    tags: Optional[List[str]] = None,
    battle_meta: Optional[dict] = None,
    source: str = "reflection_agent",
    phase: str = "",
    target_type: str = "",
    symptom: str = "",
    trigger: str = "",
    recommended_action: str = "",
    score_pattern: str = "",
    cost_risk: str = "",
) -> dict:
    tags = tags or []
    battle_meta = battle_meta or {}
    normalized = re.sub(r"\s+", " ", str(lesson).strip().lower())
    lesson_id = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]
    return {
        "lesson_id": lesson_id,
        "type": lesson_type or "general",
        "observation": observation or "",
        "lesson": lesson or "",
        "tags": tags,
        "phase": phase or "",
        "target_type": target_type or "",
        "symptom": symptom or "",
        "trigger": trigger or "",
        "score_pattern": score_pattern or "",
        "cost_risk": cost_risk or "",
        "battle_meta": battle_meta,
        "source": source,
        "schema_version": 3,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
