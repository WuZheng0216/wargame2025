import asyncio
import json
import logging
import os
import queue
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from threading import Event
from typing import List, Optional

from jsqlsim import GameFaction, GameState
from jsqlsim.enum import FactionEnum
from jsqlsim.world.action import Action

from action_freshness_guard import ActionFreshnessGuard
from engagement_memory_manager import EngagementMemoryManager
from event_logger import EventLogger
from graph import build_agent_graph
from llm_manager import LLMManager
from ltm_retriever import LongTermMemoryRetriever
from memory_manager import ShortTermMemoryManager
from red_trace_helper import (
    RedTraceLogger,
    set_global_trace_logger,
    summarize_engine_actions,
    summarize_lessons,
    truncate_text,
)
from reflection_agent import PostBattleReflectionAgent
from runtime_paths import ensure_output_dir
from state_access import extract_score_dict
from situation_summarizer import SituationSummarizer
from state_access import get_side_score
from trajectory_diagnostics import TrajectoryDiagnosticsRecorder

logger = logging.getLogger(__name__)


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        return float(raw)
    except Exception:
        logger.warning("Invalid %s=%s, fallback to %s", name, raw, default)
        return default


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        return int(raw)
    except Exception:
        logger.warning("Invalid %s=%s, fallback to %s", name, raw, default)
        return default


def _read_bool_env(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return raw not in {"0", "false", "no", "off", ""}


class BaseCommander(ABC):
    def __init__(self, faction: FactionEnum, shutdown_event: Event, llm_manager: LLMManager):
        self._faction_enum = faction
        self.faction: Optional[GameFaction] = None
        self.shutdown_event = shutdown_event
        self.llm_manager = llm_manager
        self.faction_name = faction.name if isinstance(faction, FactionEnum) else "BLUE"

        self.score = 0.0
        self.previous_state: Optional[GameState] = None
        self.event_logger: Optional[EventLogger] = None
        self.knowledge_base_path = f"test/{self.faction_name.lower()}_reflections.jsonl"
        self.reflection_agent = PostBattleReflectionAgent(
            self.llm_manager.get_llm_model(self.faction_name), self.knowledge_base_path
        )
        self.summarizer = SituationSummarizer()

        self.llm_call_interval = 60.0
        self.last_llm_trigger_time = 5.0 - self.llm_call_interval

        enabled_sides = {
            side.strip().upper()
            for side in os.getenv("LANGGRAPH_ENABLED_SIDES", "RED").split(",")
            if side.strip()
        }
        has_system2_parser = callable(getattr(self, "parse_system2_actions", None))
        self.enable_system2 = has_system2_parser and (self.faction_name.upper() in enabled_sides)

        self.agent_graph = None
        self.action_queue = None
        self.is_thinking = False
        self.loop = None
        self.sys2_thread = None
        self.stm_manager = None
        self.ltm_retriever = None
        self.freshness_guard = None
        self.engagement_manager = None
        self.trajectory_recorder = None

        self.trace_logger: Optional[RedTraceLogger] = None
        self._trace_counter = 0
        self._active_trace_context: Optional[dict] = None
        self.red_task_board = {}
        self._cleanup_lock = threading.Lock()
        self._cleanup_done = False
        self.reflection_timeout_seconds = _read_float_env("POST_BATTLE_REFLECTION_TIMEOUT_SECONDS", 0.0)
        self.semantic_replan_backoff_seconds = _read_float_env("RED_SEMANTIC_REPLAN_BACKOFF_SECONDS", 25.0)
        self.semantic_replan_backoff_threshold = _read_int_env("RED_SEMANTIC_REPLAN_BACKOFF_THRESHOLD", 2)
        self._semantic_replan_failures = 0
        self._semantic_replan_backoff_until = -1.0
        self._last_replan_backoff_log_time = -1.0
        self.slow_waste_soft_threshold = _read_int_env("RED_SLOW_WASTE_SOFT_THRESHOLD", 2)
        self.slow_waste_soft_suppress_seconds = _read_float_env("RED_SLOW_WASTE_SOFT_SUPPRESS_SECONDS", 60.0)
        self.slow_waste_hard_threshold = _read_int_env("RED_SLOW_WASTE_HARD_THRESHOLD", 4)
        self.slow_waste_hard_suppress_seconds = _read_float_env("RED_SLOW_WASTE_HARD_SUPPRESS_SECONDS", 120.0)
        self.stale_replan_min_gap_seconds = _read_float_env("RED_STALE_REPLAN_MIN_GAP_SECONDS", 20.0)
        self.slow_interval_on_soft_waste_seconds = _read_float_env("RED_SLOW_INTERVAL_ON_SOFT_WASTE_SECONDS", 60.0)
        self.slow_interval_on_hard_waste_seconds = _read_float_env("RED_SLOW_INTERVAL_ON_HARD_WASTE_SECONDS", 90.0)
        self._slow_control_lock = threading.Lock()
        self._wasted_slow_trace_count = 0
        self._slow_periodic_suppress_until = -1.0
        self._slow_waste_context_signature = ""
        self._last_slow_periodic_suppress_log_time = -1.0
        self._last_slow_semantic_suppress_log_time = -1.0
        self._last_stale_replan_gap_log_time = -1.0
        self._last_stale_replan_time = -1.0
        self._pending_graph_futures: set[Future] = set()
        self._pending_graph_futures_lock = threading.Lock()

        if self.faction_name.upper() == "RED":
            self.trace_logger = RedTraceLogger()
            set_global_trace_logger(self.trace_logger)
            logger.info("[%s] unified trace file=%s", self.faction_name, self.trace_logger.file_path)

        if self.enable_system2:
            self.agent_graph = build_agent_graph(llm_manager, self.faction_name)
            self.action_queue: "queue.Queue[dict]" = queue.Queue()
            self.stm_manager = ShortTermMemoryManager()
            self.ltm_retriever = LongTermMemoryRetriever(self.faction_name.lower())
            self.freshness_guard = ActionFreshnessGuard()
            if self.faction_name.upper() == "RED":
                self.engagement_manager = EngagementMemoryManager()

            self.loop = asyncio.new_event_loop()
            self.sys2_thread = threading.Thread(
                target=self._run_system2_loop,
                daemon=True,
                name=f"{self.faction_name}-Sys2",
            )
            self.sys2_thread.start()
            logger.info("[%s] LangGraph System2 enabled with STM+LTM+FreshnessGuard.", self.faction_name)
            logger.info(
                "[%s] System2 config: stm_window=%s stm_max_events=%s stale_max=%s stale_force_replan=%s ltm_topk=%s ltm_alpha=%s ltm_store=%s",
                self.faction_name,
                getattr(self.stm_manager, "window_seconds", None),
                getattr(self.stm_manager, "max_events", None),
                getattr(self.freshness_guard, "max_staleness", None),
                getattr(self.freshness_guard, "force_replan_enabled", None),
                getattr(self.ltm_retriever, "top_k", None),
                getattr(self.ltm_retriever, "alpha", None),
                getattr(self.ltm_retriever, "structured_store_path", None),
            )
            if self.faction_name.upper() == "RED":
                logger.info(
                    "[%s] semantic replan backoff config threshold=%s backoff=%.1fs",
                    self.faction_name,
                    self.semantic_replan_backoff_threshold,
                    self.semantic_replan_backoff_seconds,
                )
                logger.info(
                    "[%s] slow waste config soft=(%s,%.1fs) hard=(%s,%.1fs) stale_gap=%.1fs periodic_intervals=(%.1fs,%.1fs)",
                    self.faction_name,
                    self.slow_waste_soft_threshold,
                    self.slow_waste_soft_suppress_seconds,
                    self.slow_waste_hard_threshold,
                    self.slow_waste_hard_suppress_seconds,
                    self.stale_replan_min_gap_seconds,
                    self.slow_interval_on_soft_waste_seconds,
                    self.slow_interval_on_hard_waste_seconds,
                )
        else:
            logger.info(
                "[%s] LangGraph System2 disabled. has_parser=%s, enabled_sides=%s",
                self.faction_name,
                has_system2_parser,
                sorted(enabled_sides),
            )

    def _run_system2_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _trace_enabled(self) -> bool:
        return self.trace_logger is not None and self.faction_name.upper() == "RED"

    def _new_trace_id(self, sim_time: int) -> Optional[str]:
        if not self._trace_enabled():
            return None
        self._trace_counter += 1
        return f"RED-{sim_time}-{self._trace_counter:04d}"

    def _start_trace(self, trace_id: Optional[str], sim_time: int, reason: str, memory_packet: dict) -> None:
        if not self._trace_enabled() or not trace_id:
            return
        self.trace_logger.start_trace(trace_id, sim_time, reason, memory_packet)

    def _trace_step(self, trace_id: Optional[str], section: str, **data) -> None:
        if not self._trace_enabled() or not trace_id:
            return
        self.trace_logger.log_step(trace_id, section, **data)

    def _trace_error(self, trace_id: Optional[str], message: str) -> None:
        if not self._trace_enabled() or not trace_id:
            return
        self.trace_logger.log_error(trace_id, message)

    def _finish_trace(self, trace_id: Optional[str], status: str, **extra) -> None:
        if not self._trace_enabled() or not trace_id:
            return
        self.trace_logger.finish_trace(trace_id, status=status, **extra)

    def _build_memory_packet(self, state: GameState, sim_time: int) -> dict:
        snapshot = getattr(self, "_summarize_state", lambda s: "None")(state)
        score_breakdown = extract_score_dict(state)
        side_score = self._safe_get_score(state)
        if self.stm_manager is None:
            return {
                "current_snapshot": snapshot,
                "window_summary": "STM disabled.",
                "event_timeline": [],
                "recent_failures": [],
                "memory_window": {"start_sim_time": sim_time, "end_sim_time": sim_time},
                "score_breakdown": score_breakdown,
                "side_score": side_score,
                "score_summary": self._format_score_summary(score_breakdown, side_score),
                "battle_phase": self._derive_battle_phase(sim_time),
            }
        packet = self.stm_manager.build_memory_packet(sim_time, snapshot)
        packet["score_breakdown"] = score_breakdown
        packet["side_score"] = side_score
        packet["score_summary"] = self._format_score_summary(score_breakdown, side_score)
        packet["battle_phase"] = self._derive_battle_phase(sim_time)
        return packet

    def _format_score_summary(self, score_breakdown: dict, side_score: float) -> str:
        if not isinstance(score_breakdown, dict) or not score_breakdown:
            return f"side_score={side_score:.1f}"

        if self.faction_name.upper() == "BLUE":
            side_score_key = "blueScore"
            side_destroy_key = "blueDestroyScore"
            side_cost_key = "blueCost"
            enemy_score_key = "redScore"
            enemy_destroy_key = "redDestroyScore"
            enemy_cost_key = "redCost"
        else:
            side_score_key = "redScore"
            side_destroy_key = "redDestroyScore"
            side_cost_key = "redCost"
            enemy_score_key = "blueScore"
            enemy_destroy_key = "blueDestroyScore"
            enemy_cost_key = "blueCost"

        return (
            f"side_score={score_breakdown.get(side_score_key, side_score)} "
            f"side_destroy={score_breakdown.get(side_destroy_key, 0)} "
            f"side_cost={score_breakdown.get(side_cost_key, 0)} "
            f"enemy_score={score_breakdown.get(enemy_score_key, 0)} "
            f"enemy_destroy={score_breakdown.get(enemy_destroy_key, 0)} "
            f"enemy_cost={score_breakdown.get(enemy_cost_key, 0)}"
        )

    def _derive_battle_phase(self, sim_time: int) -> str:
        total_simtime = max(1, int(_read_float_env("AUTO_STOP_SIMTIME_SECONDS", 1800.0)))
        ratio = float(sim_time) / float(total_simtime)
        if sim_time <= 300 or ratio <= 0.2:
            return "opening"
        if ratio >= 0.7:
            return "endgame"
        return "midgame"

    def _build_ltm_context(self, memory_packet: dict, graph_context: dict) -> dict:
        ltm_context = dict(memory_packet or {})
        ltm_context["engagement_summary"] = graph_context.get("engagement_summary", "")
        ltm_context["engagement_memory"] = graph_context.get("engagement_memory", [])
        ltm_context["repeat_high_cost_targets"] = graph_context.get("repeat_high_cost_targets", [])
        ltm_context["pending_bda_targets"] = graph_context.get("pending_bda_targets", [])
        ltm_context["recent_attack_cost"] = graph_context.get("recent_attack_cost", 0.0)
        ltm_context["top_targets"] = (graph_context.get("target_table", []) or [])[:6]
        return ltm_context

    def _build_graph_context(self, state: GameState, sim_time: int, memory_packet: dict) -> dict:
        return {}

    def _get_system2_base_interval(self) -> float:
        return self.llm_call_interval

    def _get_system2_trigger_reason(self, state: GameState, sim_time: int) -> Optional[str]:
        return None

    def _on_system2_scheduled(self, state: GameState, sim_time: int, reason: str, graph_context: dict) -> None:
        return None

    def _get_current_slow_context_signature(self, state: GameState, sim_time: int) -> Optional[dict]:
        return None

    def _reset_semantic_replan_backoff(self) -> None:
        self._semantic_replan_failures = 0
        self._semantic_replan_backoff_until = -1.0
        self._last_replan_backoff_log_time = -1.0

    def _signature_token(self, signature: Optional[dict]) -> str:
        if not signature:
            return ""
        try:
            return json.dumps(signature, ensure_ascii=False, sort_keys=True)
        except Exception:
            return ""

    def _record_slow_trace_outcome(self, status: str, sim_time: int, slow_context_signature: Optional[dict]) -> None:
        if self.faction_name.upper() != "RED":
            return

        normalized_signature = self._signature_token(slow_context_signature)
        wasted_statuses = {"semantic_replan", "no_actions", "no_submit"}

        cooldown_start = None
        recovery_count = 0
        same_context = False
        new_count = 0
        return_log_recovery = False
        with self._slow_control_lock:
            if status == "submitted":
                recovery_count = self._wasted_slow_trace_count
                self._wasted_slow_trace_count = 0
                self._slow_periodic_suppress_until = -1.0
                self._slow_waste_context_signature = normalized_signature
                self._last_slow_periodic_suppress_log_time = -1.0
                self._last_slow_semantic_suppress_log_time = -1.0
                return_log_recovery = recovery_count > 0
                if not return_log_recovery:
                    return
            elif status in wasted_statuses:
                same_context = bool(normalized_signature) and normalized_signature == self._slow_waste_context_signature
                self._wasted_slow_trace_count = self._wasted_slow_trace_count + 1 if same_context else 1
                self._slow_waste_context_signature = normalized_signature
                new_count = self._wasted_slow_trace_count

                suppress_seconds = 0.0
                if same_context:
                    if new_count >= self.slow_waste_hard_threshold:
                        suppress_seconds = self.slow_waste_hard_suppress_seconds
                    elif new_count >= self.slow_waste_soft_threshold:
                        suppress_seconds = self.slow_waste_soft_suppress_seconds

                if suppress_seconds > 0:
                    suppress_until = sim_time + suppress_seconds
                    if suppress_until > self._slow_periodic_suppress_until:
                        self._slow_periodic_suppress_until = suppress_until
                        cooldown_start = (new_count, suppress_until, suppress_seconds)
            else:
                return

        if return_log_recovery:
            logger.info("[%s] slow_trace_recovered sim_time=%s previous_wasted=%s", self.faction_name, sim_time, recovery_count)
            return

        logger.info(
            "[%s] wasted_slow_trace_count sim_time=%s status=%s count=%s slow_context_unchanged=%s",
            self.faction_name,
            sim_time,
            status,
            new_count,
            same_context,
        )
        if cooldown_start is not None:
            count, suppress_until, suppress_seconds = cooldown_start
            logger.warning(
                "[%s] slow_periodic_cooldown_start sim_time=%s suppress_until=%s suppress_for=%.1fs wasted_slow_trace_count=%s",
                self.faction_name,
                sim_time,
                int(suppress_until),
                suppress_seconds,
                count,
            )

    def _effective_system2_interval(self, base_interval: float, slow_context_signature: Optional[dict]) -> float:
        if self.faction_name.upper() != "RED":
            return base_interval

        normalized_signature = self._signature_token(slow_context_signature)
        with self._slow_control_lock:
            if not normalized_signature or normalized_signature != self._slow_waste_context_signature:
                return base_interval
            wasted_count = self._wasted_slow_trace_count

        if wasted_count >= self.slow_waste_hard_threshold:
            return max(base_interval, self.slow_interval_on_hard_waste_seconds)
        if wasted_count >= 1:
            return max(base_interval, self.slow_interval_on_soft_waste_seconds)
        return base_interval

    def _should_schedule_periodic(self, sim_time: int, slow_context_signature: Optional[dict]) -> bool:
        if self.faction_name.upper() != "RED":
            return True

        normalized_signature = self._signature_token(slow_context_signature)
        with self._slow_control_lock:
            if (
                normalized_signature
                and normalized_signature == self._slow_waste_context_signature
                and sim_time < self._slow_periodic_suppress_until
            ):
                remaining = max(0.0, self._slow_periodic_suppress_until - sim_time)
                if self._last_slow_periodic_suppress_log_time != sim_time:
                    logger.info(
                        "[%s] slow_periodic_suppressed sim_time=%s remaining=%.1fs wasted_slow_trace_count=%s",
                        self.faction_name,
                        sim_time,
                        remaining,
                        self._wasted_slow_trace_count,
                    )
                    self._last_slow_periodic_suppress_log_time = sim_time
                return False
        return True

    def _record_semantic_replan_failure(self, sim_time: int) -> None:
        self._semantic_replan_failures += 1
        if self.semantic_replan_backoff_threshold <= 0 or self.semantic_replan_backoff_seconds <= 0:
            return
        if self._semantic_replan_failures < self.semantic_replan_backoff_threshold:
            return

        backoff_until = sim_time + self.semantic_replan_backoff_seconds
        if backoff_until <= self._semantic_replan_backoff_until:
            return

        self._semantic_replan_backoff_until = backoff_until
        logger.warning(
            "[%s] semantic_replan_backoff_start failures=%s backoff_until=%s backoff=%.1fs",
            self.faction_name,
            self._semantic_replan_failures,
            int(self._semantic_replan_backoff_until),
            self.semantic_replan_backoff_seconds,
        )

    def _should_schedule_replan(
        self,
        sim_time: int,
        reason: str,
        slow_context_signature: Optional[dict] = None,
    ) -> bool:
        if reason == "stale_replan":
            if self.stale_replan_min_gap_seconds > 0 and self._last_stale_replan_time >= 0:
                remaining = (self._last_stale_replan_time + self.stale_replan_min_gap_seconds) - sim_time
                if remaining > 0:
                    if self._last_stale_replan_gap_log_time != sim_time:
                        logger.info(
                            "[%s] stale_replan_suppressed sim_time=%s remaining=%.1fs min_gap=%.1fs",
                            self.faction_name,
                            sim_time,
                            remaining,
                            self.stale_replan_min_gap_seconds,
                        )
                        self._last_stale_replan_gap_log_time = sim_time
                    return False
            self._last_stale_replan_time = sim_time
            return True

        if reason != "semantic_replan":
            return True
        if sim_time < self._semantic_replan_backoff_until:
            if self._last_replan_backoff_log_time != sim_time:
                remaining = max(0.0, self._semantic_replan_backoff_until - sim_time)
                logger.info(
                    "[%s] semantic_replan_suppressed sim_time=%s remaining=%.1fs consecutive_failures=%s",
                    self.faction_name,
                    sim_time,
                    remaining,
                    self._semantic_replan_failures,
                )
                self._last_replan_backoff_log_time = sim_time
            return False

        if self.faction_name.upper() != "RED":
            return True

        normalized_signature = self._signature_token(slow_context_signature)
        with self._slow_control_lock:
            if (
                normalized_signature
                and normalized_signature == self._slow_waste_context_signature
                and sim_time < self._slow_periodic_suppress_until
            ):
                remaining = max(0.0, self._slow_periodic_suppress_until - sim_time)
                if self._last_slow_semantic_suppress_log_time != sim_time:
                    logger.info(
                        "[%s] slow_semantic_replan_suppressed sim_time=%s remaining=%.1fs wasted_slow_trace_count=%s",
                        self.faction_name,
                        sim_time,
                        remaining,
                        self._wasted_slow_trace_count,
                    )
                    self._last_slow_semantic_suppress_log_time = sim_time
                return False
        return True

    async def _run_graph_task(self, sim_time: int, memory_packet: dict, graph_context: dict, trace_id: Optional[str]):
        try:
            logger.info("[%s] System2 thinking starts at sim_time=%s trace_id=%s", self.faction_name, sim_time, trace_id)

            ltm_lessons = []
            ltm_lessons_structured = []
            ltm_block = "No long-term lessons."
            ltm_query = ""
            if self.ltm_retriever is not None:
                ltm_context = self._build_ltm_context(memory_packet, graph_context)
                ltm_query = self.ltm_retriever.build_query_for_context(ltm_context)
                ltm_lessons = self.ltm_retriever.retrieve_for_context(ltm_context)
                ltm_block = self.ltm_retriever.format_lessons_block(ltm_lessons)
                ltm_lessons_structured = self.ltm_retriever.format_lessons_structured(ltm_lessons)
                self._trace_step(
                    trace_id,
                    "LTM",
                    ltm_query_preview=truncate_text(ltm_query, 420),
                    hit_count=len(ltm_lessons),
                    lessons_summary=summarize_lessons(ltm_lessons),
                    retrieved_lesson_ids=[str(item.get("lesson_id", "")) for item in ltm_lessons[:5]],
                    retrieved_lesson_scores=[round(float(item.get("hybrid_score", 0.0)), 3) for item in ltm_lessons[:5]],
                    retrieved_lesson_tags=[item.get("tags", []) for item in ltm_lessons[:5]],
                    retrieved_lessons=ltm_lessons_structured[:5],
                )

            inputs = {
                "sim_time": sim_time,
                "faction_name": self.faction_name,
                "trace_id": trace_id or "",
                "state_summary": memory_packet.get("current_snapshot", ""),
                "window_summary": memory_packet.get("window_summary", ""),
                "event_timeline": memory_packet.get("event_timeline", []),
                "ltm_lessons": ltm_block,
                "ltm_lessons_structured": ltm_lessons_structured,
                "memory_window": memory_packet.get(
                    "memory_window", {"start_sim_time": sim_time, "end_sim_time": sim_time}
                ),
                "unit_roster": graph_context.get("unit_roster", []),
                "target_table": graph_context.get("target_table", []),
                "task_board": graph_context.get("task_board", {}),
                "engagement_summary": graph_context.get("engagement_summary", ""),
                "engagement_memory": graph_context.get("engagement_memory", []),
                "key_findings": "",
                "intent": "",
                "allocation_plan": {},
                "allocation_summary": "",
                "critique": "",
                "planned_at_sim_time": sim_time,
                "actions_json": [],
                "is_valid": False,
                "retry_count": 0,
                "retry_stage": "",
                "logs": [],
                "trace_events": [],
            }

            if graph_context.get("engagement_summary") or graph_context.get("engagement_memory"):
                self._trace_step(
                    trace_id,
                    "STM",
                    engagement_summary=truncate_text(graph_context.get("engagement_summary", ""), 600),
                    repeat_high_cost_targets=graph_context.get("repeat_high_cost_targets", []),
                    pending_bda_targets=graph_context.get("pending_bda_targets", []),
                    recent_attack_cost=graph_context.get("recent_attack_cost", 0.0),
                )

            final_state = await self.agent_graph.ainvoke(inputs)
            for event in final_state.get("trace_events", []) or []:
                self._trace_step(trace_id, "Graph Nodes", **event)

            actions_json = final_state.get("actions_json", [])
            planned_at = int(final_state.get("planned_at_sim_time", sim_time))
            if isinstance(actions_json, list) and actions_json:
                payload = {
                    "trace_id": trace_id,
                    "planned_at": planned_at,
                    "actions_json": actions_json,
                    "trace": {
                        "memory_window": memory_packet.get("memory_window", {}),
                        "ltm_hit_count": len(ltm_lessons),
                        "ltm_query_preview": truncate_text(ltm_query, 420),
                        "retrieved_lesson_ids": [str(item.get("lesson_id", "")) for item in ltm_lessons[:5]],
                        "retrieved_lesson_scores": [round(float(item.get("hybrid_score", 0.0)), 3) for item in ltm_lessons[:5]],
                        "retrieved_lesson_tags": [item.get("tags", []) for item in ltm_lessons[:5]],
                        "retrieved_lessons": ltm_lessons_structured[:5],
                        "graph_logs": final_state.get("logs", []),
                        "allocation_plan": final_state.get("allocation_plan", {}),
                        "allocation_summary": final_state.get("allocation_summary", ""),
                        "unit_roster": graph_context.get("unit_roster", []),
                        "target_table": graph_context.get("target_table", []),
                        "task_board": graph_context.get("task_board", {}),
                        "engagement_summary": graph_context.get("engagement_summary", ""),
                        "engagement_memory": graph_context.get("engagement_memory", []),
                        "repeat_high_cost_targets": graph_context.get("repeat_high_cost_targets", []),
                        "pending_bda_targets": graph_context.get("pending_bda_targets", []),
                        "recent_attack_cost": graph_context.get("recent_attack_cost", 0.0),
                        "slow_context_signature": graph_context.get("slow_context_signature", {}),
                    },
                }
                self.action_queue.put(payload)
            else:
                self._record_slow_trace_outcome("no_actions", sim_time, graph_context.get("slow_context_signature"))
                self._trace_step(
                    trace_id,
                    "Submit",
                    submitted_action_count=0,
                    submitted_summary="none",
                    note="no actions generated by graph",
                )
                self._finish_trace(trace_id, status="no_actions")
        except Exception as e:
            logger.error("[%s] System2 graph error: %s", self.faction_name, e, exc_info=True)
            self._trace_error(trace_id, f"graph_error={e}")
            self._finish_trace(trace_id, status="graph_error")
        finally:
            self.is_thinking = False

    def _schedule_graph_task(self, state: GameState, sim_time: int, reason: str):
        if not self.enable_system2 or self.loop is None or self.is_thinking:
            return

        memory_packet = self._build_memory_packet(state, sim_time)
        graph_context = self._build_graph_context(state, sim_time, memory_packet)
        trace_id = self._new_trace_id(sim_time)
        self._start_trace(trace_id, sim_time, reason, memory_packet)

        self.is_thinking = True
        self.last_llm_trigger_time = sim_time
        logger.info(
            "[%s] schedule graph task: reason=%s memory_window=%s trace_id=%s",
            self.faction_name,
            reason,
            memory_packet.get("memory_window"),
            trace_id,
        )
        try:
            self._on_system2_scheduled(state, sim_time, reason, graph_context)
        except Exception:
            logger.debug("[%s] _on_system2_scheduled hook failed", self.faction_name, exc_info=True)
        future = asyncio.run_coroutine_threadsafe(
            self._run_graph_task(sim_time, memory_packet, graph_context, trace_id),
            self.loop,
        )
        with self._pending_graph_futures_lock:
            self._pending_graph_futures.add(future)
        future.add_done_callback(self._on_graph_future_done)

    def _on_graph_future_done(self, future: Future) -> None:
        with self._pending_graph_futures_lock:
            self._pending_graph_futures.discard(future)

    async def _cancel_pending_loop_tasks(self):
        current = asyncio.current_task()
        tasks = [task for task in asyncio.all_tasks() if task is not current and not task.done()]
        if not tasks:
            return {"cancelled": 0, "remaining": 0}
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        remaining = sum(1 for task in tasks if not task.done())
        return {"cancelled": len(tasks), "remaining": remaining}

    def _shutdown_system2_loop(self):
        if not self.enable_system2 or self.loop is None:
            return

        with self._pending_graph_futures_lock:
            pending_futures = list(self._pending_graph_futures)

        cancelled_futures = 0
        for future in pending_futures:
            if future.done():
                continue
            if future.cancel():
                cancelled_futures += 1

        cancel_summary = {"cancelled": 0, "remaining": 0}
        if self.loop.is_running():
            try:
                loop_future = asyncio.run_coroutine_threadsafe(self._cancel_pending_loop_tasks(), self.loop)
                cancel_summary = loop_future.result(timeout=5)
            except FutureTimeoutError:
                logger.warning("[%s] system2_loop_cancel_timeout", self.faction_name)
            except Exception:
                logger.debug("[%s] system2_loop_cancel_failed", self.faction_name, exc_info=True)

        logger.info(
            "[%s] system2_loop_shutdown futures_cancelled=%s tasks_cancelled=%s tasks_remaining=%s",
            self.faction_name,
            cancelled_futures,
            cancel_summary.get("cancelled", 0),
            cancel_summary.get("remaining", 0),
        )

        try:
            if self.loop.is_running():
                self.loop.call_soon_threadsafe(self.loop.stop)
        except Exception:
            logger.debug("[%s] loop.stop failed", self.faction_name, exc_info=True)

        if self.sys2_thread is not None:
            self.sys2_thread.join(timeout=2)

    @abstractmethod
    def execute_rules(self, state: GameState) -> List[Action]:
        raise NotImplementedError

    def _safe_get_score(self, state: GameState) -> float:
        return get_side_score(state, self.faction_name)

    def _decision_source_label(self, trace_results: List[dict]) -> str:
        blue_mode = getattr(self, "blue_decision_mode", None)
        if blue_mode:
            return str(blue_mode)
        if self.enable_system2 and trace_results:
            return "system2"
        return "rule"

    def _decision_trace_summary(self, trace_results: List[dict]) -> str:
        if not trace_results:
            return "rule_only"

        trace_ids = [str(item.get("trace_id")) for item in trace_results if item.get("trace_id")]
        guard_drop_total = sum(int((item.get("freshness_report") or {}).get("drop_count", 0)) for item in trace_results)
        semantic_reject_total = sum(int((item.get("parse_context") or {}).get("semantic_rejected_count", 0)) for item in trace_results)

        parts = []
        if trace_ids:
            parts.append(f"trace_ids={','.join(trace_ids)}")
        if guard_drop_total:
            parts.append(f"guard_drop_total={guard_drop_total}")
        if semantic_reject_total:
            parts.append(f"semantic_reject_total={semantic_reject_total}")
        return "; ".join(parts) if parts else "trace_results_present"

    def filter_actions_before_submit(self, actions: List[Action], state: GameState, sim_time: int) -> tuple[List[Action], dict]:
        return actions, {"dropped_count": 0, "drop_reasons": {}}

    def _filter_submitted_engagement_events(self, parse_context: dict, submitted_action_ids: set[int]) -> List[dict]:
        events = []
        for raw_event in parse_context.get("engagement_events", []) or []:
            if not isinstance(raw_event, dict):
                continue
            event = dict(raw_event)
            launch_actions = []
            for launch in raw_event.get("launch_actions", []) or []:
                if not isinstance(launch, dict):
                    continue
                try:
                    action_id = int(launch.get("action_id"))
                except Exception:
                    continue
                if action_id in submitted_action_ids:
                    launch_actions.append(dict(launch))
            if launch_actions or str(event.get("action_type")) in {"MoveToEngage", "GuideAttack"}:
                event["launch_actions"] = launch_actions
                event["high_cost_launch_count"] = sum(
                    1 for launch in launch_actions if str(launch.get("weapon_type")) == "HighCostAttackMissile"
                )
                event["low_cost_launch_count"] = sum(
                    1 for launch in launch_actions if str(launch.get("weapon_type")) == "LowCostAttackMissile"
                )
                event["estimated_attack_cost"] = sum(float(launch.get("estimated_cost") or 0.0) for launch in launch_actions)
                event["estimated_impact_until"] = max(
                    [float(launch.get("impact_until") or 0.0) for launch in launch_actions],
                    default=0.0,
                )
                events.append(event)
        return events

    def _setup_battle(self, initial_state: GameState):
        logger.info("[%s] --- new battle started ---", self.faction_name)
        self.event_logger = EventLogger(faction_name=self.faction_name, log_dir=ensure_output_dir("battle_logs"))
        if self.trajectory_recorder is None and _read_bool_env("TRAJECTORY_DIAGNOSTICS_ENABLED", False):
            try:
                self.trajectory_recorder = TrajectoryDiagnosticsRecorder(
                    self.faction_name,
                    output_dir=ensure_output_dir("diagnostics"),
                )
            except Exception:
                logger.error("[%s] trajectory_diagnostics_init_failed", self.faction_name, exc_info=True)
        self.previous_state = initial_state
        self.score = self._safe_get_score(initial_state)
        try:
            self.event_logger.log_event(
                "SCORE_SNAPSHOT",
                f"Initial score snapshot for {self.faction_name}",
                extra={
                    "stage": "battle_start",
                    "side_score": self.score,
                    "score_breakdown": extract_score_dict(initial_state),
                },
                sim_time=initial_state.simtime(),
            )
        except Exception:
            logger.debug("[%s] initial score snapshot log failed", self.faction_name, exc_info=True)

    def _run_reflection_with_timeout(self, event_log_path: str, final_score_str: str):
        result = {"status": "done", "error": None}

        def _worker():
            try:
                self.reflection_agent.reflect(event_log_path, final_score_str)
            except Exception as exc:
                result["status"] = "error"
                result["error"] = str(exc)

        reflection_thread = threading.Thread(
            target=_worker,
            name=f"{self.faction_name}-Reflection",
            daemon=True,
        )
        started_at = time.perf_counter()
        reflection_thread.start()
        if self.reflection_timeout_seconds > 0:
            reflection_thread.join(timeout=self.reflection_timeout_seconds)
        else:
            reflection_thread.join()
        duration_ms = int((time.perf_counter() - started_at) * 1000)

        if reflection_thread.is_alive():
            return "timeout", duration_ms, None
        return result["status"], duration_ms, result["error"]

    def _cleanup_battle(self):
        with self._cleanup_lock:
            if self._cleanup_done:
                logger.info("[%s] cleanup already completed, skipping duplicate call", self.faction_name)
                return
            self._cleanup_done = True

        logger.info("[%s] --- battle ended ---", self.faction_name)
        if self.trace_logger is not None:
            try:
                logger.info("[%s] trace_finalize_start", self.faction_name)
                self.trace_logger.finalize_all(status="shutdown")
                logger.info("[%s] trace_finalize_done", self.faction_name)
            except Exception:
                logger.error("[%s] trace_finalize_error", self.faction_name, exc_info=True)
        if not self.event_logger:
            if self.trajectory_recorder is not None:
                try:
                    self.trajectory_recorder.close()
                except Exception:
                    logger.debug("[%s] trajectory recorder close failed", self.faction_name, exc_info=True)
            return

        event_log_path = self.event_logger.filepath
        try:
            if self.previous_state is not None:
                try:
                    self.event_logger.log_event(
                        "SCORE_SNAPSHOT",
                        f"Final score snapshot for {self.faction_name}",
                        extra={
                            "stage": "battle_end",
                            "side_score": self.score,
                            "score_breakdown": extract_score_dict(self.previous_state),
                        },
                        sim_time=self.previous_state.simtime(),
                    )
                except Exception:
                    logger.debug("[%s] final score snapshot log failed", self.faction_name, exc_info=True)
            logger.info("[%s] battle_log_finalize_start path=%s", self.faction_name, event_log_path)
            self.event_logger.save_log()
            final_score_str = f"Final Score: {self.score}"
            logger.info("[%s] running post-battle reflection...", self.faction_name)
            if self.reflection_timeout_seconds > 0:
                logger.info(
                    "[%s] reflection_start path=%s final_score=%s timeout=%.1fs",
                    self.faction_name,
                    event_log_path,
                    final_score_str,
                    self.reflection_timeout_seconds,
                )
            else:
                logger.info(
                    "[%s] reflection_start path=%s final_score=%s timeout=disabled",
                    self.faction_name,
                    event_log_path,
                    final_score_str,
                )
            status, duration_ms, error_msg = self._run_reflection_with_timeout(event_log_path, final_score_str)
            if status == "done":
                logger.info(
                    "[%s] reflection_done path=%s duration_ms=%d final_score=%s",
                    self.faction_name,
                    event_log_path,
                    duration_ms,
                    final_score_str,
                )
            elif status == "timeout":
                logger.warning(
                    "[%s] reflection_timeout path=%s duration_ms=%d final_score=%s note=this battle may not write lessons",
                    self.faction_name,
                    event_log_path,
                    duration_ms,
                    final_score_str,
                )
            else:
                logger.error(
                    "[%s] reflection_error path=%s duration_ms=%d final_score=%s error=%s",
                    self.faction_name,
                    event_log_path,
                    duration_ms,
                    final_score_str,
                    error_msg,
                )
        except Exception as e:
            logger.error("[%s] cleanup_or_reflection_failed: %s", self.faction_name, e, exc_info=True)
        finally:
            if self.trajectory_recorder is not None:
                try:
                    self.trajectory_recorder.close()
                except Exception:
                    logger.debug("[%s] trajectory recorder close failed", self.faction_name, exc_info=True)

    def run(self):
        if self.faction is None:
            self.faction = GameFaction.new(self._faction_enum)

        first_state = None
        try:
            for state in self.faction.iter_next_states(end_simtime=None):
                if self.shutdown_event.is_set():
                    break
                if state is None:
                    continue

                if first_state is None:
                    first_state = state
                    self._setup_battle(state)

                sim_time = state.simtime()
                new_score = self._safe_get_score(state)
                if self.event_logger is not None:
                    try:
                        self.event_logger.previous_state = self.previous_state
                        self.event_logger.check_events(state, new_score, self.score)
                    except Exception:
                        logger.debug("[%s] event check failed", self.faction_name, exc_info=True)
                self.previous_state = state
                self.score = new_score
                score_breakdown = extract_score_dict(state)
                if self.trajectory_recorder is not None:
                    try:
                        self.trajectory_recorder.record_state(state)
                    except Exception:
                        logger.debug("[%s] trajectory diagnostics record failed", self.faction_name, exc_info=True)

                if self.enable_system2 and self.stm_manager is not None:
                    self.stm_manager.record(state, self.faction_name)
                if self.engagement_manager is not None:
                    try:
                        self.engagement_manager.update(state, sim_time, self.score, score_breakdown)
                    except Exception:
                        logger.debug("[%s] engagement memory update failed", self.faction_name, exc_info=True)

                actions_to_submit = self.execute_rules(state)

                if self.enable_system2:
                    force_replan = False
                    replan_reason = None
                    trace_results = []
                    semantic_failure_seen = False
                    semantic_success_seen = False

                    while not self.action_queue.empty():
                        try:
                            payload = self.action_queue.get_nowait()
                            if not isinstance(payload, dict):
                                continue

                            trace_id = payload.get("trace_id")
                            cmd_json = payload.get("actions_json", [])
                            planned_at = int(payload.get("planned_at", sim_time))

                            filtered_json = cmd_json
                            freshness_report = {
                                "planned_at": planned_at,
                                "now": sim_time,
                                "drop_count": 0,
                                "drop_reason_dist": {},
                                "force_replan": False,
                            }
                            if self.freshness_guard is not None:
                                filtered_json, freshness_report = self.freshness_guard.filter_actions(
                                    cmd_json, state, planned_at
                                )

                            self._trace_step(
                                trace_id,
                                "Guard",
                                planned_at=freshness_report.get("planned_at"),
                                now=freshness_report.get("now"),
                                drop_count=freshness_report.get("drop_count"),
                                drop_reason_dist=freshness_report.get("drop_reason_dist"),
                                force_replan=freshness_report.get("force_replan"),
                            )

                            if freshness_report.get("drop_count", 0) > 0:
                                logger.info(
                                    "[%s] stale_action_guard trace_id=%s planned_at=%s now=%s drop_count=%s reasons=%s force_replan=%s",
                                    self.faction_name,
                                    trace_id,
                                    freshness_report.get("planned_at"),
                                    freshness_report.get("now"),
                                    freshness_report.get("drop_count"),
                                    freshness_report.get("drop_reason_dist"),
                                    freshness_report.get("force_replan"),
                                )

                            parser = getattr(self, "parse_system2_actions", None)
                            self._active_trace_context = {
                                "trace_id": trace_id,
                                "sim_time": sim_time,
                                **(payload.get("trace") or {}),
                            }
                            llm_actions = parser(filtered_json, state) if callable(parser) else []
                            parse_context = dict(self._active_trace_context or {})
                            self._active_trace_context = None

                            if llm_actions:
                                actions_to_submit.extend(llm_actions)
                                semantic_success_seen = True

                            trace_results.append(
                                {
                                    "trace_id": trace_id,
                                    "parsed_actions": llm_actions,
                                    "freshness_report": freshness_report,
                                    "parse_context": parse_context,
                                    "actions_json": filtered_json,
                                    "slow_context_signature": parse_context.get("slow_context_signature"),
                                }
                            )
                            if parse_context.get("semantic_force_replan"):
                                replan_reason = replan_reason or "semantic_replan"
                                if not llm_actions:
                                    semantic_failure_seen = True
                            if freshness_report.get("force_replan"):
                                replan_reason = replan_reason or "stale_replan"
                            force_replan = (
                                force_replan
                                or bool(freshness_report.get("force_replan"))
                                or bool(parse_context.get("semantic_force_replan"))
                            )
                        except queue.Empty:
                            break
                        finally:
                            self._active_trace_context = None

                    if semantic_success_seen:
                        self._reset_semantic_replan_backoff()
                    elif semantic_failure_seen:
                        self._record_semantic_replan_failure(sim_time)

                    for trace_result in trace_results:
                        parsed_actions = trace_result.get("parsed_actions", [])
                        freshness_report = trace_result.get("freshness_report", {})
                        parse_context = trace_result.get("parse_context", {})
                        status = "submitted" if parsed_actions else "dropped"
                        if not parsed_actions and freshness_report.get("drop_count", 0) == 0:
                            status = "no_submit"
                        if parse_context.get("semantic_force_replan") and not parsed_actions:
                            status = "semantic_replan"
                        trace_result["status"] = status
                        self._record_slow_trace_outcome(
                            status,
                            sim_time,
                            trace_result.get("slow_context_signature"),
                        )

                    base_interval = self._get_system2_base_interval()
                    current_slow_context_signature = self._get_current_slow_context_signature(state, sim_time)
                    effective_interval = self._effective_system2_interval(base_interval, current_slow_context_signature)
                    event_trigger_reason = None
                    try:
                        event_trigger_reason = self._get_system2_trigger_reason(state, sim_time)
                    except Exception:
                        logger.debug("[%s] _get_system2_trigger_reason hook failed", self.faction_name, exc_info=True)

                    periodic_due = sim_time >= self.last_llm_trigger_time + effective_interval
                    if force_replan and not self.is_thinking:
                        next_reason = replan_reason or "stale_replan"
                        if self._should_schedule_replan(sim_time, next_reason, current_slow_context_signature):
                            self._schedule_graph_task(state, sim_time, reason=next_reason)
                    elif event_trigger_reason and not self.is_thinking:
                        self._schedule_graph_task(state, sim_time, reason=event_trigger_reason)
                    elif periodic_due and not self.is_thinking:
                        if self._should_schedule_periodic(sim_time, current_slow_context_signature):
                            self._schedule_graph_task(state, sim_time, reason="periodic")
                else:
                    trace_results = []
                    if hasattr(self, "execute_llm_strategy"):
                        if sim_time >= self.last_llm_trigger_time + self.llm_call_interval:
                            self.last_llm_trigger_time = sim_time
                            llm_actions = self.execute_llm_strategy(state)
                            if llm_actions:
                                actions_to_submit.extend(llm_actions)

                if actions_to_submit:
                    actions_to_submit, pre_submit_report = self.filter_actions_before_submit(actions_to_submit, state, sim_time)
                    if pre_submit_report.get("dropped_count", 0) > 0:
                        logger.info(
                            "[%s] pre_submit_filter sim_time=%s dropped_count=%s reasons=%s kept=%s",
                            self.faction_name,
                            sim_time,
                            pre_submit_report.get("dropped_count"),
                            pre_submit_report.get("drop_reasons", {}),
                            len(actions_to_submit),
                        )
                if actions_to_submit:
                    submitted_action_ids = {id(action) for action in actions_to_submit}
                    self.faction.submit_actions(actions_to_submit)
                    if self.engagement_manager is not None:
                        for trace_result in trace_results:
                            try:
                                submitted_events = self._filter_submitted_engagement_events(
                                    trace_result.get("parse_context", {}),
                                    submitted_action_ids,
                                )
                                if submitted_events:
                                    self.engagement_manager.record_submitted_engagements(
                                        submitted_events,
                                        sim_time=sim_time,
                                        side_score=self.score,
                                        score_breakdown=score_breakdown,
                                    )
                            except Exception:
                                logger.debug("[%s] engagement memory record failed", self.faction_name, exc_info=True)
                    record_assignments = getattr(self, "record_system2_assignments", None)
                    if callable(record_assignments):
                        for trace_result in trace_results:
                            if trace_result.get("parsed_actions"):
                                try:
                                    record_assignments(
                                        parse_context=trace_result.get("parse_context", {}),
                                        sim_time=sim_time,
                                        trace_id=trace_result.get("trace_id"),
                                    )
                                except Exception:
                                    logger.debug("[%s] record_system2_assignments failed", self.faction_name, exc_info=True)
                    if self.event_logger is not None:
                        self.event_logger.log_decision(
                            sim_time=sim_time,
                            analysis=(
                                f"source={self._decision_source_label(trace_results)}; "
                                f"{self._decision_trace_summary(trace_results)}; "
                                f"summary={summarize_engine_actions(actions_to_submit)}"
                            ),
                            actions=actions_to_submit,
                            source=self._decision_source_label(trace_results),
                            trace={
                                "summary": summarize_engine_actions(actions_to_submit),
                                "trace_summary": self._decision_trace_summary(trace_results),
                            },
                        )

                for trace_result in trace_results:
                    trace_id = trace_result.get("trace_id")
                    parsed_actions = trace_result.get("parsed_actions", [])
                    status = trace_result.get("status", "submitted" if parsed_actions else "no_submit")
                    self._trace_step(
                        trace_id,
                        "Submit",
                        submitted_action_count=len(parsed_actions),
                        submitted_summary=summarize_engine_actions(parsed_actions),
                        note="submitted via main loop batch",
                    )
                    self._finish_trace(trace_id, status=status)

                self.shutdown_event.wait(0.05)

        except Exception as e:
            logger.error("[%s] main loop error: %s", self.faction_name, e, exc_info=True)
        finally:
            self._cleanup_battle()
            self._shutdown_system2_loop()
