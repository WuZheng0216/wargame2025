import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Callable, Optional

import httpx
import json_repair
from openai import OpenAI
from jsqlsim.llm.model.chat_model import Base as ChatModelBase

from prompt_library import get_strategic_prompt
from red_trace_helper import get_global_trace_logger, truncate_text
from runtime_paths import ensure_output_dir
from situation_summarizer import SituationSummarizer

logger = logging.getLogger(__name__)


class LLMManager:
    def __init__(self):
        self.key = os.getenv("LLM_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.model_name = os.getenv("MODEL_NAME")
        if not all([self.key, self.base_url, self.model_name]):
            raise ValueError("LLM_KEY, BASE_URL, MODEL_NAME are required.")

        self.red_fast_model_name = os.getenv("RED_FAST_MODEL_NAME", self.model_name)
        self.red_slow_model_name = os.getenv("RED_SLOW_MODEL_NAME", self.model_name)
        self.red_enable_structured_output = self._read_bool_env("RED_ENABLE_STRUCTURED_OUTPUT", True)
        self.red_enable_context_cache = self._read_bool_env("RED_ENABLE_CONTEXT_CACHE", True)
        self.red_disable_thinking = self._read_bool_env("RED_DISABLE_THINKING", True)
        self.llm_trust_env_proxy = self._read_bool_env("LLM_TRUST_ENV_PROXY", False)
        self.llm_timeout_seconds = int(os.getenv("LM_TIMEOUT_SECONDS", "600"))

        self.models = {}
        self.summarizer = SituationSummarizer()

        llm_output_dir = ensure_output_dir("llm_outputs")
        self.log_dir = llm_output_dir
        self.sys2_log_dir = llm_output_dir

        logger.info(
            "[LLMManager] Initialized. default_model=%s red_fast_model=%s red_slow_model=%s structured=%s cache=%s disable_thinking=%s trust_env_proxy=%s timeout=%ss",
            self.model_name,
            self.red_fast_model_name,
            self.red_slow_model_name,
            self.red_enable_structured_output,
            self.red_enable_context_cache,
            self.red_disable_thinking,
            self.llm_trust_env_proxy,
            self.llm_timeout_seconds,
        )

    def _read_bool_env(self, name: str, default: bool) -> bool:
        raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
        return raw not in {"0", "false", "no", "off", ""}

    def _normalize_side(self, faction_name: str) -> str:
        name = str(faction_name or "").lower()
        return "red" if "red" in name else "blue"

    def _resolve_model_name(self, faction_name: str, role_profile: Optional[str] = None) -> str:
        side = self._normalize_side(faction_name)
        role = str(role_profile or "").strip().lower()
        if side == "red":
            if role == "fast_router":
                return self.red_fast_model_name
            if role in {"analyst", "commander", "allocator", "operator", "reflection"}:
                return self.red_slow_model_name
            return self.red_slow_model_name
        return self.model_name

    def _build_openai_client(self) -> OpenAI:
        http_client = httpx.Client(
            timeout=self.llm_timeout_seconds,
            trust_env=self.llm_trust_env_proxy,
        )
        return OpenAI(
            api_key=self.key,
            base_url=self.base_url,
            timeout=self.llm_timeout_seconds,
            http_client=http_client,
        )

    def get_llm_model(self, faction_name: str, role_profile: Optional[str] = None):
        model_name = self._resolve_model_name(faction_name, role_profile=role_profile)
        if model_name not in self.models:
            model = ChatModelBase(key=self.key, model_name=model_name, base_url=self.base_url)
            try:
                # Force our own client so httpx does not silently inherit a broken system proxy.
                model.client = self._build_openai_client()
            except Exception as e:
                logger.warning("Failed to override OpenAI client for model=%s: %s", model_name, e)
            self.models[model_name] = model
        return self.models[model_name]

    def _save_log(
        self,
        prompt: str,
        response: str,
        faction: str,
        sim_time: int,
        success: bool,
        model_name: Optional[str] = None,
        log_dir: Optional[str] = None,
        prefix: str = "",
    ) -> str:
        target_dir = log_dir or self.log_dir
        os.makedirs(target_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_prefix = f"{prefix}_" if prefix else ""
        file_path = os.path.join(target_dir, f"{name_prefix}{faction}_sim{sim_time}_{ts}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "sim_time": sim_time,
                    "faction": faction,
                    "model": model_name or self.model_name,
                    "timestamp": datetime.now().isoformat(),
                    "success": success,
                    "prompt": prompt,
                    "raw_response": response,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        logger.debug("[%s@%s] Saved log -> %s", faction, sim_time, file_path)
        return file_path

    def _sanitize_stream_delta(self, text: str) -> str:
        cleaned = re.sub(r"Messages\s*generated.*?:", "", text)
        cleaned = cleaned.replace("[RED]", "").replace("[BLUE]", "")
        return cleaned

    def _normalize_content_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(content or "")

    def _should_use_context_cache(self, faction_name: str, role_profile: Optional[str]) -> bool:
        return self._normalize_side(faction_name) == "red" and self.red_enable_context_cache and bool(role_profile)

    def _should_disable_thinking(self, faction_name: str, role_profile: Optional[str]) -> bool:
        return self._normalize_side(faction_name) == "red" and self.red_disable_thinking and bool(role_profile)

    def _supports_json_schema(self, model_name: str) -> bool:
        name = str(model_name or "").lower()
        return (
            ("qwen-plus" in name or "qwen-flash" in name or "qwen3-max" in name)
            and "qwen3.5" not in name
        )

    def _supports_json_object(self, model_name: str) -> bool:
        name = str(model_name or "").lower()
        return "qwen" in name

    def _build_prompt_text(
        self,
        prompt: Optional[str] = None,
        static_prefix: Optional[str] = None,
        dynamic_payload: Optional[str] = None,
    ) -> str:
        if prompt is not None:
            return prompt
        return (
            "[STATIC_PREFIX]\n"
            f"{static_prefix or ''}\n\n"
            "[DYNAMIC_PAYLOAD]\n"
            f"{dynamic_payload or ''}"
        )

    def _build_messages(
        self,
        prompt: Optional[str] = None,
        static_prefix: Optional[str] = None,
        dynamic_payload: Optional[str] = None,
        use_context_cache: bool = False,
    ) -> list:
        if prompt is not None:
            return [{"role": "system", "content": prompt}]

        static_prefix = static_prefix or ""
        dynamic_payload = dynamic_payload or ""
        if use_context_cache:
            return [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": static_prefix,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
                {"role": "user", "content": dynamic_payload},
            ]
        return [
            {"role": "system", "content": static_prefix},
            {"role": "user", "content": dynamic_payload},
        ]

    def _build_extra_body(self, faction_name: str, role_profile: Optional[str]) -> Optional[dict]:
        if self._should_disable_thinking(faction_name, role_profile):
            return {"enable_thinking": False}
        return None

    def _build_json_schema_response_format(self, schema: dict, schema_name: str) -> dict:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": schema,
            },
        }

    def _trace_llm_call(
        self,
        trace_ctx: Optional[dict],
        component: str,
        duration_ms: int,
        success: bool,
        prompt_chars: int,
        response_chars: int,
        file_path: str,
        summary: str,
    ) -> None:
        trace_ctx = trace_ctx or {}
        trace_id = trace_ctx.get("trace_id")
        trace_logger = get_global_trace_logger()
        if trace_logger is not None and trace_id:
            trace_logger.log_step(
                trace_id,
                "LLM Calls",
                component=component,
                sim_time=int(trace_ctx.get("sim_time", 0)),
                duration_ms=duration_ms,
                success=success,
                prompt_chars=prompt_chars,
                response_chars=response_chars,
                file_path=file_path,
                summary=summary,
            )

    def _sync_chat_request(
        self,
        *,
        faction_name: str,
        role_profile: Optional[str] = None,
        trace_ctx: Optional[dict] = None,
        prompt: Optional[str] = None,
        static_prefix: Optional[str] = None,
        dynamic_payload: Optional[str] = None,
        response_format: Optional[dict] = None,
        structured_mode: str = "text",
    ) -> dict:
        trace_ctx = trace_ctx or {}
        model = self.get_llm_model(faction_name, role_profile=role_profile)
        use_context_cache = self._should_use_context_cache(faction_name, role_profile) and bool(static_prefix)
        extra_body = self._build_extra_body(faction_name, role_profile)
        prompt_text = self._build_prompt_text(prompt=prompt, static_prefix=static_prefix, dynamic_payload=dynamic_payload)
        messages = self._build_messages(
            prompt=prompt,
            static_prefix=static_prefix,
            dynamic_payload=dynamic_payload,
            use_context_cache=use_context_cache,
        )

        request_kwargs = {
            "model": model.model_name,
            "messages": messages,
        }
        if response_format is not None:
            request_kwargs["response_format"] = response_format
        if extra_body:
            request_kwargs["extra_body"] = extra_body

        started = time.perf_counter()
        content = ""
        success = False
        fallback_used = False
        try:
            response = model.client.chat.completions.create(**request_kwargs)
            content = self._normalize_content_text(response.choices[0].message.content)
            success = bool(content.strip())
        except Exception as e:
            logger.warning(
                "Chat request failed with role=%s model=%s structured_mode=%s cache=%s, retrying plain request: %s",
                role_profile,
                model.model_name,
                structured_mode,
                use_context_cache,
                e,
            )
            fallback_used = True
            try:
                response = model.client.chat.completions.create(
                    model=model.model_name,
                    messages=self._build_messages(
                        prompt=prompt_text,
                        static_prefix=None,
                        dynamic_payload=None,
                        use_context_cache=False,
                    ),
                )
                content = self._normalize_content_text(response.choices[0].message.content)
                success = bool(content.strip())
            except Exception as inner:
                logger.error("Sync chat error: %s", inner, exc_info=True)

        duration_ms = int((time.perf_counter() - started) * 1000)
        sim_time = int(trace_ctx.get("sim_time", 0))
        file_path = self._save_log(
            prompt_text,
            content,
            faction_name,
            sim_time,
            success,
            model_name=model.model_name,
            log_dir=self.sys2_log_dir,
            prefix="sys2",
        )

        component = trace_ctx.get("component", role_profile or "llm")
        summary = truncate_text(
            (
                f"model={model.model_name} mode={structured_mode} cache={use_context_cache} "
                f"fallback={fallback_used} {content}"
            ),
            220,
        )
        self._trace_llm_call(
            trace_ctx=trace_ctx,
            component=component,
            duration_ms=duration_ms,
            success=success,
            prompt_chars=len(prompt_text),
            response_chars=len(content),
            file_path=file_path,
            summary=summary,
        )

        return {
            "content": content,
            "success": success,
            "duration_ms": duration_ms,
            "file_path": file_path,
            "prompt_chars": len(prompt_text),
            "response_chars": len(content),
            "model_name": model.model_name,
            "structured_mode": structured_mode,
            "cache_enabled": use_context_cache,
            "fallback_used": fallback_used,
        }

    def _run_streamly(self, model: ChatModelBase, on_delta: Optional[Callable[[str], None]] = None) -> str:
        try:
            response = model.client.chat.completions.create(
                model=model.model_name,
                messages=model.history,
                stream=True,
            )
            full = ""
            for chunk in response:
                choice = chunk.choices[0].delta
                delta = getattr(choice, "content", None)
                if not delta:
                    continue
                full += self._sanitize_stream_delta(delta)
            if on_delta and full.strip():
                on_delta(full)
            return full
        except Exception as e:
            logger.error("[STREAM ERROR] %s", e, exc_info=True)
            return ""

    def get_llm_decision(
        self,
        state,
        faction_name: str,
        sim_time: int,
        show_fn: Optional[Callable[[str], None]] = None,
    ) -> Optional[dict]:
        side = self._normalize_side(faction_name)
        model = self.get_llm_model(side)
        logger.debug("[LLMDecision] faction=%s side=%s sim_time=%s", faction_name, side, sim_time)

        state_summary = self.summarizer.summarize_state(state, faction_name)
        prompt = get_strategic_prompt(
            faction_name=faction_name,
            sim_time=sim_time,
            state_summary=state_summary,
        )
        model.history = [{"role": "system", "content": prompt}]
        resp_str = self._run_streamly(model, on_delta=show_fn)

        if not resp_str.strip():
            self._save_log(prompt, resp_str, faction_name, sim_time, False, model_name=model.model_name, log_dir=self.log_dir)
            return None

        try:
            result = json_repair.loads(resp_str)
            success = True
        except Exception as e:
            success = False
            result = None
            if show_fn:
                show_fn(f"\n[WARN] JSON parse failed: {e}\n")

        self._save_log(prompt, resp_str, faction_name, sim_time, success, model_name=model.model_name, log_dir=self.log_dir)
        return result

    async def _async_chat_details(self, prompt: str, faction_name: str, trace_ctx: Optional[dict] = None) -> dict:
        return await asyncio.to_thread(self._sync_chat_call, prompt, faction_name, trace_ctx)

    async def async_chat(self, prompt: str, faction_name: str, trace_ctx: Optional[dict] = None) -> str:
        details = await self._async_chat_details(prompt, faction_name, trace_ctx)
        return details.get("content", "")

    def _sync_chat_call(self, prompt: str, faction_name: str, trace_ctx: Optional[dict] = None) -> dict:
        return self._sync_chat_request(
            faction_name=faction_name,
            trace_ctx=trace_ctx,
            prompt=prompt,
        )

    async def async_role_chat(
        self,
        static_prefix: str,
        dynamic_payload: str,
        faction_name: str,
        role_profile: str,
        trace_ctx: Optional[dict] = None,
    ) -> str:
        details = await asyncio.to_thread(
            self._sync_chat_request,
            faction_name=faction_name,
            role_profile=role_profile,
            trace_ctx=trace_ctx,
            static_prefix=static_prefix,
            dynamic_payload=dynamic_payload,
        )
        return details.get("content", "")

    async def async_structured_gen(
        self,
        *,
        schema: dict,
        static_prefix: str,
        dynamic_payload: str,
        faction_name: str,
        role_profile: str,
        trace_ctx: Optional[dict] = None,
    ) -> dict:
        trace_ctx = trace_ctx or {}
        model_name = self._resolve_model_name(faction_name, role_profile=role_profile)
        request_modes = []
        if self.red_enable_structured_output and self._supports_json_schema(model_name):
            request_modes.append(
                (
                    "json_schema",
                    self._build_json_schema_response_format(schema, schema.get("title", f"{role_profile}_schema")),
                )
            )
        if self.red_enable_structured_output and self._supports_json_object(model_name):
            request_modes.append(("json_object", {"type": "json_object"}))
        request_modes.append(("plain_text", None))

        last_details = None
        parse_error = ""
        for mode_name, response_format in request_modes:
            details = await asyncio.to_thread(
                self._sync_chat_request,
                faction_name=faction_name,
                role_profile=role_profile,
                trace_ctx=trace_ctx,
                static_prefix=static_prefix,
                dynamic_payload=dynamic_payload,
                response_format=response_format,
                structured_mode=mode_name,
            )
            last_details = details
            resp_str = (details.get("content") or "").strip()
            if not resp_str:
                parse_error = "empty_response"
                continue
            try:
                parse_started = time.perf_counter()
                parsed = json.loads(resp_str)
                parse_duration_ms = int((time.perf_counter() - parse_started) * 1000)
                self._trace_llm_call(
                    trace_ctx=trace_ctx,
                    component=f"{trace_ctx.get('component', role_profile)}.json_parse",
                    duration_ms=parse_duration_ms,
                    success=True,
                    prompt_chars=details.get("prompt_chars", 0),
                    response_chars=details.get("response_chars", 0),
                    file_path=details.get("file_path", ""),
                    summary=f"json_parse_ok mode={mode_name}",
                )
                return parsed if isinstance(parsed, dict) else {"actions": parsed} if isinstance(parsed, list) else {}
            except Exception as e:
                parse_error = str(e)
                logger.warning(
                    "Structured output parse failed. role=%s model=%s mode=%s error=%s",
                    role_profile,
                    model_name,
                    mode_name,
                    e,
                )
                self._trace_llm_call(
                    trace_ctx=trace_ctx,
                    component=f"{trace_ctx.get('component', role_profile)}.json_parse",
                    duration_ms=int((time.perf_counter() - parse_started) * 1000),
                    success=False,
                    prompt_chars=details.get("prompt_chars", 0),
                    response_chars=details.get("response_chars", 0),
                    file_path=details.get("file_path", ""),
                    summary=truncate_text(f"json_parse_failed mode={mode_name} error={e}", 220),
                )
                continue

        if last_details is not None:
            self._trace_llm_call(
                trace_ctx=trace_ctx,
                component=f"{trace_ctx.get('component', role_profile)}.json_parse",
                duration_ms=0,
                success=False,
                prompt_chars=last_details.get("prompt_chars", 0),
                response_chars=last_details.get("response_chars", 0),
                file_path=last_details.get("file_path", ""),
                summary=truncate_text(f"structured_exhausted error={parse_error}", 220),
            )
        return {}

    async def async_json_gen(self, prompt: str, faction_name: str, trace_ctx: Optional[dict] = None) -> list:
        details = await self._async_chat_details(prompt, faction_name, trace_ctx)
        resp_str = details.get("content", "")
        trace_ctx = trace_ctx or {}
        trace_id = trace_ctx.get("trace_id")
        component = trace_ctx.get("component", "llm")
        trace_logger = get_global_trace_logger()

        try:
            parse_started = time.perf_counter()
            parsed = json_repair.loads(resp_str)
            parse_duration_ms = int((time.perf_counter() - parse_started) * 1000)
            if trace_logger is not None and trace_id:
                action_count = len(parsed) if isinstance(parsed, list) else len(parsed.get("actions", [])) if isinstance(parsed, dict) else 0
                trace_logger.log_step(
                    trace_id,
                    "LLM Calls",
                    component=f"{component}.json_parse",
                    sim_time=trace_ctx.get("sim_time"),
                    duration_ms=parse_duration_ms,
                    success=True,
                    prompt_chars=details.get("prompt_chars", 0),
                    response_chars=details.get("response_chars", 0),
                    file_path=details.get("file_path"),
                    summary=f"json_parse_ok action_count={action_count}",
                )
            return parsed
        except Exception as e:
            logger.error("Async JSON parse error: %s", e)
            if trace_logger is not None and trace_id:
                trace_logger.log_step(
                    trace_id,
                    "LLM Calls",
                    component=f"{component}.json_parse",
                    sim_time=trace_ctx.get("sim_time"),
                    duration_ms=int((time.perf_counter() - parse_started) * 1000),
                    success=False,
                    prompt_chars=details.get("prompt_chars", 0),
                    response_chars=details.get("response_chars", 0),
                    file_path=details.get("file_path"),
                    summary=truncate_text(str(e), 200),
                )
            return []
