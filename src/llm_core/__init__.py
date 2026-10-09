"""LLM Core â€” streaming, non-streaming, and fallback LLM calls with multi-provider support."""

from src.llm_core._core import (
    _FallbackIneligibleHTTPException,
    _alias_harmony_tools,
    _annotate_usage_model,
    _get_cached_response_model,
    _model_actual_event,
    _normalize_http_status,
    _normalize_usage_counts,
    _reported_model_name,
    _scrub_openai_chat_tool_reasoning,
    _unalias_harmony_tool_name,
    apply_kimi_code_headers_async,
    ANTHROPIC_MODELS,
    DEAD_HOST_COOLDOWN,
    KIMI_CODE_USER_AGENT,
    KIMI_CODE_USER_AGENTS,
    LLMConfig,
    _DegenerateStreamGuard,
    _HarmonyStreamRouter,
    _MISTRAL_REASONING_EFFORT,
    _THINKING_MODEL_PATTERNS,
    _apply_local_cache_affinity,
    _apply_local_generation_stability,
    _anthropic_rejects_temperature,
    _build_anthropic_headers,
    _build_anthropic_payload,
    _build_chatgpt_responses_payload,
    _build_ollama_payload,
    _call_timeout,
    _clear_host_dead,
    _detect_provider,
    _format_chatgpt_subscription_error,
    _format_upstream_error,
    _get_cache_key,
    _get_cached_response,
    _get_http_client,
    _host_key,
    _host_match,
    _is_host_dead,
    _is_kimi_code_access_denied,
    _is_kimi_code_url,
    _is_self_hosted_openai_compatible,
    _is_ollama_native_url,
    _is_ollama_openai_compat_url,
    _local_model_slot,
    _kimi_code_base_key,
    _kimi_code_ua_cache,
    _kimi_code_ua_candidates,
    _mark_host_dead,
    _moonshot_rejects_custom_temperature,
    _normalize_anthropic_url,
    _normalize_chatgpt_subscription_url,
    _normalize_mistral_content,
    _normalize_ollama_url,
    _normalize_openai_chat_url,
    _ollama_api_root,
    _omit_temperature,
    _parse_anthropic_response,
    _parse_ollama_response,
    _provider_headers,
    _provider_label,
    _restricts_temperature,
    _same_model_identity,
    _sanitize_llm_messages,
    _set_cached_response,
    _stream_delta_event,
    _stream_timeout,
    _strip_visible_chat_template_artifacts,
    _supports_thinking,
    _uses_max_completion_tokens,
    apply_kimi_code_headers,
    httpx_get_kimi_aware,
    httpx_post_kimi_aware,
    httpx_post_kimi_aware_async,
    list_model_ids,
    normalize_model_id,
    note_model_activity,
    _remember_kimi_code_user_agent,
    _response_cache,
    seconds_since_model_activity,
    _LOCAL_MODEL_LOCK,
    _LOCAL_MODEL_WAITING_FOREGROUND,
    _LOCAL_MODEL_CURRENT,
    _local_model_gate_enabled,
    _gate_workload,
    _cache_header_identity,
    _response_model_cache,
    _HOST_FAIL_THRESHOLD,
    _dead_hosts,
    _host_fails,
    _host_health_lock,
    _model_activity,
    _HARMONY_MARKER_RE,
    _HARMONY_MARKERS,
    _HARMONY_MAX_MARKER_LEN,
    _VISIBLE_CHAT_TEMPLATE_ARTIFACT_RE,
    _harmony_suffix_hold_len,
    _DEGENERATE_WORD_RE,
    _model_activity_key,
    _http_client,
    _http_limits,
    _ollama_normalize_messages,
    _ollama_normalize_tool_messages,
    _is_local_minimax_mlx_request,
    _is_openai_hosted_chat_url,
    _model_disallows_reasoning_effort_with_chat_tools,
    _HARMONY_TOOL_ALIASES,
    _HARMONY_TOOL_ALIASES_REVERSE,
    _is_harmony_model,
    _message_content_as_text,
    _chatgpt_subscription_instructions,
    _MAX_COMPLETION_TOKENS_MODELS,
    _FIXED_TEMPERATURE_MODELS,
    _convert_openai_content_to_anthropic,
    _as_content_blocks,
    _is_untrusted_context_content,
    _REFERENCE_CONTEXT_BOUNDARY,
    _model_list_base,
    _parse_model_cache,
    _configured_cached_model_ids,
)

# Stdlib imports needed by the API functions
import httpx
import asyncio
import time
import json
import logging
import hashlib
import threading
import re
import os
from contextlib import asynccontextmanager
from fastapi import HTTPException
from typing import Optional, Dict, List, Tuple
from src.model_context import get_context_length, DEFAULT_CONTEXT, is_local_endpoint
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Imports from the helpers/providers/params sub-module


# ============================================================
# Public API
# ============================================================

def llm_call(url: str, model: str, messages: List[Dict], temperature: float = LLMConfig.DEFAULT_TEMPERATURE,
             max_tokens: int = LLMConfig.DEFAULT_MAX_TOKENS, headers: Optional[Dict] = None,
             timeout: int = LLMConfig.DEFAULT_TIMEOUT, prompt_type: Optional[str] = None) -> str:
    """Synchronous LLM call with optional prompt type enhancement."""
    h = _provider_headers(_detect_provider(url))
    # Tolerate headers that arrive as a JSON string (some sessions stored them
    # double-encoded) â€” otherwise h.update() throws "dictionary update sequence
    # element #0 has length 1; 2 is required".
    if isinstance(headers, str):
        try:
            headers = json.loads(headers)
        except Exception:
            headers = None
    if isinstance(headers, dict):
        h.update(headers)

    messages_copy = _sanitize_llm_messages(messages)

    # Consolidate multiple system messages into one at the start.
    sys_parts = []
    non_sys = []
    for m in messages_copy:
        if m.get("role") == "system":
            sys_parts.append(m.get('content') or '')
        else:
            non_sys.append(m)
    if sys_parts:
        messages_copy = [{"role": "system", "content": "\n\n".join(sys_parts)}] + non_sys
    else:
        messages_copy = non_sys

    provider = _detect_provider(url)
    cache_key = _get_cache_key(
        url, model, messages_copy, temperature, max_tokens, headers=headers,
    )
    cached_response = _get_cached_response(cache_key)
    if cached_response:
        logger.debug(f"Returning cached response for key: {cache_key}")
        return cached_response

    if provider == "anthropic":
        target_url = _normalize_anthropic_url(url)
        h = _build_anthropic_headers(headers)
        payload = _build_anthropic_payload(model, messages_copy, temperature, max_tokens)
    elif provider == "ollama":
        target_url = _normalize_ollama_url(url)
        payload = _build_ollama_payload(
            model, messages_copy, temperature, max_tokens,
            stream=False, num_ctx=get_context_length(url, model),
        )
    else:
        target_url = _normalize_openai_chat_url(url)
        if provider == "copilot":
            from src.copilot import apply_request_headers
            apply_request_headers(h, messages_copy)
        payload = {
            "model": model,
            "messages": messages_copy,
            "temperature": temperature,
        }
        if _omit_temperature(provider, model):
            payload.pop("temperature", None)
        if max_tokens and max_tokens > 0:
            tok_key = "max_completion_tokens" if _uses_max_completion_tokens(model) else "max_tokens"
            payload[tok_key] = max_tokens
        _apply_local_generation_stability(payload, target_url, model)
        if provider == "mistral" and _supports_thinking(model):
            payload["reasoning_effort"] = _MISTRAL_REASONING_EFFORT
    try:
        note_model_activity(target_url, model)
        r = httpx_post_kimi_aware(target_url, h, json=payload, timeout=timeout)
    except Exception as e:
        raise HTTPException(502, f"POST {target_url} failed: {e}")
    if not r.is_success:
        raise HTTPException(502, f"Upstream {target_url} -> {r.status_code}: {r.text}")
    data = r.json()
    try:
        if provider == "anthropic":
            response = _parse_anthropic_response(data)
        elif provider == "ollama":
            response = _parse_ollama_response(data)
        else:
            msg = data["choices"][0]["message"]
            content = msg.get("content")
            if isinstance(content, list):
                # Mistral structured content â€” extract thinking + text
                text_part, thinking_part = _normalize_mistral_content(content)
                if thinking_part:
                    response = thinking_part + "\n\n" + (text_part or "")
                else:
                    response = text_part or msg.get("reasoning_content") or ""
            else:
                response = content or msg.get("reasoning_content") or ""
        _set_cached_response(cache_key, response)
        return response
    except Exception:
        raise HTTPException(502, f"Unexpected schema from {target_url}: {str(data)[:400]}")


def _dedupe_candidates(candidates):
    """Filter malformed entries and drop a later repeat of an already-seen
    ``(url, model)`` route, preserving order (first occurrence wins).

    The chain is the primary target followed by the configured fallbacks, so a
    fallback that repeats the session's current model â€” a common misconfiguration,
    since callers prepend the live ``(url, model)`` to ``default_model_fallbacks``
    â€” would otherwise make the chain re-attempt the very route that just failed:
    a wasted round-trip plus a spurious ``fallback`` notice for a switch that did
    not happen. Headers are not part of the key; the first tuple (with its
    headers) is the one kept.
    """
    seen = set()
    out = []
    for c in candidates or []:
        if not c or not c[0] or not c[1]:
            continue
        key = (c[0], c[1])
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def llm_call_with_fallback(candidates, messages, **kwargs) -> str:
    """Sync `llm_call` with an ordered fallback chain.

    `candidates` is a list of (url, model, headers). The first one that returns
    without an exception wins. Connection / 5xx-style failures fall through to
    the next candidate. The dead-host cooldown inside `llm_call` makes repeat
    attempts at an offline primary effectively free.
    """
    cands = dedupe_model_candidates(candidates)
    if not cands:
        raise HTTPException(503, "No model endpoint configured")
    last_err = None
    for i, (url, model, headers) in enumerate(cands):
        try:
            return llm_call(url, model, messages, headers=headers, **kwargs)
        except Exception as e:
            last_err = e
            tag = "primary" if i == 0 else "candidate"
            logger.warning(f"[fallback] {tag} {model} failed ({type(e).__name__}); trying next")
            continue
    raise last_err if last_err else HTTPException(503, "All fallback candidates failed")


async def llm_call_async_with_fallback(candidates, messages, **kwargs) -> str:
    """Async variant of `llm_call_with_fallback` â€” same semantics."""
    cands = dedupe_model_candidates(candidates)
    if not cands:
        raise HTTPException(503, "No model endpoint configured")
    last_err = None
    for i, (url, model, headers) in enumerate(cands):
        try:
            return await llm_call_async(url, model, messages, headers=headers, **kwargs)
        except Exception as e:
            last_err = e
            tag = "primary" if i == 0 else "candidate"
            logger.warning(f"[fallback] {tag} {model} failed ({type(e).__name__}); trying next")
            continue
    raise last_err if last_err else HTTPException(503, "All fallback candidates failed")


async def llm_call_async(
    url: str,
    model: str,
    messages: List[Dict],
    temperature: float = LLMConfig.DEFAULT_TEMPERATURE,
    max_tokens: int = LLMConfig.DEFAULT_MAX_TOKENS,
    headers: Optional[Dict] = None,
    timeout: int = LLMConfig.STREAM_TIMEOUT,
    max_retries: int = LLMConfig.MAX_RETRIES,
    prompt_type: Optional[str] = None,
    session_id: Optional[str] = None,
    workload: str = "foreground",
    availability_only_transport: bool = False,
    return_model_metadata: bool = False,
) -> str | tuple[str, str]:
    """Asynchronous LLM call using httpx with connection pooling, timeout, retry logic, and performance logging."""
    provider = _detect_provider(url)
    messages_copy = _sanitize_llm_messages(messages)

    # Consolidate multiple system messages into one at the start.
    sys_parts = []
    non_sys = []
    for m in messages_copy:
        if m.get("role") == "system":
            sys_parts.append(m.get('content') or '')
        else:
            non_sys.append(m)
    if sys_parts:
        messages_copy = [{"role": "system", "content": "\n\n".join(sys_parts)}] + non_sys
    else:
        messages_copy = non_sys

    cache_key = _get_cache_key(
        url, model, messages_copy, temperature, max_tokens, headers=headers,
    )
    cached_response = _get_cached_response(cache_key)
    if cached_response:
        logger.debug(f"Returning cached response for key: {cache_key}")
        if return_model_metadata:
            return cached_response, (_get_cached_response_model(cache_key) or model)
        return cached_response

    if provider == "chatgpt-subscription":
        # Codex Subscription requires streamed Responses requests even for callers
        # that want a plain string (auto-title, memory extraction, etc.).
        # Reuse stream_llm's validated Codex SSE path and collect deltas.
        parts: List[str] = []
        actual_model = model
        async for chunk in stream_llm(
            url,
            model,
            messages_copy,
            temperature=temperature,
            max_tokens=max_tokens,
            headers=headers,
            timeout=timeout,
            workload=workload,
        ):
            event_is_error = False
            for line in str(chunk).splitlines():
                if line.startswith("event:"):
                    event_is_error = line[6:].strip() == "error"
                    continue
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw:
                    continue
                if raw == "[DONE]":
                    response = "".join(parts)
                    _set_cached_response(
                        cache_key,
                        response,
                        actual_model=actual_model,
                    )
                    return (
                        (response, actual_model)
                        if return_model_metadata
                        else response
                    )
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if event_is_error or data.get("error") or (data.get("status") and data.get("text")):
                    status = int(data.get("status") or 502)
                    text = data.get("text") or data.get("error") or "Codex Subscription request failed"
                    error_type = (
                        _FallbackIneligibleHTTPException
                        if data.get("fallback_eligible") is False
                        else HTTPException
                    )
                    raise error_type(status, text)
                if data.get("type") == "model_actual":
                    reported_model = data.get("model")
                    if isinstance(reported_model, str) and reported_model.strip():
                        actual_model = reported_model.strip()
                delta = data.get("delta")
                if isinstance(delta, str):
                    parts.append(delta)
        response = "".join(parts)
        _set_cached_response(cache_key, response, actual_model=actual_model)
        return (response, actual_model) if return_model_metadata else response

    if provider == "anthropic":
        target_url = _normalize_anthropic_url(url)
        h = _build_anthropic_headers(headers)
        payload = _build_anthropic_payload(model, messages_copy, temperature, max_tokens)
    elif provider == "ollama":
        target_url = _normalize_ollama_url(url)
        h = {"Content-Type": "application/json"}
        if headers:
            h.update(headers)
        payload = _build_ollama_payload(
            model, messages_copy, temperature, max_tokens,
            stream=False, num_ctx=get_context_length(url, model),
        )
    else:
        target_url = _normalize_openai_chat_url(url)
        h = _provider_headers(provider, headers)
        if provider == "copilot":
            from src.copilot import apply_request_headers
            apply_request_headers(h, messages_copy)
        payload = {
            "model": model,
            "messages": messages_copy,
            "temperature": temperature,
        }
        if _omit_temperature(provider, model):
            payload.pop("temperature", None)
        if max_tokens and max_tokens > 0:
            tok_key = "max_completion_tokens" if _uses_max_completion_tokens(model) else "max_tokens"
            payload[tok_key] = max_tokens
        # Suppress thinking for qwen3/gemma4 on Ollama /v1 â€” same as stream_llm.
        if _is_ollama_openai_compat_url(url) and _supports_thinking(model):
            payload["think"] = False
        if provider == "mistral" and _supports_thinking(model):
            payload["reasoning_effort"] = _MISTRAL_REASONING_EFFORT
        _apply_local_cache_affinity(payload, url, session_id)
        _apply_local_generation_stability(payload, target_url, model)

    if _is_host_dead(target_url):
        raise HTTPException(503, f"Upstream {_host_key(target_url)} marked unreachable (cooldown active)")

    call_timeout = _call_timeout(timeout)
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        start = time.time()
        try:
            async with _local_model_slot(target_url, model, workload):
                note_model_activity(target_url, model)
                client = _get_http_client()
                r = await httpx_post_kimi_aware_async(client, target_url, h, json=payload, timeout=call_timeout)
            duration = time.time() - start
            if not r.is_success:
                friendly = _format_upstream_error(r.status_code, r.text, target_url)
                logger.warning(
                    f"LLM async call to {target_url} failed in {duration:.2f}s "
                    f"(attempt {attempt}): HTTP {r.status_code} {friendly}"
                )
                if r.status_code in (429, 502, 503, 504) and attempt < max_retries:
                    await asyncio.sleep(LLMConfig.RETRY_DELAY)
                    continue
                raise HTTPException(r.status_code, friendly)
            logger.info(f"LLM async call to {target_url} succeeded in {duration:.2f}s (attempt {attempt})")
            _clear_host_dead(target_url)
            data = r.json()
            if isinstance(data, dict) and data.get("error"):
                provider_error = data["error"]
                status = _provider_stream_error_status(provider_error, default=400)
                if isinstance(provider_error, dict):
                    detail = provider_error.get("message") or provider_error.get("type") or str(provider_error)
                else:
                    detail = str(provider_error)
                raise HTTPException(status, detail or "Upstream request failed")
            try:
                reported_model = data.get("model") if isinstance(data, dict) else None
                actual_model = (
                    reported_model.strip()
                    if isinstance(reported_model, str) and reported_model.strip()
                    else model
                )
                if provider == "anthropic":
                    response = _parse_anthropic_response(data)
                elif provider == "ollama":
                    response = _parse_ollama_response(data)
                else:
                    msg = data["choices"][0]["message"]
                    content = msg.get("content")
                    if isinstance(content, list):
                        # Mistral structured content â€” extract thinking + text
                        # (same contract as llm_call / stream_llm; see #5435).
                        text_part, thinking_part = _normalize_mistral_content(content)
                        if thinking_part:
                            response = thinking_part + "\n\n" + (text_part or "")
                        else:
                            response = text_part or msg.get("reasoning_content") or ""
                    else:
                        response = content or msg.get("reasoning_content") or ""
                _set_cached_response(
                    cache_key,
                    response,
                    actual_model=actual_model,
                )
                return (
                    (response, actual_model)
                    if return_model_metadata
                    else response
                )
            except HTTPException:
                raise
            except Exception:
                raise _FallbackIneligibleHTTPException(
                    502,
                    f"Unexpected schema from {target_url}: {str(data)[:400]}",
                )
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            _cooled = _mark_host_dead(target_url)
            duration = time.time() - start
            _tail = f" â€” host cooled for {DEAD_HOST_COOLDOWN:.0f}s" if _cooled else " â€” transient, will retry"
            logger.warning(f"LLM async connect to {target_url} failed after {duration:.2f}s: {e}{_tail}")
            if _cooled or attempt >= max_retries:
                raise HTTPException(503, f"Cannot reach {_host_key(target_url)}: {e}")
            await asyncio.sleep(LLMConfig.RETRY_DELAY)
        except httpx.ReadTimeout as e:
            duration = time.time() - start
            logger.warning(f"LLM async read timed out after {duration:.2f}s: {e}")
            if attempt >= max_retries:
                raise HTTPException(504, f"POST {target_url} timed out after {max_retries} attempts")
            await asyncio.sleep(LLMConfig.RETRY_DELAY)
        except httpx.PoolTimeout as e:
            duration = time.time() - start
            logger.warning(f"LLM async connection pool timed out after {duration:.2f}s: {e}")
            if availability_only_transport:
                raise HTTPException(
                    504,
                    f"POST {target_url} could not acquire an upstream connection",
                )
            if attempt >= max_retries:
                raise HTTPException(504, f"POST {target_url} timed out after {max_retries} attempts")
            await asyncio.sleep(LLMConfig.RETRY_DELAY)
        except httpx.WriteTimeout as e:
            duration = time.time() - start
            logger.warning(f"LLM async upstream timeout after {duration:.2f}s: {e}")
            if availability_only_transport:
                raise _FallbackIneligibleHTTPException(
                    504,
                    f"POST {target_url} failed during request delivery",
                )
            if attempt >= max_retries:
                raise HTTPException(504, f"POST {target_url} timed out after {max_retries} attempts")
            await asyncio.sleep(LLMConfig.RETRY_DELAY)
        except httpx.ProtocolError as e:
            duration = time.time() - start
            logger.warning(f"LLM async protocol failure after {duration:.2f}s: {e}")
            if availability_only_transport:
                raise _FallbackIneligibleHTTPException(
                    502,
                    f"POST {target_url} failed with a protocol error",
                )
            if attempt >= max_retries:
                raise HTTPException(502, f"POST {target_url} failed after {max_retries} attempts: {e}")
            await asyncio.sleep(LLMConfig.RETRY_DELAY)
        except httpx.NetworkError as e:
            duration = time.time() - start
            logger.warning(f"LLM async network failure after {duration:.2f}s: {e}")
            if availability_only_transport:
                raise _FallbackIneligibleHTTPException(
                    502,
                    f"POST {target_url} failed with a network error",
                )
            if attempt >= max_retries:
                raise HTTPException(502, f"POST {target_url} failed after {max_retries} attempts: {e}")
            await asyncio.sleep(LLMConfig.RETRY_DELAY)
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else 502
            raise HTTPException(status, str(e))
        except httpx.RequestError as e:
            duration = time.time() - start
            logger.warning(f"LLM async request configuration failed after {duration:.2f}s: {e}")
            raise _FallbackIneligibleHTTPException(
                502,
                f"POST {target_url} could not be configured: {e}",
            )

def _stream_target_url(url: str) -> str:
    provider = _detect_provider(url)
    if provider == "anthropic":
        return _normalize_anthropic_url(url)
    if provider == "ollama":
        return _normalize_ollama_url(url)
    if provider == "chatgpt-subscription":
        return _normalize_chatgpt_subscription_url(url)
    return _normalize_openai_chat_url(url)


async def stream_llm(url: str, model: str, messages: List[Dict], temperature: float = LLMConfig.DEFAULT_TEMPERATURE,
                     max_tokens: int = LLMConfig.DEFAULT_MAX_TOKENS, headers: Optional[Dict] = None,
                     timeout: int = LLMConfig.STREAM_TIMEOUT, prompt_type: Optional[str] = None,
                     tools: Optional[List[Dict]] = None, session_id: Optional[str] = None,
                     tool_choice_none: bool = False, workload: str = "foreground"):
    target_url = _stream_target_url(url)
    async with _local_model_slot(target_url, model, workload):
        async for chunk in _stream_llm_inner(
            url,
            model,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            headers=headers,
            timeout=timeout,
            prompt_type=prompt_type,
            tools=tools,
            session_id=session_id,
            tool_choice_none=tool_choice_none,
        ):
            yield chunk


async def _stream_llm_inner(url: str, model: str, messages: List[Dict], temperature: float = LLMConfig.DEFAULT_TEMPERATURE,
                            max_tokens: int = LLMConfig.DEFAULT_MAX_TOKENS, headers: Optional[Dict] = None,
                            timeout: int = LLMConfig.STREAM_TIMEOUT, prompt_type: Optional[str] = None,
                            tools: Optional[List[Dict]] = None, session_id: Optional[str] = None,
                            tool_choice_none: bool = False):
    """Stream LLM responses with improved error handling.

    Yields SSE chunks:
      - data: {"delta": "text"}           â€” text content
      - data: {"type": "tool_calls", ...}  â€” accumulated native tool calls (before DONE)
      - event: error                       â€” errors
      - data: [DONE]                       â€” end of stream
    """
    provider = _detect_provider(url)
    messages_copy = _sanitize_llm_messages(messages)

    # Consolidate multiple system messages into one at the start.
    # Some models (e.g. Qwen3.5) reject system messages that aren't first.
    sys_parts = []
    non_sys = []
    for m in messages_copy:
        if m.get("role") == "system":
            sys_parts.append(m.get('content') or '')
        else:
            non_sys.append(m)
    if sys_parts:
        messages_copy = [{"role": "system", "content": "\n\n".join(sys_parts)}] + non_sys
    else:
        messages_copy = non_sys

    if provider == "anthropic":
        target_url = _normalize_anthropic_url(url)
        h = _build_anthropic_headers(headers)
        payload = _build_anthropic_payload(model, messages_copy, temperature, max_tokens, stream=True, tools=tools)
    elif provider == "ollama":
        target_url = _normalize_ollama_url(url)
        h = {"Content-Type": "application/json"}
        if headers:
            h.update(headers)
        payload = _build_ollama_payload(
            model, messages_copy, temperature, max_tokens,
            stream=True, tools=tools, num_ctx=get_context_length(url, model),
        )
    elif provider == "chatgpt-subscription":
        target_url = _normalize_chatgpt_subscription_url(url)
        h = _provider_headers(provider, headers)
        payload = _build_chatgpt_responses_payload(model, messages_copy, temperature, max_tokens, stream=True)
    else:
        target_url = _normalize_openai_chat_url(url)
        payload = {
            "model": model,
            "messages": messages_copy,
            "temperature": temperature,
            "stream": True,
        }
        if _omit_temperature(provider, model):
            payload.pop("temperature", None)
        if provider not in {"openrouter", "groq"}:
            payload["stream_options"] = {"include_usage": True}
        if max_tokens and max_tokens > 0:
            tok_key = "max_completion_tokens" if _uses_max_completion_tokens(model) else "max_tokens"
            payload[tok_key] = max_tokens
        if tools:
            payload["tools"] = _alias_harmony_tools(tools, model)
        elif tool_choice_none:
            payload["tool_choice"] = "none"
        # Mistral thinking-capable models â€” send reasoning_effort so Mistral
        # activates thinking mode and returns structured reasoning_content.
        # Effort level is configurable via ODYSSEUS_MISTRAL_REASONING_EFFORT
        # (high / medium / low / none); default "high".
        if provider == "mistral" and _supports_thinking(model):
            payload["reasoning_effort"] = _MISTRAL_REASONING_EFFORT
        # For Ollama's OpenAI-compat /v1 endpoint with thinking models (qwen3,
        # gemma4, etc.), suppress thinking so tool calls aren't swallowed inside
        # <think> blocks. Ollama /v1 accepts "think": false as a top-level param.
        if _is_ollama_openai_compat_url(url) and _supports_thinking(model):
            payload["think"] = False
        _apply_local_cache_affinity(payload, url, session_id)
        _apply_local_generation_stability(payload, target_url, model)
        _scrub_openai_chat_tool_reasoning(payload, target_url, model)
        h = _provider_headers(provider, headers)
        if provider == "copilot":
            from src.copilot import apply_request_headers
            apply_request_headers(h, messages_copy)

    # Connect budget from LLMConfig.CONNECT_TIMEOUT (env LLM_CONNECT_TIMEOUT).
    # The dead-host cooldown still bounds a genuinely unreachable upstream, so a
    # wider connect budget only affects first contact and stops a brief cold
    # connect blip (offshore/public endpoints) surfacing as a 503 on this stream
    # path, which -- unlike llm_call -- does not retry the connect.
    stream_timeout = _stream_timeout(timeout)

    if _is_host_dead(target_url):
        yield f'event: error\ndata: {json.dumps({"error": f"Upstream {_host_key(target_url)} unreachable (cooldown active)", "status": 503})}\n\n'
        return
    note_model_activity(target_url, model)
    degenerate_guard = _DegenerateStreamGuard(model)

    # â”€â”€ Codex Subscription Responses streaming â”€â”€
    if provider == "chatgpt-subscription":
        event_name = ""
        input_tokens = 0
        output_tokens = 0
        _responses_actual_model = ""
        _responses_model_announced = False
        try:
            client = _get_http_client()
            async with client.stream('POST', target_url, json=payload, headers=h, timeout=stream_timeout) as r:
                _clear_host_dead(target_url)
                if r.status_code != 200:
                    raw = (await r.aread()).decode(errors="replace")
                    friendly = _format_chatgpt_subscription_error(r.status_code, raw)
                    yield f'event: error\ndata: {json.dumps({"status": r.status_code, "text": friendly, "raw": raw[:500]})}\n\n'
                    return
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("event:"):
                        event_name = line[6:].strip()
                        continue
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw:
                        continue
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    evt = data.get("type") or event_name
                    response_data = data.get("response") or {}
                    reported_model = (
                        response_data.get("model")
                        if isinstance(response_data, dict)
                        else None
                    )
                    reported_model = _reported_model_name(
                        reported_model or data.get("model")
                    )
                    if reported_model:
                        _responses_actual_model = reported_model
                        if not _responses_model_announced:
                            model_event = _model_actual_event(model, reported_model)
                            if model_event:
                                _responses_model_announced = True
                                yield model_event
                    if evt == "response.output_text.delta":
                        delta = data.get("delta") or ""
                        if delta:
                            _degenerate = degenerate_guard.check(delta)
                            if _degenerate:
                                yield _degenerate
                                return
                            yield f'data: {json.dumps({"delta": delta})}\n\n'
                    elif evt == "response.completed":
                        usage = (data.get("response") or {}).get("usage") or data.get("usage") or {}
                        if isinstance(usage, dict):
                            raw_input = (
                                usage.get("input_tokens")
                                if "input_tokens" in usage
                                else usage.get("prompt_tokens", input_tokens)
                            )
                            raw_output = (
                                usage.get("output_tokens")
                                if "output_tokens" in usage
                                else usage.get("completion_tokens", output_tokens)
                            )
                            normalized_usage = _normalize_usage_counts(
                                raw_input,
                                raw_output,
                            )
                            if normalized_usage and (
                                "input_tokens" in usage
                                or "prompt_tokens" in usage
                                or "output_tokens" in usage
                                or "completion_tokens" in usage
                            ):
                                _annotate_usage_model(
                                    normalized_usage,
                                    model,
                                    _responses_actual_model,
                                )
                                yield f'data: {json.dumps({"type": "usage", "data": normalized_usage})}\n\n'
                        yield "data: [DONE]\n\n"
                        return
                    elif evt in ("response.failed", "error"):
                        err = data.get("error") or (data.get("response") or {}).get("error") or {}
                        if evt == "error" and not err:
                            # Responses API ``error`` events carry code/message
                            # at the top level, unlike ``response.failed``.
                            err = {
                                key: data[key]
                                for key in ("type", "code", "message", "status", "status_code", "http_status")
                                if key in data
                            }
                        text = err.get("message") if isinstance(err, dict) else str(err or "Codex Subscription request failed")
                        status = _provider_stream_error_status(err, default=400)
                        yield f'event: error\ndata: {json.dumps({"status": status, "text": text})}\n\n'
                        return
                yield "data: [DONE]\n\n"
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            _cooled = _mark_host_dead(target_url)
            _tail = f" â€” host cooled for {DEAD_HOST_COOLDOWN:.0f}s" if _cooled else " â€” transient, will retry"
            logger.warning(f"Codex Subscription stream connect to {target_url} failed: {e}{_tail}")
            yield f'event: error\ndata: {json.dumps({"error": f"Cannot reach {_host_key(target_url)}", "status": 503})}\n\n'
        except httpx.ReadTimeout:
            yield f'event: error\ndata: {json.dumps({"error": "Read timeout", "status": 504})}\n\n'
        except httpx.PoolTimeout:
            yield f'event: error\ndata: {json.dumps({"error": "Connection pool timeout", "status": 504})}\n\n'
        except httpx.WriteTimeout:
            yield f'event: error\ndata: {json.dumps({"error": "Upstream timeout", "status": 504, "fallback_eligible": False})}\n\n'
        except httpx.ProtocolError:
            yield f'event: error\ndata: {json.dumps({"error": "Upstream protocol error", "status": 502, "fallback_eligible": False})}\n\n'
        except httpx.NetworkError:
            yield f'event: error\ndata: {json.dumps({"error": "Network error", "status": 502, "fallback_eligible": False})}\n\n'
        except Exception as e:
            logger.error(f"Codex Subscription stream error: {e}")
            yield f'event: error\ndata: {json.dumps({"error": str(e), "status": 502, "fallback_eligible": False})}\n\n'
        return

    # â”€â”€ Native Ollama streaming â”€â”€
    if provider == "ollama":
        _ollama_tool_calls: List[Dict] = []
        _harmony_router = _HarmonyStreamRouter()
        _ollama_actual_model = ""
        _ollama_model_announced = False
        try:
            client = _get_http_client()
            async with client.stream('POST', target_url, json=payload, headers=h, timeout=stream_timeout) as r:
                _clear_host_dead(target_url)
                if r.status_code != 200:
                    raw = (await r.aread()).decode(errors="replace")
                    friendly = _format_upstream_error(r.status_code, raw, target_url)
                    yield f'event: error\ndata: {json.dumps({"status": r.status_code, "text": friendly, "raw": raw[:500]})}\n\n'
                    return
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    try:
                        j = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if j.get("error"):
                        err = j.get("error")
                        status = _provider_stream_error_status(err, default=400)
                        text = err.get("message") if isinstance(err, dict) else str(err)
                        yield f'event: error\ndata: {json.dumps({"error": text or "Ollama request failed", "status": status})}\n\n'
                        return
                    reported_model = _reported_model_name(j.get("model"))
                    if reported_model:
                        _ollama_actual_model = reported_model
                        if not _ollama_model_announced:
                            model_event = _model_actual_event(model, reported_model)
                            if model_event:
                                _ollama_model_announced = True
                                yield model_event
                    message = j.get("message") or {}
                    thinking = message.get("thinking") or ""
                    if thinking:
                        yield _stream_delta_event(thinking, thinking=True)
                    content = message.get("content") or ""
                    if content:
                        for part, is_thinking in _harmony_router.feed(content):
                            yield _stream_delta_event(part, thinking=is_thinking)
                    for tc in message.get("tool_calls") or []:
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            _ollama_tool_calls.append({
                                "id": tc.get("id") or f"call_{len(_ollama_tool_calls)}",
                                "name": _unalias_harmony_tool_name(fn.get("name") or "", model),
                                "arguments": json.dumps(fn.get("arguments") or {}),
                            })
                    if j.get("done"):
                        for part, is_thinking in _harmony_router.flush():
                            yield _stream_delta_event(part, thinking=is_thinking)
                        if _ollama_tool_calls:
                            yield f'data: {json.dumps({"type": "tool_calls", "calls": _ollama_tool_calls})}\n\n'
                        if j.get("prompt_eval_count") is not None or j.get("eval_count") is not None:
                            normalized_usage = _normalize_usage_counts(
                                j.get("prompt_eval_count", 0),
                                j.get("eval_count", 0),
                            )
                            if normalized_usage:
                                _annotate_usage_model(
                                    normalized_usage,
                                    model,
                                    _ollama_actual_model,
                                )
                                yield f'data: {json.dumps({"type": "usage", "data": normalized_usage})}\n\n'
                        yield "data: [DONE]\n\n"
                        return
                for part, is_thinking in _harmony_router.flush():
                    yield _stream_delta_event(part, thinking=is_thinking)
                yield "data: [DONE]\n\n"
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            _cooled = _mark_host_dead(target_url)
            _tail = f" â€” host cooled for {DEAD_HOST_COOLDOWN:.0f}s" if _cooled else " â€” transient, will retry"
            logger.warning(f"Ollama stream connect to {target_url} failed: {e}{_tail}")
            yield f'event: error\ndata: {json.dumps({"error": f"Cannot reach {_host_key(target_url)}", "status": 503})}\n\n'
        except httpx.ReadTimeout:
            yield f'event: error\ndata: {json.dumps({"error": "Read timeout", "status": 504})}\n\n'
        except httpx.PoolTimeout:
            yield f'event: error\ndata: {json.dumps({"error": "Connection pool timeout", "status": 504})}\n\n'
        except httpx.WriteTimeout:
            yield f'event: error\ndata: {json.dumps({"error": "Upstream timeout", "status": 504, "fallback_eligible": False})}\n\n'
        except httpx.ProtocolError:
            yield f'event: error\ndata: {json.dumps({"error": "Upstream protocol error", "status": 502, "fallback_eligible": False})}\n\n'
        except httpx.NetworkError:
            yield f'event: error\ndata: {json.dumps({"error": "Network error", "status": 502, "fallback_eligible": False})}\n\n'
        except Exception as e:
            logger.error(f"Ollama stream error: {e}")
            yield f'event: error\ndata: {json.dumps({"error": str(e), "status": 502, "fallback_eligible": False})}\n\n'
        return

    # â”€â”€ Anthropic streaming â”€â”€
    if provider == "anthropic":
        _anth_input_tokens = 0
        _anth_output_tokens = 0
        _anth_usage_seen = False
        _anth_actual_model = ""
        _anth_model_announced = False
        # Track tool_use blocks: {index: {id, name, arguments_json}}
        _anth_tool_blocks: Dict[int, Dict] = {}
        _anth_block_idx = -1
        _anth_block_type = ""
        try:
            client = _get_http_client()
            async with client.stream('POST', target_url, json=payload, headers=h, timeout=stream_timeout) as r:
                _clear_host_dead(target_url)
                if r.status_code != 200:
                    raw = (await r.aread()).decode(errors="replace")
                    friendly = _format_upstream_error(r.status_code, raw, target_url)
                    yield f'event: error\ndata: {json.dumps({"status": r.status_code, "text": friendly, "raw": raw[:500]})}\n\n'
                    return
                async for line in r.aiter_lines():
                    # SSE allows "data:value" with no space after the colon
                    # (the space is optional per the spec). Some gateways and
                    # local servers omit it; gating on "data: " dropped their
                    # entire stream.
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or not data.startswith("{"):
                        continue
                    try:
                        j = json.loads(data)
                        evt = j.get("type", "")
                        if evt == "content_block_start":
                            _anth_block_idx = j.get("index", _anth_block_idx + 1)
                            cb = j.get("content_block") or {}
                            _anth_block_type = cb.get("type", "text")
                            if _anth_block_type == "tool_use":
                                _anth_tool_blocks[_anth_block_idx] = {
                                    "id": cb.get("id") or f"call_{_anth_block_idx}",
                                    "name": cb.get("name") or "",
                                    "arguments": "",
                                }
                        elif evt == "content_block_delta":
                            delta = j.get("delta") or {}
                            delta_type = delta.get("type", "")
                            if delta_type == "text_delta":
                                text = delta.get("text") or ""
                                if text:
                                    yield f'data: {json.dumps({"delta": text})}\n\n'
                            elif delta_type == "input_json_delta":
                                # Accumulate tool arguments JSON
                                idx = j.get("index", _anth_block_idx)
                                if idx in _anth_tool_blocks:
                                    partial = delta.get("partial_json") or ""
                                    _anth_tool_blocks[idx]["arguments"] += partial
                                    # Stream tool arg deltas for doc tools
                                    if partial and _anth_tool_blocks[idx].get("name") in ("create_document", "update_document", "edit_document"):
                                        yield f'data: {json.dumps({"type": "tool_call_delta", "index": idx, "name": _anth_tool_blocks[idx]["name"], "arg_delta": partial})}\n\n'
                        elif evt == "message_start":
                            message_data = j.get("message") or {}
                            reported_model = _reported_model_name(
                                message_data.get("model")
                                if isinstance(message_data, dict)
                                else None
                            )
                            if reported_model:
                                _anth_actual_model = reported_model
                                if not _anth_model_announced:
                                    model_event = _model_actual_event(model, reported_model)
                                    if model_event:
                                        _anth_model_announced = True
                                        yield model_event
                            _u = (
                                message_data.get("usage")
                                if isinstance(message_data, dict)
                                else {}
                            ) or {}
                            if not isinstance(_u, dict):
                                _u = {}
                            if "input_tokens" in _u:
                                _anth_usage_seen = True
                            _anth_input_tokens = _u.get("input_tokens", 0)
                            # Surface prompt-cache effectiveness: cache_read > 0 means the
                            # stable system+tools prefix was served from cache this round.
                            _c_read = _u.get("cache_read_input_tokens", 0)
                            _c_write = _u.get("cache_creation_input_tokens", 0)
                            if _c_read or _c_write:
                                logger.info(
                                    "[anthropic-cache] read=%s write=%s fresh_input=%s",
                                    _c_read, _c_write, _anth_input_tokens,
                                )
                        elif evt == "message_delta":
                            _u = j.get("usage") or {}
                            if not isinstance(_u, dict):
                                _u = {}
                            if "output_tokens" in _u:
                                _anth_usage_seen = True
                            _anth_output_tokens = _u.get("output_tokens", 0)
                        elif evt == "message_stop":
                            # Emit accumulated tool calls in OpenAI-compatible format
                            if _anth_tool_blocks:
                                calls = []
                                for idx in sorted(_anth_tool_blocks):
                                    tb = _anth_tool_blocks[idx]
                                    calls.append({
                                        "id": tb["id"],
                                        "name": tb["name"],
                                        "arguments": tb["arguments"],
                                    })
                                yield f'data: {json.dumps({"type": "tool_calls", "calls": calls})}\n\n'
                            normalized_usage = _normalize_usage_counts(
                                _anth_input_tokens,
                                _anth_output_tokens,
                            )
                            if normalized_usage and _anth_usage_seen:
                                _annotate_usage_model(
                                    normalized_usage,
                                    model,
                                    _anth_actual_model,
                                )
                                yield f'data: {json.dumps({"type": "usage", "data": normalized_usage})}\n\n'
                            yield "data: [DONE]\n\n"
                            return
                        elif evt == "error":
                            err = j.get("error") or {}
                            err_msg = err.get("message", "Unknown error") if isinstance(err, dict) else str(err)
                            status = _provider_stream_error_status(err, default=400)
                            yield f'event: error\ndata: {json.dumps({"error": err_msg, "status": status})}\n\n'
                            return
                    except json.JSONDecodeError:
                        continue
                yield "data: [DONE]\n\n"
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            _cooled = _mark_host_dead(target_url)
            _tail = f" â€” host cooled for {DEAD_HOST_COOLDOWN:.0f}s" if _cooled else " â€” transient, will retry"
            logger.warning(f"Anthropic stream connect to {target_url} failed: {e}{_tail}")
            yield f'event: error\ndata: {json.dumps({"error": f"Cannot reach {_host_key(target_url)}", "status": 503})}\n\n'
        except httpx.ReadTimeout:
            yield f'event: error\ndata: {json.dumps({"error": "Read timeout", "status": 504})}\n\n'
        except httpx.PoolTimeout:
            yield f'event: error\ndata: {json.dumps({"error": "Connection pool timeout", "status": 504})}\n\n'
        except httpx.WriteTimeout:
            yield f'event: error\ndata: {json.dumps({"error": "Upstream timeout", "status": 504, "fallback_eligible": False})}\n\n'
        except httpx.ProtocolError:
            yield f'event: error\ndata: {json.dumps({"error": "Upstream protocol error", "status": 502, "fallback_eligible": False})}\n\n'
        except httpx.NetworkError:
            yield f'event: error\ndata: {json.dumps({"error": "Network error", "status": 502, "fallback_eligible": False})}\n\n'
        except Exception as e:
            logger.error(f"Anthropic stream error: {e}")
            yield f'event: error\ndata: {json.dumps({"error": str(e), "status": 502, "fallback_eligible": False})}\n\n'
        return

    # â”€â”€ OpenAI-compatible streaming â”€â”€
    # Accumulate native tool_calls across streaming chunks
    _tc_acc: Dict[int, Dict] = {}  # index -> {id, name, arguments}
    _tc_last_idx = [-1]  # most-recently-touched slot, for providers that omit `index`
    # For thinking models: prepend <think> to first content delta so frontend
    # can detect thinking-in-progress (some models output </think> but no <think>)
    _thinking_model = _supports_thinking(model)
    _first_content_sent = False
    _in_think_tag = False        # True while consuming <think>â€¦</think> content
    _think_open_stripped = False  # opening <think> tag already removed
    _harmony_router = _HarmonyStreamRouter()
    _harmony_active = False       # sticky: gpt-oss harmony <|channel|> stream detected
    _actual_model = ""
    _actual_model_announced = False

    def _emit_tool_calls():
        """Build the tool_calls event string if any were accumulated."""
        if not _tc_acc:
            return None
        calls = [_tc_acc[i] for i in sorted(_tc_acc)]
        return f'data: {json.dumps({"type": "tool_calls", "calls": calls})}\n\n'

    def _format_routed_content(parts: List[Tuple[str, bool]]) -> List[str]:
        nonlocal _first_content_sent
        events = []
        for part, is_thinking in parts:
            if is_thinking:
                events.append(_stream_delta_event(part, thinking=True))
                continue
            # Some thinking backends start normal content with a stray closing
            # tag. Repair only that shape; do not wrap every first token for
            # model families like MiniMax, which often stream ordinary answers.
            if _thinking_model and not _first_content_sent and part.lstrip().lower().startswith("</think"):
                part = "<think>" + part
            _first_content_sent = True
            events.append(_stream_delta_event(part))
        return events

    try:
        client = _get_http_client()
        h = await apply_kimi_code_headers_async(client, h, target_url)
        async with client.stream('POST', target_url, json=payload, headers=h, timeout=stream_timeout) as r:
            _clear_host_dead(target_url)
            if r.status_code != 200:
                raw = (await r.aread()).decode(errors="replace")
                friendly = _format_upstream_error(r.status_code, raw, target_url)
                yield f'event: error\ndata: {json.dumps({"status": r.status_code, "text": friendly, "raw": raw[:500]})}\n\n'
                return

            async for line in r.aiter_lines():
                if not line:
                    continue

                # SSE allows "data:value" with no space after the colon; gating
                # on "data: " silently dropped content + usage from providers
                # that omit it.
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data == "[DONE]":
                        for event in _format_routed_content(_harmony_router.flush()):
                            yield event
                        tc_event = _emit_tool_calls()
                        if tc_event:
                            yield tc_event
                        yield "data: [DONE]\n\n"
                        return

                    try:
                        if data.strip():
                            if data.startswith("{"):
                                j = json.loads(data)
                                if j.get("error"):
                                    err = j.get("error")
                                    status = _provider_stream_error_status(err, default=400)
                                    text = err.get("message") if isinstance(err, dict) else str(err)
                                    yield f'event: error\ndata: {json.dumps({"error": text or "Upstream request failed", "status": status})}\n\n'
                                    return
                                chunk_model = j.get("model")
                                if isinstance(chunk_model, str) and chunk_model.strip():
                                    _actual_model = chunk_model.strip()
                                    if (
                                        not _actual_model_announced
                                        and not _same_model_identity(_actual_model, model)
                                    ):
                                        _actual_model_announced = True
                                        yield f'data: {json.dumps({"type": "model_actual", "requested_model": model, "model": _actual_model})}\n\n'
                                # Usage chunk (from stream_options)
                                _choices = j.get("choices") or []
                                _delta0 = _choices[0].get("delta") if (_choices and _choices[0] is not None) else None
                                # Capture usage whenever the chunk carries it and
                                # the delta has no actual output. Some gateways /
                                # local servers attach usage to the FINAL delta,
                                # which also carries role/finish_reason (so it is
                                # not exactly None/{}/{"content": None}); gating on
                                # those exact shapes discarded their token counts.
                                _delta_has_output = isinstance(_delta0, dict) and (
                                    _delta0.get("content")
                                    or _delta0.get("reasoning_content")
                                    or _delta0.get("reasoning")
                                    or _delta0.get("thinking")
                                    or _delta0.get("tool_calls")
                                )
                                u = j.get("usage")
                                _has_genuine_usage = (
                                    isinstance(u, dict)
                                    and (
                                        "prompt_tokens" in u
                                        or "completion_tokens" in u
                                    )
                                )
                                if _has_genuine_usage and not _delta_has_output:
                                    _usage_data = _normalize_usage_counts(
                                        u.get("prompt_tokens", 0),
                                        u.get("completion_tokens", 0),
                                    )
                                    if _usage_data is None:
                                        continue
                                    # llama.cpp puts a `timings` block alongside `usage` with the
                                    # TRUE generation speed (predicted_per_second) â€” pure decode,
                                    # excluding prefill/network. Pass it through so the UI shows the
                                    # real gen t/s instead of recomputing tokens/wall-clock (which
                                    # includes prefill and reads ~20-40% low). Prefill speed too.
                                    _tm = j.get("timings")
                                    if isinstance(_tm, dict):
                                        if _tm.get("predicted_per_second"):
                                            _usage_data["gen_tps"] = round(_tm["predicted_per_second"], 2)
                                        if _tm.get("prompt_per_second"):
                                            _usage_data["prefill_tps"] = round(_tm["prompt_per_second"], 2)
                                    if _actual_model:
                                        _usage_data["model"] = _actual_model
                                        if not _same_model_identity(_actual_model, model):
                                            _usage_data["requested_model"] = model
                                    yield f'data: {json.dumps({"type": "usage", "data": _usage_data})}\n\n'
                                elif "choices" in j:
                                    _c0 = (j["choices"] or [None])[0]
                                    if _c0 is None:
                                        continue
                                    delta = _c0.get("delta") or {}
                                    if isinstance(delta, dict):
                                        # Text content
                                        # Reasoning tokens (VLLM --reasoning-parser, e.g. Qwen3/DeepSeek-R1, Nemotron). vLLM 0.20.2 / NIM emit the field as `reasoning`; older builds use `reasoning_content`. Some OpenAI-compatible Ollama builds use `thinking`.
                                        reasoning = delta.get("reasoning_content") or delta.get("reasoning") or delta.get("thinking") or ""
                                        content = delta.get("content") or ""
                                        # Mistral structured content: content is a list of typed blocks
                                        # ({"type": "thinking", ...}, {"type": "text", ...}). Split into
                                        # reasoning + text so thinking streams into the thinking panel.
                                        if isinstance(content, list):
                                            text_part, thinking_part = _normalize_mistral_content(content)
                                            if thinking_part:
                                                reasoning = (reasoning + thinking_part) if reasoning else thinking_part
                                            content = text_part
                                        if reasoning:
                                            _degenerate = degenerate_guard.check(reasoning)
                                            if _degenerate:
                                                yield _degenerate
                                                return
                                            yield _stream_delta_event(reasoning, thinking=True)
                                        if content:
                                            content = _strip_visible_chat_template_artifacts(content)
                                            if not content:
                                                continue
                                            _degenerate = degenerate_guard.check(content)
                                            if _degenerate:
                                                yield _degenerate
                                                return
                                            content = re.sub(r"<mm:think(\s+[^>]*)?>", r"<think\1>", content, flags=re.IGNORECASE)
                                            content = re.sub(r"</mm:think>", "</think>", content, flags=re.IGNORECASE)
                                            stripped = content.lstrip()
                                            # gpt-oss harmony format (<|channel|>analysis/final): route via the harmony
                                            # stream router. Sticky once the first marker appears â€” distinct from the
                                            # <think> path below (handled in the else, preserving #2588 behaviour).
                                            if _harmony_active or "<|" in content:
                                                _harmony_active = True
                                                for event in _format_routed_content(_harmony_router.feed(content)):
                                                    yield event
                                            else:
                                                # Auto-detect <think>â€¦</think> in content stream.
                                                # Covers Qwen3-derived models (Qwopus, QwQ forks) whose
                                                # names don't match _THINKING_MODEL_PATTERNS but still
                                                # emit literal <think> markup via llama.cpp --jinja.
                                                if not _first_content_sent and not _thinking_model and not _in_think_tag and stripped.lower().startswith("<think"):
                                                    _thinking_model = True
                                                    _in_think_tag = True
                                                if _in_think_tag:
                                                    close_idx = content.lower().find("</think>")
                                                    if close_idx != -1:
                                                        # Split: up-to-</think> â†’ thinking, remainder â†’ content
                                                        think_part = content[:close_idx]
                                                        if not _think_open_stripped:
                                                            # Strip the opening <think[...] > from the first chunk.
                                                            # Use a dedicated flag â€” _first_content_sent stays False
                                                            # throughout the think block, so it must not be reused.
                                                            tag_end = think_part.lower().find(">")
                                                            if tag_end != -1:
                                                                think_part = think_part[tag_end + 1:]
                                                            _think_open_stripped = True
                                                        regular_part = content[close_idx + len("</think>"):]
                                                        _in_think_tag = False
                                                        if think_part:
                                                            yield f'data: {json.dumps({"delta": think_part, "thinking": True})}\n\n'
                                                        if regular_part:
                                                            _first_content_sent = True
                                                            yield f'data: {json.dumps({"delta": regular_part})}\n\n'
                                                    else:
                                                        # Still inside <think>: route to thinking channel
                                                        if not _think_open_stripped:
                                                            # Strip the opening <think[...] > tag (first chunk only)
                                                            tag_end = stripped.lower().find(">")
                                                            if tag_end != -1:
                                                                content = stripped[tag_end + 1:]
                                                            _think_open_stripped = True
                                                        if content:
                                                            yield f'data: {json.dumps({"delta": content, "thinking": True})}\n\n'
                                                else:
                                                    # Some thinking backends start normal content with a
                                                    # stray closing tag. Repair only that shape; do not
                                                    # wrap every first token for model families like
                                                    # MiniMax, which often stream ordinary answers.
                                                    if _thinking_model and not _first_content_sent and stripped.lower().startswith("</think"):
                                                        content = "<think>" + content
                                                    _first_content_sent = True
                                                    yield f'data: {json.dumps({"delta": content})}\n\n'
                                        # Native tool calls â€” accumulate across chunks
                                        for tc in delta.get("tool_calls") or []:
                                            if tc is None:
                                                continue
                                            func = tc.get("function") or {}
                                            raw_idx = tc.get("index")
                                            if raw_idx is None:
                                                # Gemini's OpenAI-compat layer omits `index` on
                                                # parallel tool calls (every delta arrives as
                                                # index=None) and sends each call complete in one
                                                # delta. Without this, all parallel calls collide
                                                # into slot 0 â€” later calls overwrite the first's
                                                # name and CORRUPT its arguments by concatenation,
                                                # so only one malformed call survives and the
                                                # follow-up round 400s. A function name marks the
                                                # start of a new call â†’ allocate a fresh slot;
                                                # an arg-only continuation attaches to the last.
                                                if func.get("name") or _tc_last_idx[0] < 0:
                                                    # Next free slot ABOVE any existing key (not
                                                    # len()), so a provider mixing integer indices
                                                    # with index=None can never collide.
                                                    idx = max(_tc_acc, default=-1) + 1
                                                else:
                                                    idx = _tc_last_idx[0]
                                            else:
                                                idx = raw_idx
                                            _tc_last_idx[0] = idx
                                            if idx not in _tc_acc:
                                                _tc_acc[idx] = {"id": "", "name": "", "arguments": ""}
                                            if tc.get("id"):
                                                _tc_acc[idx]["id"] = tc["id"]
                                            # Gemini 3 returns an opaque thought_signature in
                                            # extra_content on the function-call delta. It MUST be
                                            # echoed back on the assistant tool_call next round or the
                                            # follow-up request 400s ("Function call is missing a
                                            # thought_signature"). Preserve it verbatim; other
                                            # providers never send it, so this is a no-op for them.
                                            if tc.get("extra_content"):
                                                _tc_acc[idx]["extra_content"] = tc["extra_content"]
                                            if func.get("name"):
                                                # Map harmony aliases back to real
                                                # tool names before anything
                                                # downstream sees them.
                                                _tc_acc[idx]["name"] = _unalias_harmony_tool_name(func["name"], model)
                                            if "arguments" in func:
                                                # Guard against a null arguments delta: `func` can be
                                                # {"arguments": None} (JSON null), and a raw `+= None`
                                                # raises TypeError that the broad except swallows,
                                                # silently dropping the rest of the chunk. Matches the
                                                # Anthropic accumulator (`partial = ... or ""`) above.
                                                _tc_acc[idx]["arguments"] += func["arguments"] or ""
                                                # Stream tool arg deltas for doc tools
                                                if func["arguments"] and _tc_acc[idx].get("name") in ("create_document", "update_document", "edit_document"):
                                                    yield f'data: {json.dumps({"type": "tool_call_delta", "index": idx, "name": _tc_acc[idx]["name"], "arg_delta": func["arguments"]})}\n\n'
                                elif "text" in j:
                                    if j["text"]:
                                        for event in _format_routed_content(_harmony_router.feed(j["text"])):
                                            yield event
                            else:
                                if data.strip():
                                    for event in _format_routed_content(_harmony_router.feed(data)):
                                        yield event
                    except Exception as e:
                        logger.error(f"Error parsing stream data: {e}")
                        continue

            # End of stream (no explicit [DONE] received)
            for event in _format_routed_content(_harmony_router.flush()):
                yield event
            tc_event = _emit_tool_calls()
            if tc_event:
                yield tc_event
            yield "data: [DONE]\n\n"

    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        _cooled = _mark_host_dead(target_url)
        _tail = f" â€” host cooled for {DEAD_HOST_COOLDOWN:.0f}s" if _cooled else " â€” transient, will retry"
        logger.warning(f"Stream connect to {target_url} failed: {e}{_tail}")
        yield f'event: error\ndata: {json.dumps({"error": f"Cannot reach {_host_key(target_url)}", "status": 503})}\n\n'
    except httpx.ReadTimeout:
        yield f'event: error\ndata: {json.dumps({"error": "Read timeout", "status": 504})}\n\n'
    except httpx.PoolTimeout:
        yield f'event: error\ndata: {json.dumps({"error": "Connection pool timeout", "status": 504})}\n\n'
    except httpx.WriteTimeout:
        yield f'event: error\ndata: {json.dumps({"error": "Upstream timeout", "status": 504, "fallback_eligible": False})}\n\n'
    except httpx.ProtocolError:
        yield f'event: error\ndata: {json.dumps({"error": "Upstream protocol error", "status": 502, "fallback_eligible": False})}\n\n'
    except httpx.NetworkError:
        yield f'event: error\ndata: {json.dumps({"error": "Network error", "status": 502, "fallback_eligible": False})}\n\n'
    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield f'event: error\ndata: {json.dumps({"error": str(e), "status": 502, "fallback_eligible": False})}\n\n'


def _summarize_stream_error(err_chunk: Optional[str]) -> str:
    """Pull a short human reason out of an `event: error` SSE chunk for the
    fallback notice. Returns a generic message if it can't be parsed."""
    if not err_chunk:
        return "primary model failed"
    try:
        for line in err_chunk.split("\n"):
            if line.startswith("data: "):
                j = json.loads(line[6:])
                txt = j.get("text") or j.get("error") or ""
                status = j.get("status")
                msg = (f"HTTP {status}: " if status else "") + str(txt)
                return msg[:200].strip() or "primary model failed"
    except Exception:
        pass
    return "primary model failed"


async def stream_llm_with_fallback(candidates, messages, **kwargs):
    """Wrap stream_llm with an ordered fallback chain.

    `candidates` is a list of (url, model, headers). Each is tried in order,
    but only retried on an eligible *pre-content* failure. Callers can restrict
    errors with ``fallback_statuses`` and disable empty-completion switching
    with ``fallback_on_empty=False``. Omitting both preserves the generic
    fallback behavior for non-foreground call sites.
    Metadata is held until substantive output commits the candidate.
    Once a candidate has emitted real output we never switch (that would
    duplicate streamed tokens); a later error from that candidate passes
    through unchanged. The dead-host cooldown in stream_llm makes repeat
    attempts at an offline primary effectively instant.

    Yields the same SSE chunk protocol as stream_llm.
    """
    fallback_statuses = kwargs.pop("fallback_statuses", None)
    fallback_on_empty = bool(kwargs.pop("fallback_on_empty", True))
    candidate_request_factory = kwargs.pop("candidate_request_factory", None)
    candidate_route_descriptors = kwargs.pop("candidate_route_descriptors", None)
    eligible_statuses = None if fallback_statuses is None else frozenset(fallback_statuses)

    raw_candidates = list(candidates or [])
    if not raw_candidates or not _candidate_is_configured(raw_candidates[0]):
        yield f'event: error\ndata: {json.dumps({"error": "Selected model endpoint is not configured", "status": 400, "fallback_eligible": False})}\n\n'
        return
    cands, route_descriptors = _dedupe_model_candidates_with_descriptors(
        raw_candidates,
        candidate_route_descriptors,
    )

    primary_model = cands[0][1]
    primary_route = route_descriptors[0]
    last_error = None
    failures = []
    for i, (url, model, headers) in enumerate(cands):
        is_last = (i == len(cands) - 1)
        emitted = False
        retried = False
        pending_metadata = []
        candidate_messages = messages
        candidate_kwargs = kwargs
        if candidate_request_factory is not None:
            try:
                request = candidate_request_factory(i, url, model, headers) or {}
                if hasattr(request, "__await__"):
                    request = await request
                candidate_messages = request.get("messages", messages)
                candidate_kwargs = {**kwargs, **(request.get("kwargs") or {})}
            except Exception as error:
                status = _nonstream_error_status(error)
                eligibility_override = getattr(error, "fallback_eligible", None)
                eligible = (
                    True
                    if eligible_statuses is None
                    else (
                        eligibility_override
                        if isinstance(eligibility_override, bool)
                        else status in eligible_statuses
                    )
                )
                error_chunk = _request_factory_error_chunk(error, status)
                if not is_last and eligible:
                    last_error = error_chunk
                    failures.append({
                        "candidate_index": i,
                        "model": model,
                        "status": status,
                        "reason": _summarize_stream_error(error_chunk),
                    })
                    tag = "primary" if i == 0 else "candidate"
                    logger.warning(
                        "[fallback] %s %s request preparation failed with "
                        "eligible status %s; trying next",
                        tag,
                        model,
                        status,
                    )
                    continue
                yield error_chunk
                return
        candidate_stream = stream_llm(
            url,
            model,
            candidate_messages,
            headers=headers,
            **candidate_kwargs,
        )
        try:
            async for chunk in candidate_stream:
                if chunk.startswith("event: error"):
                    status = _stream_error_status(chunk)
                    eligibility_override = _stream_error_fallback_override(chunk)
                    eligible = (
                        True
                        if eligible_statuses is None
                        else (
                            eligibility_override
                            if eligibility_override is not None
                            else status in eligible_statuses
                        )
                    )
                    if not emitted and not is_last and eligible:
                        # Pre-content failure with fallbacks left â€” swallow and
                        # move to the next candidate.
                        last_error = chunk
                        failures.append({
                            "candidate_index": i,
                            "model": model,
                            "status": status,
                            "reason": _summarize_stream_error(chunk),
                        })
                        retried = True
                        if i == 0:
                            logger.warning(f"[fallback] primary {model} failed before output; trying fallback")
                        else:
                            logger.warning(f"[fallback] candidate {model} failed; trying next")
                        break
                    if not emitted:
                        # A last-candidate error is already the clearest terminal
                        # result; do not append an empty-completion error as well.
                        yield chunk
                        return
                    yield chunk
                    continue

                event_data = {}
                is_done = chunk.startswith("data: [DONE]")
                if chunk.startswith("data: ") and not is_done:
                    try:
                        event_data = json.loads(chunk[6:])
                    except Exception:
                        pass

                delta = event_data.get("delta")
                event_type = event_data.get("type")
                substantive = (
                    isinstance(delta, str) and bool(delta.strip())
                ) or (
                    event_type == "tool_calls"
                    and bool(event_data.get("calls"))
                )

                if substantive and not emitted:
                    # First real output from a NON-primary candidate: tell the client
                    # the selected model failed and another answered. Without this the
                    # fallback is invisible â€” a misconfigured provider looks like it
                    # works because the reply is shown under the originally selected
                    # model's name (e.g. a Bedrock/Claude endpoint that 400s every
                    # request but appears fine because another model silently answered).
                    if i > 0:
                        primary_reason = (
                            failures[0]["reason"]
                            if failures
                            else _summarize_stream_error(last_error)
                        )
                        yield ('data: ' + json.dumps({
                            "type": "fallback",
                            "selected_model": primary_model,
                            "answered_by": model,
                            "selected_endpoint_id": primary_route.get("endpoint_id"),
                            "selected_endpoint_label": primary_route.get("endpoint_label"),
                            "selected_endpoint_cost_tracked": primary_route.get("endpoint_cost_tracked"),
                            "answered_by_endpoint_id": route_descriptors[i].get("endpoint_id"),
                            "answered_by_endpoint_label": route_descriptors[i].get("endpoint_label"),
                            "answered_by_endpoint_cost_tracked": route_descriptors[i].get("endpoint_cost_tracked"),
                            "candidate_index": i,
                            "reason": primary_reason,
                            "failures": [
                                {
                                    "candidate_index": failure["candidate_index"],
                                    "model": failure["model"],
                                    "status": failure["status"],
                                }
                                for failure in failures
                            ],
                        }) + '\n\n')
                    # Metadata must not commit a candidate. Once real output arrives,
                    # flush it after any fallback notice and before the output itself.
                    for metadata_chunk in pending_metadata:
                        yield metadata_chunk
                    pending_metadata.clear()
                    emitted = True

                if substantive or emitted:
                    yield chunk
                elif not is_done:
                    pending_metadata.append(chunk)
        finally:
            close_candidate = getattr(candidate_stream, "aclose", None)
            if callable(close_candidate):
                try:
                    await close_candidate()
                except Exception as close_error:
                    logger.warning(
                        "[fallback] failed to close candidate %s stream: %s",
                        model,
                        type(close_error).__name__,
                    )

        if emitted:
            return
        if retried:
            continue
        if not is_last and fallback_on_empty:
            last_error = f'event: error\ndata: {json.dumps({"error": f"Model {model} returned no substantive output", "status": 502})}\n\n'
            failures.append({
                "candidate_index": i,
                "model": model,
                "status": 502,
                "reason": _summarize_stream_error(last_error),
            })
            tag = "primary" if i == 0 else "candidate"
            logger.warning(f"[fallback] {tag} {model} returned no substantive output; trying next")
            continue
        if not is_last:
            yield f'event: error\ndata: {json.dumps({"error": f"Model {model} returned no substantive output", "status": 502})}\n\n'
            return
        yield f'event: error\ndata: {json.dumps({"error": "All model candidates returned no substantive output", "status": 502})}\n\n'
        return


def _candidate_is_configured(candidate) -> bool:
    return bool(
        isinstance(candidate, (tuple, list))
        and len(candidate) == 3
        and isinstance(candidate[0], str)
        and candidate[0].strip()
        and isinstance(candidate[1], str)
        and candidate[1].strip()
    )


def _safe_route_descriptor(value) -> dict:
    value = value if isinstance(value, dict) else {}
    endpoint_id = value.get("endpoint_id")
    endpoint_label = value.get("endpoint_label")
    endpoint_cost_tracked = value.get("endpoint_cost_tracked")
    return {
        "endpoint_id": endpoint_id if isinstance(endpoint_id, str) and endpoint_id else None,
        "endpoint_label": (
            endpoint_label
            if isinstance(endpoint_label, str) and endpoint_label.strip()
            else "Selected route"
        ),
        "endpoint_cost_tracked": (
            endpoint_cost_tracked
            if isinstance(endpoint_cost_tracked, bool)
            else None
        ),
    }


def _dedupe_model_candidates_with_descriptors(candidates, descriptors=None):
    """Dedupe routes and their parallel non-secret descriptors together."""

    seen = []
    out = []
    out_descriptors = []
    descriptors = list(descriptors or [])
    for index, candidate in enumerate(candidates or []):
        if not _candidate_is_configured(candidate):
            continue
        route = (candidate[0], candidate[1], candidate[2] or {})
        if any(route == prior for prior in seen):
            continue
        seen.append(route)
        out.append(candidate)
        raw_descriptor = descriptors[index] if index < len(descriptors) else {}
        out_descriptors.append(_safe_route_descriptor(raw_descriptor))
    return out, out_descriptors


def dedupe_model_candidates(candidates):
    """Filter malformed entries and drop a later repeat of an already-seen
    ``(url, model, headers)`` route, preserving order (first occurrence wins).

    The chain is the primary target followed by any caller-authorized
    fallbacks.  A fallback that repeats the session's current model would
    otherwise make the chain re-attempt the very route that just failed: a
    wasted round-trip plus a spurious ``fallback`` notice for a switch that did
    not happen. Credentials are part of route identity: two configured
    endpoints may intentionally use the same provider URL/model with different
    keys, and rate limiting on one must not discard the other candidate.
    """
    out, _descriptors = _dedupe_model_candidates_with_descriptors(candidates)
    return out


def _nonstream_error_status(error: Exception) -> Optional[int]:
    """Normalize a non-stream provider failure for explicit fallback policy."""

    status = getattr(error, "status_code", None)
    if not isinstance(status, bool) and status is not None:
        return _normalize_http_status(status)
    if isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout)):
        return 503
    if isinstance(error, httpx.ReadTimeout):
        return 504
    return None


async def llm_call_async_with_route_fallback(
    candidates,
    messages,
    *,
    fallback_statuses,
    **kwargs,
):
    """Call an ordered non-stream route chain and return route provenance.

    Unlike the legacy utility helper, this advances only for an explicitly
    eligible status.  A successful empty response still commits the current
    candidate; empty output is not availability evidence.  The third return
    value is the provider-reported model when available, otherwise the exact
    configured candidate model.
    """

    raw_candidates = list(candidates or [])
    if not raw_candidates or not _candidate_is_configured(raw_candidates[0]):
        raise _FallbackIneligibleHTTPException(400, "Selected model endpoint is not configured")
    candidate_request_factory = kwargs.pop("candidate_request_factory", None)
    cands = dedupe_model_candidates(raw_candidates)
    if not cands:
        raise HTTPException(503, "No model endpoint configured")
    eligible_statuses = frozenset(fallback_statuses or ())
    for index, candidate in enumerate(cands):
        url, model, headers = candidate
        try:
            candidate_messages = messages
            candidate_kwargs = kwargs
            if candidate_request_factory is not None:
                request = candidate_request_factory(index, url, model, headers) or {}
                if hasattr(request, "__await__"):
                    request = await request
                candidate_messages = request.get("messages", messages)
                candidate_kwargs = {**kwargs, **(request.get("kwargs") or {})}
            candidate_kwargs = {
                **candidate_kwargs,
                "availability_only_transport": True,
            }
            response = await llm_call_async(
                url,
                model,
                candidate_messages,
                headers=headers,
                return_model_metadata=True,
                **candidate_kwargs,
            )
            actual_model = model
            if (
                isinstance(response, tuple)
                and len(response) == 2
                and isinstance(response[0], str)
            ):
                response, reported_model = response
                if isinstance(reported_model, str) and reported_model.strip():
                    actual_model = reported_model.strip()
            return response, candidate, actual_model
        except Exception as error:
            if getattr(error, "fallback_eligible", None) is False:
                raise
            status = _nonstream_error_status(error)
            if index >= len(cands) - 1 or status not in eligible_statuses:
                raise
            tag = "primary" if index == 0 else "candidate"
            logger.warning(
                "[fallback] %s %s failed with eligible status %s; trying next",
                tag,
                model,
                status,
            )

    raise HTTPException(503, "All fallback candidates failed")


def _stream_error_status(err_chunk: Optional[str]) -> Optional[int]:
    """Return the integer status from an SSE error chunk when present."""

    if not err_chunk:
        return None
    try:
        for line in err_chunk.split("\n"):
            if not line.startswith("data: "):
                continue
            status = json.loads(line[6:]).get("status")
            return _normalize_http_status(status)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return None


def _stream_error_fallback_override(err_chunk: Optional[str]) -> Optional[bool]:
    """Return an adapter's explicit eligibility decision when present."""

    if not err_chunk:
        return None
    try:
        for line in err_chunk.split("\n"):
            if not line.startswith("data: "):
                continue
            value = json.loads(line[6:]).get("fallback_eligible")
            return value if isinstance(value, bool) else None
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return None


def _request_factory_error_chunk(error: Exception, status: Optional[int]) -> str:
    """Convert route-request preparation failures into a safe SSE error."""

    wire_status = status if status is not None else 500
    payload = {
        "error": f"Model request preparation failed (HTTP {wire_status})",
        "status": wire_status,
    }
    override = getattr(error, "fallback_eligible", None)
    if isinstance(override, bool):
        payload["fallback_eligible"] = override
    elif status is None:
        # An unclassified internal/configuration failure must never become an
        # availability fallback merely because its safe wire status is 500.
        payload["fallback_eligible"] = False
    return f'event: error\ndata: {json.dumps(payload)}\n\n'


_SYMBOLIC_RATE_LIMIT_STATUSES = frozenset({
    "RATE_LIMITED",
    "RATE_LIMIT_EXCEEDED",
    "RESOURCE_EXHAUSTED",
})


def _provider_stream_error_status(error, *, default: int = 400) -> int:
    """Classify structured provider stream errors without making them eligible by default.

    Some streaming APIs report an HTTP 200 handshake and put the real failure
    in a later event. Unknown application errors are request failures, not
    availability evidence; only explicit transient/server markers become 5xx
    or rate-limit statuses.
    """

    if isinstance(error, dict):
        # A structured numeric status is authoritative. Text heuristics are
        # only a fallback for providers that omit it.
        saw_explicit_status = False
        symbolic_rate_limited = False
        for key in ("status", "status_code", "http_status", "code"):
            if key not in error:
                continue
            value = error.get(key)
            if value is None:
                continue
            # Google-style errors use a symbolic ``status`` together with a
            # numeric HTTP ``code``.  A symbolic ``code`` remains part of the
            # marker heuristics below; it is not itself an explicit status.
            if key != "code":
                saw_explicit_status = True
            if (
                key == "status"
                and isinstance(value, str)
                and value.strip().upper() in _SYMBOLIC_RATE_LIMIT_STATUSES
            ):
                # Only availability evidence when no numeric status follows:
                # a payload pairing a symbolic status with e.g. code=401 must
                # surface the numeric truth, not advance fallback.
                symbolic_rate_limited = True
                continue
            status = _normalize_http_status(value)
            if status is not None:
                return status
        if symbolic_rate_limited:
            return 429
        if saw_explicit_status:
            return default
        marker = " ".join(str(error.get(key) or "") for key in ("type", "code", "message")).lower()
    else:
        marker = str(error or "").lower()

    if "insufficient_quota" in marker or "billing" in marker:
        return 402
    if any(token in marker for token in ("authentication", "unauthorized", "invalid api key", "invalid_api_key")):
        return 401
    if any(token in marker for token in ("permission", "forbidden")):
        return 403
    if any(token in marker for token in ("not_found", "not found", "unknown model")):
        return 404
    if any(token in marker for token in ("invalid_request", "invalid request", "unsupported", "malformed", "bad request")):
        return 400
    if any(token in marker for token in ("rate_limit", "rate limit", "too many requests")):
        return 429
    if any(token in marker for token in ("overloaded", "over capacity")):
        return 529
    if any(token in marker for token in ("timeout", "timed out")):
        return 504
    if any(token in marker for token in ("api_error", "server_error", "server error", "internal error", "temporarily unavailable")):
        return 500

    return default



# Compatibility exports from the former single-module implementation.
