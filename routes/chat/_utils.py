"""Chat routes — /api/chat, /api/chat_stream, /api/inject_context, /api/search."""

import asyncio
import json
import os
import re
import time
import logging
from datetime import datetime
from typing import Dict, Any, AsyncGenerator, List, Optional

from fastapi import APIRouter, Request, HTTPException, Form, Query, Depends
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from core.models import ChatMessage
from src.request_models import ChatRequest
from src.llm_core import (
    _normalize_http_status,
    llm_call_async,
    llm_call_async_with_route_fallback,
    stream_llm,
    stream_llm_with_fallback,
)
from src.agent_loop import stream_agent_loop
from src import agent_runs
from src.model_context import estimate_tokens
from src.context_compactor import (
    apply_compaction_state,
    maybe_compact,
    trim_for_context,
)
from src.chat_helpers import coerce_message_and_session
from src.endpoint_resolver import normalize_base as _normalize_base, build_chat_url
from src.foreground_model_routing import (
    build_foreground_model_candidates,
    build_foreground_route_descriptors,
    resolve_foreground_model_policy,
)
from src.session_search import search_session_messages
from src.prompt_security import untrusted_context_message
from core.exceptions import SessionNotFoundError
from src.auth_helpers import (
    effective_user,
    get_current_user,
    is_delegated_credential,
    require_api_token_scope,
    require_chat_api_token_scope,
)
from routes.session_routes import _verify_session_owner
from routes.document_helpers import _owner_session_filter
from core.database import SessionLocal, get_session_mode, set_session_mode
from core.database import Session as DBSession, ChatMessage as DBChatMessage
from core.database import Document as DBDocument, ModelEndpoint
from core.log_safety import redact_url
from routes.research_routes import _resolve_research_endpoint
from routes.model import _visible_models
from routes.chat_helpers import (
    resolve_session_auth,
    build_chat_context,
    save_assistant_response,
    run_post_response_tasks,
    accumulate_token_usage,
    clean_thinking_for_save,
    _allowed_models_for_request,
    _enforce_chat_privileges,
)
from src.action_intents import ToolIntent, classify_tool_intent as _classify_tool_intent
from src.image_model_ids import looks_like_image_generation_model
from src.tool_policy import (
    WEB_TOOL_NAMES,
    build_effective_tool_policy,
    is_web_search_explicitly_denied,
    web_search_enabled_for_turn,
)
from src.tool_approvals import tool_approval_store
from src.tool_approval_scopes import stamp_chat_session_grant
from src.tool_security import delegated_credential_blocked_tools

logger = logging.getLogger(__name__)

# Track active streams for partial-save safety net
_active_streams: Dict[str, dict] = {}
_IMAGE_MODEL_PREFIXES = (
    "gpt-image",
    "dall-e",
    "chatgpt-image",
    "flux",
    "qwen-image",
    "imagen",
    "models/imagen",
)
_VIDEO_MODEL_HINTS = (
    "text-to-video",
    "image-to-video",
    "video-generation",
    "t2v",
    "i2v",
    "veo-",
    "sora",
    "wan2",
    "wanx",
    "wan-ai",
    "kling",
    "runway",
    "luma",
    "vidu",
    "pixverse",
    "hailuo",
    "happyhorse",
)
_MEDIA_RESULT_KEYS = (
    "media_url", "media_id", "media_type", "media_prompt", "media_model",
    "media_size", "media_quality", "media_files",
    "image_url", "image_id", "image_prompt", "image_model", "image_size",
    "image_quality",
)



def _stream_failure_status(chunk: str) -> Optional[int]:
    """Extract a provider status without retaining provider-supplied detail."""

    try:
        for line in str(chunk or "").splitlines():
            if not line.startswith("data: "):
                continue
            status = json.loads(line[6:]).get("status")
            return _normalize_http_status(status)
    except json.JSONDecodeError:
        return None
    return None


def _reject_delegated_tool_approval(request: Request) -> None:
    """Refuse an approval answered by a bearer API token.

    A tool approval records that a HUMAN authorized one dangerous action. A
    token is a delegated credential handed to an integration, so when it
    answers the prompt it triggered, nobody is asked and the gate collapses
    into an extra round trip. Owner and session already match here: the token
    is answering on behalf of the account that minted it.
    """
    if is_delegated_credential(request):
        raise HTTPException(
            403,
            "Tool approvals require an interactive session. "
            "API tokens cannot authorize a gated action.",
        )


def _mark_tool_approval_resolved(sess, approval_id: Any, decision: Any) -> bool:
    """Persist a consumed approval decision on its existing tool event."""

    approval_key = str(approval_id or "")
    normalized_decision = str(decision or "").strip().lower()
    if not approval_key or normalized_decision not in {"approve", "approve_task", "deny"}:
        return False

    message_id = None
    resolved_metadata = None
    for item in reversed(getattr(sess, "history", []) or []):
        metadata = getattr(item, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        tool_events = metadata.get("tool_events")
        if not isinstance(tool_events, list):
            continue
        for event in reversed(tool_events):
            ask_user = event.get("ask_user") if isinstance(event, dict) else None
            if not isinstance(ask_user, dict):
                continue
            if str(ask_user.get("approval_id") or "") != approval_key:
                continue
            ask_user["resolved"] = normalized_decision
            stamp_chat_session_grant(
                ask_user,
                getattr(sess, "id", ""),
                normalized_decision,
            )
            message_id = metadata.get("_db_id")
            resolved_metadata = {
                key: value for key, value in metadata.items() if key != "_db_id"
            }
            break
        if resolved_metadata is not None:
            break

    if resolved_metadata is None or not message_id:
        return False

    db = SessionLocal()
    try:
        db_message = db.query(DBChatMessage).filter(
            DBChatMessage.id == message_id,
            DBChatMessage.session_id == str(getattr(sess, "id", "")),
        ).first()
        if db_message is None:
            return False
        db_message.meta_data = json.dumps(resolved_metadata)
        db.commit()
        return True
    except Exception:
        db.rollback()
        logger.exception("Failed to persist tool approval resolution")
        return False
    finally:
        db.close()


async def _tool_approval_resolution_stream(decision: str) -> AsyncGenerator[str, None]:
    yield f"data: {json.dumps({'type': 'tool_approval_resolved', 'decision': decision})}\n\n"
    yield "data: [DONE]\n\n"


def _chat_candidate_request_factory(
    messages,
    fallback_context_length: int = 0,
    *,
    session=None,
    owner: Optional[str] = None,
):
    """Shape one route-neutral Chat prompt for each candidate window."""

    state = {
        "requests": {},
        "context_lengths": {},
        "trim_stats": {},
        "compactions": {},
        "was_compacted": {},
    }

    async def factory(index, candidate_url, candidate_model, candidate_headers):
        compaction_state = {}
        candidate_messages, context_length, was_compacted = await maybe_compact(
            session,
            candidate_url,
            candidate_model,
            list(messages),
            candidate_headers,
            owner=owner,
            persist=False,
            compaction_state=compaction_state,
        )
        if not context_length:
            context_length = fallback_context_length
        request_messages = trim_for_context(candidate_messages, context_length)
        state["requests"][index] = request_messages
        state["context_lengths"][index] = context_length
        state["compactions"][index] = compaction_state
        state["was_compacted"][index] = was_compacted
        state["trim_stats"][index] = {
            "messages_before": len(messages),
            "messages_after": len(request_messages),
            "tokens_before": estimate_tokens(messages),
            "tokens_after": estimate_tokens(request_messages),
        }
        return {"messages": request_messages}

    return factory, state


def _candidate_index(candidates, actual_candidate) -> int:
    for index, candidate in enumerate(candidates):
        if candidate == actual_candidate:
            return index
    return 0


def _stream_set(session_id: str, **fields) -> None:
    """Update fields on the active-stream entry for `session_id`, or
    no-op if the entry has already been popped. Using .get() avoids a
    KeyError race between `if x in d` and `d[x]["k"] = v` if a sibling
    finally pops the key in between (which becomes possible the moment
    a coroutine cancellation reaches an inner cleanup before the
    outermost cleanup runs)."""
    rec = _active_streams.get(session_id)
    if rec is None:
        return
    rec.update(fields)


def _message_plain_text(content: Any) -> str:
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts)
    return str(content or "")


def _last_user_plain_text(messages: List[Dict[str, Any]]) -> str:
    for msg in reversed(messages or []):
        if msg.get("role") == "user":
            return _message_plain_text(msg.get("content"))
    return ""


def _ensure_current_request_is_latest_user(messages: List[Dict[str, Any]], current_message: str) -> List[Dict[str, Any]]:
    """Defensively keep detached streams grounded on the request that created them."""
    current = str(current_message or "").strip()
    if not current:
        return messages
    latest = _last_user_plain_text(messages).strip()
    if latest == current or current in latest or latest in current:
        return messages
    logger.warning(
        "[chat_stream] latest user context mismatch; appending current request for model call. latest=%r current=%r",
        latest[:120],
        current[:120],
    )
    repaired = list(messages or [])
    repaired.append({"role": "user", "content": current})
    return repaired


_WEB_FOLLOWUP_RE = re.compile(
    r"^\s*(?:(?:can|could|would|will)\s+you\s+)?"
    r"(?:check|try\s+again|look(?:\s+now|\s+it\s+up)?|search(?:\s+now|\s+online|\s+it)?|"
    r"do\s+it|again|approved|approve(?:d)?|yes|ok(?:ay)?|proceed|go\s+ahead|"
    r"send(?:\s+it)?|submit(?:\s+it)?|email(?:\s+them|\s+it)?)\??\s*$",
    re.I,
)
_RECENT_WEB_CONTEXT_RE = re.compile(
    r"\b(?:weather|forecast|rain|raining|hourly|news|headlines|rate|exchange|currency|"
    r"price|current|latest|search|look\s+up|online)\b",
    re.I,
)
_RECENT_BROWSER_CONTEXT_RE = re.compile(
    r"\b(?:browser|browse|open\s+(?:the\s+)?(?:site|page|url|link)|click|"
    r"fill(?:\s+out)?|submit|send\s+(?:the\s+)?form|contact\s+form|web\s*form|"
    r"form\s+submission|playwright|automation)\b",
    re.I,
)
_BROWSER_MCP_TOOLS = {
    "mcp__builtin_browser__browser_navigate",
    "mcp__builtin_browser__browser_snapshot",
    "mcp__builtin_browser__browser_click",
    "mcp__builtin_browser__browser_type",
    "mcp__builtin_browser__browser_fill_form",
    "mcp__builtin_browser__browser_select_option",
    "mcp__builtin_browser__browser_press_key",
    "mcp__builtin_browser__browser_wait_for",
    "mcp__builtin_browser__browser_take_screenshot",
    "mcp__builtin_browser__browser_drag",
    "mcp__builtin_browser__browser_navigate_back",
    "mcp__builtin_browser__browser_close",
}


def _recent_session_text(sess, limit: int = 8, max_chars: int = 2000) -> str:
    history = getattr(sess, "history", None) or getattr(sess, "_history", None) or []
    chunks: List[str] = []
    for msg in history[-limit:]:
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        text = _message_plain_text(content).strip()
        if text:
            chunks.append(text)
    return " ".join(chunks)[-max_chars:]


def _is_contextual_web_followup(message: str, sess) -> bool:
    """Treat short retry/check replies as web lookups when recent context was web."""
    if not message or not _WEB_FOLLOWUP_RE.search(message):
        return False
    return bool(_RECENT_WEB_CONTEXT_RE.search(_recent_session_text(sess)))


def _is_contextual_browser_followup(message: str, sess) -> bool:
    """Treat short retry replies as browser tasks when recent context was forms/browser automation."""
    if not message or not _WEB_FOLLOWUP_RE.search(message):
        return False
    return bool(_RECENT_BROWSER_CONTEXT_RE.search(_recent_session_text(sess, limit=12, max_chars=4000)))


def _resolve_request_workspace(request, raw_value) -> tuple:
    """Resolve the posted workspace for this request: (workspace, rejected).

    Privilege is checked BEFORE the path ever touches the filesystem. Only
    admin/single-user callers can use the workspace-backed file/shell tools,
    so only they get vet_workspace() and the workspace_rejected signal. For
    any other caller the submitted value is dropped uniformly, with no vetting
    and no event: otherwise the presence/absence of workspace_rejected would
    let a non-admin chat caller probe which host paths exist.

    vet_workspace rejects non-directories, sensitive roots (.ssh, .gnupg,
    ...), and filesystem roots; on rejection there is no confinement and the
    default tool-path allowlist applies. The rejected value is surfaced so the
    stream can tell an admin client (which believes a workspace is active)
    that it was dropped.
    """
    requested = (raw_value or "").strip()
    if not requested:
        return "", ""
    from src.tool_security import owner_is_admin_or_single_user
    if not owner_is_admin_or_single_user(get_current_user(request)):
        return "", ""
    from src.tool_execution import vet_workspace
    workspace = vet_workspace(requested) or ""
    return workspace, (requested if not workspace else "")


_WORKSPACE_FILE_EXT = (
    r"(?:py|pyw|js|mjs|cjs|ts|tsx|jsx|java|kt|kts|xml|json|jsonc|toml|"
    r"ya?ml|txt|md|css|scss|html?|gradle|properties|lock|sh|bash|ps1|"
    r"bat|cmd|sql|rs|go|c|cc|cpp|h|hpp|cs|rb|php|lua)"
)
_WORKSPACE_LOCAL_PATH_RE = re.compile(
    rf"(?:[A-Za-z]:[\\/][^\s`'\"<>]+|(?:\.{{1,2}}[\\/]|~[\\/]|/)[^\s`'\"<>]+|"
    rf"(?:[\w.-]+[\\/])+[^\s`'\"<>]+|\b[\w.-]+\.{_WORKSPACE_FILE_EXT}\b)"
)
_WORKSPACE_FILE_ACTION_RE = re.compile(
    r"\b(?:open|read|show|list|browse|inspect|check|search|find|edit|modify|"
    r"change|write|save|create|delete|remove|rename|move|fix|update|patch|refactor|"
    r"build|run|test|look\s+(?:at|in|through)|work\s+on)\b",
    re.I,
)
_WORKSPACE_FILE_TARGET_RE = re.compile(
    r"\b(?:files?|folders?|director(?:y|ies)|repo|repository|project|workspace|"
    r"codebase|source|tree|path)\b",
    re.I,
)


def _message_mentions_workspace_files(message: Any) -> bool:
    text = str(message or "")
    if not text.strip():
        return False
    if _WORKSPACE_LOCAL_PATH_RE.search(text):
        return True
    return bool(
        _WORKSPACE_FILE_ACTION_RE.search(text)
        and _WORKSPACE_FILE_TARGET_RE.search(text)
    )

_ABS_PATH_RE = re.compile(r"(?<!\S)(~?/[^\"'\s`<>]+)")
_LOCAL_FILE_TASK_RE = re.compile(
    r"\b(?:file|folder|directory|path|workspace|repo|project|movie|video|"
    r"subtitle|subtitles|srt|vtt|ass|download|save|rename|move|copy|extract|"
    r"convert|ffmpeg|run|execute|open|read|inspect|fix|debug|test|build)\b",
    re.IGNORECASE,
)


def _resolve_workspace_from_message_path(request, message: str) -> tuple[str, str]:
    """Auto-bind a workspace only when the user names an explicit safe path.

    This is intentionally deterministic rather than LLM/RAG-driven: RAG can
    choose the tool family, but filesystem binding must not let a prompt infer
    or probe arbitrary host paths. For a file path, bind its parent directory.
    For a directory path, bind that directory.
    """
    text = str(message or "")
    if not text or not _LOCAL_FILE_TASK_RE.search(text):
        return "", ""

    from src.tool_security import owner_is_admin_or_single_user
    if not owner_is_admin_or_single_user(get_current_user(request)):
        return "", ""

    from src.tool_execution import vet_workspace

    for match in _ABS_PATH_RE.finditer(text):
        raw = match.group(1).rstrip(".,;:)]}")
        expanded = os.path.realpath(os.path.expanduser(raw))
        candidates = [expanded]
        if os.path.isfile(expanded):
            candidates.insert(0, os.path.dirname(expanded))
        for candidate in candidates:
            workspace = vet_workspace(candidate) or ""
            if workspace:
                return workspace, ""
    return "", ""


def _session_url_matches_endpoint(session_url: str, endpoint_base: str) -> bool:
    if not session_url or not endpoint_base:
        return False
    sess = session_url.rstrip("/")
    base = _normalize_base(endpoint_base).rstrip("/")
    variants = {
        base,
        base + "/chat/completions",
        build_chat_url(base).rstrip("/"),
    }
    return sess in variants or sess.startswith(base + "/")


def _clear_orphaned_session_endpoint(sess, owner: str | None = None) -> bool:
    """Clear a session model if its endpoint was deleted from ModelEndpoint."""
    if not getattr(sess, "endpoint_url", ""):
        return False
    db = SessionLocal()
    try:
        q = db.query(ModelEndpoint).filter(ModelEndpoint.is_enabled == True)
        if owner:
            from src.auth_helpers import owner_filter
            q = owner_filter(q, ModelEndpoint, owner)
        endpoints = q.all()
        for ep in endpoints:
            if _session_url_matches_endpoint(sess.endpoint_url or "", ep.base_url or ""):
                return False
        db_session = db.query(DBSession).filter(DBSession.id == sess.id).first()
        if db_session:
            db_session.endpoint_url = ""
            db_session.model = ""
            db_session.updated_at = datetime.utcnow()
            db.commit()
        sess.endpoint_url = ""
        sess.model = ""
        sess.headers = {}
        return True
    except Exception as e:
        logger.warning("Failed to clear orphaned session endpoint", exc_info=e)
        db.rollback()
        return False
    finally:
        db.close()


def _endpoint_cache_contains_model(endpoint, model: str) -> bool:
    """Return True when a populated endpoint model cache includes ``model``.

    Empty/malformed caches are treated as unknown rather than a negative match
    so older image endpoints without cached models still work.
    """
    raw = getattr(endpoint, "cached_models", None)
    if not raw:
        return True
    try:
        models = json.loads(raw) if isinstance(raw, str) else raw
    except Exception as e:
        logger.warning("Failed to parse cached models list, treating as containing model", exc_info=e)
        return True
    if not isinstance(models, list) or not models:
        return True
    wanted = (model or "").strip()
    return wanted in {str(item).strip() for item in models}


def _is_image_generation_session(sess, owner: str | None = None) -> bool:
    """Whether this chat session should bypass text chat and generate images.

    Model-name prefixes are explicit image models. Endpoint type is only used
    when the current session endpoint actually matches that image endpoint, and
    when a populated endpoint model cache includes the selected model. This
    prevents an image endpoint on the same host from misrouting ordinary text
    models into the image-generation path.
    """
    model = (getattr(sess, "model", "") or "").strip()
    model_l = model.lower()
    if looks_like_image_generation_model(model):
        return True
    if "image" in model_l and any(term in model_l for term in ("gemini", "qwen", "flux", "imagen")):
        return True

    endpoint_url = (getattr(sess, "endpoint_url", "") or "").strip()
    if not endpoint_url:
        return False

    db = SessionLocal()
    try:
        q = db.query(ModelEndpoint).filter(ModelEndpoint.is_enabled == True)
        if owner:
            from src.auth_helpers import owner_filter
            q = owner_filter(q, ModelEndpoint, owner)
        endpoints = q.all()
        for endpoint in endpoints:
            if (getattr(endpoint, "model_type", None) or "llm") != "image":
                continue
            if not _session_url_matches_endpoint(endpoint_url, getattr(endpoint, "base_url", "") or ""):
                continue
            if _endpoint_cache_contains_model(endpoint, model):
                return True
    except Exception:
        return False
    finally:
        db.close()
    return False


def _is_video_generation_session(sess) -> bool:
    """Whether the selected session model is explicitly a video model."""
    model = (getattr(sess, "model", "") or "").strip().lower()
    if model.startswith("models/"):
        model = model.split("/", 1)[1]
    if not model or "embedding" in model:
        return False
    if "wan2.7-image" in model or "wan2-7-image" in model:
        return False
    if "minimax" in model and "video" in model:
        return True
    return any(hint in model for hint in _VIDEO_MODEL_HINTS) or (
        "video" in model and not model.startswith("gpt-image")
    )


def _looks_like_existing_media_question(message: str) -> bool:
    text = " ".join((message or "").lower().split())
    if not text:
        return False
    media = r"(?:image|picture|pic|photo|photograph|screenshot|screen\s*shot|gallery|photos)"
    reference = r"(?:above|attached|uploaded|shown|visible|this|that|the|previous|last|my)"
    observe = r"(?:see|view|look\s+at|access|inspect|describe|analy[sz]e|read|check|recognize|tell\s+me)"
    if re.search(rf"\b(?:can|could|do|did)\s+you\b.{{0,80}}\b{observe}\b.{{0,120}}\b{media}\b", text):
        return True
    if re.search(rf"\b{observe}\b.{{0,80}}\b{reference}\s+{media}\b", text):
        return True
    if re.search(rf"\b{media}\s+{reference}\b", text) and re.search(r"\b(?:can|could|do|did|what|who|where|why|how|is|are)\b", text):
        return True
    if re.search(rf"\b(?:what(?:'s| is)|who|where|why|how)\b.{{0,120}}\b(?:in|on|inside|shown|visible)\b.{{0,80}}\b{media}\b", text):
        return True
    return False


def _content_part_is_image(part: Any) -> bool:
    if not isinstance(part, dict):
        return False
    ptype = str(part.get("type") or "").lower()
    if ptype in {"image", "image_url", "input_image"}:
        return True
    if "image" in ptype:
        return True
    return any(key in part for key in ("image", "image_url", "input_image"))


def _messages_have_image_content(messages: list | None) -> bool:
    """True when the actual LLM payload contains image content blocks."""
    for item in messages or []:
        content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
        if isinstance(content, list) and any(_content_part_is_image(part) for part in content):
            return True
    return False


def _resolve_configured_vision_primary_candidate(owner: str | None = None) -> list:
    try:
        from src.settings import get_user_setting, load_settings
        from src.endpoint_resolver import resolve_endpoint_by_id

        settings = load_settings()
        owner_key = owner or ""
        endpoint_id = (get_user_setting("vision_endpoint_id", owner_key, settings.get("vision_endpoint_id", "")) or "").strip()
        model = (get_user_setting("vision_model", owner_key, settings.get("vision_model", "")) or "").strip()
        if not endpoint_id:
            return []
        resolved = resolve_endpoint_by_id(endpoint_id, model, owner=owner)
        return [resolved] if resolved else []
    except Exception as e:
        logger.debug("Could not resolve configured vision primary candidate: %s", e)
        return []


def _resolve_auto_vision_fallback_candidates(owner: str | None = None, limit: int = 3) -> list:
    """Find enabled vision-capable chat endpoints when no explicit chain exists."""
    try:
        from src.endpoint_resolver import resolve_auto_vision_candidates
        return resolve_auto_vision_candidates(owner=owner, limit=limit)
    except Exception as e:
        logger.debug("Could not resolve auto vision fallback candidates: %s", e)
        return []


def _resolve_chat_stream_fallback_candidates(messages: list | None, owner: str | None = None) -> list:
    """Resolve fallbacks for this stream.

    Image-attached turns should try configured vision fallbacks before generic
    chat fallbacks. Otherwise an incompatible regional vision endpoint can 400
    with no usable recovery path even though the user has a working vision
    model configured.
    """
    candidates = []
    if _messages_have_image_content(messages):
        candidates.extend(_resolve_configured_vision_primary_candidate(owner=owner))
        try:
            from src.endpoint_resolver import resolve_vision_fallback_candidates
            candidates.extend(resolve_vision_fallback_candidates(owner=owner))
        except Exception as e:
            logger.debug("Could not resolve vision fallback candidates: %s", e)
        candidates.extend(_resolve_auto_vision_fallback_candidates(owner=owner))
    try:
        policy = resolve_foreground_model_policy(owner=owner)
        if policy.enabled:
            candidates.extend(policy.fallback_candidates)
    except Exception as e:
        logger.debug("Could not resolve foreground fallback candidates: %s", e)
    return candidates


def _should_prefer_vision_fallback(messages: list | None, sess, fallback_candidates: list | None) -> bool:
    """Selected vision model first; otherwise route image turns to vision fallback."""
    if not fallback_candidates or not _messages_have_image_content(messages):
        return False
    try:
        from src.chat_helpers import model_supports_vision
        return not model_supports_vision(
            getattr(sess, "model", "") or "",
            getattr(sess, "endpoint_url", "") or "",
        )
    except Exception:
        return True


def _direct_media_request_for_session(
    sess,
    message: str,
    messages: list | None,
    owner: str | None = None,
) -> tuple[str, str] | None:
    if _looks_like_existing_media_question(message):
        return None
    if _is_video_generation_session(sess):
        return "video", message or ""
    if _is_image_generation_session(sess, owner=owner):
        return "image", message or ""
    return None


async def _generate_direct_media(kind: str, prompt: str, session_id: str, owner: str | None) -> dict:
    if kind == "image":
        from src.ai_interaction import do_generate_image
        from src.runcomfy_media import generate_runcomfy_media, wants_runcomfy_media

        if wants_runcomfy_media(prompt):
            return await generate_runcomfy_media("image", prompt, owner=owner, session_id=session_id)
        return await do_generate_image(prompt, session_id, owner=owner)

    from src.runcomfy_media import generate_runcomfy_media
    media_kind = "music" if kind == "audio" else kind

    return await generate_runcomfy_media(
        media_kind,
        prompt,
        owner=owner,
        session_id=session_id,
    )

def _first_image_attachment(chat_handler, att_ids: List[str], owner: str | None = None) -> Optional[Dict[str, Any]]:
    """Return the first attached image file that this owner can read."""
    upload_handler = getattr(chat_handler, "upload_handler", None)
    if not upload_handler:
        return None
    for att_id in att_ids or []:
        try:
            info = upload_handler.resolve_upload(att_id, owner=owner)
        except Exception as e:
            logger.warning("Failed to resolve image edit upload %s", att_id, exc_info=e)
            continue
        if not info:
            continue
        name = info.get("name") or info.get("original_name") or info.get("id") or ""
        mime = info.get("mime", "")
        try:
            if upload_handler.is_image_file(name, mime):
                return info
        except Exception:
            continue
    return None


def _recover_empty_session_model(sess, session_id: str, owner: str | None = None) -> bool:
    """Re-populate sess.model from the matching endpoint's cached models.

    Covers the window between endpoint setup and the first chat send: the
    picker showed a model in the dropdown but the session record never got
    written (Issue #587 — UI uses the cached endpoint list, not s.model).
    For Codex Subscription, also repairs stale OpenAI API model names such as
    ``gpt-5`` that are not accepted by the Codex-backed ChatGPT account route.
    """
    current_model = (getattr(sess, "model", "") or "").strip()
    endpoint_url = (getattr(sess, "endpoint_url", "") or "").strip()
    is_chatgpt_subscription = False
    if current_model:
        try:
            from src.chatgpt_subscription import is_chatgpt_subscription_base
            is_chatgpt_subscription = is_chatgpt_subscription_base(endpoint_url)
            if not is_chatgpt_subscription:
                return False
        except Exception:
            return False
    db = SessionLocal()
    try:
        # Prefer the endpoint whose base URL matches the session — we know the
        # user already pointed this session at that endpoint, so its first
        # cached model is the most defensible default.
        ep = None
        if getattr(sess, "endpoint_url", ""):
            q = db.query(ModelEndpoint).filter(ModelEndpoint.is_enabled == True)
            if owner:
                from src.auth_helpers import owner_filter
                q = owner_filter(q, ModelEndpoint, owner)
            endpoints = q.all()
            for cand in endpoints:
                if _session_url_matches_endpoint(sess.endpoint_url or "", cand.base_url or ""):
                    ep = cand
                    break
        if not ep:
            return False
        if not is_chatgpt_subscription:
            try:
                from src.chatgpt_subscription import is_chatgpt_subscription_base
                is_chatgpt_subscription = is_chatgpt_subscription_base(getattr(ep, "base_url", "") or endpoint_url)
            except Exception:
                is_chatgpt_subscription = False
        try:
            cached = json.loads(ep.cached_models) if isinstance(ep.cached_models, str) else (ep.cached_models or [])
        except Exception as e:
            logger.warning("Failed to parse cached_models for endpoint %r", getattr(ep, "id", "?"), exc_info=e)
            cached = []
        if not cached:
            visible = []
        else:
            try:
                visible = _visible_models(cached, getattr(ep, "hidden_models", None))
            except Exception:
                visible = cached
        if current_model and current_model in {str(item).strip() for item in visible}:
            return False
        if is_chatgpt_subscription:
            live_models = []
            if getattr(ep, "provider_auth_id", None):
                try:
                    from src.chatgpt_subscription import fetch_available_models
                    from src.endpoint_resolver import resolve_endpoint_runtime
                    _base, api_key = resolve_endpoint_runtime(ep, owner=owner)
                    if api_key:
                        live_models = fetch_available_models(api_key)
                        if live_models:
                            ep.cached_models = json.dumps(live_models)
                            db.commit()
                except Exception:
                    live_models = []
            # Codex Subscription recovery must use the live Codex catalog.
            # Cached rows are only trusted above to avoid revalidating a model
            # that is already present in the visible picker list.
            cached = live_models
            if not cached:
                return False
            try:
                visible = _visible_models(cached, getattr(ep, "hidden_models", None))
            except Exception:
                visible = cached
            if current_model and current_model in {str(item).strip() for item in visible}:
                return False
        if not visible:
            return False
        model = visible[0]
        if not isinstance(model, str) or not model.strip():
            return False
        model = model.strip()
        # Persist so the next request, websocket reconnect, or page reload
        # picks up the same model (we'd otherwise re-pick on every send
        # and silently switch on the user if the cached order shifts).
        db_session_q = db.query(DBSession).filter(DBSession.id == session_id)
        if owner:
            db_session_q = db_session_q.filter(DBSession.owner == owner)
        db_session = db_session_q.first()
        if db_session:
            db_session.model = model
            db_session.updated_at = datetime.utcnow()
            db.commit()
        sess.model = model
        logger.info(
            "Recovered session model for %s — picked %r from endpoint %s",
            session_id, model, ep.id,
        )
        return True
    except Exception as e:
        db.rollback()
        logger.warning("Failed to recover empty session model for %s: %s", session_id, e)
    return False


def _reconcile_selected_route_from_request(
    request: Request,
    sess,
    session_id: str,
    form_data,
    owner: str | None = None,
) -> bool:
    """Apply the model route the browser selected before streaming.

    The frontend creates a pending chat first and only materializes it on first
    send. Startup/default-model refreshes can race with that UI state, so the
    stream request includes the route that was selected at click/send time.
    Trust only registered endpoint ids, or the session's existing endpoint URL.
    """
    selected_model = str(form_data.get("selected_model") or "").strip()
    selected_endpoint_id = str(form_data.get("selected_endpoint_id") or "").strip()
    selected_endpoint_url = str(form_data.get("selected_endpoint_url") or "").strip()
    if not selected_model:
        return False

    endpoint_url = ""
    headers = None
    if selected_endpoint_id or selected_endpoint_url:
        try:
            from src.auth_helpers import owner_filter
            from src.endpoint_resolver import build_headers, normalize_base
            db = SessionLocal()
            try:
                q = db.query(ModelEndpoint).filter(ModelEndpoint.is_enabled == True)
                if selected_endpoint_id:
                    q = q.filter(ModelEndpoint.id == selected_endpoint_id)
                if owner:
                    q = owner_filter(q, ModelEndpoint, owner)
                candidates = q.all() if selected_endpoint_url and not selected_endpoint_id else [q.first()]
                ep = None
                for cand in candidates:
                    if not cand:
                        continue
                    if selected_endpoint_id or _session_url_matches_endpoint(selected_endpoint_url, cand.base_url or ""):
                        ep = cand
                        break
                if not ep:
                    return False
                endpoint_url = build_chat_url(normalize_base(ep.base_url or ""))
                headers = build_headers(ep.api_key or "", ep.base_url or "") if ep.api_key else {}
            finally:
                db.close()
        except Exception as e:
            logger.warning("Failed to resolve selected endpoint %s/%s for %s: %s", selected_endpoint_id, selected_endpoint_url, session_id, e)
            return False

    if not endpoint_url:
        return False

    if (
        selected_model == (getattr(sess, "model", "") or "")
        and endpoint_url == (getattr(sess, "endpoint_url", "") or "")
    ):
        return False

    sess.model = selected_model
    sess.endpoint_url = endpoint_url
    sess.headers = headers or {}
    db = SessionLocal()
    try:
        db_session = db.query(DBSession).filter(DBSession.id == session_id).first()
        if db_session:
            db_session.model = selected_model
            db_session.endpoint_url = endpoint_url
            db_session.headers = sess.headers or {}
            db_session.updated_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()
    logger.info("Reconciled selected route for %s: model=%r endpoint=%s", session_id, selected_model, redact_url(endpoint_url))
    return True


def _set_user_time_from_request(request: Request) -> None:
    """Copy browser timezone headers into the per-request context.

    This is intentionally ephemeral: it is used only while building prompts
    and running tools for this request. It is not persisted or logged.
    """
    try:
        tz_offset = request.headers.get("x-tz-offset")
        tz_name = request.headers.get("x-tz-name")
        from src.user_time import clear_user_time_context, set_user_tz_name, set_user_tz_offset

        clear_user_time_context()
        if tz_offset is not None:
            set_user_tz_offset(tz_offset)
        if tz_name:
            set_user_tz_name(tz_name)
    except Exception:
        pass
