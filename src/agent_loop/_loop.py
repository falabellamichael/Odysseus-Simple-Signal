"""Agent loop: tool resolution, metrics, stream_agent_loop."""

from src.tool_approvals import ExactToolApproval, document_content_digest, tool_approval_store
from src.tool_capabilities import (
    ResultIntegrity,
    ToolRunSecurityContext,
    capabilities_for_action,
    messages_contain_external_untrusted_context,
    tool_result_is_successful,
    tool_result_should_arm_gate,
)
from src.tool_policy import WEB_TOOL_NAMES, GUIDE_ONLY_DIRECTIVE, ToolPolicy
from src.agent_loop._prompts import (
    _WORKSPACE_TERMINUS_TOOLS,
    _agent_route_tool_mode,
    _calendar_list_summary_from_tool_output,
    _email_list_summary_from_tool_output,
    _email_read_summary_from_tool_output,
    _expand_browser_mcp_tools,
    _ody_qwen_terminal_tool_summary,
    TOOL_SECTIONS,
    get_builtin_overrides,
    _assemble_prompt,
    _build_system_prompt,
    _build_base_prompt,
    _ADMIN_SCHEMA_NAMES,
    _ADMIN_TOOLS,
    _API_HOSTS,
    _MCP_KEYWORDS,
    _TOOL_SELECTION_TIMEOUT_SECONDS,
    _looks_like_notes_list_request,
    _note_list_summary_from_tool_output,
    _load_mcp_disabled_map,
    _domain_rules_for_tools,
    _active_workspace_note,
    _DOMAIN_TOOL_MAP,
    _DOMAIN_RULES,
    _LINK_RULES,
)
from src.agent_loop._helpers import (
    _document_stream_events,
    _explicitly_references_missing_workspace,
    _is_odysseus_qwen_model,
    _looks_like_destructive_request,
    _looks_like_local_computer_request,
    _looks_like_notes_calendar_followup,
    _looks_like_success_claim,
    _looks_like_workspace_coding_request,
    _minimal_recent_notes_tool_context_message,
    _normalize_ody_qwen_text_artifacts,
    _ody_qwen_temperature_cap,
    _prepend_agent_directive,
    _resolved_tool_event_name,
    _strip_agent_injected_messages,
    _is_ollama_openai_compat_url,
    _is_local_openai_compat_url,
    _endpoint_lookup_keys,
    _detect_admin_intent,
    _extract_last_user_message,
    _user_turn_count,
    _insert_before_latest_user,
    _uploaded_files_context_message,
    _strip_think_blocks,
    _is_casual_low_signal,
    _is_explicit_continuation,
    _is_continuation_request,
    _recent_tool_events,
    _recent_tool_event_names,
    _message_text,
    _is_injected_context_message,
    _is_real_conversation_message,
    _collect_tool_events_for_brief,
    _recent_turns_for_brief,
    _build_conversation_brief_message,
    _build_continuation_trace_message,
    _assistant_requested_followup,
    _classify_agent_request,
    _turn_targets_active_document,
    _is_email_document_obj,
    _minimal_saved_memory_message,
    _compact_email_draft_context,
    _minimal_odysseus_doc_messages,
    _looks_like_notes_turn,
    _minimal_odysseus_notes_messages,
    _looks_like_memory_identity_turn,
    _minimal_odysseus_general_messages,
    _strip_doc_model_artifacts,
    _normalize_truncated_document_tool_fences,
    _normalize_stream_document_fences,
    _recent_context_for_retrieval,
)
from src.llm_core import (
    _normalize_http_status,
    _normalize_usage_counts,
    dedupe_model_candidates,
    stream_llm,
    stream_llm_with_fallback,
    _is_ollama_native_url,
)
from src.context_compactor import (
    apply_compaction_state,
    apply_compaction_state_for_session,
    maybe_compact,
)
from src.tool_security import (
    delegated_credential_blocked_tools,
    email_tool_policy_names,
    blocked_tools_for_owner,
    plan_mode_disabled_tools,
)
import asyncio
import collections
import itertools
import json
import re
import time
import logging
from typing import AsyncGenerator, List, Dict, Optional, Set
from urllib.parse import urlparse

from src.model_context import estimate_tokens
from src.settings import get_setting
from src.prompt_security import untrusted_context_message
from src.tool_utils import _truncate, get_mcp_manager
from src.agent_tools import (
    parse_tool_blocks,
    strip_tool_blocks,
    execute_tool_block,
    format_tool_result,
    set_active_document,
    set_active_model,
    function_call_to_tool_block,
    FUNCTION_TOOL_SCHEMAS,
    TOOL_TAGS,
    ToolBlock,
    MAX_AGENT_ROUNDS,
)

# Submodule imports

logger = logging.getLogger(__name__)

def _resolve_tool_blocks(
    round_response: str,
    native_tool_calls: list,
    round_num: int,
    is_api_model: bool = False,
    allow_fenced_for_api: bool = False,
):
    """Choose native function calls or fenced code block parsing. Returns (tool_blocks, used_native)."""
    used_native = False
    converted_calls = []  # native calls that converted, ALIGNED with tool_blocks
    if native_tool_calls:
        tool_blocks = []
        for tc in native_tool_calls:
            tc_name = tc.get("name", "")
            tc_args = tc.get("arguments", "{}")
            block = function_call_to_tool_block(tc_name, tc_args)
            if block:
                tool_blocks.append(block)
                converted_calls.append(tc)
                logger.info(f"  -> converted: {tc_name} -> {block.tool_type}")
            else:
                logger.warning(f"  -> FAILED to convert native call: {tc_name} args={tc_args[:200]}")
        if tool_blocks:
            used_native = True
    if not used_native:
        # Native function-calling models (GPT/Claude/Grok/Qwen3/DeepSeek-V, etc.)
        # have a reliable structured channel for real tool invocations. When such
        # a model emits no native tool_calls, any ```bash/```python/```json fence
        # in its prose is virtually always an illustrative example for the user
        # (e.g. "here's the command you'd run"), not an attempted tool call â€”
        # executing it causes accidental runs and clarification loops (#3222).
        #
        # Gate ONLY that fenced-block pattern for native models, not the whole
        # parser: explicit [TOOL_CALL]/<invoke>/<tool_code>/DSML markup that
        # leaks into content as text is never illustrative â€” it's a real call
        # the model couldn't emit on its structured channel (e.g. DeepSeek-V
        # falling back to DSML). Dropping the whole parser would silently lose
        # those too. Non-native / textual-only models keep every pattern,
        # fenced blocks included, since that's their *only* tool channel.
        tool_blocks = parse_tool_blocks(round_response, skip_fenced=(is_api_model and not allow_fenced_for_api))
        if tool_blocks:
            logger.info(f"Agent round {round_num}: {len(tool_blocks)} fenced tool block(s) detected")

    resp_preview = round_response[:200].replace('\n', '\\n') if round_response else "(empty)"
    logger.info(f"Agent round {round_num} summary: {len(round_response)} chars, "
                f"{len(native_tool_calls)} native calls, "
                f"{len(tool_blocks)} tool blocks. Preview: {resp_preview}")

    return tool_blocks, used_native, converted_calls


def _append_tool_results(
    messages: List[Dict],
    round_response: str,
    native_tool_calls: list,
    tool_results: list,
    tool_result_texts: list,
    used_native: bool,
    round_num: int,
    round_reasoning: str = "",
    tool_result_records: Optional[list] = None,
):
    """Append tool execution results back into the message history for the next LLM round.

    `round_reasoning` (DeepSeek / vLLM reasoning-parser deltas) is echoed
    back via `reasoning_content` on the assistant message â€” DeepSeek's API
    rejects follow-up requests in thinking mode that don't include the
    prior reasoning.

    NOTE: it is NOT universally ignored. Nemotron's chat template re-injects
    EVERY prior `reasoning_content` as a <think> block, and this agent loop is
    trimmed only once (before the loop), so across rounds the reasoning piles
    up unbounded â€” bloating context and feeding the model its own prior
    reasoning, which reinforces repetition/looping. So keep reasoning_content
    on the MOST RECENT assistant turn only: enough for DeepSeek continuity,
    without the per-round accumulation.
    """
    tool_result_records = tool_result_records or []
    # Strip reasoning_content from earlier assistant turns; only the newest keeps it.
    for _m in messages:
        if _m.get("role") == "assistant":
            _m.pop("reasoning_content", None)
    if used_native and native_tool_calls:
        assistant_msg = {"role": "assistant"}
        # When the model emitted ONLY tool calls (no prose), content must be
        # null, NOT an empty string. Google Gemini's OpenAI-compatible endpoint
        # and Ollama both reject an assistant message that carries tool_calls
        # alongside empty-string content with HTTP 400 ("contents is not
        # specified" / a JSON parse error), which aborts every tool-using turn
        # at the follow-up round. null (i.e. omitted text) is the spec-correct
        # form the OpenAI SDK itself emits, and OpenAI/Anthropic accept it too.
        assistant_msg["content"] = round_response if round_response.strip() else None
        if round_reasoning:
            assistant_msg["reasoning_content"] = round_reasoning
        assistant_msg["tool_calls"] = [
            {
                "id": tc.get("id", f"call_{round_num}_{j}"),
                "type": "function",
                "function": {
                    "name": tc.get("name", ""),
                    "arguments": tc.get("arguments", "{}"),
                },
                # Gemini 3 requires the opaque thought_signature it returned with
                # each function call to be echoed back on the follow-up turn, or
                # the next request 400s. Replay it when present; other providers
                # never emit it (their payload builders just ignore the field).
                **({"extra_content": tc["extra_content"]} if tc.get("extra_content") else {}),
            }
            for j, tc in enumerate(native_tool_calls)
        ]
        messages.append(assistant_msg)
        for j, tc in enumerate(native_tool_calls):
            result_text = tool_result_texts[j] if j < len(tool_result_texts) else ""
            record = tool_result_records[j] if j < len(tool_result_records) else {}
            tool_name = record.get("tool_name", tc.get("name", ""))
            tool_content = record.get("content", tc.get("arguments", ""))
            result = record.get(
                "result",
                tool_results[j] if j < len(tool_results) else None,
            )
            result_message = {
                "role": "tool",
                "tool_call_id": tc.get("id", f"call_{round_num}_{j}"),
                "content": result_text,
            }
            capabilities = capabilities_for_action(tool_name, tool_content)
            should_arm_gate = tool_result_should_arm_gate(
                tool_name,
                result,
                tool_content,
            )
            if (
                capabilities.result_integrity is not ResultIntegrity.SYSTEM
                or should_arm_gate
            ):
                result_message["metadata"] = {
                    "trusted": False,
                    "source": f"tool result: {tool_name}",
                    "tool_gate_untrusted": should_arm_gate,
                }
            messages.append(result_message)
    else:
        tool_output_text = "\n\n".join(tool_results)
        # An approved-action replay injects the sealed tool result with no
        # assistant prose for that round, which used to append an assistant turn
        # whose content was "". Anthropic's Messages API rejects a non-final
        # assistant message with empty content (HTTP 400), so the resumed turn
        # died before the model saw the result. A turn carrying neither prose nor
        # reasoning has nothing to say to any provider, so skip it entirely.
        if round_response.strip() or round_reasoning:
            msg = {"role": "assistant", "content": round_response}
            if round_reasoning:
                msg["reasoning_content"] = round_reasoning
            messages.append(msg)
        # Tool output (shell/python stdout, file reads, fetched pages, email
        # bodies, MCP results) is sourced from outside the server. Wrap it as
        # untrusted data so prompt-injection inside a tool result is treated as
        # data, not instructions â€” same hardening as skills (#788) and the
        # web/RAG context. THREAT_MODEL.md lists tool output as a surface that
        # must go through untrusted_context_message.
        arm_tool_gate = any(
            tool_result_should_arm_gate(
                record.get("tool_name"),
                record.get("result"),
                record.get("content"),
            )
            for record in tool_result_records
        )
        messages.append(
            untrusted_context_message(
                "tool execution results",
                tool_output_text,
                arm_tool_gate=arm_tool_gate,
            )
        )


def _compute_final_metrics(
    messages: List[Dict],
    full_response: str,
    total_duration: float,
    time_to_first_token,
    context_length: int,
    real_input_tokens: int,
    real_output_tokens: int,
    has_real_usage: bool,
    tool_events: list,
    round_texts: list,
    model: str = "",
    round_models: Optional[list] = None,
    round_endpoint_ids: Optional[list] = None,
    round_endpoint_labels: Optional[list] = None,
    last_round_input_tokens: int = 0,
    request_context_tokens: int = 0,
    prep_timings: Optional[Dict[str, float]] = None,
    backend_gen_tps: float = 0,
    backend_prefill_tps: float = 0,
) -> dict:
    """Compute token counts, TPS, and build the final metrics dict."""
    if has_real_usage:
        input_tokens = real_input_tokens
        output_tokens = real_output_tokens
    else:
        input_content = ""
        for msg in messages:
            if isinstance(msg.get("content"), str):
                input_content += msg["content"] + "\n"
        input_tokens = len(input_content) // 4
        output_tokens = len(full_response) // 4
    # Prefer the backend's true generation speed (llama.cpp
    # timings.predicted_per_second) â€” pure decode, no prefill/tool/network time.
    # Fall back to tokens/wall-clock only when the backend didn't report it
    # (e.g. cloud APIs without timings); that figure reads low because
    # total_duration includes prefill + agent overhead.
    if backend_gen_tps and backend_gen_tps > 0:
        tps = backend_gen_tps
    else:
        tps = output_tokens / total_duration if total_duration > 0 else 0
    # Context % should describe the prompt Odysseus assembled, not provider
    # billing/usage counters. Some providers report only the final agent round
    # or cache-adjusted input, which made the displayed context jump from e.g.
    # 44% to 5% even when the session history had not meaningfully changed.
    if request_context_tokens:
        ctx_tokens = request_context_tokens
    elif last_round_input_tokens:
        ctx_tokens = last_round_input_tokens
    elif has_real_usage:
        ctx_tokens = real_input_tokens
    else:
        ctx_tokens = estimate_tokens(messages)
    ctx_pct = min(round((ctx_tokens / context_length) * 100, 1), 100.0) if context_length else 0

    metrics = {
        "response_time": round(total_duration, 2),
        "time_to_first_token": round(time_to_first_token, 2) if time_to_first_token else 0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_per_second": round(tps, 2),
        # True decode speed when the backend reported it; "computed" = the
        # tokens/wall-clock fallback (reads low â€” includes prefill/overhead).
        "tps_source": "backend" if (backend_gen_tps and backend_gen_tps > 0) else "computed",
        "total_tokens": input_tokens + output_tokens,
        "request_context_tokens": ctx_tokens,
        "context_length": context_length,
        "context_percent": ctx_pct,
        "usage_source": "real" if has_real_usage else "estimated",
        "model": model,
    }
    if backend_prefill_tps and backend_prefill_tps > 0:
        metrics["prefill_tps"] = round(backend_prefill_tps, 2)
    if prep_timings:
        prep_total = round(sum(prep_timings.values()), 3)
        metrics["agent_prep_time"] = prep_total
        metrics["agent_model_wait_time"] = round(max((time_to_first_token or 0) - prep_total, 0), 3)
        metrics["agent_prep_breakdown"] = {
            key: round(value, 3) for key, value in prep_timings.items()
        }
    if tool_events:
        metrics["tool_events"] = tool_events
    if round_texts:
        metrics["round_texts"] = round_texts
        metrics["round_models"] = list(round_models or [])
        metrics["round_endpoint_ids"] = list(round_endpoint_ids or [])
        metrics["round_endpoint_labels"] = list(round_endpoint_labels or [])
    return metrics


# â”€â”€ Completion verifier â”€â”€
# Tools whose effects produce a checkable artifact. A turn that used one of
# these is "effectful" and worth an independent completion check; pure
# read-only / Q&A turns are not.
_VERIFIER_EFFECTFUL_TOOLS = {
    "create_document", "update_document", "edit_document",
    "bash", "python", "write_file",
    "edit_image",
}
_VERIFIER_MAX_ROUNDS = 2  # cap re-verify cycles per turn â€” never loop forever


def _build_actions_snapshot(tool_events: list, limit: int = 8000) -> str:
    """Compact record of what the agent actually did this turn, for the
    verifier to judge against. One block per tool execution: the command and
    a head of its output."""
    parts = []
    for ev in tool_events:
        tool = ev.get("tool", "?")
        cmd = (ev.get("command") or "").strip()
        out = (ev.get("output") or "").strip()
        rc = ev.get("exit_code")
        head = f"[{tool}] {cmd}" if cmd else f"[{tool}]"
        rc_s = f" (exit {rc})" if rc not in (None, 0) else ""
        body = (out[:1200] + " â€¦") if len(out) > 1200 else (out or "(no output)")
        parts.append(f"{head}{rc_s}\n-> {body}")
    snap = "\n\n".join(parts)
    return snap[:limit] if len(snap) > limit else snap


async def _run_verifier_subagent(
    instruction: str, actions_snapshot: str,
    *, endpoint_url: str, model: str, headers: dict,
) -> list:
    """Fresh-context completion verifier. A second model instance with NO
    shared history reads the user's request + a record of what the agent did
    and judges whether the task is genuinely complete. The independent context
    is the whole point: a model checking its own work rationalizes; one that
    didn't do the work reads it cold. Returns a list of failure reasons
    (empty = pass, or silently empty on any error so it can't block a valid
    completion)."""
    from src.llm_core import llm_call_async
    prompt = (
        "You are an independent verifier. Another assistant just claimed the "
        "following task is complete. Using ONLY the request and the record of "
        "what it actually did, decide whether that claim is correct. Be strict: "
        "only say SUCCESS if the work genuinely satisfies the request.\n\n"
        f"<user_request>\n{(instruction or '')[:4000]}\n</user_request>\n\n"
        f"<actions_taken>\n{actions_snapshot[:8000]}\n</actions_taken>\n\n"
        "<checklist>\n"
        "1. Every concrete deliverable the request asked for was actually produced\n"
        "2. Outputs/edits match what was asked â€” nothing missing, no extra or unrequested changes\n"
        "3. Tool results show success, not errors or empty output that got ignored\n"
        "4. Anything the request said to leave alone was left unchanged\n"
        "</checklist>\n\n"
        "Reason briefly (2-3 sentences max). Then output EXACTLY one of:\n"
        "  VERIFICATION: SUCCESS\n"
        "  VERIFICATION: FAIL: <one short sentence per issue, semicolon-separated>\n"
        "Output nothing after the VERIFICATION line."
    )
    try:
        raw = await llm_call_async(
            url=endpoint_url, model=model,
            messages=[{"role": "user", "content": prompt}],
            headers=headers, temperature=0.0, max_tokens=600, timeout=60,
        )
    except Exception as e:
        logger.warning(f"[agent] verifier subagent failed: {e}")
        return []
    raw = _strip_think_blocks(raw or "")
    last_v = None
    for line in raw.splitlines():
        if "VERIFICATION:" in line:
            last_v = line.strip()
    if not last_v or "VERIFICATION: FAIL:" not in last_v:
        return []
    reasons = last_v.split("VERIFICATION: FAIL:", 1)[1].strip()
    return [r.strip() for r in reasons.split(";") if r.strip()]


def _empty_response_fallback(
    full_response: str,
    round_reasoning: str,
    tool_events: list,
) -> tuple:
    """Return (final_response, sse_chunk_or_none) for the end-of-loop empty-response guard.

    When a thinking model routes all tokens to reasoning_content (leaving
    content=""), full_response is empty but round_reasoning has content.
    The reasoning was already streamed as {thinking:true} chunks â€” do not
    re-emit it as a normal delta.  Just persist it and yield nothing.

    Returns:
        (final_response: str, chunk: str | None)
            chunk is the SSE string to yield, or None if nothing should be emitted.
    """
    if full_response.strip() or tool_events:
        return full_response, None
    if round_reasoning.strip():
        return round_reasoning, None
    _error_msg = "The model returned an empty response. Please try again or switch to a different model."
    return _error_msg, f'data: {json.dumps({"delta": _error_msg})}\n\n'


PLAN_MODE_DIRECTIVE = (
    "## PLAN MODE â€” OVERRIDES EVERYTHING ELSE BELOW\n"
    "You are in PLAN MODE. Your ONLY job this turn is to PROPOSE a plan. You have "
    "NOT done anything yet. Do NOT claim you created, wrote, ran, sent, or changed "
    "anything â€” that would be a lie.\n"
    "\n"
    "ABSOLUTE RULE â€” DO NOT MUTATE ANYTHING. Every write/state-changing tool, "
    "including the shell (`bash`/`python`), is disabled this turn and will be "
    "rejected â€” only read-only tools remain available. Use the read-only tools "
    "listed below (read files, search code, browse the project, web lookups) to "
    "ground the plan. If the task is 'write a file', your plan is to DESCRIBE "
    "writing it â€” you do NOT write it now.\n"
    "\n"
    "OUTPUT: present the plan as a GitHub-style checklist, one concrete step per line:\n"
    "- [ ] first action you will take once approved\n"
    "- [ ] next action\n"
    "Each item = one concrete action (file to create/edit, command to run, side "
    "effect). Do not execute. Do not end with 'Done' or anything implying the work "
    "is finished. End your turn with the checklist."
)


def build_active_plan_note(approved_plan: str) -> str:
    """System note that pins an approved plan during execution.

    Sent back by the frontend each turn so a long plan on a weak model survives
    history truncation â€” the agent can always re-read it. Returns "" for empty
    input.
    """
    if not approved_plan or not approved_plan.strip():
        return ""
    return (
        "## ACTIVE PLAN (approved â€” execute this)\n"
        "You are executing a plan the user already approved. THE FULL PLAN IS "
        "BELOW â€” it is always provided here every turn. Do NOT say you lost it, "
        "and do NOT look for it in tasks, notes, memory, files, or the API; just "
        "read it below. Work through it IN ORDER. After finishing each step, call "
        "the `update_plan` tool with the full checklist and that step marked "
        "`- [x]` so progress stays visible in the user's plan window. If the user "
        "asks to change the plan, call `update_plan` with the revised checklist. "
        "Do the next unchecked item until all are done. Do not skip, reorder, or "
        "invent steps; if a step is genuinely impossible, say so and stop.\n\n"
        "Current plan:\n"
        + approved_plan.strip()
    )


def _detect_runaway_call(call_freq, threshold=15):
    """Tool name of a call signature repeated >= ``threshold`` times â€” a real
    runaway loop. Counts IDENTICAL repeated calls (same tool AND args), so a
    legitimate batch of distinct calls to one tool (e.g. creating 18 calendar
    events at once) is NOT flagged. Returns ``None`` when nothing is runaway.

    ``call_freq`` is a Counter keyed by ``"{tool_type}:{content[:120]}"``.
    """
    sig = next((s for s, n in call_freq.items() if n >= threshold), None)
    return sig.split(":", 1)[0] if sig else None


async def stream_agent_loop(
    endpoint_url: str,
    model: str,
    messages: List[Dict],
    headers: Optional[Dict] = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    prompt_type: Optional[str] = None,
    max_rounds: int = MAX_AGENT_ROUNDS,
    max_tool_calls: int = 0,
    context_length: int = 0,
    active_document=None,
    active_email: Optional[Dict[str, str]] = None,
    session_id: Optional[str] = None,
    disabled_tools: Optional[Set[str]] = None,
    owner: Optional[str] = None,
    relevant_tools: Optional[Set[str]] = None,
    fallbacks: Optional[List[tuple]] = None,
    prefer_fallbacks_first: bool = False,
    route_descriptors: Optional[List[dict]] = None,
    fallback_statuses: Optional[Set[int]] = None,
    fallback_on_empty: bool = True,
    plan_mode: bool = False,
    approved_plan: Optional[str] = None,
    tool_policy: Optional[ToolPolicy] = None,
    workspace: Optional[str] = None,
    forced_tools: Optional[Set[str]] = None,
    uploaded_files: Optional[List[Dict]] = None,
    upload_handler=None,
    workload: str = "foreground",
    external_untrusted_context_seen: bool = False,
    delegated_credential: bool = False,
    exact_approval: Optional[ExactToolApproval] = None,
    _is_teacher_run: bool = False,
    history_session=None,
    defer_context_shaping: bool = False,
) -> AsyncGenerator[str, None]:
    """Streaming agent loop generator.

    Yields SSE events:
      - data: {"delta": "text"}                             (text chunks)
      - data: {"type": "tool_start", "tool": "...", ...}    (before execution)
      - data: {"type": "tool_output", "tool": "...", ...}   (after execution)
      - data: {"type": "agent_step", "round": N}            (next round)
      - data: {"type": "metrics", "data": {...}}            (final metrics)
      - data: [DONE]                                        (end)
    """

    run_security = ToolRunSecurityContext(
        external_untrusted_context_seen=(
            bool(external_untrusted_context_seen)
            or bool(
                exact_approval
                and exact_approval.pending.external_untrusted_context_seen
            )
            or messages_contain_external_untrusted_context(messages)
        ),
        approval_gate_bypassed=bool(
            exact_approval and exact_approval.allow_remaining_actions
        ),
        delegated_credential=bool(delegated_credential),
    )
    mcp_mgr = get_mcp_manager()
    prep_timings: Dict[str, float] = {}
    disabled_tools = set(disabled_tools or [])
    route_descriptors = list(route_descriptors or [])
    while len(route_descriptors) < 1 + len(fallbacks or []):
        route_descriptors.append({})
    requested_route = route_descriptors[0] if route_descriptors else {}
    requested_endpoint_id = requested_route.get("endpoint_id")
    requested_endpoint_label = requested_route.get("endpoint_label") or "Selected route"
    requested_endpoint_cost_tracked = requested_route.get("endpoint_cost_tracked")
    if not isinstance(requested_endpoint_cost_tracked, bool):
        requested_endpoint_cost_tracked = None
    if tool_policy:
        disabled_tools.update(tool_policy.all_disabled_names())
        if tool_policy.disable_mcp:
            mcp_mgr = None
    guide_only = bool(tool_policy and tool_policy.mode == "guide_only")
    public_blocked_tools = blocked_tools_for_owner(owner)
    if delegated_credential:
        # owner is the admin who minted the token, so the call above returns
        # nothing. Cap the run regardless of who it acts for.
        public_blocked_tools.update(delegated_credential_blocked_tools())
    if public_blocked_tools:
        disabled_tools.update(public_blocked_tools)
        # MCP tools are namespaced dynamically, so hide all MCP schemas for
        # public/non-admin users rather than trying to enumerate every tool.
        mcp_mgr = None

    if plan_mode:
        # Plan mode: investigate read-only, propose a plan, don't execute. The
        # route also unions the read-only-disabled set, but enforce here too so
        # the loop is safe regardless of caller. MCP stays available but is
        # filtered to read-only tools below (after the disabled map is loaded).
        disabled_tools.update(plan_mode_disabled_tools())

    uploaded_files = uploaded_files or []
    _upload_msg = _uploaded_files_context_message(uploaded_files)
    if _upload_msg:
        messages = _insert_before_latest_user(messages, _upload_msg)

    _t0 = time.time()
    _needs_admin = _detect_admin_intent(messages)
    _last_user = _extract_last_user_message(messages)
    _ody_qwen_finetune_model = _is_odysseus_qwen_model(model)
    # The caller's temperature survives for non-qwen routes; the qwen cap is
    # applied per candidate (here for the primary, in the candidate request
    # factories for fallbacks), so neither direction of a mixed qwen/non-qwen
    # fallback chain inherits the other's value.
    _requested_temperature = temperature
    if _ody_qwen_finetune_model:
        temperature = _ody_qwen_temperature_cap(temperature)
    _ody_memory_identity_turn = _looks_like_memory_identity_turn(_last_user)
    _intent = _classify_agent_request(messages, _last_user)
    _low_signal_turn = bool(_intent.get("low_signal"))
    _casual_low_signal_turn = _is_casual_low_signal(_last_user)
    _existing_conversation = _user_turn_count(messages) > 1
    _active_document_relevant = _turn_targets_active_document(_intent, _last_user, active_document)
    _active_email_draft_relevant = _active_document_relevant and _is_email_document_obj(active_document)
    if _active_email_draft_relevant:
        disabled_tools.update({
            "list_email_accounts", "list_emails", "read_email", "scan_email_unsubscribes",
            "mcp__email__list_emails", "mcp__email__read_email", "mcp__email__scan_email_unsubscribes",
        })
    _prompt_active_document = active_document if _active_document_relevant else None
    _direct_low_signal = (
        _low_signal_turn
        and not _existing_conversation
        and not bool(_intent.get("continuation"))
        and not plan_mode
        and not approved_plan
        and not guide_only
        and (_casual_low_signal_turn or not _active_document_relevant)
        and (_casual_low_signal_turn or not active_email)
        and (_casual_low_signal_turn or not workspace)
        and not forced_tools
        and not relevant_tools
    )
    # Tool retrieval uses the latest message by default. It may inherit recent
    # user turns only for explicit continuations ("yes", "do it", "1").
    _retrieval_query = str(_intent.get("retrieval_query") or _last_user)
    if _explicitly_references_missing_workspace(_retrieval_query, workspace):
        msg = (
            "No active workspace is set. Use `/workspace pick` or "
            "`/workspace set /absolute/path`, then rerun the request."
        )
        yield f"data: {json.dumps({'delta': msg})}\n\n"
        metrics = {
            "model": model,
            "requested_model": model,
            "input_tokens": estimate_tokens(messages),
            "output_tokens": max(len(msg) // 4, 1),
            "total_time": 0,
            "response_time": 0,
            "agent_rounds": 0,
            "tool_calls": 0,
            "missing_workspace": True,
        }
        yield f"data: {json.dumps({'type': 'metrics', 'data': metrics})}\n\n"
        yield "data: [DONE]\n\n"
        return
    logger.info(
        "[agent-intent] latest=%r continuation=%s low_signal=%s domains=%s active_doc_relevant=%s retrieval_query=%r",
        _last_user[:120],
        bool(_intent.get("continuation")),
        _low_signal_turn,
        sorted(_intent.get("domains") or []),
        _active_document_relevant,
        _retrieval_query[:200],
    )
    if _low_signal_turn and _existing_conversation:
        logger.info(
            "[agent] keeping contextual path for low-signal turn in existing conversation latest=%r",
            _last_user[:80],
        )
    _mcp_disabled_map = _load_mcp_disabled_map() if mcp_mgr else {}
    if _direct_low_signal:
        logger.info("[agent] direct low-signal reply path for latest=%r", _last_user[:80])
        direct_messages = (
            _minimal_odysseus_general_messages(
                messages,
                include_memory=True,
            )
            if _ody_qwen_finetune_model
            else [{"role": "user", "content": _last_user}]
        )
        direct_response = ""
        direct_start = time.time()
        direct_actual_model = model
        direct_actual_endpoint_id = requested_endpoint_id
        direct_actual_endpoint_label = requested_endpoint_label
        direct_actual_endpoint_cost_tracked = requested_endpoint_cost_tracked
        direct_actual_messages = direct_messages
        direct_candidate_messages = {0: direct_messages}
        direct_reasoning = ""
        real_input_tokens = 0
        real_output_tokens = 0
        direct_has_real_usage = False

        def _direct_candidate_request(_index, _url, candidate_model, _headers):
            candidate_is_qwen = _is_odysseus_qwen_model(candidate_model)
            candidate_messages = (
                _minimal_odysseus_general_messages(messages, include_memory=True)
                if candidate_is_qwen
                else [{"role": "user", "content": _last_user}]
            )
            direct_candidate_messages[_index] = candidate_messages
            return {
                "messages": candidate_messages,
                "kwargs": {
                    "temperature": (
                        _ody_qwen_temperature_cap(_requested_temperature)
                        if candidate_is_qwen
                        else _requested_temperature
                    ),
                },
            }

        def _direct_terminal_event(terminal_status, failure_message):
            """Build truthful partial-history metadata for direct-path failure."""
            if not (direct_response.strip() or direct_reasoning.strip()):
                return None
            direct_usage = _usage_bucket(
                round_num=1,
                model=direct_actual_model,
                endpoint_id=direct_actual_endpoint_id,
                endpoint_label=direct_actual_endpoint_label,
                endpoint_cost_tracked=direct_actual_endpoint_cost_tracked,
                input_tokens=(
                    real_input_tokens
                    if direct_has_real_usage
                    else estimate_tokens(direct_actual_messages)
                ),
                output_tokens=(
                    real_output_tokens
                    if direct_has_real_usage
                    else max(len(direct_response + direct_reasoning) // 4, 0)
                ),
                usage_source="real" if direct_has_real_usage else "estimated",
            )
            failure_note = f"[Agent stopped: {failure_message}]"
            terminal_round = (
                f"{direct_response.strip()}\n\n{failure_note}"
                if direct_response.strip()
                else failure_note
            )
            terminal_metadata = {
                "failed": True,
                "failure": {
                    "status": terminal_status,
                    "message": failure_message,
                },
                "model": direct_actual_model,
                "requested_model": model,
                "endpoint_id": direct_actual_endpoint_id,
                "endpoint_label": direct_actual_endpoint_label,
                "requested_endpoint_id": requested_endpoint_id,
                "requested_endpoint_label": requested_endpoint_label,
                "round_texts": [terminal_round],
                "round_models": [direct_actual_model],
                "round_endpoint_ids": [direct_actual_endpoint_id],
                "round_endpoint_labels": [direct_actual_endpoint_label],
                **_usage_bucket_summary([direct_usage]),
            }
            if direct_reasoning.strip():
                terminal_metadata["thinking"] = direct_reasoning.strip()
            if isinstance(direct_actual_endpoint_cost_tracked, bool):
                terminal_metadata["endpoint_cost_tracked"] = (
                    direct_actual_endpoint_cost_tracked
                )
            return f'data: {json.dumps({"type": "agent_terminal", "data": terminal_metadata})}\n\n'

        try:
            async for chunk in stream_llm_with_fallback(
                [(endpoint_url, model, headers)] + list(fallbacks or []),
                direct_messages,
                temperature=temperature,
                max_tokens=min(max_tokens or 128, 128),
                prompt_type=None,
                tools=None,
                timeout=int(get_setting("agent_stream_timeout_seconds", 300) or 300),
                session_id=session_id,
                workload=workload,
                fallback_statuses=fallback_statuses,
                fallback_on_empty=fallback_on_empty,
                candidate_request_factory=_direct_candidate_request,
                candidate_route_descriptors=route_descriptors,
            ):
                if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
                    try:
                        data = json.loads(chunk[6:])
                    except json.JSONDecodeError:
                        yield chunk
                        continue
                    if data.get("type") == "usage":
                        usage = data.get("data", {}) or {}
                        direct_actual_model = usage.get("model") or direct_actual_model
                        normalized_usage = _normalize_usage_counts(
                            usage.get("input_tokens", 0),
                            usage.get("output_tokens", 0),
                        )
                        if normalized_usage is None:
                            logger.warning("[agent] ignoring malformed direct usage event")
                            continue
                        real_input_tokens += normalized_usage["input_tokens"]
                        real_output_tokens += normalized_usage["output_tokens"]
                        direct_has_real_usage = True
                        continue
                    if data.get("type") == "model_actual":
                        direct_actual_model = data.get("model") or direct_actual_model
                        data["requested_model"] = model
                        data["requested_endpoint_id"] = requested_endpoint_id
                        data["requested_endpoint_label"] = requested_endpoint_label
                        data["endpoint_id"] = direct_actual_endpoint_id
                        data["endpoint_label"] = direct_actual_endpoint_label
                        yield f"data: {json.dumps(data)}\n\n"
                        continue
                    if data.get("type") == "fallback":
                        direct_actual_model = data.get("answered_by") or direct_actual_model
                        direct_actual_endpoint_id = data.get("answered_by_endpoint_id")
                        direct_actual_endpoint_label = (
                            data.get("answered_by_endpoint_label") or direct_actual_endpoint_label
                        )
                        if isinstance(data.get("answered_by_endpoint_cost_tracked"), bool):
                            direct_actual_endpoint_cost_tracked = data.get(
                                "answered_by_endpoint_cost_tracked"
                            )
                        candidate_index = data.get("candidate_index")
                        if isinstance(candidate_index, int):
                            direct_actual_messages = direct_candidate_messages.get(
                                candidate_index,
                                direct_actual_messages,
                            )
                        yield chunk
                        continue
                    if "delta" in data:
                        if data.get("thinking"):
                            direct_reasoning += data.get("delta", "")
                        else:
                            direct_response += data.get("delta", "")
                        yield chunk
                        continue
                    yield chunk
                elif chunk.startswith("event: error"):
                    # A provider/request error is terminal here too.  Do not
                    # replace it with the casual-response fallback or emit
                    # success metrics/[DONE].
                    terminal_status = None
                    try:
                        error_line = next(
                            line[6:]
                            for line in chunk.splitlines()
                            if line.startswith("data: ")
                        )
                        terminal_status = _normalize_http_status(
                            json.loads(error_line).get("status")
                        )
                    except (StopIteration, json.JSONDecodeError):
                        terminal_status = None
                    failure_message = (
                        f"Model request failed (HTTP {terminal_status})"
                        if terminal_status is not None
                        else "Model request failed"
                    )
                    terminal_event = _direct_terminal_event(
                        terminal_status,
                        failure_message,
                    )
                    if terminal_event:
                        yield terminal_event
                    yield chunk
                    return
                elif chunk.startswith("event: "):
                    yield chunk
        except Exception as _direct_err:
            logger.warning("[agent] direct low-signal path failed: %s", _direct_err)
            failure_message = "Model request failed"
            terminal_event = _direct_terminal_event(None, failure_message)
            if terminal_event:
                yield terminal_event
            yield (
                "event: error\n"
                f"data: {json.dumps({'error': failure_message, 'status': 500, 'fallback_eligible': False})}\n\n"
            )
            return

        if not direct_response.strip():
            failure_message = "Model returned an empty response"
            terminal_event = _direct_terminal_event(None, failure_message)
            if terminal_event:
                yield terminal_event
            yield (
                "event: error\n"
                f"data: {json.dumps({'error': failure_message, 'status': 502, 'fallback_eligible': False})}\n\n"
            )
            return

        duration = time.time() - direct_start
        direct_usage = _usage_bucket(
            round_num=1,
            model=direct_actual_model,
            endpoint_id=direct_actual_endpoint_id,
            endpoint_label=direct_actual_endpoint_label,
            endpoint_cost_tracked=direct_actual_endpoint_cost_tracked,
            input_tokens=(
                real_input_tokens
                if direct_has_real_usage
                else estimate_tokens(direct_actual_messages)
            ),
            output_tokens=(
                real_output_tokens
                if direct_has_real_usage
                else max(len(direct_response) // 4, 1)
            ),
            usage_source="real" if direct_has_real_usage else "estimated",
        )
        metrics = {
            "model": direct_actual_model,
            "requested_model": model,
            "endpoint_id": direct_actual_endpoint_id,
            "endpoint_label": direct_actual_endpoint_label,
            "requested_endpoint_id": requested_endpoint_id,
            "requested_endpoint_label": requested_endpoint_label,
            "input_tokens": real_input_tokens or estimate_tokens(direct_actual_messages),
            "output_tokens": real_output_tokens or max(len(direct_response) // 4, 1),
            "total_time": round(duration, 2),
            "response_time": round(duration, 2),
            "agent_rounds": 0,
            "tool_calls": 0,
            "direct_low_signal": True,
            **_usage_bucket_summary([direct_usage]),
        }
        if isinstance(direct_actual_endpoint_cost_tracked, bool):
            metrics["endpoint_cost_tracked"] = direct_actual_endpoint_cost_tracked
        yield f"data: {json.dumps({'type': 'metrics', 'data': metrics})}\n\n"
        yield "data: [DONE]\n\n"
        return

    if plan_mode and mcp_mgr:
        # Allow read-only MCP tools to investigate, block write/unknown ones:
        # hide them from the schemas AND reject them at runtime by qualified name.
        _mcp_block_map, _mcp_block_q = mcp_mgr.plan_mode_blocked_mcp()
        for _sid, _names in _mcp_block_map.items():
            _mcp_disabled_map.setdefault(_sid, set()).update(_names)
        disabled_tools.update(_mcp_block_q)
    prep_timings["request_setup"] = time.time() - _t0

    # RAG-based tool selection: retrieve relevant tools for this query.
    # If caller provided a pre-computed set (e.g. task_scheduler), use that.
    _relevant_tools = relevant_tools
    _t1 = time.time()
    if _relevant_tools:
        logger.info(f"[tool-rag] Using caller-provided relevant_tools ({len(_relevant_tools)} tools)")
    if not guide_only and not _relevant_tools and _low_signal_turn:
        from src.tool_index import ALWAYS_AVAILABLE
        if workspace:
            # An active workspace IS the file-work signal: a vague "look at the
            # project" means explore this folder. Surface only the READ-ONLY file
            # tools (intersection with the plan-mode read-only allowlist) so the
            # agent can investigate; write/shell tools stay out until the request
            # actually calls for them (RAG retrieval adds those on a real ask).
            _relevant_tools = set(ALWAYS_AVAILABLE)
            from src.tool_security import PLAN_MODE_READONLY_TOOLS
            _relevant_tools |= (_DOMAIN_TOOL_MAP["files"] & PLAN_MODE_READONLY_TOOLS)
            logger.info("[tool-rag] Low-signal but workspace active; including read-only file tools")
        else:
            # Don't short-circuit: fall through to RAG retrieval below.
            # Non-English queries are flagged low_signal by the English-only
            # intent classifier, but fastembed retrieval works across languages.
            logger.info("[tool-rag] Low-signal query; will run RAG retrieval")
    if not guide_only and not _relevant_tools:
        try:
            from src.tool_index import get_tool_index, ALWAYS_AVAILABLE
            try:
                tool_idx = await asyncio.wait_for(
                    asyncio.to_thread(get_tool_index),
                    timeout=_TOOL_SELECTION_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "[tool-rag] Tool index init exceeded %.1fs; falling back to always-available tools",
                    _TOOL_SELECTION_TIMEOUT_SECONDS,
                )
                tool_idx = None
                _relevant_tools = set(ALWAYS_AVAILABLE)
            if tool_idx:
                if mcp_mgr:
                    try:
                        await asyncio.wait_for(
                            asyncio.to_thread(tool_idx.index_mcp_tools, mcp_mgr, _mcp_disabled_map),
                            timeout=_TOOL_SELECTION_TIMEOUT_SECONDS,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "[tool-rag] MCP tool indexing exceeded %.1fs; continuing without reindex",
                            _TOOL_SELECTION_TIMEOUT_SECONDS,
                        )
                if _retrieval_query:
                    try:
                        _relevant_tools = await asyncio.wait_for(
                            asyncio.to_thread(tool_idx.get_tools_for_query, _retrieval_query, 8),
                            timeout=_TOOL_SELECTION_TIMEOUT_SECONDS,
                        )
                        logger.info(f"[tool-rag] Retrieved tools for query: {sorted(_relevant_tools - ALWAYS_AVAILABLE)}")
                    except asyncio.TimeoutError:
                        # Leave _relevant_tools unset so the keyword fallback
                        # below still runs. Hard-coding ALWAYS_AVAILABLE here
                        # skipped the deterministic keyword hints whenever the
                        # embedding backend was slow (e.g. a remote endpoint
                        # cold-loading its model), silently stripping email/
                        # calendar tools from queries that named them outright.
                        logger.warning(
                            "[tool-rag] Retrieval exceeded %.1fs; falling back to keyword tool selection",
                            _TOOL_SELECTION_TIMEOUT_SECONDS,
                        )
                        _relevant_tools = None
        except Exception as e:
            logger.warning(f"[tool-rag] Retrieval failed, using keyword fallback: {e}")
            _relevant_tools = None

    # Fallback: if RAG unavailable, use keyword-based tool selection
    # instead of sending ALL tools (which overwhelms the model).
    if not guide_only and not _relevant_tools and _retrieval_query:
        from src.tool_index import ALWAYS_AVAILABLE, ToolIndex
        _relevant_tools = set(ALWAYS_AVAILABLE)
        ql = _retrieval_query.lower()
        for keywords, tools in ToolIndex._KEYWORD_HINTS.items():
            if any(kw in ql for kw in keywords):
                _relevant_tools.update(tools)
        logger.info(f"[tool-rag] Keyword fallback selected: {sorted(_relevant_tools - ALWAYS_AVAILABLE)}")

    # If deterministic domain detection fired, seed the corresponding domain
    # tools into the selected tool set. This is not direct prompt-pack
    # injection: `_assemble_prompt()` still derives domain rules from the final
    # tool names. It prevents obvious requests like "last 5 emails" from
    # collapsing to only ask_user/manage_memory when vector retrieval misses or
    # times out.
    if not guide_only and _relevant_tools is not None:
        for _domain in (_intent.get("domains") or set()):
            _relevant_tools.update(_DOMAIN_TOOL_MAP.get(str(_domain), set()))
        if "cookbook" in (_intent.get("domains") or set()):
            _relevant_tools.update({
                "list_served_models",
                "list_downloads",
                "list_cached_models",
                "list_cookbook_servers",
                "list_serve_presets",
            })
        if "email" in (_intent.get("domains") or set()):
            _relevant_tools.add("ui_control")
        if "web" in (_intent.get("domains") or set()):
            _relevant_tools.update(WEB_TOOL_NAMES)
            _blocked_web_tools = sorted(WEB_TOOL_NAMES & disabled_tools)
            if _blocked_web_tools:
                logger.info(
                    "[agent-intent] web domain selected but search tools remain disabled=%s",
                    _blocked_web_tools,
                )
        if "ui" in (_intent.get("domains") or set()):
            _relevant_tools.add("ui_control")
        if (
            (
                (
                    workspace
                    and _looks_like_workspace_coding_request(_retrieval_query or _last_user)
                )
                or _looks_like_local_computer_request(_retrieval_query or _last_user)
            )
            and not _active_document_relevant
            and not active_email
        ):
            _relevant_tools = set(_WORKSPACE_TERMINUS_TOOLS)
            logger.info("[tool-rag] Workspace file/terminal request; using Odysseus Terminus toolset")

    # If this turn targets the open document, keep editing tools available
    # regardless of which selection path (RAG, keyword, caller-provided) ran.
    # Do not leak document tools into unrelated turns just because the editor
    # panel is open.
    if _relevant_tools is not None and _active_document_relevant:
        _relevant_tools.update({"edit_document", "update_document", "suggest_document"})
        if _active_email_draft_relevant:
            # The open compose document already contains the recipient,
            # subject, source UID, and quoted previous-message excerpt. Reading
            # the same email again through IMAP/MCP is slow, token-heavy, and
            # can hang. Keep draft editing tools, drop email fetch tools.
            _email_fetch_tools = {
                "list_email_accounts", "list_emails", "read_email", "scan_email_unsubscribes",
                "mcp__email__list_emails", "mcp__email__read_email", "mcp__email__scan_email_unsubscribes",
            }
            removed = sorted(_relevant_tools & _email_fetch_tools)
            if removed:
                _relevant_tools.difference_update(_email_fetch_tools)
                logger.info("[agent-intent] active email draft pruned fetch tools=%s", removed)

    # Current and historical chat uploads use the dedicated owner-checked
    # reader rather than exposing their host paths to generic file tools.
    if not guide_only and uploaded_files:
        if _relevant_tools is None:
            from src.tool_index import ALWAYS_AVAILABLE
            _relevant_tools = set(ALWAYS_AVAILABLE)
        _relevant_tools.add("read_attachment")

    # Per-request forced tools are stronger than retrieval. Explicit search
    # settings make web tools visible even when tool RAG misses them;
    # route-level disabled_tools decides what remains allowed.
    if not guide_only and forced_tools:
        forced_set = {t for t in forced_tools if t not in disabled_tools}
        if _relevant_tools is None:
            from src.tool_index import ALWAYS_AVAILABLE
            _relevant_tools = set(ALWAYS_AVAILABLE)
        _relevant_tools.update(forced_set)

    if not guide_only and _relevant_tools is not None:
        _relevant_tools = _expand_browser_mcp_tools(_relevant_tools, mcp_mgr)

    # The skill index injected by _build_system_prompt tells the model to
    # call `manage_skills action=view`, and Jaccard-matched skills are pasted
    # into the prompt as procedures to follow â€” but neither path goes through
    # tool selection, so the model can be handed a procedure naming tools
    # (grep, read_file, ...) that aren't in its schema list. Keep the schemas
    # in lockstep: manage_skills is callable whenever any skill is indexed,
    # and a matched skill's declared requires_toolsets ride along with it.
    if not guide_only and _relevant_tools is not None and not _low_signal_turn:
        try:
            from services.memory.skills import SkillsManager
            from src.constants import DATA_DIR
            _skills_on = True
            try:
                from routes.prefs_routes import _load_for_user as _load_prefs
                _skills_on = (_load_prefs(owner) or {}).get("skills_enabled", True)
            except Exception:
                pass
            _sm = SkillsManager(DATA_DIR)
            _owner_skills = _sm.load(owner=owner) if _skills_on else []
            if _owner_skills:
                _relevant_tools.add("manage_skills")
                if _retrieval_query:
                    # Validate against every known executable tool, not just
                    # TOOL_SECTIONS â€” code-nav tools (grep/glob/ls) ship as
                    # schemas without a prompt-prose section.
                    from src.tool_policy import known_tool_names
                    _known = known_tool_names()
                    for _sk in _sm.get_relevant_skills(
                        _retrieval_query, skills=_owner_skills,
                        threshold=0.25, max_items=3,
                    ):
                        _relevant_tools.update(
                            t for t in (_sk.get("requires_toolsets") or [])
                            if t in _known
                        )
        except Exception as _e:
            logger.debug(f"[tool-rag] skill-aware tool include skipped: {_e}")

    _intent_domains = set(_intent.get("domains") or set())
    _base_relevant_tools = None if _relevant_tools is None else set(_relevant_tools)
    _runtime_skill_tools: Set[str] = set()

    def _route_finetune_modes(candidate_model: str):
        is_ody = _is_odysseus_qwen_model(candidate_model)
        doc_mode = (
            is_ody
            and not _runtime_skill_tools
            and (
                "documents" in _intent_domains
                or _active_document_relevant
                or _prompt_active_document is not None
            )
            and "files" not in _intent_domains
            and not guide_only
        )
        notes_mode = (
            is_ody
            and not _runtime_skill_tools
            and not doc_mode
            and (
                "notes_calendar_tasks" in _intent_domains
                or _looks_like_notes_turn(_last_user)
                or (
                    _looks_like_notes_calendar_followup(_last_user)
                    and _minimal_recent_notes_tool_context_message(messages) is not None
                )
            )
            and "files" not in _intent_domains
            and not guide_only
        )
        general_no_tool_mode = (
            is_ody
            and not _runtime_skill_tools
            and not doc_mode
            and not notes_mode
            and not guide_only
        )
        return (
            is_ody,
            doc_mode,
            notes_mode,
            doc_mode and _prompt_active_document is None,
            general_no_tool_mode,
        )

    def _route_relevant_tools(candidate_model: str):
        route_tools = None if _base_relevant_tools is None else set(_base_relevant_tools)
        (
            _is_ody,
            doc_mode,
            notes_mode,
            _stream_create,
            general_no_tool_mode,
        ) = _route_finetune_modes(candidate_model)
        if doc_mode and route_tools is not None:
            if _prompt_active_document is not None:
                route_tools = {
                    "edit_document", "update_document", "suggest_document",
                    "ask_user", "update_plan",
                }
            else:
                route_tools = {"create_document", "ask_user", "update_plan"}
        elif notes_mode and route_tools is not None:
            route_tools = {
                "manage_notes", "manage_calendar", "manage_tasks",
                "ask_user", "update_plan",
            }
        elif general_no_tool_mode:
            route_tools = set()
        return route_tools

    (
        _ody_qwen_finetune_model,
        _ody_doc_finetune_mode,
        _ody_notes_finetune_mode,
        _ody_doc_stream_create_mode,
        _ody_general_no_tool_mode,
    ) = _route_finetune_modes(model)
    _relevant_tools = _route_relevant_tools(model)
    if _ody_doc_finetune_mode and _relevant_tools is not None:
        logger.info("[agent-intent] odysseus doc finetune tool clamp=%s", sorted(_relevant_tools))
    elif _ody_notes_finetune_mode and _relevant_tools is not None:
        disabled_tools.difference_update({
            "manage_notes", "manage_calendar", "manage_tasks",
        })
        logger.info("[agent-intent] odysseus notes finetune tool clamp=%s", sorted(_relevant_tools))
    elif _ody_general_no_tool_mode:
        try:
            from src.tool_policy import known_tool_names
            disabled_tools.update(known_tool_names())
        except Exception:
            pass
        logger.info("[agent-intent] odysseus general no-tool clamp active")

    if (
        _relevant_tools is not None
        and _active_document_relevant
        and "files" not in _intent_domains
        and not uploaded_files
        and not workspace
    ):
        _doc_irrelevant_file_tools = {
            "append_file",
            "bash",
            "edit_file",
            "glob",
            "grep",
            "ls",
            "read_file",
            "replace_file",
            "run_shell",
            "write_file",
        }
        if _base_relevant_tools is not None:
            _base_relevant_tools.difference_update(_doc_irrelevant_file_tools)
        _removed_doc_file_tools = sorted(_relevant_tools & _doc_irrelevant_file_tools)
        if _removed_doc_file_tools:
            _relevant_tools.difference_update(_doc_irrelevant_file_tools)
            logger.info(
                "[agent-intent] active document turn removed file tools=%s",
                _removed_doc_file_tools,
            )

    if _relevant_tools is not None:
        logger.info("[agent-intent] selected_tools=%s", sorted(_relevant_tools)[:50])

    prep_timings["tool_selection"] = time.time() - _t1

    _t2 = time.time()
    _route_context_lengths = {}

    def _trim_route_request_messages(candidate_url, candidate_model, route_messages):
        """Apply the candidate route's own context budget to its request."""

        def _without_protection(items):
            # Route markers remain internal for later prompt rebuilding;
            # protection metadata is only needed during trimming.
            return [{k: v for k, v in message.items() if k != "_protected"} for message in items]

        try:
            from src.context_compactor import trim_for_context
            from src.context_budget import (
                compute_input_token_budget,
                DEFAULT_BUDGET,
                DEFAULT_HARD_MAX,
                budget_is_explicit as _budget_is_explicit,
            )
            from src.model_context import budget_context_for_model

            candidate_context = budget_context_for_model(
                candidate_url,
                candidate_model,
                fallback=context_length,
            )
            _route_context_lengths[(candidate_url, candidate_model)] = candidate_context
            soft_budget = int(get_setting("agent_input_token_budget", DEFAULT_BUDGET) or 0)
            if soft_budget <= 0:
                return _without_protection(route_messages)
            before_trim_tokens = estimate_tokens(route_messages)
            reserve_tokens = min(max(max_tokens or 1024, 512), 2048)
            try:
                hard_max = int(
                    get_setting("agent_input_token_hard_max", DEFAULT_HARD_MAX)
                    or DEFAULT_HARD_MAX
                )
            except (TypeError, ValueError):
                hard_max = DEFAULT_HARD_MAX
            if hard_max <= 0:
                hard_max = DEFAULT_HARD_MAX
            budget_is_explicit = _budget_is_explicit(soft_budget)
            effective_budget = compute_input_token_budget(
                soft_budget,
                candidate_context,
                budget_is_explicit,
                hard_max=hard_max,
            )
            trimmed_messages = trim_for_context(
                route_messages,
                effective_budget,
                reserve_tokens=reserve_tokens,
            )
            after_trim_tokens = estimate_tokens(trimmed_messages)
            if after_trim_tokens < before_trim_tokens:
                logger.info(
                    "[agent] soft-trimmed route model=%s context: %s -> %s tokens "
                    "(budget=%s, reserve=%s)",
                    candidate_model,
                    before_trim_tokens,
                    after_trim_tokens,
                    effective_budget,
                    reserve_tokens,
                )
            return _without_protection(trimmed_messages)
        except Exception as e:
            logger.warning(
                "[agent] Soft context trim skipped for route model=%s: %s",
                candidate_model,
                e,
            )
            return _without_protection(route_messages)

    async def _build_route_request_state(candidate_url, candidate_model, candidate_headers, source_messages):
        compaction_state: Dict = {}
        compacted_source = list(source_messages)
        was_compacted = False
        if defer_context_shaping or fallbacks:
            compacted_source, _candidate_context, was_compacted = await maybe_compact(
                None,
                candidate_url,
                candidate_model,
                compacted_source,
                candidate_headers,
                owner=owner,
                persist=False,
                compaction_state=compaction_state,
            )
        (
            is_ody,
            doc_mode,
            notes_mode,
            stream_create_mode,
            _general_no_tool_mode,
        ) = _route_finetune_modes(candidate_model)
        route_tools = _route_relevant_tools(candidate_model)
        is_api, is_native_ollama, is_ollama_compat = _agent_route_tool_mode(
            candidate_url,
            candidate_model,
            owner,
            headers=candidate_headers,
        )
        route_messages, route_mcp_schemas = _build_system_prompt(
            _strip_agent_injected_messages(compacted_source),
            candidate_model,
            _prompt_active_document,
            mcp_mgr,
            disabled_tools,
            needs_admin=_needs_admin,
            relevant_tools=route_tools,
            mcp_disabled_map=_mcp_disabled_map,
            compact=is_api or is_native_ollama or is_ollama_compat,
            owner=owner,
            suppress_local_context=guide_only,
            suppress_skills=_low_signal_turn,
            active_email=active_email,
            workspace=workspace,
        )
        if doc_mode and not plan_mode and not approved_plan and not guide_only:
            route_messages = _minimal_odysseus_doc_messages(
                route_messages,
                _prompt_active_document,
                stream_create=stream_create_mode,
            )
            route_mcp_schemas = []
        elif notes_mode and not plan_mode and not approved_plan and not guide_only:
            route_messages = _minimal_odysseus_notes_messages(route_messages)
            route_mcp_schemas = []
        elif (
            is_ody
            and not _runtime_skill_tools
            and not plan_mode
            and not approved_plan
            and not guide_only
        ):
            route_messages = _minimal_odysseus_general_messages(route_messages, include_memory=True)
            route_mcp_schemas = []
        if plan_mode and not guide_only:
            _prepend_agent_directive(route_messages, PLAN_MODE_DIRECTIVE)
        elif approved_plan and approved_plan.strip() and not guide_only:
            _prepend_agent_directive(route_messages, build_active_plan_note(approved_plan))
        if guide_only:
            _prepend_agent_directive(route_messages, GUIDE_ONLY_DIRECTIVE)
        return {
            "messages": route_messages,
            "mcp_schemas": route_mcp_schemas,
            "relevant_tools": route_tools,
            "is_api_model": is_api,
            "is_ollama_native": is_native_ollama,
            "ollama_openai_compat": is_ollama_compat,
            "ody_qwen_finetune_model": is_ody,
            "ody_doc_finetune_mode": doc_mode,
            "ody_notes_finetune_mode": notes_mode,
            "ody_doc_stream_create_mode": stream_create_mode,
            "compaction_state": compaction_state,
            "was_compacted": was_compacted,
        }

    _initial_route_source_messages = messages
    _route_state = await _build_route_request_state(
        endpoint_url,
        model,
        headers,
        _initial_route_source_messages,
    )
    messages = _route_state["messages"]
    mcp_schemas = _route_state["mcp_schemas"]
    _relevant_tools = _route_state["relevant_tools"]
    _is_api_model = _route_state["is_api_model"]
    _is_ollama_native = _route_state["is_ollama_native"]
    _ollama_openai_compat = _route_state["ollama_openai_compat"]
    if approved_plan and approved_plan.strip() and not guide_only:
        logger.info("[plan] pinned approved plan (%d chars) for execution turn", len(approved_plan))
    prep_timings["prompt_build"] = time.time() - _t2

    _t3 = time.time()
    _initial_route_request_messages = _trim_route_request_messages(
        endpoint_url,
        model,
        messages,
    )
    _initial_route_context_length = _route_context_lengths.get(
        (endpoint_url, model),
        context_length,
    )
    prep_timings["context_trim"] = time.time() - _t3

    run_security.observe_messages(_initial_route_request_messages)
    agent_prompt_tokens = estimate_tokens(_initial_route_request_messages)
    logger.info(
        "[agent-timing] prep_done model=%s prompt_tokens=%s context_length=%s prep=%s",
        model,
        agent_prompt_tokens,
        context_length,
        {k: round(v, 3) for k, v in prep_timings.items()},
    )
    yield f"data: {json.dumps({'type': 'agent_prep', 'data': {k: round(v, 3) for k, v in prep_timings.items()}})}\n\n"

    full_response = ""
    total_start = time.time()
    time_to_first_token = None
    first_token_received = False
    tool_events = []   # Persist tool executions for history reload
    round_texts = []   # Cleaned text per round for history reload
    round_models = []  # Actual model for each corresponding round
    round_endpoint_ids = []
    round_endpoint_labels = []
    # Completion-verifier state (mechanism 3a). _effectful_used flips on when
    # a tool that produces a checkable artifact runs; the verifier only fires
    # on such turns and at most _VERIFIER_MAX_ROUNDS times.
    _effectful_used = False
    _verifier_rounds = 0
    _verifier_instruction = _extract_last_user_message(messages)
    real_input_tokens = 0   # Accumulated real usage from API
    real_output_tokens = 0
    last_round_input_tokens = 0  # Last round's input tokens (for context % peak)
    has_real_usage = False
    backend_gen_tps = 0      # backend-reported true gen speed (llama.cpp timings)
    backend_prefill_tps = 0  # backend-reported prefill speed
    requested_model = model
    actual_model = model
    actual_endpoint_id = requested_endpoint_id
    actual_endpoint_label = requested_endpoint_label
    actual_endpoint_cost_tracked = requested_endpoint_cost_tracked
    usage_buckets = []
    total_tool_calls = 0  # for budget enforcement
    _ody_notes_tool_completed = False
    _pinned_fallback_candidate = None
    _pinned_fallback_route = None
    _last_route_request_messages = _initial_route_request_messages
    _last_route_context_length = _initial_route_context_length

    # Loop-breaker state. Small models (e.g. deepseek-v4-flash) can get
    # stuck firing the same tool call over and over with no text â€” burns
    # all 20 rounds, looks like the chat "died". Track recent call
    # signatures + consecutive no-text tool rounds to bail early.
    _recent_call_sigs = collections.deque(maxlen=6)
    _stuck_rounds = 0
    # Frequency of each exact call signature (tool + args), for the runaway
    # backstop. Counting identical repeats â€” not distinct same-tool calls â€”
    # lets a legit batch (e.g. 18 calendar events at once) through.
    _call_freq: collections.Counter = collections.Counter()
    _force_answer = False  # set by loop-breaker â†’ next round runs with NO tools
    # Supervisor: how many times we've nudged the model after it announced
    # an action without emitting the tool call. Capped to prevent a model
    # that *can't* call the tool from looping forever.
    _intent_nudge_count = 0
    _MAX_INTENT_NUDGES = 2

    # "I said I would, then didn't" detector. The pattern that breaks debug
    # loops on weak models (deepseek-v4-flash mid-2026): the model writes
    # "Let me tail the output to see the error" and then ends the turn with
    # no tool_calls. The intent is sincere but the function call gets dropped.
    # Match the common phrasings + an action verb that maps to an available
    # tool, so we don't nudge on harmless transitional text like "let me
    # know what you think".
    _INTENT_RE = re.compile(
        r"(?:^|\n)\s*(?:let me|i'?ll|i will|i need to|we need to|need to|"
        r"i should|we should|i must|we must|going to|let's)\s+"
        r"(?:tail|check|investigate|look at|see|tail|read|fetch|inspect|"
        r"verify|diagnose|examine|debug|capture|grab|pull|view|run|call|"
        r"trigger|launch|start|kick off|stop|kill|restart|adopt|serve|"
        r"register|adopt|list|search|find|query|hit|ping|test|use|perform|do)"
        r"\b[^.\n]{0,140}",
        re.IGNORECASE,
    )
    _awaiting_user = False  # set by ask_user â†’ end the turn and wait for a choice

    _doc_stream_create_completed = False
    _ody_doc_tool_completed = False

    # Set when a positive round cap runs out while the agent was still actively
    # using tools â€” i.e. it was cut off, not finished. Drives a "Continue" event
    # so the user can resume instead of the turn silently stalling.
    _exhausted_rounds = False
    try:
        _round_cap = int(max_rounds)
    except (TypeError, ValueError):
        _round_cap = MAX_AGENT_ROUNDS

    def _filter_route_tool_schemas(schemas):
        # Keep candidate actions visible after taint so the model can propose
        # the exact call that the server will seal for user approval.  Schema
        # visibility is not authority: both the loop and dispatcher still gate
        # execution, and only a one-use server record can cross that boundary.
        return schemas

    def _tool_schemas_for_route(route_state):
        route_mcp_schemas = route_state["mcp_schemas"]
        route_relevant_tools = route_state["relevant_tools"]
        if _force_answer:
            return []
        if route_state["is_api_model"]:
            if route_relevant_tools:
                schema_names = set(route_relevant_tools)
                if _needs_admin:
                    schema_names |= _ADMIN_TOOLS
                base_schemas = [
                    schema for schema in FUNCTION_TOOL_SCHEMAS
                    if schema.get("function", {}).get("name") in schema_names
                ]
                mcp_filtered = [
                    schema for schema in route_mcp_schemas
                    if schema.get("function", {}).get("name") in route_relevant_tools
                ]
                schemas = base_schemas + mcp_filtered
            else:
                base_schemas = FUNCTION_TOOL_SCHEMAS if _needs_admin else [
                    schema for schema in FUNCTION_TOOL_SCHEMAS
                    if schema.get("function", {}).get("name") not in _ADMIN_SCHEMA_NAMES
                ]
                schemas = base_schemas + route_mcp_schemas
            if route_state["ody_qwen_finetune_model"]:
                schemas = []
            if disabled_tools:
                schemas = [
                    schema for schema in schemas
                    if schema.get("function", {}).get("name") not in disabled_tools
                    and schema.get("name") not in disabled_tools
                ]
            return _filter_route_tool_schemas(schemas)

        wants_mcp = any(keyword in _last_user.lower() for keyword in _MCP_KEYWORDS)
        schemas = route_mcp_schemas if wants_mcp and route_mcp_schemas else []
        return _filter_route_tool_schemas(schemas)

    _approved_result_injected = False
    if exact_approval is not None:
        approved = exact_approval.pending
        approved_block = ToolBlock(approved.tool_name, approved.content)
        approved_display = approved.content.strip()
        approval_matches = exact_approval.matches(
            owner=owner,
            session_id=session_id,
            tool_name=approved.tool_name,
            content=approved.content,
            workspace=workspace,
        )
        if approval_matches:
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "tool_start",
                        "tool": approved.tool_name,
                        "command": approved_display[:240],
                        "full_command": approved_display,
                        "round": 0,
                        "approved": True,
                    }
                )
                + "\n\n"
            )
        approved_progress_q: asyncio.Queue = asyncio.Queue()

        async def _push_approved_progress(payload):
            await approved_progress_q.put(payload)

        async def _run_approved_tool():
            try:
                return await execute_tool_block(
                    approved_block,
                    session_id=session_id,
                    disabled_tools=disabled_tools,
                    tool_policy=tool_policy,
                    owner=owner,
                    progress_cb=_push_approved_progress,
                    workspace=workspace,
                    security_context=run_security,
                    exact_approval=exact_approval,
                    upload_handler=upload_handler,
                )
            finally:
                await approved_progress_q.put(None)

        approved_tool_task = asyncio.create_task(_run_approved_tool())
        try:
            while True:
                progress_event = await approved_progress_q.get()
                if progress_event is None:
                    break
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "type": "tool_progress",
                            "tool": approved.tool_name,
                            "round": 0,
                            "approved": True,
                            **progress_event,
                        }
                    )
                    + "\n\n"
                )
            desc, approved_result = await approved_tool_task
        finally:
            if not approved_tool_task.done():
                approved_tool_task.cancel()
                try:
                    await approved_tool_task
                except (asyncio.CancelledError, Exception):
                    pass
        total_tool_calls += 1

        if tool_result_is_successful(approved_result):
            for doc_event in _document_stream_events(approved_block):
                yield f"data: {json.dumps(doc_event)}\n\n"
        if approved_result.get("action") == "suggest":
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "doc_suggestions",
                        "doc_id": approved_result.get("doc_id"),
                        "suggestions": approved_result.get("suggestions", []),
                    }
                )
                + "\n\n"
            )
        elif approved_result.get("doc_id") and approved_result.get("content") is not None:
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "doc_update",
                        "doc_id": approved_result["doc_id"],
                        "title": approved_result.get("title", ""),
                        "language": approved_result.get("language", ""),
                        "content": approved_result.get("content", ""),
                        "version": approved_result.get("version", 1),
                    }
                )
                + "\n\n"
            )
        if approved_result.get("ui_event"):
            yield (
                "data: "
                + json.dumps({"type": "ui_control", "data": approved_result})
                + "\n\n"
            )

        approved_output = str(
            approved_result.get("output")
            or approved_result.get("stdout")
            or approved_result.get("response")
            or approved_result.get("results")
            or approved_result.get("content")
            or approved_result.get("error")
            or "(no output)"
        )
        approved_event = {
            "type": "tool_output",
            "tool": approved.tool_name,
            "command": approved_display[:240] if approval_matches else "",
            "output": _truncate(approved_output),
            "exit_code": approved_result.get("exit_code"),
            "approved": True,
        }
        for key in (
            "image_url",
            "image_id",
            "image_prompt",
            "image_model",
            "image_size",
            "image_quality",
            "doc_id",
            "title",
            "language",
            "content",
            "version",
            "action",
            "ui_event",
            "diff",
        ):
            if key in approved_result:
                approved_event[key] = approved_result[key]
        if approved_result.get("images"):
            approved_image = approved_result["images"][0]
            approved_event["screenshot"] = (
                f"data:{approved_image['mimeType']};base64,{approved_image['data']}"
            )
        yield "data: " + json.dumps(approved_event) + "\n\n"
        if approved_result.get("image_url"):
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "generated_image",
                        "url": approved_result["image_url"],
                        **{
                            key: approved_result[key]
                            for key in (
                                "image_url",
                                "image_id",
                                "image_prompt",
                                "image_model",
                                "image_size",
                                "image_quality",
                            )
                            if key in approved_result
                        },
                    }
                )
                + "\n\n"
            )

        approved_research_id = approved_result.get("research_session_id")
        if approved_research_id:
            approved_anchor = (
                f"\n\n[Open in Deep Research](#research-{approved_research_id})\n"
            )
            full_response += approved_anchor
            yield "data: " + json.dumps({"delta": approved_anchor}) + "\n\n"
        approved_note_id = approved_result.get("note_id")
        if approved_note_id and approved.tool_name == "manage_notes":
            approved_note_title = str(
                approved_result.get("note_title") or ""
            ).strip()
            approved_note_label = (
                f"View note: {approved_note_title}"
                if approved_note_title
                else "View note"
            )
            approved_anchor = (
                f"\n\n[{approved_note_label}](#note-{approved_note_id})\n"
            )
            full_response += approved_anchor
            yield "data: " + json.dumps({"delta": approved_anchor}) + "\n\n"

        approved_tool_event = {
            "round": 0,
            "tool": approved.tool_name,
            "desc": desc,
            "command": approved_display[:240] if approval_matches else "",
            "output": _truncate(approved_output),
            "exit_code": approved_result.get("exit_code"),
            "approved": True,
            "approval_digest": approved.digest[:16],
        }
        for key in (
            "image_url",
            "image_prompt",
            "image_model",
            "image_size",
            "image_quality",
            "diff",
        ):
            if approved_result.get(key):
                approved_tool_event[key] = approved_result[key]
        if approved_result.get("doc_id"):
            approved_tool_event["doc_id"] = approved_result["doc_id"]
            approved_tool_event["doc_title"] = approved_result.get("title", "")
        tool_events.append(approved_tool_event)
        if approved.tool_name in _VERIFIER_EFFECTFUL_TOOLS:
            _effectful_used = True
        formatted_approved_result = format_tool_result(desc, approved_result)
        _append_tool_results(
            messages,
            "",
            [],
            [formatted_approved_result],
            [formatted_approved_result],
            False,
            0,
            tool_result_records=[
                {
                    "tool_name": approved.tool_name,
                    "content": approved.content,
                    "result": approved_result,
                    "text": formatted_approved_result,
                }
            ],
        )
        _approved_result_injected = True

    for round_num in (itertools.count(1) if _round_cap <= 0 else range(1, _round_cap + 1)):
        round_response = ""
        round_reasoning = ""  # reasoning_content deltas (DeepSeek-thinking, vLLM --reasoning-parser)
        native_tool_calls = []  # populated if model uses function calling

        _active_route_state = {
            "messages": messages,
            "mcp_schemas": mcp_schemas,
            "relevant_tools": _relevant_tools,
            "is_api_model": _is_api_model,
            "is_ollama_native": _is_ollama_native,
            "ollama_openai_compat": _ollama_openai_compat,
            "ody_qwen_finetune_model": _ody_qwen_finetune_model,
            "ody_doc_finetune_mode": _ody_doc_finetune_mode,
            "ody_notes_finetune_mode": _ody_notes_finetune_mode,
            "ody_doc_stream_create_mode": _ody_doc_stream_create_mode,
            "compaction_state": (
                _route_state.get("compaction_state", {}) if round_num == 1 else {}
            ),
        }
        if round_num == 1 and not _approved_result_injected:
            _active_route_state["request_messages"] = _initial_route_request_messages
        all_tool_schemas = _tool_schemas_for_route(_active_route_state)
        agent_stream_timeout = int(get_setting("agent_stream_timeout_seconds", 300) or 300)

        _tool_names_sent = [t.get("function", {}).get("name") for t in (all_tool_schemas or []) if t.get("function")]
        logger.info(f"[agent-debug] round={round_num} model={model} _is_api_model={_is_api_model} tools_sent={len(_tool_names_sent)} tool_names={_tool_names_sent[:15]} relevant_tools={sorted(_relevant_tools)[:15] if _relevant_tools else 'ALL'}")

        # Once a fallback produces substantive output, keep that exact route
        # pinned for every later tool round instead of retrying the primary.
        if _pinned_fallback_candidate:
            _raw_candidates = [_pinned_fallback_candidate]
            _raw_route_descriptors = [_pinned_fallback_route or {}]
        else:
            _raw_candidates = [(endpoint_url, model, headers)] + list(fallbacks or [])
            _raw_route_descriptors = list(route_descriptors or [])
            if prefer_fallbacks_first and fallbacks:
                _raw_candidates = _raw_candidates[1:] + _raw_candidates[:1]
                if _raw_route_descriptors:
                    _raw_route_descriptors = _raw_route_descriptors[1:] + _raw_route_descriptors[:1]
        _candidates = dedupe_model_candidates(_raw_candidates)
        _candidate_route_descriptors = []
        for candidate in _candidates:
            source_index = next(
                (
                    index
                    for index, source in enumerate(_raw_candidates)
                    if source == candidate
                ),
                0,
            )
            _candidate_route_descriptors.append(
                _raw_route_descriptors[source_index]
                if source_index < len(_raw_route_descriptors)
                else {}
            )
        _candidate_request_states = {0: _active_route_state}

        async def _candidate_request(index, candidate_url, candidate_model, candidate_headers):
            nonlocal _last_route_request_messages, _last_route_context_length
            if index == 0 and (candidate_url, candidate_model, candidate_headers) == (endpoint_url, model, headers):
                state = _active_route_state
            else:
                candidate_source_messages = (
                    _initial_route_source_messages if round_num == 1 else messages
                )
                state = await _build_route_request_state(
                    candidate_url,
                    candidate_model,
                    candidate_headers,
                    candidate_source_messages,
                )
            request_messages = state.get("request_messages")
            if request_messages is None:
                request_messages = _trim_route_request_messages(
                    candidate_url,
                    candidate_model,
                    state["messages"],
                )
                state["request_messages"] = request_messages
            _last_route_request_messages = request_messages
            state["context_length"] = _route_context_lengths.get(
                (candidate_url, candidate_model),
                context_length,
            )
            _last_route_context_length = state["context_length"]
            run_security.observe_messages(request_messages)
            candidate_tools = _tool_schemas_for_route(state)
            state["tools"] = candidate_tools
            _candidate_request_states[index] = state
            return {
                "messages": request_messages,
                "kwargs": {
                    "tools": candidate_tools or None,
                    "tool_choice_none": state["ody_doc_finetune_mode"],
                    "temperature": (
                        _ody_qwen_temperature_cap(_requested_temperature)
                        if _is_odysseus_qwen_model(candidate_model)
                        else _requested_temperature
                    ),
                },
            }

        def _apply_candidate_compaction(index: int) -> bool:
            state = _candidate_request_states.get(index) or {}
            if history_session is not None:
                return apply_compaction_state(
                    history_session,
                    state.get("compaction_state"),
                )
            return apply_compaction_state_for_session(
                session_id,
                state.get("compaction_state"),
            )
        # stream_llm enforces a per-read INACTIVITY timeout (httpx read=timeout),
        # which kills a wedged/silent endpoint. This wall-clock deadline is the
        # complementary cap for the rare stream that trickles bytes forever and
        # so never trips the inactivity timeout. Generous â€” only catches runaway.
        _round_deadline = time.time() + max(agent_stream_timeout * 4, 1200)
        _round_start = time.time()
        _round_first_event_logged = False
        _round_first_token_logged = False
        _round_actual_model = model
        _round_actual_endpoint_id = actual_endpoint_id
        _round_actual_endpoint_label = actual_endpoint_label
        _round_real_input_tokens = 0
        _round_real_output_tokens = 0
        _round_has_real_usage = False
        _round_usage_finalized = False
        candidate_index = 0

        def _finalize_round_usage(*, include_empty: bool = True):
            nonlocal _round_usage_finalized
            if _round_usage_finalized:
                return
            _round_usage_finalized = True
            if (
                not include_empty
                and not _round_has_real_usage
                and not round_response
                and not round_reasoning
                and not native_tool_calls
            ):
                return
            if _round_has_real_usage:
                round_input_tokens = _round_real_input_tokens
                round_output_tokens = _round_real_output_tokens
                usage_source = "real"
            else:
                round_input_tokens = estimate_tokens(_last_route_request_messages)
                round_output_tokens = max(
                    len(round_response + round_reasoning) // 4,
                    0,
                )
                usage_source = "estimated"
            usage_buckets.append(_usage_bucket(
                round_num=round_num,
                model=_round_actual_model,
                endpoint_id=_round_actual_endpoint_id,
                endpoint_label=_round_actual_endpoint_label,
                endpoint_cost_tracked=actual_endpoint_cost_tracked,
                input_tokens=round_input_tokens,
                output_tokens=round_output_tokens,
                usage_source=usage_source,
            ))
        logger.info(
            "[agent-timing] round_start round=%s model=%s endpoint=%s prompt_tokens=%s tools=%s native_tools=%s timeout=%s",
            round_num,
            model,
            endpoint_url,
            estimate_tokens(messages),
            len(_tool_names_sent),
            bool(all_tool_schemas),
            agent_stream_timeout,
        )
        async for chunk in stream_llm_with_fallback(
            _candidates,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt_type=prompt_type if round_num == 1 else None,
            tools=all_tool_schemas if all_tool_schemas else None,
            tool_choice_none=_ody_doc_finetune_mode,
            timeout=agent_stream_timeout,
            session_id=session_id,
            workload=workload,
            fallback_statuses=fallback_statuses,
            fallback_on_empty=fallback_on_empty,
            candidate_request_factory=_candidate_request,
            candidate_route_descriptors=_candidate_route_descriptors,
        ):
            if not _round_first_event_logged:
                _round_first_event_logged = True
                logger.info(
                    "[agent-timing] first_event round=%s elapsed=%.3fs kind=%s",
                    round_num,
                    time.time() - _round_start,
                    "error" if chunk.startswith("event: error") else "data",
                )
            if time.time() > _round_deadline:
                logger.warning(
                    "[agent-timing] round_deadline round=%s elapsed=%.3fs deadline_s=%s",
                    round_num,
                    time.time() - _round_start,
                    max(agent_stream_timeout * 4, 1200),
                )
                break
            # Forward error events from stream_llm to the frontend
            if chunk.startswith("event: error"):
                logger.warning(
                    "[agent-timing] stream_error round=%s elapsed=%.3fs chunk=%r",
                    round_num,
                    time.time() - _round_start,
                    chunk[:500],
                )
                terminal_status = None
                try:
                    error_line = next(
                        line[6:]
                        for line in chunk.splitlines()
                        if line.startswith("data: ")
                    )
                    error_data = json.loads(error_line)
                    terminal_status = _normalize_http_status(
                        error_data.get("status")
                    )
                except Exception:
                    pass
                terminal_error = {
                    "message": (
                        f"Model request failed (HTTP {terminal_status})"
                        if terminal_status is not None
                        else "Model request failed"
                    ),
                    "status": terminal_status,
                }
                if full_response.strip() or round_reasoning.strip() or tool_events or round_texts:
                    _finalize_round_usage(include_empty=False)
                    partial_round = strip_tool_blocks(
                        round_response,
                        skip_fenced=(
                            _is_api_model
                            and not native_tool_calls
                            and not guide_only
                        ),
                    ).strip()
                    if _ody_qwen_finetune_model:
                        partial_round = _strip_doc_model_artifacts(partial_round).strip()
                    failure_note = f"[Agent stopped: {terminal_error['message']}]"
                    terminal_round = (
                        f"{partial_round}\n\n{failure_note}"
                        if partial_round
                        else failure_note
                    )
                    terminal_metadata = {
                        "failed": True,
                        "failure": terminal_error,
                        "model": actual_model,
                        "requested_model": requested_model,
                        "endpoint_id": actual_endpoint_id,
                        "endpoint_label": actual_endpoint_label,
                        "requested_endpoint_id": requested_endpoint_id,
                        "requested_endpoint_label": requested_endpoint_label,
                        "tool_events": tool_events,
                        "round_texts": [*round_texts, terminal_round],
                        "round_models": [*round_models, _round_actual_model],
                        "round_endpoint_ids": [*round_endpoint_ids, _round_actual_endpoint_id],
                        "round_endpoint_labels": [*round_endpoint_labels, _round_actual_endpoint_label],
                        **_usage_bucket_summary(usage_buckets),
                    }
                    if round_reasoning.strip():
                        terminal_metadata["thinking"] = round_reasoning.strip()
                    if isinstance(actual_endpoint_cost_tracked, bool):
                        terminal_metadata["endpoint_cost_tracked"] = (
                            actual_endpoint_cost_tracked
                        )
                    yield f'data: {json.dumps({"type": "agent_terminal", "data": terminal_metadata})}\n\n'
                yield chunk
                # A terminal provider/request failure is not a completed Agent
                # round.  Stop before empty-response synthesis, metrics,
                # teacher escalation, post-processing, or a success [DONE].
                return
            if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
                try:
                    data = json.loads(chunk[6:])
                    # IMPORTANT: check type-based events BEFORE "delta" key,
                    # because tool_call_delta also has an "arg_delta" field.
                    if data.get("type") == "tool_call_delta":
                        # Tool-call argument deltas are model proposals, not an
                        # authorization decision.  Document UI events are built
                        # from the parsed ToolBlock only after successful dispatch.
                        continue
                    elif data.get("type") == "tool_calls":
                        if _apply_candidate_compaction(candidate_index):
                            yield f'data: {json.dumps({"type": "compacted", "context_length": _last_route_context_length})}\n\n'
                        native_tool_calls = data.get("calls", [])
                        logger.info(f"Agent round {round_num}: received {len(native_tool_calls)} native tool call(s)")
                    elif data.get("type") == "usage":
                        u = data.get("data", {})
                        actual_model = u.get("model") or actual_model
                        _round_actual_model = u.get("model") or _round_actual_model
                        normalized_usage = _normalize_usage_counts(
                            u.get("input_tokens", 0),
                            u.get("output_tokens", 0),
                        )
                        if normalized_usage is None:
                            logger.warning(
                                "[agent] ignoring malformed usage event in round %s",
                                round_num,
                            )
                            continue
                        round_input = normalized_usage["input_tokens"]
                        round_output = normalized_usage["output_tokens"]
                        real_input_tokens += round_input
                        real_output_tokens += round_output
                        _round_real_input_tokens += round_input
                        _round_real_output_tokens += round_output
                        last_round_input_tokens = round_input
                        has_real_usage = True
                        _round_has_real_usage = True
                        # Backend-reported TRUE generation speed (llama.cpp
                        # timings.predicted_per_second) â€” pure decode, excludes
                        # prefill/network. Preferred over tokens/wall-clock, which
                        # reads low. Keep the last round's value (the gen phase).
                        if u.get("gen_tps"):
                            backend_gen_tps = u["gen_tps"]
                        if u.get("prefill_tps"):
                            backend_prefill_tps = u["prefill_tps"]
                    elif data.get("type") == "fallback":
                        # The selected model failed and another answered; surface
                        # the notice so a misconfigured provider isn't masked.
                        actual_model = data.get("answered_by") or actual_model
                        actual_endpoint_id = data.get("answered_by_endpoint_id")
                        actual_endpoint_label = (
                            data.get("answered_by_endpoint_label") or actual_endpoint_label
                        )
                        if isinstance(data.get("answered_by_endpoint_cost_tracked"), bool):
                            actual_endpoint_cost_tracked = data.get(
                                "answered_by_endpoint_cost_tracked"
                            )
                        candidate_index = data.get("candidate_index")
                        if (
                            _pinned_fallback_candidate is None
                            and isinstance(candidate_index, int)
                            and 0 < candidate_index < len(_candidates)
                        ):
                            _pinned_fallback_candidate = _candidates[candidate_index]
                            _pinned_fallback_route = (
                                _candidate_route_descriptors[candidate_index]
                                if candidate_index < len(_candidate_route_descriptors)
                                else {}
                            )
                            endpoint_url, model, headers = _pinned_fallback_candidate
                            answering_state = _candidate_request_states.get(candidate_index)
                            if answering_state is None:
                                answering_state = await _build_route_request_state(
                                    endpoint_url,
                                    model,
                                    headers,
                                    messages,
                                )
                                answering_state["request_messages"] = _trim_route_request_messages(
                                    endpoint_url,
                                    model,
                                    answering_state["messages"],
                                )
                                answering_state["context_length"] = _route_context_lengths.get(
                                    (endpoint_url, model),
                                    context_length,
                                )
                            messages = answering_state["messages"]
                            mcp_schemas = answering_state["mcp_schemas"]
                            _relevant_tools = answering_state["relevant_tools"]
                            _is_api_model = answering_state["is_api_model"]
                            _is_ollama_native = answering_state["is_ollama_native"]
                            _ollama_openai_compat = answering_state["ollama_openai_compat"]
                            _ody_qwen_finetune_model = answering_state["ody_qwen_finetune_model"]
                            _ody_doc_finetune_mode = answering_state["ody_doc_finetune_mode"]
                            _ody_notes_finetune_mode = answering_state["ody_notes_finetune_mode"]
                            _ody_doc_stream_create_mode = answering_state["ody_doc_stream_create_mode"]
                            if _ody_notes_finetune_mode:
                                # Mirror the primary-route clamp: the answering
                                # candidate's notes mode must re-enable the
                                # personal managers in the shared execution
                                # blocklist, or its tool calls are rejected.
                                disabled_tools.difference_update({
                                    "manage_notes", "manage_calendar", "manage_tasks",
                                })
                            data["pinned_for_run"] = True
                        if _apply_candidate_compaction(candidate_index):
                            yield f'data: {json.dumps({"type": "compacted", "context_length": _last_route_context_length})}\n\n'
                        _round_actual_model = data.get("answered_by") or model
                        _round_actual_endpoint_id = actual_endpoint_id
                        _round_actual_endpoint_label = actual_endpoint_label
                        data["round"] = round_num
                        logger.warning(f"[agent] round {round_num} fell back: "
                                       f"{data.get('selected_model')} -> {data.get('answered_by')}")
                        yield f"data: {json.dumps(data)}\n\n"
                    elif data.get("type") == "model_actual":
                        if _apply_candidate_compaction(
                            candidate_index if isinstance(candidate_index, int) else 0
                        ):
                            yield f'data: {json.dumps({"type": "compacted", "context_length": _last_route_context_length})}\n\n'
                        actual_model = data.get("model") or actual_model
                        _round_actual_model = data.get("model") or _round_actual_model
                        data["requested_model"] = requested_model
                        data["requested_endpoint_id"] = requested_endpoint_id
                        data["requested_endpoint_label"] = requested_endpoint_label
                        data["endpoint_id"] = _round_actual_endpoint_id
                        data["endpoint_label"] = _round_actual_endpoint_label
                        data["round"] = round_num
                        yield f"data: {json.dumps(data)}\n\n"
                    elif "delta" in data:
                        if _apply_candidate_compaction(
                            candidate_index if isinstance(candidate_index, int) else 0
                        ):
                            yield f'data: {json.dumps({"type": "compacted", "context_length": _last_route_context_length})}\n\n'
                        if not first_token_received:
                            time_to_first_token = time.time() - total_start
                            first_token_received = True
                        if not _round_first_token_logged:
                            _round_first_token_logged = True
                            logger.info(
                                "[agent-timing] first_visible_token round=%s elapsed=%.3fs total_elapsed=%.3fs thinking=%s",
                                round_num,
                                time.time() - _round_start,
                                time.time() - total_start,
                                bool(data.get("thinking")),
                            )
                        # Keep reasoning deltas in a separate accumulator so
                        # we can echo them back via `reasoning_content` on the
                        # next request (DeepSeek requires this; harmless for
                        # other vendors). Regular content still flows into
                        # round_response unchanged.
                        if data.get("thinking"):
                            round_reasoning += data["delta"]
                        else:
                            _delta_text = (
                                _strip_doc_model_artifacts(data["delta"])
                                if _ody_qwen_finetune_model
                                else data["delta"]
                            )
                            if _ody_qwen_finetune_model:
                                _delta_text = _normalize_ody_qwen_text_artifacts(_delta_text)
                            round_response += _delta_text
                            full_response += _delta_text
                            data["delta"] = _delta_text
                        if not _ody_qwen_finetune_model or data.get("thinking"):
                            yield f"data: {json.dumps(data)}\n\n"
                    elif data.get("error"):
                        err_msg = data.get("error", "unknown")
                        logger.error(f"Agent round {round_num}: stream error: {err_msg}")
                        yield f'data: {json.dumps({"delta": chr(10) + chr(10) + "*[Stream error: " + str(err_msg) + "]*"})}\n\n'
                except json.JSONDecodeError:
                    if round_num == 1:
                        yield chunk
            elif chunk.startswith("event: "):
                # Forward error events to frontend as visible text
                yield chunk
            # Intercept [DONE] â€” don't forward until all rounds finish

        logger.info(
            "[agent-timing] round_stream_done round=%s elapsed=%.3fs text_chars=%s tool_calls=%s first_event=%s first_token=%s",
            round_num,
            time.time() - _round_start,
            len(round_response),
            len(native_tool_calls),
            _round_first_event_logged,
            _round_first_token_logged,
        )
        _finalize_round_usage()
        _normalized_doc_round = (
            _normalize_stream_document_fences(
                round_response,
                "create_document" if _ody_doc_stream_create_mode else "update_document",
            )
            if _ody_doc_finetune_mode
            else round_response
        )
        tool_blocks, used_native, converted_calls = _resolve_tool_blocks(
            _normalized_doc_round,
            native_tool_calls,
            round_num,
            is_api_model=(_is_api_model and not guide_only),
            allow_fenced_for_api=_ody_doc_finetune_mode,
        )
        if _ody_doc_stream_create_mode and tool_blocks:
            create_idx = next(
                (idx for idx, block in enumerate(tool_blocks) if block.tool_type == "create_document"),
                None,
            )
            if create_idx is None:
                logger.info(
                    "[agent] odysseus doc stream-create discarded non-create tool call(s): %s",
                    [block.tool_type for block in tool_blocks],
                )
                tool_blocks = []
                converted_calls = []
            else:
                if len(tool_blocks) > 1 or create_idx != 0:
                    logger.info(
                        "[agent] odysseus doc stream-create keeping first create_document and dropping extras: %s",
                        [block.tool_type for block in tool_blocks],
                    )
                tool_blocks = [tool_blocks[create_idx]]
                converted_calls = (
                    [converted_calls[create_idx]]
                    if create_idx < len(converted_calls)
                    else converted_calls[:1]
                )

        if _ody_qwen_finetune_model and tool_blocks:
            _allowed_memory_write_actions = {"add", "edit", "update", "delete", "delete_all"}
            _explicit_memory_browse = bool(re.search(
                r"\b(search|list|show|open|view)\b.{0,40}\b(memories|memory|brain)\b",
                _last_user.lower(),
            ))
            _filtered_tool_blocks = []
            _filtered_converted_calls = []
            _dropped_memory_lookup = False
            for _idx, _block in enumerate(tool_blocks):
                if _block.tool_type != "manage_memory":
                    _filtered_tool_blocks.append(_block)
                    if _idx < len(converted_calls):
                        _filtered_converted_calls.append(converted_calls[_idx])
                    continue
                _action = ""
                try:
                    _args = json.loads(_block.content or "{}")
                    if isinstance(_args, dict):
                        _action = str(_args.get("action") or "").lower()
                except Exception:
                    _action = ""
                if _action in {"list", "search", "view", "get", "read"} and not _explicit_memory_browse:
                    _dropped_memory_lookup = True
                elif _action in _allowed_memory_write_actions and re.search(
                    r"\b(remember|forget|preference|prefer|save this about me|update memory|delete memory)\b",
                    _last_user.lower(),
                ):
                    _filtered_tool_blocks.append(_block)
                    if _idx < len(converted_calls):
                        _filtered_converted_calls.append(converted_calls[_idx])
                else:
                    _dropped_memory_lookup = True
            if _dropped_memory_lookup:
                logger.info(
                    "[agent-intent] odysseus qwen dropped manage_memory lookup; answering from compact memory"
                )
                tool_blocks = _filtered_tool_blocks
                converted_calls = _filtered_converted_calls
                if used_native:
                    native_tool_calls = _filtered_converted_calls
                if not tool_blocks:
                    _force_answer = True
                    messages.append({
                        "role": "system",
                        "content": (
                            "Answer the user's identity/personal-memory question from the compact "
                            "saved memory facts already provided. Do not call manage_memory or any tool."
                        ),
                    })
                    yield f'data: {json.dumps({"type": "agent_step", "round": round_num + 1})}\n\n'
                    continue

        # Force-answer round: we told the model to STOP calling tools and
        # answer. If it ignored that and emitted a (possibly DSML) tool
        # call anyway, discard it â€” don't execute, don't re-loop. Keep
        # only the prose; if there's none, emit a graceful fallback.
        if _force_answer:
            if tool_blocks:
                logger.info(f"[agent] force-answer round {round_num}: discarding {len(tool_blocks)} ignored tool call(s)")
            tool_blocks = []
            if not _strip_think_blocks(strip_tool_blocks(round_response)).strip():
                # The model burned its budget gathering data but never wrote a
                # final answer (common with weaker models on multi-source
                # briefings). Salvage it: one blunt non-streaming synthesis call
                # over the full conversation (which already holds every tool
                # result) before falling back to the canned apology.
                _synth = ""
                try:
                    from src.llm_core import llm_call_async
                    _synth_messages = list(messages) + [{
                        "role": "user",
                        "content": (
                            "Using ONLY the information already gathered above, write "
                            "the final answer for the user now. Do NOT call any tools, "
                            "do NOT explain your reasoning â€” output the finished response "
                            "directly. If some data couldn't be fetched, just work with "
                            "what you have and note what's missing in one short line."
                        ),
                    }]
                    _raw = await llm_call_async(
                        url=endpoint_url, model=model, messages=_synth_messages,
                        headers=headers, temperature=0.3, max_tokens=max_tokens, timeout=60,
                    )
                    _raw_text = _raw or ""
                    _synth = _strip_think_blocks(strip_tool_blocks(_raw_text)).strip()
                    usage_buckets.append(_usage_bucket(
                        round_num=round_num,
                        model=model,
                        endpoint_id=_round_actual_endpoint_id,
                        endpoint_label=_round_actual_endpoint_label,
                        endpoint_cost_tracked=actual_endpoint_cost_tracked,
                        input_tokens=estimate_tokens(_synth_messages),
                        output_tokens=max(len(_raw_text) // 4, 0),
                        usage_source="estimated",
                    ))
                except Exception as _e:
                    logger.warning(f"[agent] grace synthesis failed: {_e}")
                if _synth:
                    yield f'data: {json.dumps({"delta": _synth})}\n\n'
                    round_response += _synth
                    full_response += _synth
                else:
                    _fb = ("I gathered some search results but couldn't pull a clean "
                           "answer together. Want me to try a more specific question, "
                           "or summarize what I did find?")
                    yield f'data: {json.dumps({"delta": _fb})}\n\n'
                    round_response += _fb
                    full_response += _fb

        # â”€â”€ Fallback: auto-create document if model dumped large code in chat â”€â”€
        # If no create_document tool was used, check for big code blocks in text
        has_doc_tool = any(
            b.tool_type in ("create_document", "update_document")
            for b in tool_blocks
        ) or any(
            tc.get("name") in ("create_document", "update_document")
            for tc in native_tool_calls
        )
        if not has_doc_tool and session_id and "create_document" not in (disabled_tools or set()):
            _code_block_re = re.compile(r'```(\w*)\n([\s\S]*?)```')
            for m in _code_block_re.finditer(round_response):
                lang_tag = m.group(1).lower()
                code_body = m.group(2).strip()
                # Skip small blocks and known tool tags
                if code_body.count('\n') < 30:
                    continue
                if lang_tag in TOOL_TAGS:
                    continue  # already handled as a tool execution
                # Auto-create a document from this code block
                lang_map = {"py": "python", "js": "javascript", "ts": "typescript", "": "text"}
                doc_lang = lang_map.get(lang_tag, lang_tag or "text")
                doc_title = f"Code ({doc_lang})"
                tb = ToolBlock("create_document", f"{doc_title}\n{doc_lang}\n{code_body}")
                tool_blocks.append(tb)
                logger.info(f"Auto-created document from {lang_tag} code block ({code_body.count(chr(10))+1} lines)")
                break  # only auto-create one document per round

        # Save cleaned round text for history persistence
        # Keep <think> blocks so they render in the thinking section on reload
        # Mirror the same fenced-pattern gate used to resolve tool_blocks above:
        # an illustrative fence that wasn't executed (because this is a native
        # model with no real native_tool_calls) must not be stripped from the
        # persisted text either â€” otherwise it streams once and then disappears
        # on reload (#3222 follow-up).
        cleaned_round = strip_tool_blocks(round_response, skip_fenced=(_is_api_model and not used_native and not guide_only)).strip()
        round_texts.append(cleaned_round)
        round_models.append(_round_actual_model)
        round_endpoint_ids.append(_round_actual_endpoint_id)
        round_endpoint_labels.append(_round_actual_endpoint_label)
        if _ody_qwen_finetune_model and not tool_blocks and cleaned_round:
            yield f'data: {json.dumps({"delta": cleaned_round})}\n\n'

        if not tool_blocks:
            # â”€â”€ Completion verifier (mechanism 3a) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # The model is finishing. If this was an effectful agentic turn,
            # have a fresh-context verifier independently check the work
            # before we accept "done". On FAIL, surface the issues and let
            # the model fix them (capped, and it must do new effectful work
            # to re-trigger). Skipped on force-answer rounds (no tools to
            # fix with), pure Q&A, and when the toggle is off.
            _claimed_done = bool(_strip_think_blocks(cleaned_round).strip())
            if (_effectful_used and not _force_answer
                    and _claimed_done
                    and _verifier_rounds < _VERIFIER_MAX_ROUNDS
                    # Default OFF: on weak local models the verifier can't judge
                    # from the action-snapshot (no doc body), so it false-rejects
                    # ("content not shown") and forces a costly extra round every
                    # effectful turn. Opt-in via setting for strong models.
                    and get_setting("agent_verifier_subagent", False)):
                # Brief "working" indicator while the verifier runs.
                yield f'data: {json.dumps({"type": "agent_step", "round": round_num})}\n\n'
                _vfail = await _run_verifier_subagent(
                    _verifier_instruction,
                    _build_actions_snapshot(tool_events),
                    endpoint_url=endpoint_url, model=model, headers=headers,
                )
                if _vfail:
                    _verifier_rounds += 1
                    logger.info(f"[agent] verifier flagged {len(_vfail)} issue(s) on round {round_num}: {_vfail}")
                    _note = "\n\n_Double-checked the work and found something to fix._\n\n"
                    yield f'data: {json.dumps({"delta": _note})}\n\n'
                    full_response += _note
                    messages.append({
                        "role": "system",
                        "content": (
                            "An independent verifier reviewed your work against the "
                            "original request and found issues that must be fixed before "
                            "this is actually done:\n- " + "\n- ".join(_vfail) +
                            "\n\nFix these now using tools, then finish."
                        ),
                    })
                    # Require fresh effectful work before verifying again, so we
                    # never re-verify an unchanged state in a loop.
                    _effectful_used = False
                    continue
            # â”€â”€ Intent-without-action supervisor â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # Catch "Let me tail the output" / "I'll check the logs" /
            # "Let me investigate" patterns where the model announces an
            # action but emits no tool_call. The bug shows up most on
            # smaller models trained to verbalize plans before acting.
            # We inject one sharp nudge ("you said you would X â€” call the
            # actual tool now") and loop again. Capped at
            # _MAX_INTENT_NUDGES so a model that genuinely cannot use the
            # tool doesn't pin us in a forever loop.
            _intent_text = _strip_think_blocks(cleaned_round).strip()
            _intent_match = _INTENT_RE.search(_intent_text) if _intent_text else None
            # Only nudge when the round REALLY looks like an unfinished
            # promise: short response (<400 chars), no fenced code/answer,
            # and an action-intent phrase was matched. Long answers that
            # happen to contain "let me know" are not stalls.
            _looks_like_promise = (
                not guide_only
                and _intent_match is not None
                and len(_intent_text) < 400
                and "```" not in _intent_text
            )
            if _looks_like_promise and _intent_nudge_count < _MAX_INTENT_NUDGES:
                _intent_nudge_count += 1
                _matched_phrase = _intent_match.group(0).strip()
                logger.info(f"[agent] intent-without-action nudge #{_intent_nudge_count} on round {round_num}: {_matched_phrase!r}")
                _lower_phrase = _matched_phrase.lower()
                _cookbook_log_hint = ""
                if any(_word in _lower_phrase for _word in ("log", "logs", "output", "tail", "status")):
                    _cookbook_log_hint = (
                        " If this is about a Cookbook/model serve, the concrete calls are: "
                        "`list_served_models` first, then `tail_serve_output` with the "
                        "session_id from the serve/list result. Never answer with "
                        "\"check logs\" when those tools are available."
                    )
                messages.append({
                    "role": "system",
                    "content": (
                        f"You just wrote: \"{_matched_phrase}\" â€” but ended the "
                        "turn without making the actual tool call. The user can "
                        "see you announced the action but didn't run it, which "
                        "is the most frustrating thing you can do. "
                        "DO IT NOW: emit the actual function call this turn. "
                        f"{_cookbook_log_hint}"
                        "If you decided not to do it after all, say so plainly in "
                        "one sentence instead of restating the plan."
                    ),
                })
                # Visible signal in the stream so the user knows we caught it.
                yield f'data: {json.dumps({"type": "agent_step", "round": round_num + 1})}\n\n'
                continue
            if _looks_like_promise:
                _matched_phrase = _intent_match.group(0).strip()
                _guard_message = (
                    "The agent stopped because it repeatedly announced a tool "
                    "action without making the tool call."
                )
                logger.warning(
                    "[agent] intent-without-action guard exhausted on round %d after %d nudges: %r",
                    round_num,
                    _intent_nudge_count,
                    _matched_phrase,
                )
                yield (
                    "data: "
                    + json.dumps({
                        "type": "intent_nudge_exhausted",
                        "reason": "intent_without_action_nudge_cap",
                        "message": _guard_message,
                        "round": round_num,
                        "nudges": _intent_nudge_count,
                        "matched": _matched_phrase,
                    })
                    + "\n\n"
                )
                break
            break  # no tools â€” done

        # â”€â”€ Loop-breaker (Terminus-style stall detector) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Stall detector for repeated no-progress tool loops.
        # A round is "useless" ONLY when it re-issues a recent tool call AND
        # writes no answer text â€” i.e. the model is going in circles.
        # Genuine exploration (new, distinct calls) is never useless, so
        # multi-step work (file hunts, multi-host ssh, buildâ†’testâ†’fix) rides
        # all the way to a real answer. We bail only on a streak of useless
        # rounds, or a single tool fired an absurd number of times (hard
        # runaway backstop). On bail we don't give up â€” we force one
        # tool-free round so the model declares done or declares blocked,
        # mirroring Terminus's explicit-completion handshake.
        _sig = "|".join(sorted(f"{b.tool_type}:{(b.content or '').strip()[:120]}" for b in tool_blocks))
        _is_repeat = _sig in _recent_call_sigs
        _recent_call_sigs.append(_sig)
        for _b in tool_blocks:
            _call_freq[f"{_b.tool_type}:{(_b.content or '').strip()[:120]}"] += 1
        # "Real" answer text = round text minus <think> blocks. Empty-think
        # rounds (just "<think>\n\n</think>" + a tool call) must not read as
        # progress, so strip think before checking.
        _real_text = _strip_think_blocks(cleaned_round).strip()
        # Circling = repeating a recent call with nothing written. Any
        # progress (a NEW distinct call, or actual answer text) resets it.
        if _is_repeat and not _real_text:
            _stuck_rounds += 1
        else:
            _stuck_rounds = 0
        # Runaway = the SAME exact call repeated an absurd number of times.
        # Distinct calls to one tool (a real batch) are legitimate work, so we
        # count identical call signatures, not raw per-tool-type totals.
        _runaway = _detect_runaway_call(_call_freq)
        if _stuck_rounds >= 4 or _runaway:
            reason = (f"calling {_runaway} with identical arguments over and over" if _runaway
                      else "repeating the same tool calls without new progress")
            logger.warning(f"[agent] loop-breaker tripped on round {round_num} ({reason}); sig={_sig[:80]!r}")
            yield (
                "data: "
                + json.dumps({
                    "type": "loop_breaker_triggered",
                    "reason": "loop_breaker_stall",
                    "message": (
                        "The loop-breaker detected repeated tool calls without "
                        "new progress, so the agent is being forced to stop "
                        "using tools and give its best final answer."
                    ),
                    "round": round_num,
                    "detail": reason,
                })
                + "\n\n"
            )
            # The model has been executing tools, so its results are already
            # in context. Force ONE tool-free round to converge: write the
            # answer from what it has, or state plainly what's blocking it.
            # The force-answer handler above salvages (grace synthesis) or
            # apologizes honestly if it still writes nothing.
            _off = [t for t in ("web_search", "bash")
                    if disabled_tools and t in disabled_tools]
            _off_note = (f" ({', '.join(_off)} is currently disabled â€” say so if "
                         f"you needed it.)" if _off else "")
            _force_answer = True
            messages.append({
                "role": "system",
                "content": (
                    "You're repeating tool calls without converging. STOP calling "
                    "tools and end the turn one of two ways: (a) write your best "
                    "final answer NOW from the information already gathered, or "
                    "(b) if you're genuinely blocked, say plainly what's blocking "
                    "you in a sentence or two." + _off_note
                ),
            })
            full_response += "\n\n"
            yield f'data: {json.dumps({"type": "agent_step", "round": round_num + 1})}\n\n'
            continue

        # Execute each tool block
        tool_results = []
        tool_result_texts = []  # plain text for native tool role messages
        tool_result_records = []  # aligned structured provenance for next round
        budget_hit = False
        for i, block in enumerate(tool_blocks):
            # --- Tool budget check ---
            if max_tool_calls > 0 and total_tool_calls >= max_tool_calls:
                yield f'data: {json.dumps({"type": "budget_exceeded", "limit": max_tool_calls, "used": total_tool_calls})}\n\n'
                budget_hit = True
                break

            total_tool_calls += 1
            # Build a short display string for the frontend tool bubble.
            # Document tools show a brief summary instead of dumping full content.
            is_doc_tool = block.tool_type in ("create_document", "update_document", "edit_document", "suggest_document")
            full_command = block.content.strip()
            if is_doc_tool:
                cmd_display = block.content.split("\n")[0].strip()[:80]
            else:
                cmd_display = full_command

            security_decision = run_security.decision_for(
                block.tool_type,
                block.content,
            )
            _ody_clamped_tool_allowed = (
                _ody_notes_finetune_mode
                and block.tool_type in {"manage_notes", "manage_calendar", "manage_tasks"}
            )
            policy_names = email_tool_policy_names(block.tool_type)
            blocked_by_tool_policy = bool(
                tool_policy
                and any(tool_policy.blocks(name) for name in policy_names)
            )
            blocked_by_disabled_tools = bool(
                disabled_tools and not policy_names.isdisjoint(disabled_tools)
            )
            if (
                (blocked_by_tool_policy or blocked_by_disabled_tools)
                and not _ody_clamped_tool_allowed
            ):
                if blocked_by_tool_policy:
                    blocked_name = next(
                        name for name in policy_names if tool_policy.blocks(name)
                    )
                    reason = tool_policy.reason_for(blocked_name)
                else:
                    reason = (
                        f"Tool '{block.tool_type}' is disabled by the current "
                        "request policy."
                    )
                desc = f"{block.tool_type}: BLOCKED"
                result = {
                    "error": reason,
                    "exit_code": 1,
                    "blocked": True,
                    "policy": "current_tool_policy",
                }
                logger.info(
                    "Tool blocked before approval by current policy: %s",
                    block.tool_type,
                )
            elif not security_decision.allowed:
                approval_document = (
                    active_document
                    if block.tool_type
                    in {"edit_document", "suggest_document", "update_document"}
                    else None
                )
                if (
                    block.tool_type
                    in {"edit_document", "suggest_document", "update_document"}
                    and (
                        approval_document is None
                        or getattr(approval_document, "id", None) is None
                        or getattr(approval_document, "version_count", None) is None
                    )
                ):
                    # These legacy tools otherwise fall back to a process-global
                    # or most-recent document at dispatch time. That target can
                    # change while an approval card is pending, so there is no
                    # exact action to seal until the user opens a real document.
                    desc = f"{block.tool_type}: BLOCKED"
                    result = {
                        "error": (
                            "Open the exact document to edit, then request this "
                            "action again so its id and version can be sealed."
                        ),
                        "exit_code": 1,
                        "blocked": True,
                        "policy": "exact_tool_approval_target",
                    }
                else:
                    # The approval click becomes a synthetic user turn. Seal the
                    # actual server-selected candidates now so that continuation
                    # does not lose memory, skills, MCP, documents, or other
                    # ToolIndex/RAG-selected tools by classifying that synthetic text.
                    approval_selected_tools = set(_relevant_tools or ())
                    approval_selected_tools.update(
                        name for name in _tool_names_sent if name
                    )
                    approval_selected_tools.add(block.tool_type)
                    approval_selected_tools.difference_update(disabled_tools)
                    pending_approval = tool_approval_store.create(
                        owner=owner,
                        session_id=session_id,
                        origin_run_id=run_security.run_id,
                        tool_name=block.tool_type,
                        content=block.content,
                        workspace=workspace,
                        document_id=getattr(approval_document, "id", None),
                        document_version=getattr(
                            approval_document,
                            "version_count",
                            None,
                        ),
                        document_digest=(
                            document_content_digest(
                                getattr(
                                    approval_document,
                                    "current_content",
                                    "",
                                )
                            )
                            if approval_document is not None
                            else None
                        ),
                        external_untrusted_context_seen=(
                            run_security.external_untrusted_context_seen
                        ),
                        selected_tools=approval_selected_tools,
                        continuation_query=_retrieval_query or _last_user,
                        capabilities=capabilities_for_action(
                            block.tool_type,
                            block.content,
                        ),
                    )
                    desc = f"{block.tool_type}: APPROVAL REQUIRED"
                    result = {
                        "output": "Waiting for an exact user approval.",
                        "exit_code": None,
                        "approval_required": True,
                        "ask_user": pending_approval.public_payload(
                            reason=security_decision.reason,
                        ),
                    }
                    logger.info(
                        "Exact approval required before tool start: %s",
                        block.tool_type,
                    )
            else:
                yield (
                    f'data: {json.dumps({"type": "tool_start", "tool": block.tool_type, "command": cmd_display, "full_command": full_command, "round": round_num})}\n\n'
                )

                # Streaming progress for long-running tools (bash, python).
                # The bash/python branches inside _direct_fallback emit
                # periodic {elapsed_s, tail} payloads via this callback;
                # we forward each one as a `tool_progress` SSE event so
                # the UI can render live elapsed-time + tail-of-output.
                _progress_q: asyncio.Queue = asyncio.Queue()
                async def _push_progress(payload):
                    await _progress_q.put(payload)

                async def _run_tool():
                    try:
                        return await execute_tool_block(
                            block,
                            session_id=session_id,
                            disabled_tools=disabled_tools,
                            tool_policy=tool_policy,
                            owner=owner,
                            progress_cb=_push_progress,
                            workspace=workspace,
                            upload_handler=upload_handler,
                            security_context=run_security,
                        )
                    finally:
                        # Sentinel so the drainer knows to stop.
                        await _progress_q.put(None)

                _tool_task = asyncio.create_task(_run_tool())
                try:
                    # Drain progress events as they arrive â€” block until the
                    # next event OR the tool finishes (sentinel = None).
                    while True:
                        evt = await _progress_q.get()
                        if evt is None:
                            break
                        yield (
                            f'data: {json.dumps({"type": "tool_progress", "tool": block.tool_type, "round": round_num, **evt})}\n\n'
                        )
                    desc, result = await _tool_task
                finally:
                    # If the SSE client disconnects (or this generator is
                    # otherwise closed) while we're awaiting a progress event
                    # above, GeneratorExit is thrown in right here and the
                    # `await _tool_task` on the line above never runs â€” the
                    # task (and any subprocess execute_tool_block spawned for
                    # bash/python tools) would otherwise keep running
                    # orphaned with nothing left to await or cancel it.
                    if not _tool_task.done():
                        _tool_task.cancel()
                        try:
                            await _tool_task
                        except (asyncio.CancelledError, Exception):
                            pass

            run_security.observe_tool_result(block.tool_type, result, block.content)

            # A skill the model just loaded can prescribe tools that weren't
            # RAG-selected this turn (declared via requires_toolsets in its
            # frontmatter). Union them into the selection so the NEXT round's
            # schema list includes them â€” otherwise the model reads "use
            # grep" from the skill it fetched but has no grep schema to call.
            if (
                block.tool_type == "manage_skills"
                and _relevant_tools is not None
                and not result.get("error")
            ):
                _ms_args = {}
                _ms_raw = (block.content or "").strip()
                if _ms_raw.startswith("{"):
                    try:
                        _ms_args = json.loads(_ms_raw)
                    except json.JSONDecodeError:
                        _ms_args = {}
                _ms_name = str(_ms_args.get("name", "") or "").strip()
                if _ms_name and _ms_args.get("action") in ("view", "view_ref"):
                    try:
                        from services.memory.skills import SkillsManager as _SkM
                        from src.constants import DATA_DIR as _DD
                        from src.tool_policy import known_tool_names as _ktn
                        _known = _ktn()
                        for _sk in _SkM(_DD).load(owner=owner):
                            if _sk.get("name") == _ms_name:
                                _new = {
                                    t for t in (_sk.get("requires_toolsets") or [])
                                    if t in _known and t not in _relevant_tools
                                }
                                if _new:
                                    _relevant_tools.update(_new)
                                    _runtime_skill_tools.update(_new)
                                    if _base_relevant_tools is not None:
                                        _base_relevant_tools.update(_new)
                                    logger.info(
                                        "[tool-rag] skill '%s' unlocked tools for next round: %s",
                                        _ms_name, sorted(_new),
                                    )
                                break
                    except Exception as _e:
                        logger.debug(f"skill requires_toolsets unlock skipped: {_e}")

            # Extract structured web sources from web_search tool output.
            # web_search returns {"output": ..., "exit_code": 0}; check "output"
            # first so the <!-- SOURCES:â€¦--> marker is found and stripped even
            # when the result doesn't carry a "results" or "stdout" key.
            _src_text = result.get("output") or result.get("results") or result.get("stdout") or ""
            if block.tool_type == "web_search" and _src_text:
                _src_marker = "<!-- SOURCES:"
                _src_idx = _src_text.find(_src_marker)
                if _src_idx >= 0:
                    _src_end = _src_text.find(" -->", _src_idx)
                    if _src_end >= 0:
                        try:
                            _extracted_sources = json.loads(_src_text[_src_idx + len(_src_marker):_src_end])
                            yield f'data: {json.dumps({"type": "web_sources", "data": _extracted_sources})}\n\n'
                            # Strip the marker from the result so it doesn't show in chat
                            _clean = _src_text[:_src_idx].rstrip()
                            if "output" in result:
                                result["output"] = _clean
                            elif "results" in result:
                                result["results"] = _clean
                            elif "stdout" in result:
                                result["stdout"] = _clean
                        except (json.JSONDecodeError, Exception):
                            pass

            # Only a successful, authorized document execution may affect the
            # editor.  Start the authorized stream before any completed-document
            # event: handleDocUpdate finalizes that stream, while sending a
            # doc_update first can enter diff mode and make the later stream
            # discard/save the stale pre-update document.
            if tool_result_is_successful(result):
                for doc_event in _document_stream_events(block):
                    yield f'data: {json.dumps(doc_event)}\n\n'

            # Emit doc-specific event for document tools â€” the frontend
            # document panel handles this; no need to show content in chat.
            if is_doc_tool and "action" in result:
                if result["action"] == "suggest":
                    yield (
                        f'data: {json.dumps({"type": "doc_suggestions", "doc_id": result["doc_id"], "suggestions": result["suggestions"]})}\n\n'
                    )
                else:
                    yield (
                        f'data: {json.dumps({"type": "doc_update", "doc_id": result["doc_id"], "content": result["content"], "version": result["version"], "title": result.get("title", ""), "language": result.get("language")})}\n\n'
                    )

            # Emit ui_control event for frontend to apply UI changes
            if "ui_event" in result:
                yield (
                    f'data: {json.dumps({"type": "ui_control", "data": result})}\n\n'
                )

            # ask_user: remember the payload now, but emit the interactive event
            # only *after* tool_output below.  Emitting it before tool_output let
            # the subsequent tool-card rewrite/scroll push the choices out of
            # view.  The payload is also copied into the persisted tool event so
            # history reload can reconstruct an unanswered card.
            _pending_ask_user_event = None
            if "ask_user" in result:
                # The question lives in the tool args. ChatMessage.to_dict()
                # replays only role+content to the model next turn â€” tool_event
                # metadata is dropped â€” so if the question is never in the saved
                # assistant text, the model can't see it already asked and will
                # loop and re-ask after the user answers. Stream it as assistant
                # text (once) so it persists and is replayed. The card shows the
                # options only, so this is the single visible copy of the question.
                _auq = result["ask_user"]
                _auq_q = (_auq.get("question") or "").strip()
                if _auq_q and _auq_q not in full_response:
                    _auq_delta = ("\n\n" if full_response.strip() else "") + _auq_q
                    full_response += _auq_delta
                    yield 'data: ' + json.dumps({"delta": _auq_delta}) + '\n\n'
                _pending_ask_user_event = _auq
                _awaiting_user = True

            # update_plan: agent wrote back to the plan (ticked a step / revised).
            # Push it to the frontend so the stored plan + docked window update
            # live. Does NOT end the turn â€” the agent keeps working.
            if "plan_update" in result:
                yield (
                    f'data: {json.dumps({"type": "plan_update", "data": result["plan_update"]})}\n\n'
                )

            # Build output for frontend tool bubble.
            # Document tools get a short summary â€” content goes to the editor panel.
            output_text = ""
            if is_doc_tool and "action" in result:
                action = result["action"]
                title = result.get("title", "")
                ver = result.get("version", "?")
                if action == "create":
                    output_text = f'Document created: "{title}" (v{ver})'
                elif action == "edit":
                    output_text = f'Document edited: "{title}" (v{ver}, {result.get("applied", 0)} edit(s))'
                elif action == "update":
                    output_text = f'Document updated: "{title}" (v{ver})'
            elif "stdout" in result:
                # On a bash/python timeout the result carries error + (often
                # empty) stdout/stderr; fall back to the error so the "timed
                # out" reason reaches the UI instead of a blank result.
                raw = result["stdout"] or result["stderr"] or result.get("error", "")
                output_text = _truncate(raw)
            elif "output" in result:
                # bash / python canonical result: {"output": ..., "exit_code": ...}
                raw = result["output"] or ""
                output_text = _truncate(raw)
            elif "response" in result:
                # AI interaction tools (chat_with_model, send_to_session)
                label = result.get("model", result.get("session_name", "AI"))
                output_text = _truncate(f"{label}: {result['response']}")
            elif "content" in result:
                output_text = _truncate(result["content"])
            elif "results" in result:
                output_text = _truncate(result["results"])
            elif "session_id" in result and "name" in result:
                output_text = f"Session created: {result['name']} (id: {result['session_id']})"
            elif "success" in result:
                output_text = (
                    f"Written: {result.get('path', '')}"
                    if result["success"]
                    else f"Error: {result.get('error', '')}"
                )
            elif "error" in result:
                output_text = _truncate(result["error"])

            # Emit tool_output (include ui_event data if present)
            tool_output_data = {"type": "tool_output", "tool": block.tool_type, "command": cmd_display, "output": output_text, "exit_code": result.get("exit_code")}
            if is_doc_tool and "action" in result:
                tool_output_data.update({
                    "doc_id": result.get("doc_id"),
                    "document_action": result.get("action"),
                    "document_title": result.get("title", ""),
                    "document_language": result.get("language", ""),
                    "document_version": result.get("version"),
                    "document_content": result.get("content", ""),
                })
            if _pending_ask_user_event:
                # Keep enough state in the streamed tool result for alternate
                # clients to render the prompt without depending on event order.
                tool_output_data["ask_user"] = _pending_ask_user_event
            if "ui_event" in result:
                tool_output_data["ui_event"] = result["ui_event"]
                for k in (
                    "toggle_name", "state", "mode", "model", "endpoint_url",
                    "theme_name", "colors",
                    # ui_control open_email_reply payload â€” without these the
                    # frontend openReplyDraft bails on undefined uid and the
                    # reply window silently never opens.
                    "uid", "folder", "account_id",
                    # Optional pre-filled body for open_email_reply so the
                    # agent can compose-and-open in one tool call.
                    "body",
                    # ui_control open_panel payload
                    "panel",
                ):
                    if k in result:
                        tool_output_data[k] = result[k]
            # Forward generated media data from image/video/music tools.
            for k in ("media_url", "media_id", "media_type", "media_prompt", "media_model", "media_size", "media_quality"):
                if k in result:
                    tool_output_data[k] = result[k]
            # Back-compat for generated image/edit-image bubbles.
            for k in ("image_url", "image_id", "image_prompt", "image_model", "image_size", "image_quality"):
                if k in result:
                    tool_output_data[k] = result[k]
            # Forward screenshots from browser tools (base64 images)
            if result.get("images"):
                img = result["images"][0]
                tool_output_data["screenshot"] = f"data:{img['mimeType']};base64,{img['data']}"
            # Forward a file-write diff for inline before/after rendering
            if "diff" in result:
                tool_output_data["diff"] = result["diff"]
            yield f'data: {json.dumps(tool_output_data)}\n\n'
            if result.get("image_url"):
                generated_image_data = {"type": "generated_image", "url": result.get("image_url")}
                for k in ("image_url", "image_id", "image_prompt", "image_model", "image_size", "image_quality"):
                    if k in result:
                        generated_image_data[k] = result[k]
                yield f'data: {json.dumps(generated_image_data)}\n\n'

            if block.tool_type == "manage_notes":
                _notes_action = ""
                try:
                    _notes_args = json.loads(block.content or "{}")
                    if isinstance(_notes_args, dict):
                        _notes_action = str(_notes_args.get("action") or "").lower()
                except Exception:
                    _notes_action = ""
                _notes_text = ""
                if not result.get("error"):
                    if _notes_action in {"list", "search", "find", "view", "lis"}:
                        _notes_text = _note_list_summary_from_tool_output(
                            result.get("output") or result.get("results") or result.get("content") or ""
                        )
                    elif _notes_action in {"add", "update", "delete", "toggle_item"}:
                        _notes_text = str(
                            result.get("response")
                            or result.get("output")
                            or result.get("results")
                            or ""
                        ).strip()
                        if _notes_text.startswith("AI: "):
                            _notes_text = _notes_text[4:].strip()
                        if _notes_text and not re.match(r"^(done|note|item|deleted)\b", _notes_text, re.IGNORECASE):
                            _notes_text = f"Done â€” {_notes_text}"
                if _notes_text:
                    _clean_current = strip_tool_blocks(full_response).strip()
                    if _notes_text not in _clean_current:
                        _prefix = "\n\n" if _clean_current else ""
                        full_response = (_clean_current + _prefix + _notes_text).strip()
                        yield f'data: {json.dumps({"delta": _prefix + _notes_text})}\n\n'
                    _ody_notes_tool_completed = True

            if block.tool_type == "manage_tasks":
                _tasks_action = ""
                try:
                    _tasks_args = json.loads(block.content or "{}")
                    if isinstance(_tasks_args, dict):
                        _tasks_action = str(_tasks_args.get("action") or "").lower()
                except Exception:
                    _tasks_action = ""
                _tasks_text = ""
                if not result.get("error"):
                    _tasks_text = str(
                        result.get("response")
                        or result.get("output")
                        or result.get("results")
                        or ""
                    ).strip()
                    if _tasks_text.startswith("AI: "):
                        _tasks_text = _tasks_text[4:].strip()
                    if _tasks_action == "list" and _tasks_text:
                        _tasks_text = _tasks_text
                    elif _tasks_text and not re.match(r"^(done|created|updated|deleted|task)\b", _tasks_text, re.IGNORECASE):
                        _tasks_text = f"Done â€” {_tasks_text}"
                if _tasks_text:
                    _clean_current = strip_tool_blocks(full_response).strip()
                    if _tasks_text not in _clean_current:
                        _prefix = "\n\n" if _clean_current else ""
                        full_response = (_clean_current + _prefix + _tasks_text).strip()
                        yield f'data: {json.dumps({"delta": _prefix + _tasks_text})}\n\n'
                    _ody_notes_tool_completed = True

            if _ody_qwen_finetune_model and not result.get("error"):
                _terminal_summary = _ody_qwen_terminal_tool_summary({
                    "tool": block.tool_type,
                    "desc": desc,
                    "command": block.content,
                    "output": result.get("output")
                    or result.get("response")
                    or result.get("results")
                    or result.get("content")
                    or output_text
                    or "",
                })
                if _terminal_summary:
                    _terminal_summary = _normalize_ody_qwen_text_artifacts(_terminal_summary).strip()
                    _clean_current = strip_tool_blocks(full_response).strip()
                    # Replace model-written summaries for list/read tools. They
                    # are the common source of doubled text and dropped-letter
                    # artifacts; the tool output is already structured enough
                    # to render deterministically.
                    full_response = _terminal_summary
                    if _terminal_summary not in _clean_current:
                        yield f'data: {json.dumps({"delta": _terminal_summary})}\n\n'
                    _ody_notes_tool_completed = True

            # This must be the final UI event for ask_user: the frontend appends
            # the card below the now-settled tool node and cancels any between-
            # round spinner.  The turn ends after the current tool batch.
            if _pending_ask_user_event:
                yield (
                    f'data: {json.dumps({"type": "ask_user", "data": _pending_ask_user_event})}\n\n'
                )

            # Native document tools open in the editor + carry the REAL doc id.
            # Emit a doc_update so the frontend opens/activates it and sends it
            # back as active_doc_id next turn (otherwise the agent can't "see"
            # the document it just created on the follow-up message).
            if block.tool_type in ("create_document", "update_document", "edit_document") and result.get("doc_id"):
                yield (
                    'data: ' + json.dumps({
                        "type": "doc_update",
                        "doc_id": result["doc_id"],
                        "title": result.get("title", ""),
                        "language": result.get("language", ""),
                        "content": result.get("content", ""),
                        "version": result.get("version", 1),
                    }) + '\n\n'
                )

            # Inline research: emit the open-link as part of the assistant's
            # actual response text â€” a `#research-<id>` anchor that chatRenderer
            # turns into a regular clickable link. Saved with the message, so it
            # PERSISTS across refresh (unlike the old ephemeral injected chip).
            _rsid = result.get("research_session_id")
            if _rsid:
                _anchor = f"\n\n[Open in Deep Research](#research-{_rsid})\n"
                yield 'data: ' + json.dumps({"delta": _anchor}) + '\n\n'

            # Same pattern for notes: when manage_notes creates a note
            # and returns note_id, drop a `[View note](#note-<id>)` link
            # into the stream so chatRenderer's click handler routes to
            # the new openNote() in notes.js â€” opens the notes panel and
            # scrolls/flashes the matching card. Without this, the agent
            # would write "View note" as a phrase with no target.
            _nid = result.get("note_id")
            if _nid and block.tool_type == "manage_notes":
                _title = (result.get("note_title") or "").strip()
                _label = f"View note: {_title}" if _title else "View note"
                _anchor = f"\n\n[{_label}](#note-{_nid})\n"
                full_response = (full_response.rstrip() + _anchor).strip()
                yield 'data: ' + json.dumps({"delta": _anchor}) + '\n\n'

            # Save for history persistence
            tool_event = {
                "round": round_num,
                "model": _round_actual_model,
                "endpoint_id": _round_actual_endpoint_id,
                "endpoint_label": _round_actual_endpoint_label,
                "tool": _resolved_tool_event_name({
                    "tool": block.tool_type,
                    "desc": desc,
                    "command": cmd_display,
                    "output": output_text,
                }),
                "desc": desc,
                "command": cmd_display,
                "output": output_text,
                "exit_code": result.get("exit_code"),
            }
            if result.get("media_url"):
                for mk in ("media_url", "media_id", "media_type", "media_prompt", "media_model", "media_size", "media_quality"):
                    if result.get(mk):
                        tool_event[mk] = result[mk]
            if result.get("image_url"):
                for ik in ("image_url", "image_id", "image_prompt", "image_model", "image_size", "image_quality"):
                    if result.get(ik):
                        tool_event[ik] = result[ik]
            if result.get("doc_id"):
                tool_event["doc_id"] = result["doc_id"]
                tool_event["doc_title"] = result.get("title", "")
            # Persist the file-write/edit diff so it re-renders on reload â€” without
            # this the diff shows live but vanishes from saved history.
            if result.get("diff"):
                tool_event["diff"] = result["diff"]
            if _pending_ask_user_event:
                # Persist the structured question with the tool event.  On a
                # reload, chatRenderer can restore the card; a later user
                # message removes it as answered.
                tool_event["ask_user"] = _pending_ask_user_event
            tool_events.append(tool_event)
            if block.tool_type in _VERIFIER_EFFECTFUL_TOOLS:
                _effectful_used = True

            formatted = format_tool_result(desc, result)
            tool_results.append(formatted)
            tool_result_texts.append(formatted)
            tool_result_records.append(
                {
                    "tool_name": block.tool_type,
                    "content": block.content,
                    "result": result,
                    "text": formatted,
                }
            )
            if (
                _ody_doc_stream_create_mode
                and block.tool_type == "create_document"
                and result.get("action") == "create"
            ):
                _doc_stream_create_completed = True
            if (
                _ody_doc_finetune_mode
                and block.tool_type in ("create_document", "update_document", "edit_document", "suggest_document")
                and not result.get("error")
            ):
                _ody_doc_tool_completed = True
            if _pending_ask_user_event:
                # An approval card is a turn boundary.  Never execute a later
                # model-supplied call from the same batch after this request.
                break

        # If budget was hit, stop the loop
        if budget_hit:
            break

        # ask_user posed a question â€” stop here and wait for the user's choice.
        # Don't feed tool results back or advance a round; the user's selection
        # arrives as the next message and the agent resumes from there. The
        # question text is already in the streamed response, so it persists.
        if _awaiting_user:
            break

        if _doc_stream_create_completed:
            if not full_response.strip():
                full_response = "Done."
                yield 'data: ' + json.dumps({"delta": "Done."}) + '\n\n'
            logger.info("[agent] odysseus doc stream-create completed after one create_document")
            break

        if _ody_doc_tool_completed:
            if not full_response.strip() or full_response.strip().startswith("```"):
                full_response = "Done."
                yield 'data: ' + json.dumps({"delta": "Done."}) + '\n\n'
            logger.info("[agent] odysseus doc tool completed after one textual tool block")
            break

        if (_ody_notes_finetune_mode or _ody_qwen_finetune_model) and _ody_notes_tool_completed:
            logger.info("[agent] odysseus completed from deterministic tool output")
            break

        # Feed results back to LLM for next round
        # Pass the CONVERTED calls (aligned 1:1 with tool_result_texts), not the
        # raw native_tool_calls: a call that failed to convert is dropped from
        # tool_blocks but stayed in native_tool_calls, so indexing results by
        # native position mis-attached each result to the wrong tool_call_id
        # (and left the real call answered empty).
        _append_tool_results(messages, round_response, converted_calls,
                             tool_results, tool_result_texts, used_native, round_num,
                             round_reasoning=round_reasoning,
                             tool_result_records=tool_result_records)

        # Emit agent_step event
        yield (
            f'data: {json.dumps({"type": "agent_step", "round": round_num + 1})}\n\n'
        )

        # Separator in accumulated response
        full_response += "\n\n"
    else:
        # The finite for-loop completed every allowed round WITHOUT an early `break`
        # (a `break` fires on "done", budget, or error). Reaching this `else`
        # means the agent kept working until it ran out of rounds â€” so offer
        # Continue instead of stopping silently. This catches ALL exhaustion
        # paths, including a verifier `continue` on the final round (the old
        # bottom-of-loop flag missed those). Unlimited mode uses itertools.count,
        # so this `else` is unreachable there.
        if _round_cap > 0:
            _exhausted_rounds = True

    # If the loop hit the round cap while still working, tell the client so it
    # can show a "Continue" affordance instead of the turn just stopping.
    if _exhausted_rounds:
        logger.info("[agent] round cap (%d) reached mid-task â€” emitting rounds_exhausted", _round_cap)
        yield f'data: {json.dumps({"type": "rounds_exhausted", "rounds": _round_cap})}\n\n'

    # If the response is completely empty and no tools were executed,
    # yield a fallback message so the user is not left hanging.
    full_response, _fallback_chunk = _empty_response_fallback(
        full_response, round_reasoning, tool_events
    )
    if _fallback_chunk:
        yield _fallback_chunk

    # Do not persist raw textual tool-call JSON / role markers as assistant
    # prose. Local finetunes may emit those before the parser catches and
    # executes them; saved history should contain only the user-facing answer.
    full_response = strip_tool_blocks(full_response).strip()
    if _ody_qwen_finetune_model:
        full_response = _normalize_ody_qwen_text_artifacts(full_response)
        if (
            not tool_events
            and _looks_like_destructive_request(_last_user)
            and _looks_like_success_claim(full_response)
        ):
            full_response = "I couldn't make that change because no matching tool action completed."
    _response_before_tool_summary = full_response
    if tool_events:
        for _ev in reversed(tool_events):
            _tool_name = _resolved_tool_event_name(_ev)
            _tool_action = ""
            try:
                _cmd_args = json.loads(_ev.get("command") or "{}")
                if isinstance(_cmd_args, dict):
                    _tool_action = str(_cmd_args.get("action") or "").lower()
            except Exception:
                _tool_action = ""
            if _tool_name == "manage_notes" and _tool_action in {"list", "search", "find", "view", "lis"}:
                _notes_summary = _note_list_summary_from_tool_output(_ev.get("output") or "")
                if _notes_summary:
                    full_response = _notes_summary
                break
            if _tool_name == "manage_calendar" and _tool_action in {"list", "list_events"}:
                _calendar_summary = _calendar_list_summary_from_tool_output(_ev.get("output") or "")
                if _calendar_summary:
                    full_response = _calendar_summary
                break
            if _tool_name == "manage_tasks" and _tool_action == "list":
                _tasks_summary = str(_ev.get("output") or "").strip()
                if _tasks_summary.startswith("AI: "):
                    _tasks_summary = _tasks_summary[4:].strip()
                if _tasks_summary:
                    full_response = _tasks_summary
                break
            if _tool_name in {"list_emails", "mcp__email__list_emails"}:
                _email_summary = _email_list_summary_from_tool_output(_ev.get("output") or "")
                if _email_summary:
                    full_response = _email_summary
                break
            if _tool_name in {"read_email", "mcp__email__read_email"}:
                _email_summary = _email_read_summary_from_tool_output(_ev.get("output") or "")
                if _email_summary:
                    full_response = _email_summary
                break

    if (
        tool_events
        and full_response.strip()
        and full_response.strip() != (_response_before_tool_summary or "").strip()
        and full_response.strip() not in (_response_before_tool_summary or "")
    ):
        _final_delta = full_response.strip()
        yield f"data: {json.dumps({'delta': _final_delta})}\n\n"

    # --- Final metrics ---
    total_duration = time.time() - total_start
    final_context_tokens = estimate_tokens(messages)
    metrics = _compute_final_metrics(
        _last_route_request_messages, full_response, total_duration, time_to_first_token,
        _last_route_context_length, real_input_tokens, real_output_tokens,
        has_real_usage, tool_events, round_texts, model=actual_model,
        round_models=round_models,
        round_endpoint_ids=round_endpoint_ids,
        round_endpoint_labels=round_endpoint_labels,
        last_round_input_tokens=last_round_input_tokens,
        request_context_tokens=final_context_tokens,
        prep_timings=prep_timings,
        backend_gen_tps=backend_gen_tps,
        backend_prefill_tps=backend_prefill_tps,
    )
    metrics["requested_model"] = requested_model
    metrics["endpoint_id"] = actual_endpoint_id
    metrics["endpoint_label"] = actual_endpoint_label
    if isinstance(actual_endpoint_cost_tracked, bool):
        metrics["endpoint_cost_tracked"] = actual_endpoint_cost_tracked
    usage_summary = _usage_bucket_summary(usage_buckets)
    if usage_summary:
        metrics.update(usage_summary)
        if not backend_gen_tps and total_duration > 0:
            metrics["tokens_per_second"] = round(
                usage_summary["output_tokens"] / total_duration,
                2,
            )
        if _last_route_context_length:
            metrics["context_percent"] = min(
                round(
                    (usage_buckets[-1]["input_tokens"] / _last_route_context_length) * 100,
                    1,
                ),
                100.0,
            )
    metrics["requested_endpoint_id"] = requested_endpoint_id
    metrics["requested_endpoint_label"] = requested_endpoint_label
    yield f"data: {json.dumps({'type': 'metrics', 'data': metrics})}\n\n"

    # Teacher-escalation: inline takeover visible in the chat stream.
    # The student just finished; if Tier 1 flags failure, the teacher
    # gets a turn (with its own tool calls forwarded to the user) and
    # a skill is saved ONLY if the teacher actually succeeds. Skipped
    # when we ARE the teacher to avoid recursion.
    if not _is_teacher_run and not guide_only and not _awaiting_user:
        try:
            from src.teacher_escalation import run_teacher_inline
            async for evt in run_teacher_inline(
                student_endpoint_url=endpoint_url,
                student_messages=messages,
                student_tool_events=tool_events,
                student_reply=full_response,
                owner=owner,
                session_id=session_id,
                workspace=workspace,
                disabled_tools=disabled_tools,
                tool_policy=tool_policy,
                active_document=active_document,
                active_email=active_email,
                external_untrusted_context_seen=(
                    run_security.external_untrusted_context_seen
                ),
                delegated_credential=delegated_credential,
            ):
                yield evt
        except Exception as _esc_err:
            logger.warning(f"teacher escalation hook failed: {_esc_err}", exc_info=True)

    yield "data: [DONE]\n\n"


def _usage_bucket(
    *,
    round_num: int,
    model: str,
    endpoint_id,
    endpoint_label,
    endpoint_cost_tracked,
    input_tokens: int,
    output_tokens: int,
    usage_source: str,
) -> dict:
    """Build non-secret usage attribution for one concrete Agent round."""

    bucket = {
        "round": round_num,
        "model": model,
        "endpoint_id": endpoint_id,
        "endpoint_label": endpoint_label,
        "input_tokens": max(int(input_tokens or 0), 0),
        "output_tokens": max(int(output_tokens or 0), 0),
        "usage_source": "real" if usage_source == "real" else "estimated",
    }
    # Persist the owner-resolved route classification so saved usage remains
    # stable even if the session later selects a different endpoint.
    if isinstance(endpoint_cost_tracked, bool):
        bucket["endpoint_cost_tracked"] = endpoint_cost_tracked
    return bucket


def _usage_bucket_summary(usage_buckets: list) -> dict:
    """Return aggregate token fields without losing per-route attribution."""

    if not usage_buckets:
        return {}
    input_tokens = sum(bucket.get("input_tokens", 0) or 0 for bucket in usage_buckets)
    output_tokens = sum(bucket.get("output_tokens", 0) or 0 for bucket in usage_buckets)
    sources = {bucket.get("usage_source") for bucket in usage_buckets}
    usage_source = next(iter(sources)) if len(sources) == 1 else "mixed"
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "usage_source": usage_source,
        "usage_buckets": [dict(bucket) for bucket in usage_buckets],
    }
