
import json
# _urgency.py -- chunked from builtin_actions.py
import logging
from typing import Tuple

from src.builtin_actions._email import _email_task_account_id
from src.builtin_actions._shared import TaskNoop
from src.constants import DATA_DIR, EMAIL_URGENCY_CACHE_DIR
from src.interactive_gate import wait_for_interactive_quiet

logger = logging.getLogger(__name__)

async def action_check_email_urgency(owner: str, **kwargs) -> Tuple[str, bool]:
    """Scan unread emails across all accounts, LLM-triage new ones, cache
    per-UID verdicts, tag the inbox, and fire a reminder when a previously
    unseen UID scores reply-soon/urgent (>=2). State persists under
    data/email_urgency_state_* so the UI can color the unread dot by tier.

    Design notes:
    - Only classifies emails newer than 7 days (first-run scale guard).
    - Cache key = `<account_id>:<uid>` so the same UID across accounts doesn't collide.
    - Re-notify gate: only when at least one UID NEW to `notified_uids` scores ≥2.
      Repeat scans where the set is unchanged stay silent.
    """
    from src.settings import load_settings

    try:
        settings = load_settings()
        import json as _json
        import email as _email_mod
        import asyncio as _aio
        import os as _os
        import re as _re
        import time as _time
        import httpx
        from datetime import datetime as _dt, timedelta as _td
        from pathlib import Path as _P
        from core.database import SessionLocal as _SL, EmailAccount as _EA
        from routes.email_helpers import _imap_connect, _decode_header
        from src.llm_core import llm_call_async_with_fallback

        # Per-owner state file so multi-user runs don't clobber each other's
        # notified_uids / urgency counts. Empty owner falls back to a generic
        # filename for single-user installs (matches prior behaviour).
        _owner_slug = "".join(c if (c.isalnum() or c in "-_.@") else "_" for c in (owner or "default"))
        STATE_PATH = _P(DATA_DIR) / f"email_urgency_state_{_owner_slug}.json"
        STATE_LOCK_DB = STATE_PATH.with_suffix(".lock.sqlite3")
        CACHE_DIR = _P(EMAIL_URGENCY_CACHE_DIR)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        AGE_CUTOFF = _dt.utcnow() - _td(days=7)
        TRIAGE_VERSION = 10
        CATEGORY_TAGS = {
            "bills", "receipt", "travel", "calendar", "action-needed",
        }
        VISIBLE_EMAIL_TAGS = CATEGORY_TAGS | {"urgent", "reply-soon"}
        MANAGED_TAGS = VISIBLE_EMAIL_TAGS | {
            "newsletter", "marketing", "notification", "finance", "security",
            "shopping", "social", "work", "personal", "legal", "support", "promo",
        }

        # Resolve with the task owner as before, but defer the availability
        # gate until after authoritative account cleanup. State retirement must
        # still run when no model is configured.
        from src.task_endpoint import resolve_task_candidates
        candidates = resolve_task_candidates(owner=owner)
        target_account_id = _email_task_account_id(kwargs)

        # ── 1. Enumerate enabled accounts. Match this task's owner AND fall
        # back to the legacy "unowned account whose imap_user / from_address
        # == this owner" pattern — same rule `_get_email_config` uses, so a
        # pre-multi-user account row still gets picked up for the seeded task.
        def _enumerate_enabled_accounts():
            db = _SL()
            try:
                from sqlalchemy import and_ as _and, or_ as _or
                q = db.query(_EA).filter(_EA.enabled == True)  # noqa: E712
                if owner:
                    unowned = _or(_EA.owner == None, _EA.owner == "")  # noqa: E711
                    same_mailbox = _or(
                        _EA.imap_user == owner,
                        _EA.from_address == owner,
                    )
                    q = q.filter(
                        _or(_EA.owner == owner, _and(unowned, same_mailbox))
                    )
                if target_account_id:
                    q = q.filter(_EA.id == target_account_id)
                return q.all()
            finally:
                db.close()

        initial_accounts = _enumerate_enabled_accounts()
        initial_account_ids = {
            str(account.id) for account in initial_accounts
        }

        # Register every account before IMAP work, including its first-ever
        # scan. A concurrent zero-account cleanup can then advance this marker
        # and fence delivery even before the scan has produced payload.
        registered_state = None
        if initial_account_ids:
            async def _register_accounts(prior):
                next_state = _merge_email_urgency_state(
                    prior,
                    owner=owner,
                    per_uid_scores={},
                    notified_uids=prior.get("notified_uids", []),
                    all_unread_keys=set(),
                    fully_scanned_account_ids=set(),
                    base_account_generations=(
                        _email_urgency_account_generations(prior)
                    ),
                    timestamp=_time.time(),
                    known_account_ids=initial_account_ids,
                )
                # Return the exact state committed by registration. This is
                # the scan's generation token: adopting a later checkpoint
                # after account cleanup would let the stale scan appear fresh.
                return next_state, next_state

            registered_state = await _run_email_urgency_state_transaction(
                STATE_PATH,
                STATE_LOCK_DB,
                _register_accounts,
            )

        # Revalidate after registration. If deletion/disable and its cleanup
        # completed before the marker was published, this second enumeration
        # observes the absence and this action retires its own marker instead
        # of starting IMAP. Accounts newly appearing between the two reads are
        # left for the next pass rather than scanned without prior registration.
        verified_accounts = _enumerate_enabled_accounts()
        enabled_account_ids = {
            str(account.id) for account in verified_accounts
        }
        accounts = [
            account
            for account in verified_accounts
            if str(account.id) in initial_account_ids
        ]

        # Capture the checkpoint basis before cleanup or IMAP. A full
        # owner-wide enumeration authoritatively retires all known state IDs
        # absent from the current enabled/visible set. A scoped task may retire
        # only its selected missing/disabled account. Existing accounts remain
        # present even if their later network scan fails, so transient IMAP
        # failure never erases their last known state.
        base_state = (
            registered_state
            if registered_state is not None
            else _read_email_urgency_state(STATE_PATH)
        )
        base_account_generations = _email_urgency_account_generations(
            base_state
        )
        base_payload_account_ids = _email_urgency_payload_account_ids(base_state)
        known_state_account_ids = _email_urgency_known_account_ids(base_state)
        if target_account_id:
            retired_account_ids = (
                {str(target_account_id)}
                if str(target_account_id) not in enabled_account_ids
                else set()
            )
        else:
            retired_account_ids = (
                known_state_account_ids - enabled_account_ids
            )

        if retired_account_ids:
            async def _retire_accounts(prior):
                next_state = _merge_email_urgency_state(
                    prior,
                    owner=owner,
                    per_uid_scores={},
                    notified_uids=prior.get("notified_uids", []),
                    all_unread_keys=set(),
                    fully_scanned_account_ids=set(),
                    base_account_generations=base_account_generations,
                    timestamp=_time.time(),
                    retired_account_ids=retired_account_ids,
                    base_payload_account_ids=base_payload_account_ids,
                )
                return None, next_state

            await _run_email_urgency_state_transaction(
                STATE_PATH,
                STATE_LOCK_DB,
                _retire_accounts,
            )
        if not accounts:
            raise TaskNoop("no email accounts configured")

        # ── 2. Account retirement above is state maintenance and does not
        # depend on model availability. Scanning still requires the utility
        # primary/fallback candidates resolved for this task owner.
        if not candidates:
            return "No LLM endpoint available", False

        urgency_prompt = settings.get("urgent_email_prompt", "")
        per_uid_scores = {}   # key = "<acc_id>:<uid>" → {"score": 0-3, "reason": "..."}
        all_unread_keys = set()
        llm_attempts = 0
        saved_classifications = 0
        failed_classifications = []
        tag_write_details = []
        scanned = 0
        fully_scanned_account_ids = set()

        def _heuristic_email_verdict(item: dict) -> dict:
            blob = (
                f"{item.get('headers','')}\n{item.get('from','')}\n"
                f"{item.get('subject','')}\n{item.get('body','')}"
            ).lower()
            response_tags = []
            type_candidates = []

            def add_response(tag: str):
                if tag in CATEGORY_TAGS and tag not in response_tags:
                    response_tags.append(tag)

            def add_type(tag: str):
                if tag in CATEGORY_TAGS and tag not in type_candidates:
                    type_candidates.append(tag)

            bulkish = bool(_re.search(
                r"\b(list-unsubscribe|list-id|mailchimp|mailchimpapp|view this email in your browser|unsubscribe|newsletter|digest|precedence:\s*bulk)\b",
                blob,
            ))
            marketingish = bool(_re.search(
                r"\b(advertisement|sponsored|promo|promotion|sale|discount|offer|limited time|deal|coupon|shop now|buy now|membership|rewards?)\b",
                blob,
            ))
            if bulkish or marketingish:
                add_type("newsletter")
            if _re.search(r"\b(receipt|order|注文|payment confirmation|delivery|shipment|tracking|お届け|購入)\b", blob):
                add_type("receipt")
            if _re.search(r"\b(bill|billing|amount due|overdue|pay by|payment due|subscription could not be renewed)\b", blob):
                add_type("bills")
            if _re.search(r"\b(court|charge|legal|lawyer|solicitor|claim|judgment|registration fee|debt)\b", blob):
                add_type("legal")
            if _re.search(r"\b(flight|hotel|booking|reservation|itinerary|train|ticket|trip|旅|予約)\b", blob):
                add_type("travel")
            if _re.search(r"\b(ticket|case|support|helpdesk|request)\b", blob):
                add_type("support")
            if _re.search(r"\b(meeting|appointment|calendar|invite|event|schedule|予定|保育園|連絡帳)\b", blob):
                add_response("calendar")
            if _re.search(
                r"\b(action required|required action|please reply|please respond|deadline|by \d{1,2} |pay within|submit|sign|confirm|approval|waiting outside|locked out|can't get in|cannot get in|invoice|bill|billing|payment|balance|debt|subscription|renewal|overdue|amount due|court|charge|legal|lawyer|solicitor|claim|judgment)\b",
                blob,
            ):
                add_response("action-needed")

            type_priority = ("bills", "receipt", "travel")
            tags = [*response_tags]
            for type_tag in type_priority:
                if type_tag in type_candidates and type_tag not in tags:
                    tags.append(type_tag)
                if len(tags) >= len(response_tags) + 2:
                    break

            score = 0
            reason = "categorized by email metadata"
            if "action-needed" in response_tags:
                score = 2
                reason = "action likely needed"
            if _re.search(r"\b(urgent|immediately|final notice|locked out|waiting outside|can't get in|cannot get in)\b", blob):
                score = 3
                reason = "urgent wording"
            if (bulkish or marketingish) and score < 2:
                score = 0
                reason = "bulk marketing/newsletter"

            _from_raw = item.get("from", "") or ""
            if "<" in _from_raw:
                _from_short = _from_raw.split("<", 1)[0].strip().strip('"') or _from_raw
            else:
                _from_short = _from_raw
            return {
                "score": max(0, min(3, score)),
                "tags": tags[:4],
                "spam": False,
                "reason": reason,
                "subject": (item.get("subject") or "")[:200],
                "from": _from_short[:120],
                "triage_version": TRIAGE_VERSION,
                "message_id": (item.get("message_id") or "").strip(),
                "unread": bool(item.get("unread")),
                "ts": _time.time(),
            }

        # ── 3. Per-account scan: pull headers + lightweight body for new UIDs
        # since 7 days ago, score via LLM, cache the verdict.
        for acc in accounts:
            cache_file = CACHE_DIR / f"{acc.id}.json"
            try:
                cache = _json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {"uids": {}}
            except Exception:
                cache = {"uids": {}}

            def _scan_one(account=acc, cache_uids=cache.get("uids", {})):
                """Sync IMAP work runs in a thread."""
                results = []
                scan_complete = True
                conn = _imap_connect(account.id)
                try:
                    select_status, _select_data = conn.select("INBOX", readonly=True)
                    if select_status != "OK":
                        return results, False
                    # Tag recent inbox mail, not only unread mail. Urgency
                    # reminders below still only notify for unread messages.
                    since_str = AGE_CUTOFF.strftime("%d-%b-%Y")
                    status, data = conn.uid("SEARCH", None, f'(SINCE {since_str})')
                    if status != "OK":
                        return results, False
                    if not data or not data[0]:
                        return results, True
                    matching_uids = data[0].split()
                    if len(matching_uids) > 30:
                        # The scale guard deliberately processes only the most
                        # recent 30. That is a partial account snapshot, so it
                        # cannot justify pruning older checkpoint facts.
                        scan_complete = False
                    uids = matching_uids[-30:]
                    for uid_b in uids:
                        uid = uid_b.decode() if isinstance(uid_b, bytes) else str(uid_b)
                        key = f"{account.id}:{uid}"
                        cached = cache_uids.get(uid)
                        cached_ok = isinstance(cached, dict) and cached.get("triage_version") == TRIAGE_VERSION
                        results.append({"key": key, "uid": uid, "cached": cached if cached_ok else None})
                        if cached_ok:
                            # Cached verdicts still need a lightweight FLAGS
                            # refresh. Without it a cached unread message looks
                            # read and its successful notification checkpoint
                            # is pruned on the next pass.
                            try:
                                st, flag_data = conn.uid("FETCH", uid_b, "(UID FLAGS)")
                                if st != "OK" or not flag_data:
                                    scan_complete = False
                                    results.pop()
                                    continue
                                flag_parts = []
                                for part in flag_data:
                                    if isinstance(part, (bytes, bytearray)):
                                        flag_parts.append(bytes(part))
                                    elif (
                                        isinstance(part, tuple)
                                        and part
                                        and isinstance(part[0], (bytes, bytearray))
                                    ):
                                        flag_parts.append(bytes(part[0]))
                                flags_blob = b" ".join(flag_parts)
                                results[-1]["unread"] = b"\\Seen" not in flags_blob
                            except Exception as _fe:
                                scan_complete = False
                                results.pop()
                                logger.debug(
                                    f"urgency: flag fetch for uid {uid} failed: {_fe}"
                                )
                            continue
                        # Pull headers + first ~800 chars of plaintext body.
                        try:
                            st, msg_data = conn.uid("FETCH", uid_b, "(UID FLAGS RFC822.HEADER BODY.PEEK[TEXT]<0.800>)")
                            if st != "OK" or not msg_data:
                                scan_complete = False
                                results.pop()
                                continue
                            flags_blob = b" ".join(
                                part[0] for part in msg_data
                                if isinstance(part, tuple) and part and isinstance(part[0], (bytes, bytearray))
                            )
                            is_unread = b"\\Seen" not in flags_blob
                            # Headers + body land in different tuples in the
                            # response — concatenate the bytes for parsing.
                            raw = b""
                            for part in msg_data:
                                if isinstance(part, tuple) and part[1]:
                                    raw += part[1] + b"\n\n"
                            if not raw:
                                scan_complete = False
                                results.pop()
                                continue
                            msg = _email_mod.message_from_bytes(raw)
                            # Skip Odysseus-generated reminders so the scanner
                            # doesn't classify its own emails as urgent and
                            # trigger a feedback loop. Match on either the
                            # stamped headers OR the subject prefix.
                            _ody_origin = (msg.get("X-Odysseus-Origin") or "").strip().lower()
                            _ody_kind = (msg.get("X-Odysseus-Kind") or "").strip().lower()
                            _raw_subj = (msg.get("Subject") or "").lower()
                            # MCP path drops custom headers (email_server's
                            # schema doesn't accept them), so we ALSO match the
                            # `[Task]` subject prefix that `_deliver_via_mcp`
                            # always stamps. Anything that looks self-generated
                            # is dropped before classification to prevent the
                            # scanner from labelling its own emails "urgent".
                            if (_ody_origin == "odysseus-ui" or _ody_kind == "reminder"
                                    or _raw_subj.startswith("reminder (odysseus):")
                                    or _raw_subj.startswith("reminder:")
                                    or _raw_subj.startswith("[task]")):
                                # Drop this candidate entirely — don't list it
                                # in results so its UID never enters the cache
                                # nor counts toward `scanned`.
                                results.pop()
                                continue
                            subject = _decode_header(msg.get("Subject") or "")
                            from_raw = _decode_header(msg.get("From") or "")
                            header_blob = "\n".join(
                                f"{name}: {msg.get(name, '')}"
                                for name in (
                                    "From", "Subject", "List-Unsubscribe", "List-ID",
                                    "Precedence", "X-Mailchimp-Campaign-Id",
                                    "X-Campaign", "X-MC-User",
                                )
                                if msg.get(name)
                            )
                            body_snippet = ""
                            try:
                                if msg.is_multipart():
                                    for part in msg.walk():
                                        if part.get_content_type() == "text/plain":
                                            body_snippet = part.get_payload(decode=True).decode("utf-8", errors="ignore")[:1600]
                                            break
                                else:
                                    body_snippet = (msg.get_payload(decode=True) or b"").decode("utf-8", errors="ignore")[:1600]
                            except Exception:
                                body_snippet = ""
                            results[-1].update({
                                "subject": subject,
                                "from": from_raw,
                                "headers": header_blob,
                                "body": body_snippet.strip(),
                                "message_id": (msg.get("Message-ID") or "").strip(),
                                "unread": is_unread,
                            })
                        except Exception as _fe:
                            scan_complete = False
                            results.pop()
                            logger.debug(f"urgency: header fetch for uid {uid} failed: {_fe}")
                finally:
                    try: conn.logout()
                    except Exception: pass
                return results, scan_complete

            try:
                items, scan_complete = await _aio.to_thread(_scan_one)
            except Exception as e:
                logger.warning(f"urgency: IMAP scan failed for account {acc.id}: {e}")
                continue
            if scan_complete:
                fully_scanned_account_ids.add(str(acc.id))

            for item in items:
                scanned += 1
                key = item["key"]
                if item.get("unread"):
                    all_unread_keys.add(key)
                if item.get("cached"):
                    cached_v = dict(item["cached"])
                    cached_v["unread"] = bool(item.get("unread"))
                    per_uid_scores[key] = cached_v
                    continue
                # Skip uids we couldn't fetch (no subject/from/body).
                if not item.get("subject") and not item.get("from"):
                    continue
                verdict = _heuristic_email_verdict(item)
                cache.setdefault("uids", {})[item["uid"]] = verdict
                per_uid_scores[key] = verdict
                saved_classifications += 1
                continue
                # ── LLM-classify. JSON-only response; bullet-proof parse.
                llm_attempts += 1
                prompt = (
                    "You are triaging ONE email. Return ONLY JSON: "
                    "{\"score\":0|1|2|3,\"tags\":[\"...\"],\"spam\":false,"
                    "\"reason\":\"one short phrase\"}.\n"
                    "0 = trivial / promotional · 1 = informational, no reply needed · "
                    "2 = should reply within a day · 3 = urgent, reply now (deadline, blocker).\n\n"
                    "Allowed visible tags: urgent, reply-soon, action-needed, calendar, bills, receipt, travel.\n"
                    "Use action-needed when the user likely needs to reply, pay, sign, book, or decide. "
                    "Use bills for bills or debts, receipt for purchases/deliveries, travel for reservations/trips, "
                    "and calendar only when a calendar event/reminder is involved. spam=true for scams, phishing, "
                    "junk, cold sales, generic ads, or no-personal-action bulk mail.\n"
                    "Important: 'I'm outside', 'I am outside', 'waiting outside', 'at the door', "
                    "'locked out', or 'can't get in' means score 3 unless clearly historical.\n\n"
                    f"User's rules:\n{urgency_prompt}\n\n"
                    f"Email:\nFrom: {item.get('from','')}\nSubject: {item.get('subject','')}\n"
                    f"Snippet:\n{item.get('body','')}\n"
                )
                try:
                    await wait_for_interactive_quiet("email urgency action")
                    raw = await llm_call_async_with_fallback(
                        candidates,
                        [{"role": "user", "content": prompt}],
                        temperature=0.1, max_tokens=220, timeout=30,
                    )
                    # Tolerant JSON-parse: strip code fences if present.
                    txt = (raw or "").strip()
                    if txt.startswith("```"):
                        txt = txt.strip("`")
                        # Drop a leading "json\n" or any tag.
                        nl = txt.find("\n")
                        if nl >= 0:
                            txt = txt[nl + 1:]
                    # Find first { ... } in the response.
                    s = txt.find("{")
                    e = txt.rfind("}")
                    if s < 0 or e <= s:
                        failed_classifications.append({
                            "subject": item.get("subject") or "(no subject)",
                            "from": item.get("from") or "",
                            "reason": "model returned no JSON",
                        })
                        continue
                    obj = _json.loads(txt[s:e + 1])
                    score = int(obj.get("score", 0))
                    reason = str(obj.get("reason", ""))[:200]
                    raw_tags = obj.get("tags") or []
                    if isinstance(raw_tags, str):
                        raw_tags = [raw_tags]
                    tags = []
                    for t in raw_tags:
                        if not isinstance(t, str):
                            continue
                        tag = t.strip().lower().replace("_", "-")
                        if tag == "promo":
                            tag = "marketing"
                        if tag in CATEGORY_TAGS and tag not in tags:
                            tags.append(tag)
                    _spam_raw = obj.get("spam")
                    if isinstance(_spam_raw, bool):
                        spam = _spam_raw
                    elif isinstance(_spam_raw, (int, float)):
                        spam = bool(_spam_raw)
                    else:
                        spam = str(_spam_raw or "").strip().lower() in {"1", "true", "yes", "y"}
                    _blob = f"{item.get('headers','')}\n{item.get('subject','')}\n{item.get('body','')}".lower()
                    if _re.search(r"\b(i'?m|i am|im|we'?re|we are)\s+outside\b", _blob) or _re.search(
                        r"\b(waiting outside|at the door|locked out|can'?t get in|cannot get in)\b", _blob
                    ):
                        if score < 3:
                            reason = "person is waiting outside"
                        score = max(score, 3)
                    bulkish = bool(_re.search(
                        r"\b(list-unsubscribe|list-id|mailchimp|mailchimpapp|view this email in your browser|unsubscribe|newsletter|digest|precedence:\s*bulk)\b",
                        _blob,
                    ))
                    marketingish = bool(_re.search(
                        r"\b(advertisement|sponsored|promo|promotion|sale|discount|offer|limited time|deal|tickets?|tour|merch|stream|purchase|sold out|low tickets|coupon|shop now|buy now)\b",
                        _blob,
                    ))
                    if (bulkish or marketingish) and score < 2:
                        score = 0
                        if not reason or "urgent" in reason.lower():
                            reason = "bulk mail; no personal reply needed"
                    # Strip "Name <addr>" to bare display name for compact summary.
                    _from_raw = item.get("from", "") or ""
                    if "<" in _from_raw:
                        _from_short = _from_raw.split("<", 1)[0].strip().strip('"') or _from_raw
                    else:
                        _from_short = _from_raw
                    verdict = {
                        "score": max(0, min(3, score)),
                        "tags": tags[:4],
                        "spam": spam,
                        "reason": reason,
                        "subject": (item.get("subject") or "")[:200],
                        "from": _from_short[:120],
                        "triage_version": TRIAGE_VERSION,
                        # Cache the message_id too so re-scans of already-cached
                        # UIDs can still write the inbox tag without re-LLM'ing.
                        "message_id": (item.get("message_id") or "").strip(),
                        "unread": bool(item.get("unread")),
                        "ts": _time.time(),
                    }
                    cache.setdefault("uids", {})[item["uid"]] = verdict
                    per_uid_scores[key] = verdict
                    saved_classifications += 1
                except Exception as e:
                    failed_classifications.append({
                        "subject": item.get("subject") or "(no subject)",
                        "from": item.get("from") or "",
                        "reason": str(e)[:120] or "classification failed",
                    })
                    logger.debug(f"urgency: LLM classify failed for {key}: {e}")
                    continue

            if scan_complete:
                # Only a complete account scan proves a cached UID left the
                # recent window. Partial/failing scans preserve prior facts.
                seen_uids = {it["uid"] for it in items}
                cache_uids = cache.get("uids", {})
                for stale in [u for u in cache_uids if u not in seen_uids]:
                    cache_uids.pop(stale, None)

            try:
                cache_file.write_text(_json.dumps(cache), encoding="utf-8")
            except Exception as e:
                logger.warning(f"urgency: cache write failed for {acc.id}: {e}")

        # ── 3.5  Mirror triage verdicts into email_tags so inbox filters and
        # pills show urgency + category tags. Runs for BOTH cached and freshly
        # classified items; message_id lives on the cached verdict so this is cheap.
        try:
            import sqlite3 as _sql3
            from routes.email_helpers import SCHEDULED_DB, _init_scheduled_db
            from datetime import datetime as _dt2
            _init_scheduled_db()
            _conn = _sql3.connect(SCHEDULED_DB)
            try:
                for _key, _v in per_uid_scores.items():
                    _msg_id = (_v.get("message_id") or "").strip()
                    _score = _v.get("score", 0)
                    if not _msg_id:
                        continue
                    _new_tags = []
                    if _score >= 3:
                        _new_tags.append("urgent")
                    elif _score >= 2:
                        _new_tags.append("reply-soon")
                    for _tag in (_v.get("tags") or []):
                        _tag = str(_tag).strip().lower().replace("_", "-")
                        if _tag == "promo":
                            _tag = "marketing"
                        if _tag == "action-needed" and any(t in _new_tags for t in ("urgent", "reply-soon")):
                            continue
                        if _tag in VISIBLE_EMAIL_TAGS and _tag not in _new_tags:
                            _new_tags.append(_tag)
                    _spam = 1 if _v.get("spam") else 0
                    # _key is "<account_id>:<uid>" — extract uid for the row.
                    _acc_id, _uid_only = (_key.split(":", 1) + [""])[:2]
                    _owner_key = owner or ""
                    _row = _conn.execute(
                        "SELECT tags FROM email_tags WHERE message_id=? AND owner=? AND account_id=?",
                        (_msg_id, _owner_key, _acc_id),
                    ).fetchone()
                    if _row:
                        try:
                            _existing = _json.loads(_row[0] or "[]")
                            if not isinstance(_existing, list):
                                _existing = []
                        except Exception:
                            _existing = []
                        # Drop previous triage-owned tags so re-classification
                        # can upgrade/downgrade/clear without touching manual tags.
                        _existing = [
                            str(t).strip().lower().replace("_", "-")
                            for t in _existing
                            if str(t).strip().lower().replace("_", "-") not in MANAGED_TAGS
                        ]
                        for _tag in _new_tags:
                            if _tag not in _existing:
                                _existing.append(_tag)
                        if _new_tags or _spam:
                            tag_write_details.append({
                                "uid": _uid_only,
                                "subject": _v.get("subject", ""),
                                "from": _v.get("from", ""),
                                "tags": list(_new_tags),
                                "spam": _spam,
                                "reason": _v.get("reason", ""),
                                "updated": True,
                            })
                        _conn.execute(
                            "UPDATE email_tags SET tags=?, spam_verdict=?, spam_reason=?, uid=?, folder=?, subject=?, sender=? "
                            "WHERE message_id=? AND owner=? AND account_id=?",
                            (_json.dumps(_existing), _spam, _v.get("reason", ""), _uid_only, "INBOX",
                             _v.get("subject", ""), _v.get("from", ""), _msg_id, _owner_key, _acc_id),
                        )
                    else:
                        if not _new_tags and not _spam:
                            continue
                        _conn.execute(
                            "INSERT INTO email_tags "
                            "(message_id, owner, account_id, uid, folder, subject, sender, tags, spam_verdict, spam_reason, created_at) "
                            "VALUES (?, ?, ?, ?, 'INBOX', ?, ?, ?, ?, ?, ?)",
                            (_msg_id, _owner_key, _acc_id, _uid_only, _v.get("subject", ""),
                             _v.get("from", ""), _json.dumps(_new_tags), _spam, _v.get("reason", ""),
                             _dt2.utcnow().isoformat()),
                        )
                        tag_write_details.append({
                            "uid": _uid_only,
                            "subject": _v.get("subject", ""),
                            "from": _v.get("from", ""),
                            "tags": list(_new_tags),
                            "spam": _spam,
                            "reason": _v.get("reason", ""),
                            "updated": False,
                        })
                _conn.commit()
            finally:
                _conn.close()
        except Exception as _te:
            logger.warning(f"urgency: bulk tag write failed: {_te}")

        # ── 4. Aggregate state. urgent = score ≥ 2.
        urgent_keys = [k for k, v in per_uid_scores.items() if v.get("score", 0) >= 2 and v.get("unread")]

        # ── 5. Fire a reminder only when a previously-unnotified UID scores
        # urgent. The read, decision, delivery, and checkpoint are serialized
        # below so two scheduler workers cannot both act on the same stale
        # state or overwrite each other's successful checkpoint.
        newly_notified = set()
        notify_failed = set()

        def _urgency_reminder_payload(reminder_keys):
            total = len(reminder_keys)
            title = "Urgent email" if total == 1 else f"{total} urgent emails"
            sorted_urgent = sorted(
                ((key, per_uid_scores[key]) for key in reminder_keys),
                key=lambda item: item[1].get("score", 0),
                reverse=True,
            )[:10]
            _pub = (settings.get("app_public_url") or "").strip().rstrip("/")
            from urllib.parse import quote as _quote
            lines = [
                f"{total} email" + ("" if total == 1 else "s")
                + " need an urgent reply:",
                "",
            ]
            for i, (key, value) in enumerate(sorted_urgent, 1):
                subj = (value.get("subject") or "(no subject)")[:160]
                frm = value.get("from") or ""
                why = value.get("reason") or ""
                uid_for_link = str(key).split(":", 1)[-1]
                hash_link = f"#email={_quote('INBOX', safe='')}:{uid_for_link}"
                open_link = f"{_pub}/{hash_link}" if _pub else hash_link
                line = f"{i}. {subj}"
                if frm:
                    line += f"  —  {frm}"
                if why:
                    line += f"  ·  {why}"
                lines.append(line)
                lines.append(f"   Open email: {open_link}")
            if total > len(sorted_urgent):
                lines.append("")
                lines.append(f"…and {total - len(sorted_urgent)} more.")
            return title, "\n".join(lines)

        async def _dispatch_urgency_reminder(reminder_keys):
            # Call dispatch_reminder directly: a scheduler has no browser
            # session cookie with which to call the HTTP endpoint.
            from routes.note_routes import dispatch_reminder
            title, body = _urgency_reminder_payload(reminder_keys)
            return await dispatch_reminder(
                title=title,
                note_body=body,
                note_id="urgent-email",
                owner=owner or "",
            )

        async def _dispatch_and_checkpoint(prior):
            notified_uids = _email_urgency_string_set(
                prior.get("notified_uids", [])
            )
            observed_accounts = {
                _email_urgency_account_key(key) for key in per_uid_scores
            } | fully_scanned_account_ids
            stale_accounts = _email_urgency_stale_accounts(
                prior,
                base_account_generations,
                observed_accounts,
            )
            # Generation fencing must happen before delivery, not only during
            # merge. A stale-only unread UID may have been removed, read, or
            # downgraded by the newer completed scan.
            deliverable_urgent = [
                key
                for key in urgent_keys
                if _email_urgency_account_key(key) not in stale_accounts
            ]
            new_urgent = [
                key
                for key in deliverable_urgent
                if key not in notified_uids
            ]
            if new_urgent:
                try:
                    dispatch_result = await _dispatch_urgency_reminder(
                        deliverable_urgent
                    )
                    channel = (settings.get("reminder_channel") or "browser").strip().lower()
                    delivered = bool(dispatch_result.get("browser_sent"))
                    if channel == "email":
                        delivered = bool(dispatch_result.get("email_sent"))
                    elif channel == "ntfy":
                        delivered = bool(dispatch_result.get("ntfy_sent"))
                    elif channel == "webhook":
                        delivered = bool(dispatch_result.get("webhook_sent"))
                    if delivered:
                        newly_notified.update(new_urgent)
                        notified_uids.update(new_urgent)
                    else:
                        notify_failed.update(new_urgent)
                        logger.warning(
                            "urgency: reminder dispatch returned no successful "
                            f"delivery path: {dispatch_result}"
                        )
                except Exception as e:
                    logger.warning(f"urgency: reminder dispatch failed: {e}")
                    notify_failed.update(new_urgent)

            next_state = _merge_email_urgency_state(
                prior,
                owner=owner,
                per_uid_scores=per_uid_scores,
                notified_uids=notified_uids,
                all_unread_keys=all_unread_keys,
                fully_scanned_account_ids=fully_scanned_account_ids,
                base_account_generations=base_account_generations,
                timestamp=_time.time(),
            )
            return notified_uids, next_state

        try:
            await _run_email_urgency_state_transaction(
                STATE_PATH,
                STATE_LOCK_DB,
                _dispatch_and_checkpoint,
            )
        except Exception as e:
            logger.warning(f"urgency: state transaction failed: {e}")

        # ── 6. Activity-log summary — counts line on top, then per-tier
        # bulleted breakdown so the user can see WHICH emails ranked where
        # (subject · sender · reason) and which ones triggered notifications.
        tier_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for v in per_uid_scores.values():
            tier_counts[v.get("score", 0)] = tier_counts.get(v.get("score", 0), 0) + 1
        if scanned == 0:
            raise TaskNoop("no unread emails in last 7 days")
        head = (
            f"scanned {scanned} · urgent {tier_counts[3]} · "
            f"reply-soon {tier_counts[2]} · info {tier_counts[1]} · trivial {tier_counts[0]} · "
            f"{saved_classifications} saved classifications"
        )
        if failed_classifications:
            head += f" · {len(failed_classifications)} failed"
        if newly_notified:
            head += f" · notified {len(newly_notified)}"
        if notify_failed:
            head += f" · notify failed {len(notify_failed)}"

        def _fmt_tag_write(v):
            subj = (v.get("subject") or "(no subject)")[:80]
            frm = v.get("from") or ""
            tags = list(v.get("tags") or [])
            if v.get("spam"):
                tags.append("spam")
            tag_txt = ", ".join(tags) if tags else "cleared managed tags"
            why = v.get("reason") or ""
            op = "updated" if v.get("updated") else "created"
            line = f"- **{subj}**" + (f" — _{frm}_" if frm else "")
            line += f" — `{tag_txt}` ({op})"
            if why:
                line += f" · {why}"
            return line

        def _fmt_one(v, newly_notified_set, failed_set, key):
            subj = (v.get("subject") or "(no subject)")[:80]
            frm = v.get("from") or ""
            why = v.get("reason") or ""
            tag = " · *notified now*" if key in newly_notified_set else (" · *notify failed*" if key in failed_set else "")
            line = f"- **{subj}**" + (f" — _{frm}_" if frm else "")
            if why:
                line += f" — {why}"
            return line + tag

        # Sort each tier by reason length (longest reason first → most info).
        by_tier = {3: [], 2: [], 1: [], 0: []}
        for k, v in per_uid_scores.items():
            by_tier.setdefault(v.get("score", 0), []).append((k, v))
        lines = [head]
        if tag_write_details:
            lines.append("")
            lines.append(f"**Applied tags ({len(tag_write_details)}):**")
            for v in tag_write_details[:16]:
                lines.append(_fmt_tag_write(v))
            if len(tag_write_details) > 16:
                lines.append(f"…and {len(tag_write_details) - 16} more")
        tier_labels = {3: "Urgent", 2: "Reply soon", 1: "Informational", 0: "Trivial"}
        for tier in (3, 2, 1, 0):
            items_t = by_tier.get(tier, [])
            if not items_t:
                continue
            lines.append("")
            lines.append(f"**{tier_labels[tier]} ({len(items_t)}):**")
            # Cap each tier at 8 rows to keep the activity entry readable.
            for k, v in items_t[:8]:
                lines.append(_fmt_one(v, newly_notified, notify_failed, k))
            if len(items_t) > 8:
                lines.append(f"…and {len(items_t) - 8} more")
        if failed_classifications:
            lines.append("")
            lines.append(f"**Unclassified ({len(failed_classifications)}):**")
            for v in failed_classifications[:8]:
                subj = (v.get("subject") or "(no subject)")[:80]
                frm = v.get("from") or ""
                why = v.get("reason") or ""
                line = f"- **{subj}**" + (f" — _{frm}_" if frm else "")
                if why:
                    line += f" — {why}"
                lines.append(line)
            if len(failed_classifications) > 8:
                lines.append(f"…and {len(failed_classifications) - 8} more")
        return "\n".join(lines), True
    except TaskNoop:
        raise
    except Exception as e:
        logger.exception("check_email_urgency action failed")
        return str(e), False


def _read_email_urgency_state(state_path):
    """Read one atomic urgency checkpoint, tolerating the legacy shape."""
    from pathlib import Path

    state_path = Path(state_path)
    try:
        state = (
            json.loads(state_path.read_text(encoding="utf-8"))
            if state_path.exists()
            else {}
        )
    except Exception:
        return {}
    return state if isinstance(state, dict) else {}


def _email_urgency_account_generations(state):
    """Return normalized per-account checkpoint/complete generations.

    Checkpoint generations fence every accepted state mutation. Complete
    generations advance only for a non-stale complete scan. Missing metadata
    is the legacy generation zero.
    """
    raw = state.get("account_generations", {}) if isinstance(state, dict) else {}
    if not isinstance(raw, dict):
        return {}

    generations = {}
    for account_id, value in raw.items():
        if isinstance(value, dict):
            checkpoint = value.get("checkpoint", 0)
            complete = value.get("complete", 0)
        else:
            # Tolerate an intermediate scalar representation as one completed
            # checkpoint generation instead of discarding its fence.
            checkpoint = value
            complete = value
        try:
            checkpoint = max(0, int(checkpoint))
        except (TypeError, ValueError):
            checkpoint = 0
        try:
            complete = max(0, int(complete))
        except (TypeError, ValueError):
            complete = 0
        generations[str(account_id)] = {
            "checkpoint": checkpoint,
            "complete": complete,
        }
    return generations


def _email_urgency_string_set(value):
    if not isinstance(value, (list, tuple, set, frozenset)):
        return set()
    return {str(item) for item in value if isinstance(item, (str, int))}


def _acquire_email_urgency_state_lock(
    state_path,
    lock_db_path,
    cancel_event,
    timeout_seconds=120,
):
    """Acquire the cross-process urgency lock without blocking the app loop."""
    import sqlite3
    import time
    from pathlib import Path

    state_path = Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds

    while not cancel_event.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise sqlite3.OperationalError("timed out waiting for urgency state lock")
        conn = sqlite3.connect(
            str(lock_db_path),
            timeout=min(0.25, max(0.01, remaining)),
            check_same_thread=False,
        )
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            conn.close()
            if "locked" not in str(exc).lower():
                raise
            cancel_event.wait(min(0.05, max(0.0, remaining)))
            continue
        except BaseException:
            conn.close()
            raise

        if cancel_event.is_set():
            conn.rollback()
            conn.close()
            return None, None
        return conn, _read_email_urgency_state(state_path)

    return None, None


def _close_email_urgency_state_lock(conn):
    if conn is None:
        return
    try:
        try:
            conn.rollback()
        except Exception:
            pass
    finally:
        conn.close()


def _commit_email_urgency_state(conn, state_path, next_state):
    """Atomically publish JSON before releasing the SQLite write lock."""
    import uuid
    from pathlib import Path

    state_path = Path(state_path)
    temp_path = state_path.with_name(
        f".{state_path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temp_path.write_text(json.dumps(next_state), encoding="utf-8")
        temp_path.replace(state_path)
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        temp_path.unlink(missing_ok=True)
        conn.close()


async def _run_email_urgency_state_transaction(
    state_path,
    lock_db_path,
    operation,
):
    """Serialize one urgency decision while keeping async work on this loop.

    Only lock acquisition waits in a worker thread. ``operation`` is awaited
    on the caller's long-lived event loop, where shared async clients, locks,
    and the browser-notification queue belong. Cancellation rolls back the
    SQLite transaction and never publishes a checkpoint.
    """
    import asyncio
    import threading

    loop = asyncio.get_running_loop()
    cancel_event = threading.Event()
    acquire_future = loop.run_in_executor(
        None,
        _acquire_email_urgency_state_lock,
        state_path,
        lock_db_path,
        cancel_event,
    )
    try:
        conn, prior = await asyncio.shield(acquire_future)
    except asyncio.CancelledError as cancelled:
        cancel_event.set()
        # The acquisition worker owns any connection until it returns. Wait
        # for its short busy-poll to observe cancellation, then close a lock it
        # may have won concurrently with the cancellation request.
        while True:
            try:
                conn, _prior = await asyncio.shield(acquire_future)
                break
            except asyncio.CancelledError:
                continue
            except Exception:
                conn = None
                break
        _close_email_urgency_state_lock(conn)
        raise cancelled

    if conn is None:
        raise asyncio.CancelledError

    try:
        result, next_state = await operation(prior)
        # Keep this small atomic publish synchronous. There is no await between
        # the successful operation and commit, so cancellation cannot be
        # observed and then followed by a checkpoint.
        try:
            _commit_email_urgency_state(conn, state_path, next_state)
        finally:
            conn = None
        return result
    except BaseException:
        _close_email_urgency_state_lock(conn)
        raise


def _email_urgency_account_key(message_key):
    return str(message_key).split(":", 1)[0]


def _email_urgency_payload_account_ids(state):
    """Return account IDs that still own user-visible urgency payload."""
    if not isinstance(state, dict):
        return set()

    per_uid = state.get("per_uid", {})
    per_uid_keys = per_uid if isinstance(per_uid, dict) else {}
    return {
        _email_urgency_account_key(key) for key in per_uid_keys
    } | {
        _email_urgency_account_key(key)
        for key in _email_urgency_string_set(state.get("notified_uids", []))
    }


def _email_urgency_known_account_ids(state):
    """Return payload owners plus generation-only active/retired markers."""
    return _email_urgency_payload_account_ids(state) | set(
        _email_urgency_account_generations(state)
    )


def _email_urgency_stale_accounts(
    prior,
    base_account_generations,
    account_ids,
):
    prior_generations = _email_urgency_account_generations(prior)
    base_generations = _email_urgency_account_generations(
        {"account_generations": base_account_generations}
    )
    return {
        str(account_id)
        for account_id in account_ids
        if prior_generations.get(str(account_id), {}).get("checkpoint", 0)
        != base_generations.get(str(account_id), {}).get("checkpoint", 0)
    }


def _merge_email_urgency_state(
    prior,
    *,
    owner,
    per_uid_scores,
    notified_uids,
    all_unread_keys,
    fully_scanned_account_ids,
    base_account_generations,
    timestamp,
    retired_account_ids=(),
    base_payload_account_ids=(),
    known_account_ids=(),
):
    """Merge a scan without letting an older snapshot erase newer facts."""
    prior_per_uid = prior.get("per_uid", {})
    if not isinstance(prior_per_uid, dict):
        prior_per_uid = {}
    complete = {str(account_id) for account_id in fully_scanned_account_ids}
    prior_generations = _email_urgency_account_generations(prior)
    retire_requested = {str(account_id) for account_id in retired_account_ids}
    observed_accounts = {
        _email_urgency_account_key(key) for key in per_uid_scores
    } | complete | retire_requested
    stale_accounts = _email_urgency_stale_accounts(
        prior,
        base_account_generations,
        observed_accounts,
    )
    prior_payload_accounts = _email_urgency_payload_account_ids(prior)
    base_payload_accounts = {
        str(account_id) for account_id in base_payload_account_ids
    }
    # A selected account can be absent from the base snapshot. If another
    # worker creates its first payload before this transaction wins the lock,
    # membership itself is a fence even when both snapshots normalize to the
    # legacy generation zero.
    retired_accounts = {
        account_id
        for account_id in retire_requested - stale_accounts
        if not (
            account_id in prior_payload_accounts
            and account_id not in base_payload_accounts
        )
    }
    fresh_complete = complete - stale_accounts - retired_accounts
    changed_accounts = set(fresh_complete)

    merged_per_uid = {
        key: value
        for key, value in prior_per_uid.items()
        if _email_urgency_account_key(key) not in retired_accounts
    }
    for key in list(merged_per_uid):
        account_id = _email_urgency_account_key(key)
        if account_id in fresh_complete:
            merged_per_uid.pop(key, None)
            changed_accounts.add(account_id)
    # Partial scans may add or refresh facts, but absence from a partial scan
    # is not evidence that another checkpoint or UI row is stale. When another
    # worker committed after this scan captured its base generation, discard
    # this account's whole stale snapshot. A key absent from the newer state
    # may have been removed/read, so even a stale-only key is not safely
    # additive without another fresh scan.
    for key, value in per_uid_scores.items():
        account_id = _email_urgency_account_key(key)
        if account_id in stale_accounts or account_id in retired_accounts:
            continue
        if merged_per_uid.get(key) != value:
            changed_accounts.add(account_id)
        merged_per_uid[key] = value

    prior_notified = _email_urgency_string_set(prior.get("notified_uids", []))
    merged_notified = {
        key
        for key in prior_notified
        if _email_urgency_account_key(key) not in retired_accounts
    }
    for key in _email_urgency_string_set(notified_uids) - prior_notified:
        account_id = _email_urgency_account_key(key)
        if account_id in stale_accounts or account_id in retired_accounts:
            continue
        merged_notified.add(key)
        changed_accounts.add(account_id)
    for key in list(merged_notified):
        if (
            _email_urgency_account_key(key) in fresh_complete
            and key not in all_unread_keys
        ):
            merged_notified.discard(key)
            changed_accounts.add(_email_urgency_account_key(key))

    next_generations = {
        account_id: dict(value)
        for account_id, value in prior_generations.items()
    }
    for account_id in changed_accounts:
        generation = next_generations.setdefault(
            account_id,
            {"checkpoint": 0, "complete": 0},
        )
        generation["checkpoint"] += 1
        if account_id in fresh_complete:
            generation["complete"] += 1
    for account_id in {str(value) for value in known_account_ids}:
        next_generations.setdefault(
            account_id,
            {"checkpoint": 0, "complete": 0},
        )
    for account_id in retired_accounts:
        # Every authoritative absence advances its generation, even when the
        # prior state is already a payload-empty tombstone. A re-enabled scan
        # may have captured that previous tombstone immediately before the
        # account was disabled/deleted again; monotonic advancement is what
        # makes that in-flight scan stale.
        generation = next_generations.setdefault(
            account_id,
            {"checkpoint": 0, "complete": 0},
        )
        generation["checkpoint"] += 1

    total_unread = 0
    total_urgent = 0
    max_score = 0
    for value in merged_per_uid.values():
        if not isinstance(value, dict):
            continue
        try:
            score = max(0, min(3, int(value.get("score", 0))))
        except (TypeError, ValueError):
            score = 0
        max_score = max(max_score, score)
        if value.get("unread"):
            total_unread += 1
            if score >= 2:
                total_urgent += 1

    return {
        "ts": timestamp,
        "owner": owner or "",
        "total_unread": total_unread,
        "total_urgent": total_urgent,
        "max_score": max_score,
        "per_uid": merged_per_uid,
        "notified_uids": sorted(merged_notified),
        "account_generations": next_generations,
    }
