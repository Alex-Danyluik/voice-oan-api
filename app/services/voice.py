import asyncio
from contextlib import nullcontext
from functools import lru_cache
import time
from typing import AsyncGenerator, Optional, Literal
import re
from fastapi import Request

import regex
# from fastapi import BackgroundTasks
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart

from agents.voice import voice_agent
from agents.tools.farmer import normalize_phone_to_mobile
from agents.services.farmer_context import get_farmer_full_context_string
from agents.tools.common import (
    get_timeout_nudge_message,
    get_tool_nudge_message,
    send_nudge_message_raya,
    set_tool_call_nudge_event,
)
from helpers.utils import get_logger, clean_output_by_language
from app.config import settings
from app.utils import (
    update_message_history,
    trim_history,
    format_message_pairs,
    clean_message_history_for_openai,
    SessionRequestOwner,
    is_session_request_owner,
    refresh_session_request_ownership,
    release_session_request_ownership,
)
from app.services.stt_signals import (
    detect_stt_signal,
    generate_stt_signal_response,
    count_consecutive_stt_signals,
)
from app.services.translation import (
    INDIAN_LANGUAGES,
    OPENAI_PRETRANSLATION_MODEL,
    translate_text,
    translate_text_stream_fast,
    translate_to_english_with_gpt5_mini,
)
# NOTE: Removing telemetry for now.
# from app.tasks.telemetry import send_telemetry
from agents.deps import FarmerContext

logger = get_logger(__name__)


class SentenceSegmenter:
    sep = 'ŽžŽžSentenceSeparatorŽžŽž'
    latin_terminals = '!?.'
    jap_zh_terminals = '。！？'
    terminals = latin_terminals + jap_zh_terminals

    def __init__(self):
        terminals = self.terminals
        self._re = [
            (regex.compile(r'(\P{N})([' + terminals + r'])(\p{Z}*)'), r'\1\2\3' + self.sep),
            (regex.compile(r'(' + terminals + r')(\P{N})'), r'\1' + self.sep + r'\2'),
        ]

    @lru_cache(maxsize=2**16)
    def __call__(self, line: str):
        for (_re, repl) in self._re:
            line = _re.sub(repl, line)
        return [t for t in line.split(self.sep) if t != '']


sentence_segmenter = SentenceSegmenter()


def extract_complete_sentences(text: str):
    if not text:
        return [], ""
    sentences = sentence_segmenter(text)
    if len(sentences) <= 1:
        return [], text
    return sentences[:-1], sentences[-1]


def _batch_starts_new_line_or_list(text: str) -> bool:
    if not text or not text.strip():
        return False
    stripped = text.lstrip()
    if text != stripped:
        return True
    if stripped.startswith(("-", "•")) and (len(stripped) == 1 or stripped[1:2].isspace() or stripped[1:2] == "."):
        return True
    if stripped.startswith("*") and (len(stripped) == 1 or stripped[1:2].isspace() or stripped[1:2] == "."):
        return True
    return bool(re.match(r"^\d+\.\s", stripped))


# ── Greeting short-circuit helpers ─────────────────────────────────────
_GREETING_TOKENS = {
    # English
    "hello", "hi", "hey", "hlo",
    # Gujarati
    "હલો", "હેલો", "નમસ્તે", "નમસ્કાર", "હા",
    # Hindi
    "नमस्ते", "हेलो", "हलो",
    # Transliteration
    "namaste", "halo", "helo",
    # Multi-word greeting combos
    "ha hello", "હા હલો", "ji", "જી", "bolo", "બોલો",
    "ha bolo", "હા બોલો", "ji bolo", "જી બોલો",
}


def _is_bare_greeting(query: str) -> bool:
    """Return True if the query is just a greeting with no real content."""
    cleaned = re.sub(r"[*\s]+", " ", query).strip().lower()
    if not cleaned:
        return False
    # Strip punctuation for matching
    cleaned = re.sub(r"[.,!?।]+$", "", cleaned).strip()
    if cleaned in _GREETING_TOKENS:
        return True
    # Collapse repeated words: "hello hello" → "hello"
    words = cleaned.split()
    if len(words) <= 4:
        deduped = " ".join(dict.fromkeys(words))
        if deduped in _GREETING_TOKENS:
            return True
    return False


_GREETING_RESPONSES = {
    "gu": "નમસ્તે, હું સરલાબેન છું. તમારા પશુ વિશે કોઈ સમસ્યા હોય તો મને જણાવો.",
    "en": "Hello, I am Sarlaben. Please tell me what issue you are facing with your animal.",
}

# ── Fragment detection (garbled / too-short input) ────────────────────────
_FRAGMENT_RESPONSES = {
    "gu": "મને તમારો પ્રશ્ન સમજાયો નથી. કૃપા કરીને તમારો પ્રશ્ન ફરીથી પૂછો.",
    "en": "I could not understand your question. Please ask your question again.",
}


def _is_fragment_query(query: str) -> bool:
    """Return True if query is too short/garbled to be a real question."""
    cleaned = re.sub(r"[*\s.,!?।]+", " ", query).strip()
    if not cleaned:
        return True
    # Single character or very short (≤3 chars) — likely noise
    if len(cleaned) <= 3:
        return True
    return False


# ── Hold message detection ─────────────────────────────────────────────
# Carrier IVR "your call is on hold" messages get picked up by STT and
# sent as user input, creating runaway loops. Detect them and respond
# with "goodbye" so the STT provider cuts the call.
_HOLD_MSG_PATTERNS_GU = [
    "હોલ્ડ પર",            # "on hold" in Gujarati
    "લાઇન પર રહો",        # "stay on the line"
    "લાઈન પર રહો",        # variant spelling
]
_HOLD_MSG_PATTERNS_EN = [
    "put your call on hold",
    "call has been put on hold",
    "call on hold",
    "please stay on the line",
    "please remain on the line",
]
_HOLD_GOODBYE = {
    "gu": "Goodbye.",
    "en": "Goodbye.",
}


def _is_hold_message(query: str) -> bool:
    """Return True if the query looks like a carrier hold/IVR message."""
    lower = query.lower()
    for pat in _HOLD_MSG_PATTERNS_GU:
        if pat in lower:
            return True
    for pat in _HOLD_MSG_PATTERNS_EN:
        if pat in lower:
            return True
    return False


def _greeting_response(target_lang: str) -> str:
    return _GREETING_RESPONSES.get(target_lang, _GREETING_RESPONSES["gu"])


def _prepare_voice_output(text: str, lang_code: str) -> str:
    """Normalize model output for voice delivery."""
    return clean_output_by_language(text, lang_code)


def should_translate_batch(batch_text: str, word_count: int) -> bool:
    min_words = 15
    max_words = 80

    if word_count < min_words:
        text_end = batch_text.rstrip()
        return text_end.endswith(('.', '!', '?')) and word_count >= 5
    if word_count >= max_words:
        return True

    text_end = batch_text.rstrip()
    if text_end.endswith('\n\n'):
        return True
    if text_end.endswith('\n') and len(batch_text.split('\n')) > 1:
        last_line = batch_text.rstrip('\n').split('\n')[-1].strip()
        if last_line.startswith(('-', '*', '•')) or re.match(r'^\d+\.', last_line):
            return True
    return text_end.endswith(('.', '!', '?'))

# Langfuse Sessions: same session_id groups all traces for one conversation (session replay, session-level metrics).
def _langfuse_session_context(session_id: str, user_id: str, process_id: Optional[str] = None):
    """Set Langfuse session_id so all agent runs for this conversation appear under one Session."""
    try:
        from app.observability import langfuse_client
        from langfuse import propagate_attributes
        if langfuse_client is None:
            return nullcontext()
        # Langfuse Sessions: session_id ≤200 chars (US-ASCII); same ID = one Session in Langfuse UI
        safe_session_id = (session_id or "").strip()[:200]
        kwargs = dict(
            session_id=safe_session_id or None,
            user_id=(user_id or "anonymous")[:200],
        )
        if process_id:
            kwargs["metadata"] = {"process_id": str(process_id)[:200]}
        return propagate_attributes(**kwargs)
    except Exception:
        return nullcontext()


async def stream_voice_message(
    query: str,
    session_id: str,
    source_lang: str,
    target_lang: str,
    user_id: str,
    history: list,
    provider: Optional[Literal['RAYA']] = None,
    process_id: Optional[str] = None,
    user_info: dict = None,
    use_translation_pipeline: bool = False,
    owner: Optional[SessionRequestOwner] = None,
    http_request: Optional[Request] = None,
#    background_tasks: BackgroundTasks,
    
) -> AsyncGenerator[str, None]:
    """Async generator for streaming chat messages."""
    request_started_at = time.monotonic()
    last_owner_refresh_at = 0.0

    async def _request_is_stale(reason: str) -> bool:
        nonlocal last_owner_refresh_at
        if http_request is not None and await http_request.is_disconnected():
            logger.info(
                "Stopping request due to client disconnect - session_id=%s process_id=%s reason=%s",
                session_id,
                process_id,
                reason,
            )
            return True

        now = time.monotonic()
        if owner is not None and (
            last_owner_refresh_at == 0.0
            or now - last_owner_refresh_at >= settings.session_owner_refresh_interval_seconds
        ):
            refreshed = await refresh_session_request_ownership(owner)
            last_owner_refresh_at = now
            if not refreshed:
                logger.info(
                    "Stopping stale request after ownership lost during refresh - session_id=%s process_id=%s epoch=%s reason=%s",
                    session_id,
                    process_id,
                    owner.epoch,
                    reason,
                )
                return True

        if owner is not None and not await is_session_request_owner(owner):
            logger.info(
                "Stopping stale request because a newer request owns the session - session_id=%s process_id=%s epoch=%s reason=%s",
                session_id,
                process_id,
                owner.epoch,
                reason,
            )
            return True

        return False

    try:
        with _langfuse_session_context(session_id, user_id, process_id):
            requested_source_lang = (source_lang or "gu").strip().lower()
            requested_target_lang = (target_lang or "gu").strip().lower()
            needs_output_translation = use_translation_pipeline and requested_target_lang in INDIAN_LANGUAGES
            nudge_lang = (requested_target_lang or "en").strip().lower()

            # ── STT signal handling (no-audio / unclear speech) ─────────────
            # These are not real user messages — skip translation & agent,
            # generate a short contextual "please repeat" via GPT-5-mini.
            stt_signal = detect_stt_signal(query)
            if stt_signal is not None:
                logger.info(
                    "STT signal detected; session_id=%s process_id=%s signal=%s",
                    session_id,
                    process_id,
                    stt_signal,
                )
                recent_text = "\n\n".join(format_message_pairs(history, 3))
                if await _request_is_stale("before_stt_signal_response"):
                    return
                prior_stt_failures = count_consecutive_stt_signals(history)
                final_attempt = (prior_stt_failures + 1) >= max(1, settings.stt_signal_retry_ceiling)
                stt_response = await generate_stt_signal_response(
                    signal=stt_signal,
                    target_lang=requested_target_lang,
                    recent_history_text=recent_text,
                    final_attempt=final_attempt,
                )
                stt_req = ModelRequest(parts=[UserPromptPart(content=query)])
                stt_resp = ModelResponse(parts=[TextPart(content=stt_response)])
                await update_message_history(session_id, [*history, stt_req, stt_resp])
                yield _prepare_voice_output(stt_response, requested_target_lang)
                return

            # ── Hold message short-circuit ────────────────────────────────
            # Carrier IVR "your call is on hold" messages get transcribed by
            # STT and sent as user input, creating runaway loops of 20+ traces.
            # Respond with "goodbye" so the STT provider disconnects the call.
            if _is_hold_message(query):
                logger.info(
                    "Hold message detected; responding with goodbye to cut call - session_id=%s process_id=%s query=%r",
                    session_id, process_id, query[:100],
                )
                goodbye = _HOLD_GOODBYE.get(requested_target_lang, _HOLD_GOODBYE["en"])
                hold_req = ModelRequest(parts=[UserPromptPart(content=query)])
                hold_resp = ModelResponse(parts=[TextPart(content=goodbye)])
                await update_message_history(session_id, [*history, hold_req, hold_resp])
                yield _prepare_voice_output(goodbye, requested_target_lang)
                return

            # ── Greeting short-circuit ────────────────────────────────────
            # Bare greetings ("hello", "હલો", "હા") should not trigger the
            # full agent pipeline or a nudge.  Respond immediately.
            if _is_bare_greeting(query):
                logger.info(
                    "Bare greeting detected; short-circuiting - session_id=%s process_id=%s query=%r",
                    session_id, process_id, query,
                )
                greeting_response = _greeting_response(requested_target_lang)
                greet_req = ModelRequest(parts=[UserPromptPart(content=query)])
                greet_resp = ModelResponse(parts=[TextPart(content=greeting_response)])
                await update_message_history(session_id, [*history, greet_req, greet_resp])
                yield _prepare_voice_output(greeting_response, requested_target_lang)
                return

            # ── Fragment short-circuit ────────────────────────────────────
            # Very short / garbled input (≤3 chars) that isn't a greeting or
            # STT signal — ask the farmer to repeat instead of routing to agent.
            if _is_fragment_query(query):
                logger.info(
                    "Fragment query detected; short-circuiting - session_id=%s process_id=%s query=%r",
                    session_id, process_id, query,
                )
                frag_response = _FRAGMENT_RESPONSES.get(requested_target_lang, _FRAGMENT_RESPONSES["gu"])
                frag_req = ModelRequest(parts=[UserPromptPart(content=query)])
                frag_resp = ModelResponse(parts=[TextPart(content=frag_response)])
                await update_message_history(session_id, [*history, frag_req, frag_resp])
                yield _prepare_voice_output(frag_response, requested_target_lang)
                return

            # ── Nudge: arm BEFORE any pre-processing ────────────────────────
            # Fires on whichever happens first:
            #   (a) the configured timer expires, OR
            #   (b) the LLM invokes a tool (signalled via tool_call_event).
            # Cancelled if first text/translated chunk reaches the client first.
            nudge_sent = False
            tool_call_event = asyncio.Event()
            set_tool_call_nudge_event(tool_call_event)

            async def send_nudge_on_trigger() -> None:
                nonlocal nudge_sent
                try:
                    elapsed = max(0.0, time.monotonic() - request_started_at)
                    remaining = max(0.0, float(settings.nudge_timeout_seconds) - elapsed)
                    logger.info(
                        "Nudge armed; session_id=%s process_id=%s elapsed=%.3fs remaining=%.3fs timeout=%.3fs",
                        session_id,
                        process_id,
                        remaining,
                        settings.nudge_timeout_seconds,
                    )

                    # Wait for EITHER the timer OR a tool-call signal
                    timer_task = asyncio.create_task(asyncio.sleep(remaining))
                    event_task = asyncio.create_task(tool_call_event.wait())
                    done, pending = await asyncio.wait(
                        {timer_task, event_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for t in pending:
                        t.cancel()

                    trigger_reason = "tool_call" if event_task in done else "timeout"
                    if await _request_is_stale("before_nudge_send"):
                        return
                    if nudge_sent:
                        return
                    nudge_sent = True
                    nudge_msg = (
                        get_tool_nudge_message(nudge_lang)
                        if trigger_reason == "tool_call"
                        else get_timeout_nudge_message(nudge_lang)
                    )
                    await send_nudge_message_raya(nudge_msg, session_id, process_id)
                    elapsed = max(0.0, time.monotonic() - request_started_at)
                    logger.info(
                        "Nudge sent (%s); session_id=%s process_id=%s total_elapsed=%.3fs",
                        trigger_reason,
                        session_id,
                        process_id,
                        elapsed,
                    )
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.warning(
                        "Nudge task failed; session_id=%s process_id=%s error=%s",
                        session_id,
                        process_id,
                        e,
                    )

            nudge_task = asyncio.create_task(send_nudge_on_trigger())
            logger.info(
                "Nudge initiated; session_id=%s process_id=%s",
                session_id,
                process_id,
            )
            # ── End nudge setup ─────────────────────────────────────────────

            processing_query = query
            processing_lang = requested_source_lang
            pretranslation_confidence = "unknown"

            if use_translation_pipeline and requested_source_lang in {"gu", "gujarati"}:
                logger.info(
                    "Translation pipeline enabled; pretranslating %s -> en with %s",
                    requested_source_lang,
                    OPENAI_PRETRANSLATION_MODEL,
                )
                if await _request_is_stale("before_query_pretranslation"):
                    return
                try:
                    processing_query, pretranslation_confidence = await translate_to_english_with_gpt5_mini(
                        text=query,
                        source_lang=requested_source_lang,
                    )
                    processing_lang = "en"
                except Exception as e:
                    logger.error(
                        "OpenAI pretranslation failed for session_id=%s source_lang=%s model=%s error=%s",
                        session_id,
                        requested_source_lang,
                        OPENAI_PRETRANSLATION_MODEL,
                        e,
                    )
                    try:
                        logger.info("Falling back to TranslateGemma pretranslation for session_id=%s", session_id)
                        processing_query = await translate_text(
                            text=query,
                            source_lang=requested_source_lang,
                            target_lang="english",
                        )
                        processing_lang = "en"
                    except Exception as fallback_error:
                        logger.error(
                            "TranslateGemma pretranslation fallback failed for session_id=%s error=%s",
                            session_id,
                            fallback_error,
                        )
                        processing_query = query
                        processing_lang = requested_source_lang

            # ── Low-confidence pretranslation filter ─────────────────────
            # When the pretranslation model reports low confidence, the
            # input was likely garbled noise. Ask the farmer to repeat
            # instead of routing a hallucinated translation to the agent.
            if (
                use_translation_pipeline
                and requested_source_lang in {"gu", "gujarati"}
                and pretranslation_confidence == "low"
            ):
                logger.info(
                    "Pretranslation confidence=low; asking to repeat - session_id=%s process_id=%s query=%r translated=%r",
                    session_id, process_id, query, processing_query,
                )
                low_conf_resp = _FRAGMENT_RESPONSES.get(requested_target_lang, _FRAGMENT_RESPONSES["gu"])
                low_conf_req = ModelRequest(parts=[UserPromptPart(content=query)])
                low_conf_rsp = ModelResponse(parts=[TextPart(content=low_conf_resp)])
                await update_message_history(session_id, [*history, low_conf_req, low_conf_rsp])
                yield low_conf_resp
                return

            if use_translation_pipeline and needs_output_translation:
                processing_lang = "en"

            mobile = normalize_phone_to_mobile(user_id)
            farmer_info = ""
            if mobile:
                try:
                    farmer_info = await get_farmer_full_context_string(mobile)
                    logger.info(f"Farmer context loaded for mobile {mobile}, length={len(farmer_info)}")
                except Exception as e:
                    logger.warning(f"Failed to load farmer context for mobile {mobile}: {e}")

            logger.info(f"User info: {user_info}")
            deps = FarmerContext(
                query=processing_query,
                lang_code=processing_lang,
                target_lang=requested_target_lang,
                provider=provider,
                session_id=session_id,
                process_id=process_id,
                farmer_info=farmer_info,
                use_translation_pipeline=use_translation_pipeline,
            )

            message_pairs = "\n\n".join(format_message_pairs(history, 3))
            logger.info(f"Message pairs: {message_pairs}")
            user_message = deps.get_user_message()
            logger.info(f"Running agent with user message: {user_message}")

            cleaned_history = clean_message_history_for_openai(history)
            if len(cleaned_history) != len(history):
                logger.warning(f"Cleaned {len(history) - len(cleaned_history)} orphaned tool calls from history")
                if not await _request_is_stale("before_cleaned_history_write"):
                    await update_message_history(session_id, cleaned_history)
                history = cleaned_history

            trimmed_history = trim_history(
                history,
                max_tokens=80_000,
                include_system_prompts=True,
                include_tool_calls=True,
            )
            logger.info(f"Trimmed history length: {len(trimmed_history)} messages")

            async with voice_agent.run_stream(
                user_prompt=user_message,
                message_history=trimmed_history,
                deps=deps,
            ) as response_stream:
                stream_iter = response_stream.stream_text(delta=True)
                first_text_chunk_received = False
                sentence_buffer = ""
                translation_batch: list[str] = []
                batch_word_count = 0

                async def _yield_translated_text(text_to_translate: str) -> AsyncGenerator[str, None]:
                    if not text_to_translate:
                        return
                    try:
                        async for translated_chunk in translate_text_stream_fast(
                            text=text_to_translate,
                            source_lang="english",
                            target_lang=requested_target_lang,
                        ):
                            if await _request_is_stale("during_output_translation"):
                                return
                            cleaned_chunk = (
                                _prepare_voice_output(translated_chunk, requested_target_lang)
                                if isinstance(translated_chunk, str) and translated_chunk
                                else translated_chunk
                            )
                            yield cleaned_chunk
                    except Exception as e:
                        logger.error(
                            "Translation pipeline output translation failed for session_id=%s error=%s",
                            session_id,
                            e,
                        )
                        yield _prepare_voice_output(text_to_translate, "en")

                try:
                    async for chunk in stream_iter:
                        if await _request_is_stale("during_agent_stream"):
                            break

                        if not use_translation_pipeline or not needs_output_translation:
                            if (
                                not first_text_chunk_received
                                and isinstance(chunk, str)
                                and chunk
                                and chunk.strip()
                            ):
                                first_text_chunk_received = True
                                if nudge_task: nudge_task.cancel()
                                logger.info(
                                    "Nudge canceled (first text chunk received); session_id=%s process_id=%s chunk_preview=%s",
                                    session_id,
                                    process_id,
                                    chunk[:50] if len(chunk) > 50 else chunk,
                                )
                                try:
                                    await nudge_task
                                except asyncio.CancelledError:
                                    pass

                            cleaned_chunk = (
                                _prepare_voice_output(chunk, requested_target_lang)
                                if isinstance(chunk, str) and chunk
                                else chunk
                            )
                            if await _request_is_stale("before_direct_yield"):
                                break
                            yield cleaned_chunk
                            continue

                        sentence_buffer += chunk
                        complete_sentences, remaining = extract_complete_sentences(sentence_buffer)
                        if complete_sentences:
                            for sentence in complete_sentences:
                                translation_batch.append(sentence)
                                batch_word_count += len(sentence.split())

                            batch_text = "".join(translation_batch)
                            if should_translate_batch(batch_text, batch_word_count):
                                async for translated_chunk in _yield_translated_text(batch_text):
                                    if (
                                        not first_text_chunk_received
                                        and isinstance(translated_chunk, str)
                                        and translated_chunk
                                        and translated_chunk.strip()
                                    ):
                                        first_text_chunk_received = True
                                        if nudge_task: nudge_task.cancel()
                                        logger.info(
                                            "Nudge canceled (first translated chunk received); session_id=%s process_id=%s",
                                            session_id,
                                            process_id,
                                        )
                                        try:
                                            await nudge_task
                                        except asyncio.CancelledError:
                                            pass
                                    if await _request_is_stale("before_translated_yield"):
                                        break
                                    yield translated_chunk
                                translation_batch = []
                                batch_word_count = 0

                            sentence_buffer = remaining

                    if use_translation_pipeline and needs_output_translation and not await _request_is_stale("before_translation_flush"):
                        if translation_batch:
                            batch_text = "".join(translation_batch)
                            async for translated_chunk in _yield_translated_text(batch_text):
                                if (
                                    not first_text_chunk_received
                                    and isinstance(translated_chunk, str)
                                    and translated_chunk
                                    and translated_chunk.strip()
                                ):
                                    first_text_chunk_received = True
                                    if nudge_task: nudge_task.cancel()
                                    logger.info(
                                        "Nudge canceled (final translated batch); session_id=%s process_id=%s",
                                        session_id,
                                        process_id,
                                    )
                                    try:
                                        if nudge_task:
                                            await nudge_task
                                    except asyncio.CancelledError:
                                        pass
                                if await _request_is_stale("before_final_translated_yield"):
                                    break
                                yield translated_chunk

                        if sentence_buffer.strip():
                            async for translated_chunk in _yield_translated_text(sentence_buffer):
                                if (
                                    not first_text_chunk_received
                                    and isinstance(translated_chunk, str)
                                    and translated_chunk
                                    and translated_chunk.strip()
                                ):
                                    first_text_chunk_received = True
                                    if nudge_task: nudge_task.cancel()
                                    logger.info(
                                        "Nudge canceled (tail translated fragment); session_id=%s process_id=%s",
                                        session_id,
                                        process_id,
                                    )
                                    try:
                                        if nudge_task:
                                            await nudge_task
                                    except asyncio.CancelledError:
                                        pass
                                if await _request_is_stale("before_tail_translated_yield"):
                                    break
                                yield translated_chunk
                except StopAsyncIteration:
                    pass
                except RuntimeError as e:
                    if "StopAsyncIteration" in str(e) or "anext()" in str(e):
                        # anext() errors occur on superseded processes during
                        # teardown — the final process_id has its own generator
                        # and is unaffected, so this is just cleanup noise.
                        logger.debug(
                            "Suppressed stream runtime error (superseded process teardown) - session_id=%s process_id=%s error=%s",
                            session_id,
                            process_id,
                            e,
                        )
                    else:
                        raise
                finally:
                    if nudge_task and not nudge_task.done():
                        if nudge_task: nudge_task.cancel()
                        logger.info(
                            "Nudge canceled (stream ended); session_id=%s process_id=%s",
                            session_id,
                            process_id,
                        )
                        try:
                            await nudge_task
                        except asyncio.CancelledError:
                            pass

                logger.info(f"Streaming complete for session {session_id}")
                new_messages = response_stream.new_messages()

            if await _request_is_stale("before_history_write"):
                return

            messages = [*history, *new_messages]
            logger.info(f"Updating message history for session {session_id} with {len(messages)} messages")
            await update_message_history(session_id, messages)
    finally:
        released = await release_session_request_ownership(owner)
        if owner is not None:
            logger.info(
                "Session ownership released - session_id=%s process_id=%s epoch=%s released=%s",
                session_id,
                process_id,
                owner.epoch,
                released,
            )
