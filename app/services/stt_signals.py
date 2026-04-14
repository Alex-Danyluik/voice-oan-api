"""
Handle special STT signals (no-audio, unclear speech) with contextual responses.

When the STT pipeline cannot transcribe the user's audio it sends sentinel
strings instead of real text.  We intercept these early, skip translation
and the main agent, and ask a small LLM (GPT-5-mini) to produce a short,
context-aware prompt back to the user (e.g. "I can't hear you, could you
speak a bit louder?").
"""

import os
import random
import re
from typing import Optional, Sequence

from openai import AsyncOpenAI
from pydantic_ai.messages import ModelMessage

from helpers.utils import get_logger
from app.config import settings

logger = get_logger(__name__)

# ── Signal detection ────────────────────────────────────────────────────
# Strip leading/trailing whitespace and surrounding * markers, then match.
_SIGNAL_NO_AUDIO = "No audio/User is speaking softly"
_SIGNAL_UNCLEAR = "Unclear Speech"


def _normalize(text: str) -> str:
    return re.sub(r"[*]", "", text).strip().lower()


_SIGNALS = {
    _normalize(s): s
    for s in (_SIGNAL_NO_AUDIO, _SIGNAL_UNCLEAR)
}


def detect_stt_signal(query: str) -> Optional[str]:
    """Return the canonical signal name if *query* is a known STT signal, else None."""
    if not query:
        return None
    return _SIGNALS.get(_normalize(query))


# ── Contextual response generation ──────────────────────────────────────
STT_RESPONSE_MODEL = os.getenv("STT_SIGNAL_MODEL", "gpt-5-mini")

_SYSTEM_PROMPT = """\
You are a friendly voice assistant on a phone call with a farmer.
The speech-to-text system reported a problem with the caller's audio.

Your ONLY job is to produce a single short sentence (in the target language) \
politely asking the user to repeat or speak louder.  Take the recent \
conversation into account so the prompt feels natural — for example, if you \
just asked a question you can say "Sorry, I couldn't hear your answer, \
could you say that again?"

Rules:
- One sentence only, conversational tone, no markdown.
- If target language is Gujarati respond in Gujarati script.
- Never answer a question or provide information — just ask to repeat."""

_openai_client: Optional[AsyncOpenAI] = None

# ── Hardcoded fallback responses ──────────────────────────────────────
# Rotated randomly so repeated failures don't sound robotic.
_FALLBACK_NO_AUDIO: dict[str, list[str]] = {
    "gu": [
        "માફ કરશો, મને તમારો અવાજ સંભળાતો નથી. કૃપા કરીને ફોન નજીક રાખીને ફરીથી બોલો.",
        "હું તમને સાંભળી શકતી નથી. થોડું મોટેથી બોલી શકશો?",
        "તમારો અવાજ આવતો નથી. કૃપા કરીને ફરીથી બોલો.",
    ],
    "en": [
        "Sorry, I can't hear you. Could you please speak a little louder?",
        "I wasn't able to hear that. Could you move closer to the phone and try again?",
        "I didn't catch any audio. Please try speaking again.",
    ],
}

_FALLBACK_UNCLEAR: dict[str, list[str]] = {
    "gu": [
        "માફ કરશો, મને તમારી વાત બરાબર સમજાઈ નહીં. કૃપા કરીને ફરીથી બોલો.",
        "તમે શું કહ્યું એ સ્પષ્ટ સંભળાયું નહીં. થોડું ધીમેથી ફરી કહેશો?",
        "મને બરાબર સમજાયું નહીં. કૃપા કરીને ફરીથી કહો.",
    ],
    "en": [
        "Sorry, I couldn't understand that clearly. Could you please repeat?",
        "That wasn't clear to me. Could you say it again, a bit slowly?",
        "I didn't quite catch what you said. Please try again.",
    ],
}

_FINAL_NO_AUDIO: dict[str, list[str]] = {
    "gu": [
        "માફ કરશો, હજુ તમારો અવાજ સંભળાતો નથી. કૃપા કરીને પછીથી ફરી પ્રયાસ કરો.",
        "હાલમાં અવાજ સ્પષ્ટ નથી. કૃપા કરીને થોડા સમય પછી ફરી કોલ કરો.",
    ],
    "en": [
        "Sorry, I still can't hear you. Please try again later.",
        "The audio is still unclear. Please call again later.",
    ],
}

_FINAL_UNCLEAR: dict[str, list[str]] = {
    "gu": [
        "માફ કરશો, તમારી વાત હજી સ્પષ્ટ સમજાઈ નથી. કૃપા કરીને પછીથી ફરી પ્રયાસ કરો.",
        "હું હજી તમારી વાત સમજી શકતી નથી. કૃપા કરીને થોડા સમય પછી ફરી કોલ કરો.",
    ],
    "en": [
        "Sorry, I still couldn't understand that clearly. Please try again later.",
        "I'm still not able to understand. Please call again later.",
    ],
}


def _pick_fallback(signal: str, target_lang: str) -> str:
    pool = _FALLBACK_NO_AUDIO if signal == _SIGNAL_NO_AUDIO else _FALLBACK_UNCLEAR
    responses = pool.get(target_lang, pool["en"])
    return random.choice(responses)


def _pick_final_fallback(signal: str, target_lang: str) -> str:
    pool = _FINAL_NO_AUDIO if signal == _SIGNAL_NO_AUDIO else _FINAL_UNCLEAR
    responses = pool.get(target_lang, pool["en"])
    return random.choice(responses)


def count_consecutive_stt_signals(messages: Sequence[ModelMessage] | None) -> int:
    """Count consecutive STT signal user messages from the tail of history."""
    count = 0
    for msg in reversed(messages or []):
        user_text = None
        for part in getattr(msg, "parts", []) or []:
            if getattr(part, "part_kind", "") == "user-prompt":
                user_text = getattr(part, "content", None)
                break
        if not user_text:
            continue
        if detect_stt_signal(user_text) is not None:
            count += 1
            continue
        break
    return count


def _get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for STT signal responses")
        _openai_client = AsyncOpenAI(api_key=api_key)
    return _openai_client


async def generate_stt_signal_response(
    signal: str,
    target_lang: str,
    recent_history_text: str = "",
    final_attempt: bool = False,
) -> str:
    """Ask GPT-5-mini for a short, contextual 'please repeat' message.

    Falls back to hardcoded responses if the OpenAI call fails.

    Args:
        signal: canonical signal name (from detect_stt_signal).
        target_lang: language code for the response (e.g. "gu", "en").
        recent_history_text: formatted recent conversation turns for context.

    Returns:
        A single-sentence response string.
    """
    if final_attempt:
        return _pick_final_fallback(signal, target_lang)

    lang_label = {"gu": "Gujarati", "en": "English"}.get(target_lang, target_lang)

    user_content = (
        f"STT signal: {signal}\n"
        f"Target language: {lang_label}\n"
    )
    if recent_history_text:
        user_content += f"\nRecent conversation:\n{recent_history_text}\n"

    try:
        client = _get_openai_client()
        response = await client.chat.completions.create(
            model=STT_RESPONSE_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_completion_tokens=150,
            temperature=0.7,
        )
        text = (response.choices[0].message.content or "").strip()
        if text:
            return text
    except Exception:
        logger.warning(
            "STT signal LLM call failed, using hardcoded fallback - signal=%s lang=%s model=%s",
            signal, target_lang, STT_RESPONSE_MODEL,
            exc_info=True,
        )

    return _pick_fallback(signal, target_lang)
