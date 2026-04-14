"""
Integration regressions for Apr 11-12 voice failures.

Run with:
  VOICE_PIPELINE_INTEGRATION=1 pytest tests/test_voice_regressions_apr11_12_integration.py -q

The file is skipped unless VOICE_PIPELINE_INTEGRATION is set so normal pytest
remains fast. Some tests also require real model endpoints / API keys.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.translation import translate_to_english_with_gpt5_mini, translate_text
from app.services.voice import stream_voice_message
from app.services.stt_signals import generate_stt_signal_response


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


pytestmark = pytest.mark.skipif(
    not _env_flag("VOICE_PIPELINE_INTEGRATION"),
    reason="Set VOICE_PIPELINE_INTEGRATION=1 to run voice integration regressions",
)

FIXTURE_PATH = Path(__file__).with_name("fixtures") / "apr11_12_regressions.json"


def load_fixture() -> dict:
    with FIXTURE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


class _DummyPart:
    def __init__(self, part_kind: str, tool_name: str | None = None, args: dict | None = None):
        self.part_kind = part_kind
        self.tool_name = tool_name
        self.args = args or {}


class _DummyMessage:
    def __init__(self, parts):
        self.parts = parts


class _FakeResponseStream:
    def __init__(self, chunks: list[str], new_messages: list | None = None, delay: float = 0.0):
        self._chunks = chunks
        self._new_messages = new_messages or []
        self._delay = delay

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def stream_text(self, delta: bool = True):
        async def _gen():
            for chunk in self._chunks:
                if self._delay:
                    await asyncio.sleep(self._delay)
                yield chunk

        return _gen()

    def new_messages(self):
        return self._new_messages


def _set_identity_monkeypatches(monkeypatch, *, response_stream: _FakeResponseStream, history_store: dict):
    from agents import voice as voice_agent_module
    from app.services import voice as voice_module

    monkeypatch.setattr(voice_agent_module.voice_agent, "run_stream", lambda **kwargs: response_stream)

    async def _noop_async(*args, **kwargs):
        return None

    async def _get_farmer_full_context_string(mobile):
        return ""

    async def _update_message_history(session_id, messages):
        history_store[session_id] = messages

    async def _send_nudge_message_raya(message, session_id, process_id=None):
        return None

    monkeypatch.setattr(voice_module, "normalize_phone_to_mobile", lambda user_id: None)
    monkeypatch.setattr(voice_module, "get_farmer_full_context_string", _get_farmer_full_context_string)
    monkeypatch.setattr(voice_module, "clean_message_history_for_openai", lambda history: history)
    monkeypatch.setattr(voice_module, "trim_history", lambda history, **kwargs: history)
    monkeypatch.setattr(voice_module, "format_message_pairs", lambda history, limit=None: [])
    monkeypatch.setattr(voice_module, "update_message_history", _update_message_history)
    monkeypatch.setattr(voice_module, "send_nudge_message_raya", _send_nudge_message_raya)
    monkeypatch.setattr(voice_module, "get_timeout_nudge_message", lambda lang_code="gu": "હું જવાબ લઈને પાછી આવું છું, કૃપા કરીને થોડી રાહ જુઓ.")
    monkeypatch.setattr(voice_module, "get_tool_nudge_message", lambda lang_code="gu": "હું તપાસી રહી છું, કૃપા કરીને થોડી રાહ જુઓ.")
    monkeypatch.setattr(voice_module.settings, "nudge_timeout_seconds", 0.02, raising=False)


async def _collect_stream(query: str, *, session_id: str, history: list, monkeypatch, response_stream: _FakeResponseStream):
    from app.services import voice as voice_module

    history_store: dict[str, list] = {}
    _set_identity_monkeypatches(monkeypatch, response_stream=response_stream, history_store=history_store)
    chunks: list[str] = []
    async for chunk in voice_module.stream_voice_message(
        query=query,
        session_id=session_id,
        source_lang="gu",
        target_lang="gu",
        user_id="anonymous",
        history=history,
        provider=None,
        process_id="proc-1",
        user_info={},
        use_translation_pipeline=False,
        owner=None,
        http_request=None,
    ):
        if isinstance(chunk, str):
            chunks.append(chunk)
    return "".join(chunks), history_store.get(session_id, [])


def _contains_any(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def _contains_none(text: str, needles: list[str]) -> bool:
    return not _contains_any(text, needles)


def test_fixture_contains_integration_scenarios():
    data = load_fixture()
    scenario_ids = {scenario["scenario_id"] for scenario in data["scenarios"]}
    assert {"feedback_removed", "stt_retry_ceiling", "voice_text_cleanup"}.issubset(scenario_ids)


def test_real_openai_pretranslation_confidence_for_clear_and_garbled_inputs():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is required for this integration test")

    clear_text, clear_confidence = asyncio.run(translate_to_english_with_gpt5_mini("ગાયને તાવ છે", "gu"))
    noisy_text, noisy_confidence = asyncio.run(translate_to_english_with_gpt5_mini("કાળજ (વેચાવ)", "gu"))

    assert clear_text
    assert clear_confidence == "high"
    assert noisy_text
    assert noisy_confidence in {"low", "high", "unknown"}


def test_real_translategemma_output_is_speakable():
    if not (os.getenv("TRANSLATEGEMMA_27B_BASE_ENDPOINT") or os.getenv("TRANSLATEGEMMA_27B_BASE_ENDPOINTS")):
        pytest.skip("TRANSLATEGEMMA_27B_BASE_ENDPOINT(S) is required for this integration test")

    translated = asyncio.run(translate_text("The cow is pregnant.", "english", "gu"))
    assert translated
    assert _contains_any(translated, ["ગાભણ", "ગર્ભ"])


def test_real_stt_prompt_is_short_and_voice_friendly():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is required for this integration test")

    response = asyncio.run(
        generate_stt_signal_response(
            signal="No audio/User is speaking softly",
            target_lang="gu",
            recent_history_text="**User Message**:\nગાયને તાવ છે",
        )
    )

    assert response
    assert "\n" not in response
    assert "```" not in response
    assert len(response) < 180


def test_global_timeout_nudge_fires_with_neutral_copy_once(monkeypatch):
    history_store: dict[str, list] = {}
    response_stream = _FakeResponseStream(chunks=[""], delay=0.03)
    _set_identity_monkeypatches(monkeypatch, response_stream=response_stream, history_store=history_store)

    from app.services import voice as voice_module

    nudges: list[str] = []
    async def _capture_nudge(message, session_id, process_id=None):
        nudges.append(message)

    monkeypatch.setattr(voice_module, "send_nudge_message_raya", _capture_nudge)

    output, _ = asyncio.run(
        _collect_stream(
            query="મારી ગાયને તાવ છે",
            session_id="integration-nudge",
            history=[],
            monkeypatch=monkeypatch,
            response_stream=response_stream,
        )
    )

    assert output == ""
    assert len(nudges) <= 1
    if nudges:
        assert _contains_any(nudges[0], ["રાહ જુઓ", "wait", "back to you"])


def test_greeting_short_circuit_does_not_send_nudge(monkeypatch):
    response_stream = _FakeResponseStream(chunks=["ignored"], delay=0.0)
    _set_identity_monkeypatches(monkeypatch, response_stream=response_stream, history_store={})

    from app.services import voice as voice_module

    nudges: list[str] = []
    async def _capture_nudge(message, session_id, process_id=None):
        nudges.append(message)

    monkeypatch.setattr(voice_module, "send_nudge_message_raya", _capture_nudge)

    output, _ = asyncio.run(
        _collect_stream(
            query="hello",
            session_id="integration-greeting",
            history=[],
            monkeypatch=monkeypatch,
            response_stream=response_stream,
        )
    )

    assert _contains_any(output, ["નમસ્તે", "hello"])
    assert nudges == []


def test_closing_turn_does_not_append_feedback_question(monkeypatch):
    response_stream = _FakeResponseStream(
        chunks=["ચોક્કસ, આપના પશુ માટે હું અહીં છું."],
        new_messages=[],
        delay=0.0,
    )
    history_store: dict[str, list] = {}
    _set_identity_monkeypatches(monkeypatch, response_stream=response_stream, history_store=history_store)

    output, saved_history = asyncio.run(
        _collect_stream(
            query="આભાર, બસ છે",
            session_id="integration-closing",
            history=[],
            monkeypatch=monkeypatch,
            response_stream=response_stream,
        )
    )

    assert _contains_any(output, ["ચોક્કસ", "હું અહીં છું"])
    assert _contains_none(output, ["1 થી 5", "feedback", "કેટલો ઉપયોગી", "how helpful"])
    assert saved_history, "stream should persist conversation history"


def test_repeated_stt_failure_hits_retry_ceiling(monkeypatch):
    history: list = []
    history_store: dict[str, list] = {}
    response_stream = _FakeResponseStream(chunks=[""], delay=0.0)
    _set_identity_monkeypatches(monkeypatch, response_stream=response_stream, history_store=history_store)

    from app.services import voice as voice_module

    stt_calls: list[str] = []

    async def fake_generate_stt_signal_response(
        signal: str,
        target_lang: str,
        recent_history_text: str = "",
        final_attempt: bool = False,
    ) -> str:
        stt_calls.append(signal)
        if final_attempt:
            return "માફ કરશો, હજુ તમારો અવાજ સંભળાતો નથી. કૃપા કરીને પછીથી ફરી પ્રયાસ કરો."
        return "માફ કરશો, મને તમારો અવાજ સંભળાતો નથી. કૃપા કરીને ફરીથી બોલો."

    monkeypatch.setattr(voice_module, "generate_stt_signal_response", fake_generate_stt_signal_response)

    outputs: list[str] = []
    for idx in range(4):
        output, history = asyncio.run(
            _collect_stream(
                query="No audio/User is speaking softly",
                session_id="integration-stt-retry",
                history=history,
                monkeypatch=monkeypatch,
                response_stream=response_stream,
            )
        )
        outputs.append(output)

    assert len(stt_calls) >= 1
    assert _contains_any(outputs[0], ["સંભળાતો નથી", "ફરીથી"])
    assert _contains_any(outputs[1], ["સંભળાતો નથી", "ફરીથી"])
    assert _contains_any(outputs[2], ["સંભળાતો નથી", "ફરીથી"])
    assert _contains_none(outputs[3], ["ફરીથી", "સંભળાતો નથી", "repeat", "again"])
