"""
Deterministic Apr 11-12 regression tests for the voice pipeline.

These tests stay runnable without external model access and act as the
lightweight guardrail for the April 11-12 feedback corpus.
"""
from __future__ import annotations

import json
import os
import sys
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.voice import voice_agent
from app.services.stt_signals import detect_stt_signal
from app.services.translation import _extract_translation_from_raw, _post_normalize_gu_translation
from app.services.voice import (
    TELEPHONY_TERMINATE_CALL_TOKEN,
    _has_meaningful_history,
    _is_bare_greeting,
    _is_fragment_query,
    _is_hold_message,
)


FIXTURE_PATH = Path(__file__).with_name("fixtures") / "apr11_12_regressions.json"


def load_fixture() -> dict:
    with FIXTURE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_gu(text: str) -> str:
    return _post_normalize_gu_translation(text, target_lang="gu", strip_outer=True)


class _FakeResponseStream:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def stream_text(self, delta: bool = True):
        async def _gen():
            if False:
                yield ""
        return _gen()

    def new_messages(self):
        return []


class TestApr11Apr12Fixture:
    def test_fixture_has_expected_shape(self):
        data = load_fixture()
        assert data["source"] == "shridhar_feedbacks_apr11_12.html"
        assert data["version"] == 1
        assert isinstance(data["scenarios"], list)
        assert len(data["scenarios"]) >= 6

    def test_comment_ids_are_unique(self):
        data = load_fixture()
        seen = []
        for scenario in data["scenarios"]:
            seen.extend(scenario["comment_ids"])
        assert len(seen) == len(set(seen)), "comment ids in fixture should not repeat across scenarios"

    def test_high_signal_scenarios_present(self):
        data = load_fixture()
        scenario_ids = {scenario["scenario_id"] for scenario in data["scenarios"]}
        expected = {
            "feedback_removed",
            "clarify_instead_of_answer",
            "no_phone_channel_hallucination",
            "no_missing_numeric_content",
            "respectful_gender_neutral_gujarati",
            "domain_disambiguation",
            "stt_retry_ceiling",
            "voice_text_cleanup",
        }
        assert expected.issubset(scenario_ids)


class TestHelperCoverage:
    @pytest.mark.parametrize("query", [
        "*No audio/User is speaking softly*",
        "No audio/User is speaking softly",
    ])
    def test_stt_signal_detection(self, query):
        assert detect_stt_signal(query) == "No audio/User is speaking softly"

    @pytest.mark.parametrize("query", [
        "*Unclear Speech*",
        "Unclear Speech",
    ])
    def test_unclear_signal_detection(self, query):
        assert detect_stt_signal(query) == "Unclear Speech"

    @pytest.mark.parametrize("query", [
        "hello",
        "હલો",
        "નમસ્તે",
        "Hi hi",
    ])
    def test_bare_greeting_helper(self, query):
        assert _is_bare_greeting(query) is True

    def test_affirmative_is_not_treated_as_bare_greeting(self):
        assert _is_bare_greeting("હા") is False

    @pytest.mark.parametrize("query", [
        "",
        "  ",
        "*",
        "ok",
        "હા",
    ])
    def test_fragment_helper(self, query):
        assert _is_fragment_query(query) is True

    @pytest.mark.parametrize("query", [
        "મારી ગાય",
        "દૂધ ઓછું",
        "ભેંસ બીમાર છે",
    ])
    def test_fragment_helper_does_not_overfire(self, query):
        assert _is_fragment_query(query) is False

    @pytest.mark.parametrize("query", [
        "તમારો કોલ હોલ્ડ પર રાખ્યો છે કૃપા કરીને લાઇન પર રહો",
        "Please stay on the line, your call has been put on hold.",
    ])
    def test_hold_message_helper(self, query):
        assert _is_hold_message(query) is True

    def test_telephony_terminate_token_stays_goodbye(self):
        assert TELEPHONY_TERMINATE_CALL_TOKEN["gu"] == "Goodbye."
        assert TELEPHONY_TERMINATE_CALL_TOKEN["en"] == "Goodbye."

    def test_meaningful_history_detected(self):
        history = [ModelRequest(parts=[UserPromptPart(content="મારી ગાયને તાવ છે")])]
        assert _has_meaningful_history(history) is True

    def test_stt_only_history_not_treated_as_meaningful(self):
        history = [ModelRequest(parts=[UserPromptPart(content="*No audio/User is speaking softly*")])]
        assert _has_meaningful_history(history) is False

    def test_voice_agent_runtime_config(self):
        assert voice_agent.end_strategy == "early"
        assert voice_agent.model_settings["max_tokens"] == 3600
        assert voice_agent.model_settings["temperature"] == 0.0
        assert voice_agent.model_settings["parallel_tool_calls"] is False

    @pytest.mark.parametrize("text, expected", [
        ("દૂધમાં ચરબી ઓછી છે.", "ફેટ"),
        ("ગાય ગર્ભવતી છે.", "ગાભણ"),
        ("સારા બળદ નો ઉપયોગ કરો.", "બુલ"),
    ])
    def test_current_gu_term_policy_still_holds(self, text, expected):
        result = normalize_gu(text)
        assert expected in result

    def test_policy_cleanup_does_not_strip_meaningful_text(self):
        result = normalize_gu("ગાયને તાવ છે.")
        assert result
        assert "તાવ" in result

    def test_extract_translation_from_raw_json(self):
        translated, confidence = _extract_translation_from_raw(
            '{"translation": "the cow has fever", "confidence": "low"}'
        )
        assert translated == "the cow has fever"
        assert confidence == "low"

    def test_low_confidence_fallback_pretranslation_still_short_circuits(self, monkeypatch):
        from app.services import voice as voice_module
        from agents import voice as voice_agent_module

        async def _openai_pretranslation(*args, **kwargs):
            raise TimeoutError("primary pretranslation failed")

        async def _fallback_pretranslation(*args, **kwargs):
            return "unclear livestock query", "low"

        async def _noop_async(*args, **kwargs):
            return None

        async def _get_farmer_full_context_string(mobile):
            return ""

        agent_called = False

        def _unexpected_run_stream(**kwargs):
            nonlocal agent_called
            agent_called = True
            return _FakeResponseStream()

        monkeypatch.setattr(voice_module, "translate_to_english_with_gpt5_mini", _openai_pretranslation)
        monkeypatch.setattr(voice_module, "translate_to_english_with_structured_fallback", _fallback_pretranslation)
        monkeypatch.setattr(voice_agent_module.voice_agent, "run_stream", _unexpected_run_stream)
        monkeypatch.setattr(voice_module, "normalize_phone_to_mobile", lambda user_id: None)
        monkeypatch.setattr(voice_module, "get_farmer_full_context_string", _get_farmer_full_context_string)
        monkeypatch.setattr(voice_module, "clean_message_history_for_openai", lambda history: history)
        monkeypatch.setattr(voice_module, "trim_history", lambda history, **kwargs: history)
        monkeypatch.setattr(voice_module, "format_message_pairs", lambda history, limit=None: [])
        monkeypatch.setattr(voice_module, "update_message_history", _noop_async)
        monkeypatch.setattr(voice_module, "send_nudge_message_raya", _noop_async)
        monkeypatch.setattr(voice_module, "set_tool_call_nudge_event", lambda event: SimpleNamespace())
        monkeypatch.setattr(voice_module.settings, "nudge_timeout_seconds", 999.0, raising=False)

        async def _run():
            chunks = []
            async for chunk in voice_module.stream_voice_message(
                query="કાળજ (વેચાવ)",
                session_id="fallback-low-confidence",
                source_lang="gu",
                target_lang="gu",
                user_id="anonymous",
                history=[],
                provider=None,
                process_id="proc-1",
                user_info={},
                use_translation_pipeline=True,
                owner=None,
                http_request=None,
            ):
                if isinstance(chunk, str):
                    chunks.append(chunk)
            return "".join(chunks)

        output = asyncio.run(_run())
        assert "ફરીથી" in output or "સમજાયો નથી" in output
        assert agent_called is False
