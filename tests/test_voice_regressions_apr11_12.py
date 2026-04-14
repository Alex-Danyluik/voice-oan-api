"""
Deterministic Apr 11-12 regression tests for the voice pipeline.

These tests stay runnable without external model access and act as the
lightweight guardrail for the April 11-12 feedback corpus.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.stt_signals import detect_stt_signal
from app.services.translation import _post_normalize_gu_translation
from app.services.voice import _is_bare_greeting, _is_fragment_query, _is_hold_message


FIXTURE_PATH = Path(__file__).with_name("fixtures") / "apr11_12_regressions.json"


def load_fixture() -> dict:
    with FIXTURE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_gu(text: str) -> str:
    return _post_normalize_gu_translation(text, target_lang="gu", strip_outer=True)


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
