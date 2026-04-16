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
from pydantic_ai.messages import ModelRequest, TextPart, UserPromptPart

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.voice import voice_agent, voice_agent_signed_in, STATIC_VOICE_SYSTEM_PROMPT
from app.services.stt_signals import detect_stt_signal
from app.services.translation import (
    GU_PREFERRED_TRANSLATION_RULES,
    _build_openai_pretranslation_messages,
    _extract_translation_from_raw,
    _post_normalize_gu_translation,
)
from app.services.voice import (
    TELEPHONY_TERMINATE_CALL_TOKEN,
    _build_compact_farmer_summary,
    _build_runtime_context_request,
    _has_meaningful_history,
    _is_bare_greeting,
    _is_fragment_query,
    _is_signed_in_session,
    _is_hold_message,
    _voice_answer_mode_for_query,
)
from agents.deps import FarmerContext
from agents.models.farmer import FarmerDataEnvelope, FarmerRecord


FIXTURE_PATH = Path(__file__).with_name("fixtures") / "apr11_12_regressions.json"


def load_fixture() -> dict:
    with FIXTURE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_gu(text: str) -> str:
    return _post_normalize_gu_translation(text, target_lang="gu", strip_outer=True)


class _FakeResponseStream:
    def __init__(self, chunks: list[str] | None = None, new_messages: list | None = None, delay: float = 0.0, on_enter=None):
        self._chunks = chunks or []
        self._new_messages = new_messages or []
        self._delay = delay
        self._on_enter = on_enter

    async def __aenter__(self):
        if self._on_enter is not None:
            await self._on_enter()
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


def _make_agent_messages(user_text: str, assistant_text: str) -> list:
    return [
        ModelRequest(parts=[UserPromptPart(content=user_text)]),
        SimpleNamespace(parts=[TextPart(content=assistant_text)]),
    ]


async def _noop_async(*args, **kwargs):
    return None


def _set_voice_monkeypatches(
    monkeypatch,
    *,
    response_stream: _FakeResponseStream,
    history_store: dict,
    nudges: list[str] | None = None,
    tool_event_box: dict | None = None,
    run_stream_override=None,
    signed_in_run_stream_override=None,
    normalize_phone_override=None,
    farmer_data_override=None,
):
    from agents import voice as voice_agent_module
    from app.services import voice as voice_module

    if run_stream_override is not None:
        monkeypatch.setattr(voice_agent_module.voice_agent, "run_stream", run_stream_override)
    else:
        monkeypatch.setattr(voice_agent_module.voice_agent, "run_stream", lambda **kwargs: response_stream)
    if signed_in_run_stream_override is not None:
        monkeypatch.setattr(voice_agent_module.voice_agent_signed_in, "run_stream", signed_in_run_stream_override)

    async def _get_or_fetch_farmer_data(_mobile):
        return None

    async def _update_message_history(session_id, messages):
        history_store[session_id] = messages

    async def _send_nudge_message_raya(message, session_id, process_id=None):
        if nudges is not None:
            nudges.append(message)

    async def _render_text_for_caller(text_en, target_lang):
        if target_lang in {"gu", "gujarati"}:
            if text_en == "Hello, I am Sarlaben. Please tell me what issue you are facing with your animal.":
                return "નમસ્તે, હું સરલાબેન છું. તમારા પશુ વિશે કોઈ સમસ્યા હોય તો મને જણાવો."
            if text_en == "I could not understand your question. Please ask your question again.":
                return "મને તમારો પ્રશ્ન સમજાયો નથી. કૃપા કરીને તમારો પ્રશ્ન ફરીથી પૂછો."
            return "મને તમારો પ્રશ્ન સમજાયો નથી. કૃપા કરીને ફરીથી પૂછો."
        return text_en

    def _capture_tool_call_event(event):
        if tool_event_box is not None:
            tool_event_box["event"] = event
        return SimpleNamespace()

    if normalize_phone_override is not None:
        monkeypatch.setattr(voice_module, "normalize_phone_to_mobile", normalize_phone_override)
    else:
        monkeypatch.setattr(voice_module, "normalize_phone_to_mobile", lambda user_id: None)
    monkeypatch.setattr(
        voice_module,
        "get_or_fetch_farmer_data",
        farmer_data_override if farmer_data_override is not None else _get_or_fetch_farmer_data,
    )
    monkeypatch.setattr(voice_module, "clean_message_history_for_openai", lambda history: history)
    monkeypatch.setattr(voice_module, "trim_history", lambda history, **kwargs: history)
    monkeypatch.setattr(voice_module, "format_message_pairs", lambda history, limit=None: [])
    monkeypatch.setattr(voice_module, "update_message_history", _update_message_history)
    monkeypatch.setattr(voice_module, "send_nudge_message_raya", _send_nudge_message_raya)
    monkeypatch.setattr(voice_module, "_render_text_for_caller", _render_text_for_caller)
    monkeypatch.setattr(voice_module, "set_tool_call_nudge_event", _capture_tool_call_event)
    monkeypatch.setattr(voice_module.settings, "nudge_timeout_seconds", 0.02, raising=False)
    return voice_module


async def _collect_stream(
    query: str,
    *,
    session_id: str,
    history: list,
    monkeypatch,
    response_stream: _FakeResponseStream,
    source_lang: str = "gu",
    target_lang: str = "gu",
    user_id: str = "anonymous",
    nudges: list[str] | None = None,
    tool_event_box: dict | None = None,
    run_stream_override=None,
    signed_in_run_stream_override=None,
    normalize_phone_override=None,
    farmer_data_override=None,
):
    history_store: dict[str, list] = {}
    voice_module = _set_voice_monkeypatches(
        monkeypatch,
        response_stream=response_stream,
        history_store=history_store,
        nudges=nudges,
        tool_event_box=tool_event_box,
        run_stream_override=run_stream_override,
        signed_in_run_stream_override=signed_in_run_stream_override,
        normalize_phone_override=normalize_phone_override,
        farmer_data_override=farmer_data_override,
    )
    chunks: list[str] = []
    async for chunk in voice_module.stream_voice_message(
        query=query,
        session_id=session_id,
        source_lang=source_lang,
        target_lang=target_lang,
        user_id=user_id,
        history=history,
        provider=None,
        process_id="proc-1",
        user_info={},
        owner=None,
        http_request=None,
    ):
        if isinstance(chunk, str):
            chunks.append(chunk)
    return "".join(chunks), history_store.get(session_id, [])


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
        assert voice_agent_signed_in.end_strategy == "early"
        assert voice_agent_signed_in.model_settings["max_tokens"] == 3600

    def test_signed_in_agent_has_farmer_tools(self):
        base_tool_names = set(voice_agent._function_tools.keys())
        signed_in_tool_names = set(voice_agent_signed_in._function_tools.keys())
        assert {"search_terms", "search_documents", "create_ai_call"}.issubset(base_tool_names)
        assert {"get_farmer_profile", "get_herd_summary", "list_animal_tags"}.issubset(signed_in_tool_names)
        assert "get_farmer_profile" not in base_tool_names

    def test_voice_system_prompt_is_static(self):
        assert "Today's date:" not in STATIC_VOICE_SYSTEM_PROMPT
        assert "## Farmer Context" not in STATIC_VOICE_SYSTEM_PROMPT
        assert "{{today_date}}" not in STATIC_VOICE_SYSTEM_PROMPT
        assert "{{farmer_context}}" not in STATIC_VOICE_SYSTEM_PROMPT

    def test_pretranslation_prompt_preserves_uncertainty(self):
        messages = _build_openai_pretranslation_messages(
            "Gujarati",
            "gu",
            "કા પણ બેની કઈ દોરણ ખાવડાવું જોઈએ",
        )
        prompt = messages[0]["content"]
        assert "faithful pretranslation" in prompt
        assert "Preserve uncertainty" in prompt
        assert "Do not infer animal species" in prompt
        assert "Set confidence to \"high\" only when the core request is clear without guessing" in prompt
        assert "unclear animal" in prompt
        assert "Never convert a doubtful token into a specific medicine, feed, disease, animal species, or service term" in prompt

    def test_pretranslation_prompt_does_not_turn_address_words_into_caller_gender(self):
        messages = _build_openai_pretranslation_messages("Gujarati", "gu", "બેન મારી ભેંસને તાવ છે")
        prompt = messages[0]["content"]
        assert "Kinship words" in prompt
        assert "Do not turn them into the caller's gender" in prompt
        assert "address marker" in prompt
        assert "mark confidence low if the word could also be an address marker" in prompt

    def test_core_prompt_requires_professional_detached_gender_neutral_tone(self):
        assert "professional, cordial, detached" in STATIC_VOICE_SYSTEM_PROMPT
        assert "Do not mirror kinship words from the translation" in STATIC_VOICE_SYSTEM_PROMPT
        assert "Never address the caller as sister" in STATIC_VOICE_SYSTEM_PROMPT
        assert "Never infer or assign the caller's gender" in STATIC_VOICE_SYSTEM_PROMPT
        assert "This is a live phone call, not a chat or article." in STATIC_VOICE_SYSTEM_PROMPT
        assert "Default to one short sentence." in STATIC_VOICE_SYSTEM_PROMPT
        assert "Do not use colons, headings, labels, hyphens, or en dashes" in STATIC_VOICE_SYSTEM_PROMPT
        assert "Do not organize the answer as \"one\", \"two\", \"three\"" in STATIC_VOICE_SYSTEM_PROMPT

    def test_gujarati_output_rules_keep_addressing_neutral_and_detached(self):
        rules = "\n".join(GU_PREFERRED_TRANSLATION_RULES)
        assert "professional, cordial, and detached" in rules
        assert "Do not translate English address markers" in rules
        assert "sister, brother, bhai, ben, madam, or sir" in rules
        assert "respectful gender-neutral 'આપ'" in rules
        assert "do not call the caller બહેન" in rules

    def test_runtime_context_request_contains_dynamic_turn_state(self):
        deps = FarmerContext(
            query="What is your name?",
            farmer_info="# Farmer Context\n\n- **Matched farmer records:** 1",
            signed_in=True,
            mobile="9723293369",
        )
        request = _build_runtime_context_request(deps)
        content = request.parts[0].content
        assert "Runtime context for this turn:" in content
        assert "Signed-in session: yes" in content
        assert "Normalized mobile: 9723293369" in content
        assert "Farmer context summary:" in content

    def test_runtime_context_adds_compact_comparison_mode(self):
        deps = FarmerContext(query="What is the difference between A2 milk and normal milk?")
        request = _build_runtime_context_request(deps)
        content = request.parts[0].content
        assert "Voice answer mode: compact comparison." in content
        assert "Give one short contrast sentence" in content
        assert "Do not enumerate." in content

    def test_runtime_context_adds_compact_explainer_mode(self):
        deps = FarmerContext(query="What is mastitis?")
        request = _build_runtime_context_request(deps)
        content = request.parts[0].content
        assert "Voice answer mode: compact explainer." in content
        assert "Do not teach the full topic." in content

    def test_runtime_context_adds_action_first_symptom_mode(self):
        deps = FarmerContext(query="My cow has fever")
        request = _build_runtime_context_request(deps)
        content = request.parts[0].content
        assert "Voice answer mode: action-first symptom response." in content
        assert "Start with the most useful immediate action" in content

    @pytest.mark.parametrize("query, expected", [
        ("What is the difference between A2 milk and normal milk?", "compact_comparison"),
        ("What is mastitis?", "compact_explainer"),
        ("Compare buffalo milk and cow milk", "compact_comparison"),
        ("My cow has fever", "action_first_symptom"),
        ("My buffalo is not eating", "action_first_symptom"),
        ("What is SNF in milk?", "compact_explainer"),
        ("Hello", None),
    ])
    def test_voice_answer_mode_for_query(self, query, expected):
        assert _voice_answer_mode_for_query(query) == expected

    def test_signed_in_session_helper(self):
        assert _is_signed_in_session({"sub": "user-1"}, "anonymous") is True
        assert _is_signed_in_session({}, "9876543210") is True
        assert _is_signed_in_session({}, "anonymous") is False

    def test_compact_farmer_summary_is_small_and_structured(self):
        envelope = FarmerDataEnvelope(
            farmers=[
                FarmerRecord(
                    farmerName="Rameshbhai",
                    societyName="Anand Dairy Society",
                    farmerCode="F123",
                    totalAnimals=6,
                    tagNumbers="1001,1002,1003",
                )
            ],
            source="cache",
        )
        summary = _build_compact_farmer_summary(envelope)
        assert "Farmer records matched: 1" in summary
        assert "Farmer data source: cache" in summary
        assert "Farmer name: Rameshbhai" in summary
        assert "Farmer code available: yes" in summary
        assert "Known animal tags: 1001, 1002, 1003" in summary
        assert "##" not in summary

    def test_translation_pipeline_prompt_has_unclear_input_confirmation_rules(self):
        prompt_path = Path(__file__).resolve().parents[1] / "assets" / "prompts" / "voice_system_translation_pipeline_en.md"
        prompt_text = prompt_path.read_text(encoding="utf-8")
        assert "If the message is fully unclear, partly clear, single-word, fragmentary, contradictory, or garbled" in prompt_text
        assert "Never open with filler phrases like \"I am checking\"" in prompt_text
        assert "Never output missing-value placeholders" in prompt_text
        assert "ask the farmer to repeat that word instead of explaining what you think it means" in prompt_text
        assert "\"feed for the pregnant animal\"" in prompt_text
        assert "\"samudri\"" in prompt_text
        assert "ask for clarification rather than assuming a brand name" in prompt_text
        assert "This is a phone call. The caller cannot see formatting." in prompt_text
        assert "Do not use colons, headings, labels, hyphens, or en dashes" in prompt_text
        assert "For comparison questions, give only the main difference first" in prompt_text
        assert "Do not append a follow-up question unless it is necessary" in prompt_text

    def test_translation_pipeline_prompt_contains_short_voice_examples(self):
        prompt_path = Path(__file__).resolve().parents[1] / "assets" / "prompts" / "voice_system_translation_pipeline_en.md"
        prompt_text = prompt_path.read_text(encoding="utf-8")
        assert "## Voice Examples" in prompt_text
        assert "User: `What is the difference between A2 milk and normal milk?`" in prompt_text
        assert "Assistant: `A2 milk differs mainly in the type of beta casein protein." in prompt_text
        assert "User: `samudri dan for buffalo`" in prompt_text
        assert "Assistant: `Please repeat that feed name once. I did not understand it clearly.`" in prompt_text
        assert "User: `No, that is all`" in prompt_text
        assert "Assistant: `All right. You can call again if you need help.`" in prompt_text

    @pytest.mark.parametrize("text, expected", [
        ("દૂધમાં ચરબી ઓછી છે.", "ફેટ"),
        ("ગાય ગર્ભવતી છે.", "ગાભણ"),
        ("સારા બળદ નો ઉપયોગ કરો.", "બુલ"),
        ("મને બૈડું ઠંડું લાગે છે.", "શરીર ઠંડું લાગે છે"),
        ("પશુના બૈડા પર સોજો છે.", "પીઠ"),
        ("લીલા ચારમાં બરબા આપો.", "બરસીમ"),
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

        agent_called = False
        history_store: dict[str, list] = {}

        async def _mark_called():
            nonlocal agent_called
            agent_called = True

        monkeypatch.setattr(voice_module, "translate_to_english_with_gpt5_mini", _openai_pretranslation)
        monkeypatch.setattr(voice_module, "translate_to_english_with_structured_fallback", _fallback_pretranslation)

        output, saved_history = asyncio.run(
            _collect_stream(
                query="કાળજ (વેચાવ)",
                session_id="fallback-low-confidence",
                history=[],
                monkeypatch=monkeypatch,
                response_stream=_FakeResponseStream(on_enter=_mark_called),
                source_lang="gu",
                target_lang="gu",
            )
        )

        assert "ફરીથી" in output or "સમજાયો નથી" in output
        assert agent_called is False
        saved_text = " ".join(
            getattr(part, "content", "")
            for msg in saved_history
            for part in getattr(msg, "parts", [])
            if isinstance(getattr(part, "content", None), str)
        )
        assert "કાળજ" not in saved_text
        assert "[unclear-user-input]" in saved_text or "I could not understand your question" in saved_text


class TestMultiTurnFlows:
    def test_greeting_then_domain_query_reaches_agent(self, monkeypatch):
        from app.services import voice as voice_module

        first_output, history = asyncio.run(
            _collect_stream(
                query="hello",
                session_id="multiturn-greeting-domain",
                history=[],
                monkeypatch=monkeypatch,
                response_stream=_FakeResponseStream(chunks=["ignored"]),
            )
        )
        assert "નમસ્તે" in first_output or "Hello" in first_output
        greeting_history_text = " ".join(
            getattr(part, "content", "")
            for msg in history
            for part in getattr(msg, "parts", [])
            if isinstance(getattr(part, "content", None), str)
        )
        assert "Hello, I am Sarlaben." in greeting_history_text
        assert "નમસ્તે" not in greeting_history_text

        agent_called = {"value": False}

        async def _mark_called():
            agent_called["value"] = True

        async def _pretranslate(*args, **kwargs):
            return "My cow has fever", "high"

        monkeypatch.setattr(voice_module, "translate_to_english_with_gpt5_mini", _pretranslate)

        second_output, _ = asyncio.run(
            _collect_stream(
                query="મારી ગાયને તાવ છે",
                session_id="multiturn-greeting-domain",
                history=history,
                monkeypatch=monkeypatch,
                response_stream=_FakeResponseStream(
                    chunks=["ગાયને તાવ છે તો પશુચિકિત્સકનો સંપર્ક કરો."],
                    new_messages=_make_agent_messages("મારી ગાયને તાવ છે", "ગાયને તાવ છે તો પશુચિકિત્સકનો સંપર્ક કરો."),
                    on_enter=_mark_called,
                ),
            )
        )

        assert agent_called["value"] is True
        assert "નમસ્તે" not in second_output
        assert "પશુચિકિત્સક" in second_output

    def test_affirmative_with_meaningful_history_does_not_restart_as_greeting(self, monkeypatch):
        from app.services import voice as voice_module

        history = _make_agent_messages("ગાય માટે કે ભેંસ માટે?", "ગાય માટે કે ભેંસ માટે?")
        agent_called = {"value": False}

        async def _mark_called():
            agent_called["value"] = True

        async def _pretranslate(*args, **kwargs):
            return "yes", "high"

        monkeypatch.setattr(voice_module, "translate_to_english_with_gpt5_mini", _pretranslate)

        output, _ = asyncio.run(
            _collect_stream(
                query="હા",
                session_id="multiturn-affirmative-history",
                history=history,
                monkeypatch=monkeypatch,
                response_stream=_FakeResponseStream(
                    chunks=["સમજાયું, ગાય માટે નોંધ્યું."],
                    new_messages=_make_agent_messages("હા", "સમજાયું, ગાય માટે નોંધ્યું."),
                    on_enter=_mark_called,
                ),
            )
        )

        assert agent_called["value"] is True
        assert "નમસ્તે" not in output
        assert "સમજાયું" in output

    def test_runtime_context_is_not_persisted_in_history(self, monkeypatch):
        output, saved_history = asyncio.run(
            _collect_stream(
                query="તમારું નામ શું છે?",
                session_id="runtime-context-not-persisted",
                history=[],
                monkeypatch=monkeypatch,
                response_stream=_FakeResponseStream(
                    chunks=["હું સરલાબેન છું."],
                    new_messages=_make_agent_messages("What is your name?", "I am Sarlaben."),
                ),
            )
        )

        assert output
        persisted_text = " ".join(
            getattr(part, "content", "")
            for msg in saved_history
            for part in getattr(msg, "parts", [])
            if isinstance(getattr(part, "content", None), str)
        )
        assert "Runtime context for this turn:" not in persisted_text

    def test_stream_builds_runtime_context_from_cached_farmer_summary(self, monkeypatch):
        from app.services import voice as voice_module

        captured = {}

        async def _fake_farmer_data(_mobile):
            return FarmerDataEnvelope(
                farmers=[
                    FarmerRecord(
                        farmerName="Rameshbhai",
                        societyName="Anand Dairy Society",
                        farmerCode="F123",
                        totalAnimals=6,
                        tagNumbers="1001,1002",
                    )
                ],
                source="cache",
            )

        def _run_stream(**kwargs):
            history = kwargs["message_history"]
            captured["runtime_context"] = history[0].parts[0].content
            return _FakeResponseStream(
                chunks=["ગાયને તાવ છે તો પશુચિકિત્સકનો સંપર્ક કરો."],
                new_messages=_make_agent_messages("My cow has fever.", "Contact a veterinarian for the cow's fever."),
            )

        async def _pretranslate(*args, **kwargs):
            return "My cow has fever.", "high"

        monkeypatch.setattr(voice_module, "translate_to_english_with_gpt5_mini", _pretranslate)

        output, _ = asyncio.run(
            _collect_stream(
                query="મારી ગાયને તાવ છે",
                session_id="runtime-context-farmer-summary",
                history=[],
                monkeypatch=monkeypatch,
                response_stream=_FakeResponseStream(),
                source_lang="gu",
                target_lang="gu",
                user_id="9723293369",
                signed_in_run_stream_override=_run_stream,
                normalize_phone_override=lambda user_id: user_id,
                farmer_data_override=_fake_farmer_data,
            )
        )

        assert output
        runtime_context = captured["runtime_context"]
        assert "Farmer data source: cache" in runtime_context
        assert "Farmer name: Rameshbhai" in runtime_context
        assert "Known animal tags: 1001, 1002" in runtime_context

    def test_signed_in_session_uses_signed_in_agent_and_higher_request_limit(self, monkeypatch):
        from app.services import voice as voice_module

        captured = {}

        async def _fake_farmer_data(_mobile):
            return FarmerDataEnvelope(
                farmers=[FarmerRecord(farmerName="Rameshbhai", tagNumbers="1001")],
                source="cache",
            )

        def _signed_in_run_stream(**kwargs):
            captured["request_limit"] = kwargs["usage_limits"].request_limit
            return _FakeResponseStream(
                chunks=["ગાયને તાવ છે તો પશુચિકિત્સકનો સંપર્ક કરો."],
                new_messages=_make_agent_messages("My cow has fever.", "Contact a veterinarian for the cow's fever."),
            )

        def _unexpected_base_run_stream(**kwargs):
            raise AssertionError("anonymous agent should not be used for signed-in session")

        monkeypatch.setattr(voice_module, "get_or_fetch_farmer_data", _fake_farmer_data)
        async def _pretranslate(*args, **kwargs):
            return "My cow has fever.", "high"
        monkeypatch.setattr(voice_module, "translate_to_english_with_gpt5_mini", _pretranslate)
        from agents import voice as voice_agent_module
        monkeypatch.setattr(voice_agent_module.voice_agent, "run_stream", _unexpected_base_run_stream)
        monkeypatch.setattr(voice_agent_module.voice_agent_signed_in, "run_stream", _signed_in_run_stream)

        output, _ = asyncio.run(
            _collect_stream(
                query="મારી ગાયને તાવ છે",
                session_id="signed-in-agent-selection",
                history=[],
                monkeypatch=monkeypatch,
                response_stream=_FakeResponseStream(),
                source_lang="gu",
                target_lang="gu",
                user_id="9723293369",
                normalize_phone_override=lambda user_id: user_id,
                farmer_data_override=_fake_farmer_data,
            )
        )

        assert output
        assert captured["request_limit"] == 6

    def test_repeated_stt_failures_hit_retry_ceiling_on_third_attempt(self, monkeypatch):
        from app.services import voice as voice_module

        history: list = []
        outputs: list[str] = []
        final_flags: list[bool] = []

        async def _fake_generate(signal: str, target_lang: str, recent_history_text: str = "", final_attempt: bool = False) -> str:
            final_flags.append(final_attempt)
            if final_attempt:
                return "માફ કરશો, હજુ તમારો અવાજ સંભળાતો નથી. કૃપા કરીને પછીથી ફરી પ્રયાસ કરો."
            return "માફ કરશો, મને તમારો અવાજ સંભળાતો નથી. કૃપા કરીને ફરીથી બોલો."

        monkeypatch.setattr(voice_module, "generate_stt_signal_response", _fake_generate)

        for _ in range(3):
            output, history = asyncio.run(
                _collect_stream(
                    query="No audio/User is speaking softly",
                    session_id="multiturn-stt-ceiling",
                    history=history,
                    monkeypatch=monkeypatch,
                    response_stream=_FakeResponseStream(),
                )
            )
            outputs.append(output)

        assert final_flags == [False, False, True]
        assert "ફરીથી" in outputs[0]
        assert "ફરીથી" in outputs[1]
        assert "પછીથી ફરી પ્રયાસ કરો" in outputs[2] or "થોડા સમય પછી ફરી કોલ કરો" in outputs[2]
        history_text = " ".join(
            getattr(part, "content", "")
            for msg in history
            for part in getattr(msg, "parts", [])
            if isinstance(getattr(part, "content", None), str)
        )
        assert "[stt:no-audio]" in history_text
        assert "No audio/User is speaking softly" not in history_text

    def test_tool_triggered_nudge_fires_once_before_first_chunk(self, monkeypatch):
        nudges: list[str] = []
        tool_event_box: dict = {}

        async def _trigger_tool_event():
            await asyncio.sleep(0)
            tool_event_box["event"].set()

        response_stream = _FakeResponseStream(
            chunks=["I will check and tell you."],
            delay=0.03,
            on_enter=_trigger_tool_event,
        )

        output, _ = asyncio.run(
            _collect_stream(
                query="What should I do for my cow?",
                session_id="multiturn-tool-nudge",
                history=[],
                monkeypatch=monkeypatch,
                response_stream=response_stream,
                source_lang="en",
                target_lang="en",
                nudges=nudges,
                tool_event_box=tool_event_box,
            )
        )

        assert "I will check" in output
        assert len(nudges) == 1
        assert "wait" in nudges[0].lower()

    def test_closing_turn_does_not_append_feedback_across_turns(self, monkeypatch):
        first_output, history = asyncio.run(
            _collect_stream(
                query="આભાર, બસ છે",
                session_id="multiturn-closing",
                history=[],
                monkeypatch=monkeypatch,
                response_stream=_FakeResponseStream(
                    chunks=["ચોક્કસ, આપના પશુ માટે હું અહીં છું."],
                    new_messages=_make_agent_messages("આભાર, બસ છે", "ચોક્કસ, આપના પશુ માટે હું અહીં છું."),
                ),
            )
        )

        assert "1 થી 5" not in first_output
        assert all("1 થી 5" not in getattr(part, "content", "") for msg in history for part in getattr(msg, "parts", []))

        second_output, second_history = asyncio.run(
            _collect_stream(
                query="hello",
                session_id="multiturn-closing",
                history=history,
                monkeypatch=monkeypatch,
                response_stream=_FakeResponseStream(chunks=["ignored"]),
            )
        )

        assert "feedback" not in second_output.lower()
        assert all("કેટલો ઉપયોગી" not in getattr(part, "content", "") for msg in second_history for part in getattr(msg, "parts", []))
