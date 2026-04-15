#!/usr/bin/env python3
"""
Direct-call TTFT benchmark for the voice pipeline.

This bypasses HTTP/JWT and measures the lower bound of the Python voice pipeline
on the current machine. It is intended for dev-server benchmarking.

Example:
  ./.venv/bin/python scripts/measure_voice_ttft.py --runs 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv()

from app.services import voice as voice_module


@dataclass
class Scenario:
    name: str
    query: str
    source_lang: str
    target_lang: str
    user_id: str = "anonymous"
    note: str = ""


SCENARIOS: dict[str, Scenario] = {
    "greeting_fast_path": Scenario(
        name="greeting_fast_path",
        query="hello",
        source_lang="en",
        target_lang="gu",
        note="Fast-path greeting; no agent expected.",
    ),
    "identity_en_en": Scenario(
        name="identity_en_en",
        query="What is your name?",
        source_lang="en",
        target_lang="en",
        note="Lower bound for direct English agent response.",
    ),
    "identity_gu_gu": Scenario(
        name="identity_gu_gu",
        query="તમારું નામ શું છે?",
        source_lang="gu",
        target_lang="gu",
        note="Pretranslation + agent + output translation.",
    ),
    "signed_in_identity_gu_gu": Scenario(
        name="signed_in_identity_gu_gu",
        query="તમારું નામ શું છે?",
        source_lang="gu",
        target_lang="gu",
        user_id="9723293369",
        note="Includes signed-in/cached farmer summary path.",
    ),
    "domain_en_en": Scenario(
        name="domain_en_en",
        query="My cow has fever. What should I do?",
        source_lang="en",
        target_lang="en",
        note="Direct English retrieval path.",
    ),
    "domain_gu_gu": Scenario(
        name="domain_gu_gu",
        query="મારી ગાયને તાવ છે, શું કરવું?",
        source_lang="gu",
        target_lang="gu",
        note="Full Gujarati retrieval path.",
    ),
}


async def _noop_nudge(*args: Any, **kwargs: Any) -> None:
    return None


async def _noop_update_history(*args: Any, **kwargs: Any) -> None:
    return None


async def _run_once(scenario: Scenario) -> dict[str, Any]:
    chunks: list[str] = []
    first_chunk_ms: float | None = None
    first_nonempty_chunk: str = ""
    started = time.perf_counter()

    async for chunk in voice_module.stream_voice_message(
        query=scenario.query,
        session_id=f"ttft-{scenario.name}-{uuid.uuid4().hex[:8]}",
        source_lang=scenario.source_lang,
        target_lang=scenario.target_lang,
        user_id=scenario.user_id,
        history=[],
        provider=None,
        process_id=f"ttft-{uuid.uuid4().hex[:8]}",
        user_info={},
        owner=None,
        http_request=None,
    ):
        if isinstance(chunk, str):
            if first_chunk_ms is None and chunk.strip():
                first_chunk_ms = (time.perf_counter() - started) * 1000.0
                first_nonempty_chunk = chunk.strip().replace("\n", " ")[:160]
            chunks.append(chunk)

    total_ms = (time.perf_counter() - started) * 1000.0
    output = "".join(chunks)
    return {
        "ttft_ms": first_chunk_ms,
        "total_ms": total_ms,
        "output_chars": len(output),
        "first_chunk_preview": first_nonempty_chunk,
    }


def _summarize(name: str, note: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    ttfts = [r["ttft_ms"] for r in results if r["ttft_ms"] is not None]
    totals = [r["total_ms"] for r in results]
    return {
        "scenario": name,
        "note": note,
        "runs": len(results),
        "ttft_ms": {
            "min": min(ttfts) if ttfts else None,
            "p50": statistics.median(ttfts) if ttfts else None,
            "max": max(ttfts) if ttfts else None,
        },
        "total_ms": {
            "min": min(totals),
            "p50": statistics.median(totals),
            "max": max(totals),
        },
        "sample_first_chunk": next((r["first_chunk_preview"] for r in results if r["first_chunk_preview"]), ""),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--scenario",
        action="append",
        choices=sorted(SCENARIOS.keys()),
        help="Scenario(s) to run. Defaults to all.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    args = parser.parse_args()

    selected = args.scenario or list(SCENARIOS.keys())

    original_nudge_fn = voice_module.send_nudge_message_raya
    original_update_history = voice_module.update_message_history
    original_timeout = voice_module.settings.nudge_timeout_seconds
    voice_module.send_nudge_message_raya = _noop_nudge
    voice_module.update_message_history = _noop_update_history
    voice_module.settings.nudge_timeout_seconds = 999.0

    try:
        all_results: list[dict[str, Any]] = []
        for scenario_name in selected:
            scenario = SCENARIOS[scenario_name]
            results: list[dict[str, Any]] = []
            for _ in range(args.runs):
                results.append(await _run_once(scenario))
            all_results.append(_summarize(scenario.name, scenario.note, results))
    finally:
        voice_module.send_nudge_message_raya = original_nudge_fn
        voice_module.update_message_history = original_update_history
        voice_module.settings.nudge_timeout_seconds = original_timeout

    if args.json:
        print(json.dumps(all_results, indent=2, ensure_ascii=False))
        return

    print("Voice TTFT benchmark")
    print("====================")
    for row in all_results:
        print(f"\nScenario: {row['scenario']}")
        if row["note"]:
            print(f"Note: {row['note']}")
        ttft = row["ttft_ms"]
        total = row["total_ms"]
        print(
            "TTFT ms: min={min:.1f} p50={p50:.1f} max={max:.1f}".format(
                min=ttft["min"] or -1,
                p50=ttft["p50"] or -1,
                max=ttft["max"] or -1,
            )
        )
        print(
            "Total ms: min={min:.1f} p50={p50:.1f} max={max:.1f}".format(
                min=total["min"],
                p50=total["p50"],
                max=total["max"],
            )
        )
        print(f"Sample first chunk: {row['sample_first_chunk']}")


if __name__ == "__main__":
    asyncio.run(main())
