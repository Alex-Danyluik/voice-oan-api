import asyncio
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.voice import stream_voice_message
from app.utils import _get_message_history, claim_session_request_ownership


async def test():
    session_id = "test-goodbye-flow-3"
    phone = "9924457046"

    # Turn 1: greeting
    print("Turn 1: sending hello...", flush=True)
    owner = await claim_session_request_ownership(session_id)
    history = await _get_message_history(session_id)
    chunks = []
    async for chunk in stream_voice_message(
        query="hello", session_id=session_id, source_lang="gu", target_lang="gu",
        user_id=phone, history=history, provider="RAYA", process_id="1",
        user_info={"user_id": phone}, owner=owner, http_request=None,
    ):
        chunks.append(chunk if isinstance(chunk, str) else "")
    resp1 = "".join(chunks).strip()
    print(f"Turn 1: {resp1}", flush=True)
    print(flush=True)

    # Turn 2: bye in Gujarati
    print("Turn 2: sending 'no I dont want anything bye'...", flush=True)
    owner = await claim_session_request_ownership(session_id)
    history = await _get_message_history(session_id)
    chunks = []
    async for chunk in stream_voice_message(
        query="no I dont want anything bye",
        session_id=session_id, source_lang="en", target_lang="gu",
        user_id=phone, history=history, provider="RAYA", process_id="2",
        user_info={"user_id": phone}, owner=owner, http_request=None,
    ):
        chunks.append(chunk if isinstance(chunk, str) else "")
    resp2 = "".join(chunks).strip()
    print(f"Turn 2: {resp2}", flush=True)
    print(flush=True)
    print(f"Contains 'Goodbye': {'Goodbye' in resp2}", flush=True)


asyncio.run(test())
