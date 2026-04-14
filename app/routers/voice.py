from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from app.auth.jwt_auth import get_current_user
from app.config import settings
from app.services.voice import stream_voice_message
from app.utils import _get_message_history, claim_session_request_ownership
from app.models.requests import ChatRequest
from helpers.utils import get_logger
import uuid

logger = get_logger(__name__)

router = APIRouter(prefix="/voice", tags=["voice"])

@router.get("/")
async def voice_endpoint(
    http_request: Request,
    request: ChatRequest = Depends(),
    user_info: dict = Depends(get_current_user),
):
    """
    Voice endpoint that streams responses back to the client.
    Requires JWT authentication (Authorization: Bearer <token>).
    JWT is validated using the public key from JWT_PUBLIC_KEY env or JWT_PUBLIC_KEY_PATH file.
    session_id is used for message history and Langfuse Sessions: same ID groups all agent runs for one conversation.
    """
    session_id = request.session_id or str(uuid.uuid4())
    logger.info(
        f"Voice request received - session_id: {session_id}, user_id: {request.user_id}, "
        f"source_lang: {request.source_lang}, "
        f"target_lang: {request.target_lang}, provider: {request.provider}, process_id: {request.process_id}, "
        f"query: {request.query}"
    )
    owner = await claim_session_request_ownership(session_id)
    logger.info(
        "Session ownership claimed - session_id=%s epoch=%s token=%s process_id=%s",
        session_id,
        owner.epoch,
        owner.request_token,
        request.process_id,
    )

    history = await _get_message_history(session_id)
    logger.debug(f"Retrieved message history for session {session_id} - length: {len(history)}")

    return StreamingResponse(
        stream_voice_message(
            query=request.query,
            session_id=session_id,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            user_id=request.user_id,
            history=history,
            provider=request.provider,
            process_id=request.process_id,
            user_info=user_info,
            owner=owner,
            http_request=http_request,
        ),
        media_type='text/event-stream'
    ) 
