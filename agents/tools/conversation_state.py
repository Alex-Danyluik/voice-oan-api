"""
Conversation state signaling tool.

Lets the LLM tell the telephony layer when the call is wrapping up
or when the caller seems frustrated, so RAYA can handle the session
lifecycle (e.g. graceful hangup after closing).
"""
from typing import Literal
from pydantic_ai import RunContext
from helpers.utils import get_logger
from agents.deps import FarmerContext

logger = get_logger(__name__)


def signal_conversation_state(
    ctx: RunContext[FarmerContext],
    event: Literal["conversation_closing", "user_frustration"],
) -> str:
    """
    Signal the current conversation state to the telephony system.

    Call conversation_closing when:
    - The farmer's question has been answered and they decline further help
    - The farmer says goodbye, thanks, or indicates they are done
    - You have delivered the closing line

    Call user_frustration when:
    - The farmer corrects you or says that is not what they meant
    - The farmer repeats the same request after you already answered
    - The farmer sounds confused or unhappy with the response

    Args:
        ctx: Run context with session info
        event: One of conversation_closing, user_frustration

    Returns:
        Confirmation string (not shown to user)
    """
    logger.info(
        "Conversation state signaled: event=%s session_id=%s",
        event,
        ctx.deps.session_id,
    )
    return f"State {event} recorded."
