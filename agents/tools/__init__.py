"""
Tools for the Sunbird VA API.
"""
import functools
import inspect

from pydantic_ai import Tool
from agents.tools.terms import search_terms
from agents.tools.search import search_documents
from agents.tools.ai_call import create_ai_call
from agents.tools.conversation_state import signal_conversation_state
from agents.tools.farmer_cached import get_farmer_profile, get_herd_summary, list_animal_tags
from agents.tools.common import fire_tool_call_nudge


def _with_nudge_signal(func):
    """Wrap a tool function so it fires the nudge event on invocation."""
    if inspect.iscoroutinefunction(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            fire_tool_call_nudge()
            return await func(*args, **kwargs)
        return wrapper
    else:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            fire_tool_call_nudge()
            return func(*args, **kwargs)
        return wrapper


BASE_TOOLS = [
    Tool(
        _with_nudge_signal(search_terms),
        takes_ctx=False,
    ),
    Tool(
        _with_nudge_signal(search_documents),
        takes_ctx=True,
    ),
    Tool(
        _with_nudge_signal(create_ai_call),
        takes_ctx=True,
        docstring_format='auto',
        require_parameter_descriptions=True,
    ),
    Tool(
        signal_conversation_state,
        takes_ctx=True,
        docstring_format='auto',
    ),
]

SIGNED_IN_FARMER_TOOLS = [
    Tool(
        _with_nudge_signal(get_farmer_profile),
        takes_ctx=True,
        docstring_format='auto',
        require_parameter_descriptions=False,
    ),
    Tool(
        _with_nudge_signal(get_herd_summary),
        takes_ctx=True,
        docstring_format='auto',
        require_parameter_descriptions=False,
    ),
    Tool(
        _with_nudge_signal(list_animal_tags),
        takes_ctx=True,
        docstring_format='auto',
        require_parameter_descriptions=False,
    ),
]

TOOLS = BASE_TOOLS
