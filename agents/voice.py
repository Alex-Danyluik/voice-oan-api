import os
from pydantic_ai import Agent
from helpers.utils import get_prompt, get_logger
from dotenv import load_dotenv
from agents.models import LLM_MODEL
from agents.tools import BASE_TOOLS, SIGNED_IN_FARMER_TOOLS
from pydantic_ai.settings import ModelSettings
from agents.deps import FarmerContext

logger = get_logger(__name__)

load_dotenv()

VOICE_SYSTEM_PROMPT_NAME = "voice_system_translation_pipeline_en"
STATIC_VOICE_SYSTEM_PROMPT = get_prompt(
    VOICE_SYSTEM_PROMPT_NAME,
    context={
        "today_date": "Provided in runtime context message.",
        "farmer_context": None,
    },
)

def _build_voice_agent(name: str, tools):
    return Agent(
        model=LLM_MODEL,
        name=name,
        instrument=True,
        output_type=str,
        deps=FarmerContext,
        retries=3,
        tools=tools,
        system_prompt=STATIC_VOICE_SYSTEM_PROMPT,
        end_strategy='early',
        model_settings=ModelSettings(
            max_tokens=3600,
            temperature=0.0,
            parallel_tool_calls=False,
       )
    )


voice_agent = _build_voice_agent("Voice Agent", BASE_TOOLS)
voice_agent_signed_in = _build_voice_agent(
    "Voice Agent Signed In",
    [*BASE_TOOLS, *SIGNED_IN_FARMER_TOOLS],
)
