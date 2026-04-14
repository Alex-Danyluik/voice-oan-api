import os
from pydantic_ai import Agent, RunContext
from datetime import datetime, timezone
from helpers.utils import get_prompt, get_today_date_str, get_logger
from dotenv import load_dotenv
from agents.models import LLM_MODEL
from agents.tools import TOOLS
from pydantic_ai.settings import ModelSettings
from agents.deps import FarmerContext

logger = get_logger(__name__)

load_dotenv()

voice_agent = Agent(
    model=LLM_MODEL,
    name="Voice Agent",
    instrument=True,
    output_type=str,
    deps=FarmerContext,
    retries=3,
    tools=TOOLS,
    end_strategy='early',
    model_settings=ModelSettings(
        max_tokens=3600,
        temperature=0.0,
        parallel_tool_calls=False,
   )
)

@voice_agent.system_prompt(dynamic=True)
def get_voice_system_prompt(ctx: RunContext[FarmerContext]) -> str:
    """Get the system prompt for the voice agent."""
    deps = ctx.deps
    farmer_context = deps.get_farmer_context_string()
    prompt_name = (
        "voice_system_translation_pipeline_en"
        if deps.use_translation_pipeline
        else f"voice_system_{deps.target_lang if deps.target_lang else 'gu'}"
    )
    logger.info("Voice prompt selected=%s translation_pipeline=%s", prompt_name, deps.use_translation_pipeline)
    return get_prompt(prompt_name, context={
        'today_date': get_today_date_str(),
        'farmer_context': farmer_context if farmer_context else None
        })
