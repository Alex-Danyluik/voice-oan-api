from typing import Optional, Literal
from pydantic import BaseModel, Field


class FarmerContext(BaseModel):
    """Context for the voice agent.

    Args:
        query: The user's question.
        lang_code: The language code of the user's question.
        target_lang: The target language for the response.
        farmer_info: Pre-built markdown string with farmer profile, animals, vet visits.
        provider: The provider for the voice service.
        session_id: The session ID for the user.
        process_id: The process ID for tracking and hold messages.
    """
    query: str = Field(description="The user's question.")
    lang_code: str = Field(description="The language code of the user's question.", default='gu')
    target_lang: str = Field(description="The target language for the response (gu=Gujarati, en=English).", default='gu')
    provider: Optional[Literal['RAYA']] = Field(default=None, description="The provider for the voice service - can be RAYA or None.")
    session_id: Optional[str] = Field(default=None, description="The session ID for the user.")
    process_id: Optional[str] = Field(default=None, description="The process ID for tracking and hold messages.")
    farmer_info: str = Field(default="", description="Pre-built markdown farmer context string.")

    def _query_string(self):
        """Get the query string for the agrinet agent."""
        return "**User:** " + '"' + self.query + '"'

    def get_farmer_context_string(self) -> Optional[str]:
        """Return the pre-built farmer context markdown string."""
        return self.farmer_info if self.farmer_info else None

    def get_user_message(self):
        """Get the user message for the agrinet agent."""
        return self._query_string()
