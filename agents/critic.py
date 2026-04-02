from datetime import date
from typing import Literal

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import config
from config import settings
from tools import web_search, read_url, knowledge_search


class CritiqueResult(BaseModel):
    verdict: Literal["APPROVE", "REVISE"]
    is_fresh: bool = Field(description="Is the data up-to-date and based on recent sources?")
    is_complete: bool = Field(description="Does the research fully cover the user's original request?")
    is_well_structured: bool = Field(description="Are findings logically organized and ready for a report?")
    strengths: list[str] = Field(description="What is good about the research")
    gaps: list[str] = Field(description="What is missing, outdated, or poorly structured")
    revision_requests: list[str] = Field(description="Specific things to fix if verdict is REVISE")


critic_agent = create_agent(
    model=ChatOpenAI(
        model=settings.model_name,
        api_key=settings.api_key,
    ),
    tools=[web_search, read_url, knowledge_search],
    system_prompt=config.CRITIC_SYSTEM_PROMPT.replace("{today}", date.today().strftime("%B %d, %Y")),
    response_format=CritiqueResult,
    middleware=[
        ModelCallLimitMiddleware(
            run_limit=settings.max_iterations
        )
    ],
)
# result["structured_response"] → validated CritiqueResult instance
