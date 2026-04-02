from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import config
from config import settings
from tools import web_search, knowledge_search


class ResearchPlan(BaseModel):
    goal: str = Field(description="What we are trying to answer")
    search_queries: list[str] = Field(description="Specific queries to execute")
    sources_to_check: list[str] = Field(description="'knowledge_base', 'web_search', or both")
    output_format: str = Field(description="What the final report should look like")


planner_agent = create_agent(
    model=ChatOpenAI(
        model=settings.model_name,
        api_key=settings.api_key,
    ),
    tools=[web_search, knowledge_search],
    system_prompt=config.PLANNER_SYSTEM_PROMPT,
    response_format=ResearchPlan,
    middleware=[
        ModelCallLimitMiddleware(
            run_limit=settings.max_iterations
        )
    ],
)
# result["structured_response"] → validated ResearchPlan instance
