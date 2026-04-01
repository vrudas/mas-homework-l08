from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from config import settings, SYSTEM_PROMPT
from tools import web_search, write_report, read_url, knowledge_search

llm = ChatOpenAI(
    model=settings.model_name,
    api_key=settings.api_key,
)

tools = [web_search, read_url, write_report, knowledge_search]

memory = InMemorySaver()

agent = create_agent(
    model=llm,
    tools=tools,
    middleware=[
        ModelCallLimitMiddleware(
            run_limit=settings.max_iterations
        )
    ],
    checkpointer=memory,
    system_prompt=SYSTEM_PROMPT,
)
