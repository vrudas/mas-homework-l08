from langchain_core.tools import tool

from agents.critic import critic_agent
from agents.planner import planner_agent
from agents.research import research_agent


@tool
def research(request: str) -> str:
    """Perform research based on the user request
    Args: request: user research request"""

    try:
        result = research_agent.invoke({"messages": [{"role": "user", "content": request}]})
        return result["messages"][-1].content
    except Exception as e:
        print(f"Error during research for: {request}, error: {e}")
        return f"Error during research for: {request}"


@tool
def critique(findings: str) -> str:
    """Critique the findings by verifying and evaluating them
    Args: findings: findings string"""

    try:
        result = critic_agent.invoke({"messages": [{"role": "user", "content": findings}]})
        return result["messages"][-1].content
    except Exception as e:
        print(f"Error during critique for: {findings}, error: {e}")
        return f"Error during critique for: {findings}"


@tool
def plan(request: str) -> str:
    """Analyze a user's research request and decompose it into a plan.
    Args: request: user research request"""

    try:
        result = planner_agent.invoke(
            {"messages": [{"role": "user", "content": request}]}
        )

        return result["messages"][-1].content
    except Exception as e:
        print(f"Error during planning for: {request}, error: {e}")
        return f"Error during planning for: {request}"
