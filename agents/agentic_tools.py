from langchain_core.tools import tool

from agents.critic import critic_agent
from agents.planner import planner_agent
from agents.research import research_agent
from schemas import ResearchPlan


@tool
def research(request: str) -> str:
    """Execute research based on a plan or request. Returns comprehensive findings.
    Args: request: research plan or specific instructions for what to investigate"""
    result = research_agent.invoke({"messages": [("user", request)]})
    messages = result.get("messages", [])
    return messages[-1].content if messages else "No findings returned."


@tool
def critique(findings: str) -> str:
    """Evaluate research findings for freshness, completeness, and structure.
    Args: findings: the research findings text to evaluate"""

    try:
        result = critic_agent.invoke({"messages": [("user", findings)]})
        return result["messages"][-1].content
    except Exception as e:
        print(f"Error during critique for: {findings}, error: {e}")
        return f"Error during critique for: {findings}"


@tool
def plan(request: str) -> str:
    """Decompose a research request into a structured ResearchPlan.
    Args: request: the user's original research request"""
    result = planner_agent.invoke({"messages": [("user", request)]})
    research_plan: ResearchPlan = result["structured_response"]
    return research_plan.model_dump_json(indent=2)
