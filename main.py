import uuid
from typing import Any

from agents.research import agent


def main():
    print("Research Agent with RAG (type 'exit' to quit)")
    print("-" * 40)

    # Create a unique thread ID for this conversation session
    thread_id = str(uuid.uuid4())

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        for chunk in agent.stream(
                {"messages": [("user", user_input)]},
                config={"configurable": {"thread_id": thread_id}}
        ):
            print_tool_calls_from_model_output(chunk)
            print_tool_results_output(chunk)


def print_tool_calls_from_model_output(chunk):
    if "model" in chunk and "messages" in chunk["model"]:
        for msg in chunk["model"]["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for call in msg.tool_calls:
                    args_str = ", ".join(f'{k}="{v}"' for k, v in call["args"].items())
                    print(f"🔧 Tool call: {call['name']}({args_str})")

            if hasattr(msg, "content") and msg.content:
                print(f"\nAgent: {msg.content}\n")


def print_tool_results_output(chunk):
    if "tools" in chunk and "messages" in chunk["tools"]:
        for msg in chunk["tools"]["messages"]:
            content = msg.content if hasattr(msg, "content") else str(msg)

            lines = content_to_lines(content)

            if lines and len(lines) > 1:
                print(f"📎 Result: {truncate_content(lines[0])}")
                for line in lines[1:4]:
                    if line.strip():
                        print(f"   - {truncate_content(line.strip())}")
                print(f"   ...")
            else:
                print(f"📎 Result: {truncate_content(content)}")
            print()


def content_to_lines(content: str | Any) -> list[str] | list[Any]:
    return content.strip().splitlines() if isinstance(content, str) else []

def truncate_content(content: str | Any) -> str | Any:
    return content[:200] + "..." if len(content) > 200 else content


if __name__ == "__main__":
    main()
