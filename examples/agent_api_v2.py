"""Agent API V2 research example using resumable start/status/result tools.

This example shows the Web Search Agent lifecycle (distinct from Extract
Templates). Long runs are resumable across turns — start returns immediately.

Requirements:
    pip install langchain-nimble langchain langchain-anthropic

Environment:
    export NIMBLE_API_KEY="your-api-key"
    export ANTHROPIC_API_KEY="your-anthropic-api-key"

Run:
    python examples/agent_api_v2.py
    python examples/agent_api_v2.py "Compare open-source LLM evaluation tools"
"""

import argparse
import asyncio
import os
import time
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent

from langchain_nimble import NimbleToolkit

load_dotenv()


async def main() -> None:
    """Run an async Agent API V2 research agent."""
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description="Run a Nimble Agent API V2 research workflow"
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Research question for the Web Search Agent (optional)",
    )
    args = parser.parse_args()

    if not os.environ.get("NIMBLE_API_KEY"):
        msg = "NIMBLE_API_KEY environment variable is required"
        raise ValueError(msg)

    toolkit = NimbleToolkit(
        include_search=False,
        include_extract=False,
        include_web_search_agents=True,
    )
    tools = toolkit.get_tools()

    agent: Any = create_agent(
        model="claude-haiku-4-5",
        tools=tools,
        system_prompt=(
            "You can run Nimble Web Search Agents (Agent API V2).\n\n"
            "Workflow:\n"
            "1. nimble_web_search_agents_list and/or "
            "nimble_web_search_agent_templates_list to discover\n"
            "2. nimble_web_search_agent_create if you need a new agent "
            "from a template\n"
            "3. nimble_web_search_agent_run_start — returns immediately; "
            "use id as run_id and web_search_agent_id as agent_id\n"
            "4. nimble_web_search_agent_run_status — check "
            "queued/running/completed\n"
            "5. nimble_web_search_agent_run_result — fetch the finished "
            "result\n\n"
            "Never pretend a run is finished before status says so. "
            "Preserve agent_id and run_id across steps."
        ),
    )

    query = args.question or (
        "Use a Nimble Web Search Agent to research recent developments in "
        "retrieval-augmented generation evaluation. List agents first, start "
        "a run, check status, and return the result when ready."
    )

    print("=" * 80)
    print("Nimble Agent API V2 Example")
    print(f"Tools loaded: {', '.join(t.name for t in tools)}")
    print("=" * 80)
    print(f"\n\n📝 Query: {query}")
    print("-" * 80)

    async for step in agent.astream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()

    print("-" * 80)
    elapsed_time = time.time() - start_time
    print(f"\n\n{'=' * 80}")
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    asyncio.run(main())
