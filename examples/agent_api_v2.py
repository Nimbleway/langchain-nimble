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
            "Prefer Mode 1 for this stateless session: call "
            "nimble_web_search_agent_run_start with agent_name + use_case="
            "research + effort=medium (+ optional skill/sources). "
            "Or Mode 2 with a persisted agent_id / create tool. "
            "Mode 3: omit both agent_id and agent_name.\n\n"
            "Workflow:\n"
            "1. Start a run (returns immediately; often 3-15 minutes total)\n"
            "2. Map id -> run_id, web_search_agent_id -> agent_id\n"
            "3. nimble_web_search_agent_run_status until completed/failed\n"
            "4. nimble_web_search_agent_run_result for text/json + trust\n\n"
            "use_case is locked after create — do not switch it on reuse. "
            "Never pretend a run finished before status says so."
        ),
    )

    query = args.question or (
        "Using Mode 1 (agent_name create-or-reuse), start a medium-effort "
        "research run about retrieval-augmented generation evaluation, "
        "poll status across turns, and return the result with citations "
        "when ready."
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
