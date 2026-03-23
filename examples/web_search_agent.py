"""Async multi-tool agent example using NimbleToolkit.

This example demonstrates how to create an agent with all Nimble tools
loaded via the NimbleToolkit.

Requirements:
    pip install langchain-nimble langchain langchain-anthropic

Environment:
    export NIMBLE_API_KEY="your-api-key"
    export ANTHROPIC_API_KEY="your-anthropic-api-key"

Run:
    # Run with sample queries
    python examples/web_search_agent.py

    # Run with a custom question
    python examples/web_search_agent.py "What are the latest AI trends?"
"""

import argparse
import asyncio
import os
import time
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent

from langchain_nimble import NimbleToolkit

# Load environment variables from .env file
load_dotenv()


async def main() -> None:
    """Run an async multi-tool web agent."""
    # Start timing
    start_time = time.time()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run a web search agent with a custom question or sample queries"
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="A single question to ask the agent (optional)",
    )
    args = parser.parse_args()

    # Check for required API keys
    if not os.environ.get("NIMBLE_API_KEY"):
        msg = "NIMBLE_API_KEY environment variable is required"
        raise ValueError(msg)

    # Create all Nimble tools via the toolkit
    toolkit = NimbleToolkit(
        include_crawl=True,
        include_map=True,
        include_agent=True,
    )
    tools = toolkit.get_tools()

    # Create agent with system prompt and all tools
    # Using Claude Haiku 4.5 for fast, cost-effective performance
    agent: Any = create_agent(
        model="claude-haiku-4-5",
        tools=tools,
        system_prompt=(
            "You are a helpful assistant with access to Nimble's web data tools.\n\n"
            "Available tools:\n"
            "- nimble_search: Search the web for information\n"
            "- nimble_extract: Extract full content from a URL as markdown\n"
            "- nimble_map: Discover all URLs on a website\n"
            "- nimble_crawl: Crawl a website to extract multiple pages\n"
            "- nimble_agent_list: List available Nimble agent templates\n"
            "- nimble_agent_get: Get an agent's required parameters\n"
            "- nimble_agent_run: Run an agent for structured data extraction\n\n"
            "For agents: use nimble_agent_list to discover agents, "
            "nimble_agent_get to check required params, then nimble_agent_run.\n\n"
            "Always cite your sources and provide comprehensive, accurate answers."
        ),
    )

    # Use custom question if provided, otherwise use sample queries
    if args.question:
        queries = [args.question]
    else:
        # Example queries demonstrating various tools
        queries = [
            "What are the latest developments in artificial intelligence?",
            (
                "Find the official Python 3.13 release notes and summarize "
                "the key new features"
            ),
            (
                "List the available Nimble agents and show me what parameters "
                "the amazon_pdp agent requires"
            ),
        ]

    print("=" * 80)
    print("Nimble Toolkit Agent Example")
    print(f"Tools loaded: {', '.join(t.name for t in tools)}")
    print("=" * 80)

    # Run the agent with example queries
    for query in queries:
        print(f"\n\n📝 Query: {query}")
        print("-" * 80)

        # Stream the agent's response for real-time output
        async for step in agent.astream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
        ):
            step["messages"][-1].pretty_print()

        print("-" * 80)

    # Print total execution time
    elapsed_time = time.time() - start_time
    print(f"\n\n{'=' * 80}")
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    asyncio.run(main())
