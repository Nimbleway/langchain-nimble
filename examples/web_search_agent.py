"""Async multi-tool agent example using NimbleSearchTool and NimbleExtractTool.

This example demonstrates how to create an agent that can both search the web
and extract content from specific URLs using Nimble's API.

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

from langchain_nimble import NimbleExtractTool, NimbleSearchTool

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

    # Create both search and extract tools
    search_tool = NimbleSearchTool()
    extract_tool = NimbleExtractTool()

    # Create agent with system prompt and both tools
    # Using Claude Haiku 4.5 for fast, cost-effective performance
    agent: Any = create_agent(
        model="claude-haiku-4-5",
        tools=[search_tool, extract_tool],
        system_prompt=(
            "You are a helpful assistant with access to real-time web "
            "information. You can search the web and extract content from "
            "specific URLs. Use the search tool to find relevant information, "
            "then use the extract tool to get detailed content from specific "
            "pages when needed. Always cite your sources and provide "
            "comprehensive, accurate answers."
        ),
    )

    # Use custom question if provided, otherwise use sample queries
    if args.question:
        queries = [args.question]
    else:
        # Example queries demonstrating both tools
        queries = [
            "What are the latest developments in artificial intelligence?",
            (
                "Find the official Python 3.13 release notes and summarize "
                "the key new features"
            ),
            (
                "Search for the LangChain documentation and extract the main "
                "concepts from the homepage"
            ),
        ]

    print("=" * 80)
    print("Multi-Tool Web Agent Example (Search + Extract)")
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
