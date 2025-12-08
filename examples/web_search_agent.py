"""Async multi-tool agent example using NimbleSearchTool and NimbleExtractTool.

This example demonstrates how to create an agent that can both search the web
and extract content from specific URLs using Nimble's API.

Requirements:
    pip install langchain-nimble langchain langchain-anthropic

Environment:
    export NIMBLE_API_KEY="your-api-key"
    export ANTHROPIC_API_KEY="your-anthropic-api-key"

Run:
    python examples/web_search_agent.py
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain.agents import create_agent

from langchain_nimble import NimbleExtractTool, NimbleSearchTool

# Load environment variables from .env file
load_dotenv()


async def main() -> None:
    """Run an async multi-tool web agent."""
    # Check for required API keys
    if not os.environ.get("NIMBLE_API_KEY"):
        msg = "NIMBLE_API_KEY environment variable is required"
        raise ValueError(msg)

    # Create both search and extract tools
    search_tool = NimbleSearchTool()
    extract_tool = NimbleExtractTool()

    # Create agent with system prompt and both tools
    # Using Claude Haiku 4.5 for fast, cost-effective performance
    agent = create_agent(
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

        # Invoke the agent asynchronously
        response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": query}]}
        )

        # Extract and print the final answer
        final_message = response["messages"][-1]
        print(f"\n🤖 Answer:\n{final_message.content}")
        print("-" * 80)


if __name__ == "__main__":
    asyncio.run(main())
