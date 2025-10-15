"""
Example script demonstrating the use of Kaggle tools with Claude Agent.

This script shows how to create an agent that can interact with Kaggle
to download datasets, list competitions, and make submissions.
"""

import asyncio
import logging

from src.mlagentfactory.agents.chat_agent import ChatAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def main():
    """Run the Kaggle agent example."""
    # Create chat agent
    agent = ChatAgent()

    try:
        # Initialize the agent
        await agent.initialize()
        logger.info("Agent initialized successfully")

        # Example 1: List available competitions
        print("\n" + "="*80)
        print("Example 1: Listing Kaggle Competitions")
        print("="*80)
        response = await agent.chat("List Kaggle competitions related to 'titanic'")
        print(f"\nAgent: {response}\n")

        # Example 2: Download a dataset
        print("\n" + "="*80)
        print("Example 2: Downloading a Kaggle Dataset")
        print("="*80)
        response = await agent.chat(
            "Download the titanic dataset from Kaggle to the './data/titanic' directory"
        )
        print(f"\nAgent: {response}\n")

        # Example 3: Check competition details
        print("\n" + "="*80)
        print("Example 3: Checking Competition Leaderboard")
        print("="*80)
        response = await agent.chat("Show me the leaderboard for the titanic competition")
        print(f"\nAgent: {response}\n")

    except Exception as e:
        logger.error(f"Error running agent: {e}")
        raise
    finally:
        # Clean up
        await agent.cleanup()
        logger.info("Agent cleaned up")


if __name__ == "__main__":
    asyncio.run(main())
