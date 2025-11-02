"""Example demonstrating UCI ML Repository tools with the ChatAgent."""
import asyncio
import sys
from src.mlagentfactory.agents.chat_agent import ChatAgent


async def main():
    """Run UCI ML Repository example."""
    print("=" * 80)
    print("UCI ML Repository Tools Example")
    print("=" * 80)
    print()

    # Create and initialize the agent
    agent = ChatAgent()
    await agent.initialize()

    try:
        # Example 1: Get information about the Iris dataset
        print("\n" + "=" * 80)
        print("Example 1: Get information about the Iris dataset (ID: 53)")
        print("=" * 80)
        print("\nSending query to agent...")

        query1 = "Get detailed information about the Iris dataset from UCI ML Repository using dataset ID 53"

        async for chunk in agent.send_message(query1):
            if chunk["type"] == "text":
                print(chunk["content"], end="", flush=True)
            elif chunk["type"] == "tool_use":
                print(f"\n[Tool: {chunk['tool_name']}]", flush=True)
            elif chunk["type"] == "tool_result" and not chunk.get("is_error"):
                # Tool results are handled internally, just show a checkmark
                print(" ✓", flush=True)

        print("\n")

        # Example 2: Fetch and save the Iris dataset
        print("\n" + "=" * 80)
        print("Example 2: Fetch and save the Iris dataset to CSV files")
        print("=" * 80)
        print("\nSending query to agent...")

        query2 = "Fetch the Iris dataset (ID: 53) and save it to CSV files in ./uci_datasets/iris/"

        async for chunk in agent.send_message(query2):
            if chunk["type"] == "text":
                print(chunk["content"], end="", flush=True)
            elif chunk["type"] == "tool_use":
                print(f"\n[Tool: {chunk['tool_name']}]", flush=True)
            elif chunk["type"] == "tool_result" and not chunk.get("is_error"):
                print(" ✓", flush=True)

        print("\n")

        # Example 3: List available datasets
        print("\n" + "=" * 80)
        print("Example 3: List available UCI ML Repository datasets")
        print("=" * 80)
        print("\nSending query to agent...")

        query3 = "List some available datasets from UCI ML Repository, particularly those related to classification tasks"

        async for chunk in agent.send_message(query3):
            if chunk["type"] == "text":
                print(chunk["content"], end="", flush=True)
            elif chunk["type"] == "tool_use":
                print(f"\n[Tool: {chunk['tool_name']}]", flush=True)
            elif chunk["type"] == "tool_result" and not chunk.get("is_error"):
                print(" ✓", flush=True)

        print("\n")

        # Show total cost
        if agent.total_cost > 0:
            print(f"\nTotal cost: ${agent.total_cost:.4f}")

    finally:
        # Cleanup
        await agent.cleanup()

    print("\n" + "=" * 80)
    print("Example completed!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
