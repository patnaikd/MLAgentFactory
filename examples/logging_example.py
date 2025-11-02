"""Example demonstrating how to record and replay ChatAgent interactions.

This example shows:
1. Recording interactions to JSONL log files using LoggingClaudeSDKClient
2. Replaying interactions from log files using FakeClaudeSDKClient
"""
import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlagentfactory.agents.chat_agent import ChatAgent


async def record_session():
    """Record a chat session to a JSONL log file."""
    print("=" * 60)
    print("RECORDING SESSION")
    print("=" * 60)

    # Create agent with logging enabled
    agent = ChatAgent(
        enable_logging=True,
        log_dir="./data/claude_logs",
        session_prefix="example_session"
    )

    await agent.initialize()

    try:
        # Send a simple message
        print("\nUser: What is 2 + 2?")
        response_parts = []

        async for chunk in agent.send_message("What is 2 + 2?"):
            if chunk["type"] == "text":
                response_parts.append(chunk["content"])

        full_response = "".join(response_parts)
        print(f"\nAgent: {full_response}")

        # Check if log file was created
        log_dir = Path("./data/claude_logs")
        log_files = sorted(log_dir.glob("example_session_*.jsonl"))

        if log_files:
            latest_log = log_files[-1]
            print(f"\n✓ Interaction logged to: {latest_log}")
            print(f"  Total cost: ${agent.total_cost:.4f}")
            print(f"  Session ID: {agent.session_id}")

            return str(latest_log)
        else:
            print("\n✗ No log file found!")
            return None

    finally:
        await agent.cleanup()


async def replay_session(log_file: str):
    """Replay a chat session from a JSONL log file."""
    print("\n" + "=" * 60)
    print("REPLAYING SESSION")
    print("=" * 60)
    print(f"Log file: {log_file}\n")

    # Create agent with replay enabled
    agent = ChatAgent(replay_log_file=log_file)

    await agent.initialize()

    try:
        # Send the same message - should replay from log
        print("User: What is 2 + 2?")
        response_parts = []

        async for chunk in agent.send_message("What is 2 + 2?"):
            if chunk["type"] == "text":
                response_parts.append(chunk["content"])
            elif chunk["type"] == "tool_use":
                print(f"  [Tool: {chunk['tool_name']}]")
            elif chunk["type"] == "session_id":
                print(f"  [Session ID: {chunk['content']}]")

        full_response = "".join(response_parts)
        print(f"\nAgent (replayed): {full_response}")

        print(f"\n✓ Successfully replayed session")
        print(f"  Total cost (from log): ${agent.total_cost:.4f}")

    finally:
        await agent.cleanup()


async def main():
    """Run the recording and replay demo."""
    print("Claude Agent Recording & Replay Demo")
    print("=" * 60)

    # Step 1: Record a session
    log_file = await record_session()

    if log_file:
        # Step 2: Replay the session
        await replay_session(log_file)

        # Show log file contents
        print("\n" + "=" * 60)
        print("LOG FILE CONTENTS (first 10 lines)")
        print("=" * 60)

        with open(log_file, 'r') as f:
            for i, line in enumerate(f, 1):
                if i > 10:
                    print("...")
                    break
                print(f"{i:3d}: {line.rstrip()[:100]}...")

    else:
        print("\n✗ Recording failed - skipping replay")


if __name__ == "__main__":
    asyncio.run(main())
