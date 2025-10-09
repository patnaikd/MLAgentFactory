"""
Hello World File Creation Agent using Claude Agent SDK.

This agent demonstrates basic file creation capabilities by creating
a simple hello world text file when invoked.
"""

import asyncio
from pathlib import Path

from claude_agent_sdk import query, create_sdk_mcp_server, ClaudeAgentOptions

from tools import write_file


async def main():
    """Run the file creation agent."""

    # Create an SDK MCP server with our file writing tool
    file_server = create_sdk_mcp_server(
        name="file_tools",
        version="1.0.0",
        tools=[write_file]
    )

    # Configure the agent with the MCP server
    options = ClaudeAgentOptions(
        mcp_servers={"files": file_server},
        allowed_tools=["write_file"],
        permission_mode="bypassPermissions"  # Allow tools to execute automatically
    )

    # Define the task for the agent
    task = """
    Create a new file called 'hello_world.txt' in the current directory
    with the content consisting of a short poem about saying hello to the world.
    """

    print("Starting File Creation Agent...")
    print(f"Task: {task.strip()}")
    print("-" * 50)

    # Run the agent with the task
    print("\nAgent Response:")
    async for message in query(prompt=task, options=options):
        if hasattr(message, 'content'):
            for block in message.content:
                if hasattr(block, 'text'):
                    print(block.text)
    print("-" * 50)

    # Verify the file was created
    hello_file = Path("hello_world.txt")
    if hello_file.exists():
        print("\n✓ File created successfully!")
        print(f"Content: {hello_file.read_text()}")
    else:
        print("\n✗ File was not created")


if __name__ == "__main__":
    asyncio.run(main())
