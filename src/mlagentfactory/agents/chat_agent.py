"""Simple conversational agent using Claude Agent SDK with MCP servers."""

import asyncio
import logging
from typing import AsyncGenerator, Dict, List, Optional

from claude_agent_sdk import (
    ClaudeSDKClient,
    AssistantMessage,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    ClaudeAgentOptions
)

from ..utils.logging_config import initialize_observability
from ..tools import file_io_tools, web_fetch_tools

logger = logging.getLogger(__name__)


class ChatAgent:
    """A simple conversational agent that maintains conversation history."""

    def __init__(self):
        """Initialize the chat agent."""
        initialize_observability(log_level="DEBUG", enable_tracing=False)

        # Create MCP servers with tools
        self.file_server = create_sdk_mcp_server(
            name="file_tools",
            version="1.0.0",
            tools=[
                file_io_tools.read_file,
                file_io_tools.write_file,
                file_io_tools.edit_file,
                file_io_tools.delete_file,
                file_io_tools.list_directory,
                file_io_tools.create_directory,
                file_io_tools.remove_directory
            ]
        )

        self.web_server = create_sdk_mcp_server(
            name="web_tools",
            version="1.0.0",
            tools=[web_fetch_tools.fetch_webpage]
        )

        # Configure agent options
        self.options = ClaudeAgentOptions(
            mcp_servers={"files": self.file_server, "web": self.web_server},
            #allowed_tools=["Read", "Write", "Edit", "Delete", "List", "Create", "Remove", "Fetch", "Bash", "Calculate", "Python", "Search", "Ask", "Lookup", "Summarize"],
            allowed_tools=[
                "read_file",
                "write_file",
                "edit_file",
                "delete_file",
                "list_directory",
                "create_directory",
                "remove_directory",
                "fetch_webpage"
            ],
            permission_mode="bypassPermissions"
        )

        self.client: Optional[ClaudeSDKClient] = None
        self._initialized = False
        logger.info("ChatAgent initialized with file I/O and web tools")

    async def initialize(self):
        """Initialize the client. Call this once before using the agent."""
        if not self._initialized:
            self.client = ClaudeSDKClient(options=self.options)
            await self.client.connect()
            self._initialized = True
            logger.info("ChatAgent client initialized and connected")

    async def cleanup(self):
        """Cleanup the client. Call this when done with the agent."""
        if self._initialized and self.client:
            await self.client.disconnect()
            self._initialized = False
            self.client = None
            logger.info("ChatAgent client cleaned up and disconnected")


    async def send_message(self, message: str) -> AsyncGenerator[Dict, None]:
        """Send a message and stream the response.

        Args:
            message: The user's message

        Yields:
            Dictionary containing response chunks with 'type' and 'content' keys
        """
        if not self.client:
            raise RuntimeError("ChatAgent must be used as an async context manager")

        logger.info(f"Sending message: {message[:100]}...")
        logger.debug(f"Input to ClaudeSDKClient.query(): {message!r}")

        # Send the query
        await self.client.query(message)

        # Stream the response
        async for msg in self.client.receive_response():
            logger.debug(f"Raw message from ClaudeSDKClient.receive_response(): {msg!r}")

            if isinstance(msg, AssistantMessage):
                logger.debug(f"AssistantMessage content blocks: {msg.content!r}")

                for block in msg.content:
                    if isinstance(block, TextBlock):
                        logger.debug(f"TextBlock: {block!r}")
                        yield {
                            "type": "text",
                            "content": block.text
                        }
                    elif isinstance(block, ToolUseBlock):
                        logger.debug(f"ToolUseBlock: {block!r}")
                        yield {
                            "type": "tool_use",
                            "content": f"Using tool: {block.name}"
                        }

    async def chat(self, message: str) -> str:
        """Send a message and return the complete response.

        Args:
            message: The user's message

        Returns:
            The complete response text
        """
        response_parts = []

        logger.info(f"Starting chat response for message: {message[:100]}")

        async for chunk in self.send_message(message):
            if chunk["type"] == "text":
                response_parts.append(chunk["content"])


        logger.info("Completed chat response", extra={"response": "".join(response_parts)[:100]})

        return "".join(response_parts)


async def demo():
    """Demo the chat agent."""
    async with ChatAgent() as agent:
        # First question
        print("User: What's the capital of France?")
        response = await agent.chat("What's the capital of France?")
        print(f"Agent: {response}\n")

        # Follow-up question - agent remembers context
        print("User: What's the population of that city?")
        response = await agent.chat("What's the population of that city?")
        print(f"Agent: {response}\n")

        # Another follow-up
        print("User: What are some famous landmarks there?")
        response = await agent.chat("What are some famous landmarks there?")
        print(f"Agent: {response}\n")


if __name__ == "__main__":
    asyncio.run(demo())
