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
from ..tools import write_file, fetch_webpage

logger = logging.getLogger(__name__)


class ChatAgent:
    """A simple conversational agent that maintains conversation history."""

    def __init__(self):
        """Initialize the chat agent."""
        initialize_observability(log_level="INFO", enable_tracing=False)

        # Create MCP servers with tools
        self.file_server = create_sdk_mcp_server(
            name="file_tools",
            version="1.0.0",
            tools=[write_file]
        )

        self.web_server = create_sdk_mcp_server(
            name="web_tools",
            version="1.0.0",
            tools=[fetch_webpage]
        )

        # Configure agent options
        self.options = ClaudeAgentOptions(
            mcp_servers={"files": self.file_server, "web": self.web_server},
            allowed_tools=["write_file", "fetch_webpage"],
            permission_mode="bypassPermissions"
        )

        self.client: Optional[ClaudeSDKClient] = None
        self._initialized = False
        logger.info("ChatAgent initialized with file and web tools")

    async def initialize(self):
        """Initialize the client. Call this once before using the agent."""
        if not self._initialized:
            self.client = ClaudeSDKClient(options=self.options)
            await self.client.__aenter__()
            self._initialized = True
            logger.info("ChatAgent client initialized")

    async def cleanup(self):
        """Cleanup the client. Call this when done with the agent."""
        if self._initialized and self.client:
            await self.client.__aexit__(None, None, None)
            self._initialized = False
            self.client = None
            logger.info("ChatAgent client cleaned up")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
        """Async context manager exit."""
        await self.cleanup()

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

        # Send the query
        await self.client.query(message)

        # Stream the response
        async for msg in self.client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        yield {
                            "type": "text",
                            "content": block.text
                        }
                    elif isinstance(block, ToolUseBlock):
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

        async for chunk in self.send_message(message):
            if chunk["type"] == "text":
                response_parts.append(chunk["content"])

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
