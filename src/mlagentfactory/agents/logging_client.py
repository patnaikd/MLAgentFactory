"""Logging wrapper and fake client for ClaudeSDKClient to enable recording and replay."""
import json
import logging
from pathlib import Path
from typing import AsyncGenerator, Optional, Any
from datetime import datetime

from claude_agent_sdk import (
    ClaudeSDKClient,
    AssistantMessage,
    UserMessage,
    SystemMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    ClaudeAgentOptions
)

logger = logging.getLogger(__name__)


def _serialize_message(msg: Any) -> dict:
    """Serialize a Claude SDK message object to a dictionary.

    Args:
        msg: Message object from Claude SDK

    Returns:
        Dictionary representation that can be JSON serialized
    """
    msg_dict = {
        "type": type(msg).__name__,
        "timestamp": datetime.utcnow().isoformat()
    }

    if isinstance(msg, (AssistantMessage, UserMessage)):
        # Serialize content blocks
        content_list = []
        for block in msg.content:
            if isinstance(block, TextBlock):
                content_list.append({
                    "block_type": "TextBlock",
                    "text": block.text
                })
            elif isinstance(block, ToolUseBlock):
                content_list.append({
                    "block_type": "ToolUseBlock",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input if hasattr(block, 'input') else {}
                })
            elif isinstance(block, ToolResultBlock):
                content_list.append({
                    "block_type": "ToolResultBlock",
                    "tool_use_id": block.tool_use_id,
                    "content": block.content,
                    "is_error": block.is_error
                })
            else:
                content_list.append({
                    "block_type": type(block).__name__,
                    "raw": str(block)
                })

        msg_dict["content"] = content_list

    elif isinstance(msg, SystemMessage):
        msg_dict["subtype"] = getattr(msg, 'subtype', None)
        msg_dict["data"] = getattr(msg, 'data', None)

    elif isinstance(msg, ResultMessage):
        msg_dict["subtype"] = msg.subtype
        msg_dict["is_error"] = msg.is_error
        msg_dict["duration_ms"] = msg.duration_ms
        msg_dict["duration_api_ms"] = msg.duration_api_ms
        msg_dict["num_turns"] = msg.num_turns
        msg_dict["total_cost_usd"] = getattr(msg, 'total_cost_usd', None)
        msg_dict["usage"] = getattr(msg, 'usage', None)

    return msg_dict


def _deserialize_message(msg_dict: dict) -> Any:
    """Deserialize a dictionary back to a Claude SDK message object.

    Args:
        msg_dict: Dictionary representation of a message

    Returns:
        Reconstructed message object
    """
    msg_type = msg_dict.get("type")

    if msg_type == "AssistantMessage":
        # Reconstruct content blocks
        content_blocks = []
        for block_dict in msg_dict.get("content", []):
            block_type = block_dict.get("block_type")

            if block_type == "TextBlock":
                content_blocks.append(TextBlock(text=block_dict["text"]))
            elif block_type == "ToolUseBlock":
                # Create ToolUseBlock with required fields
                tool_block = ToolUseBlock(
                    id=block_dict["id"],
                    name=block_dict["name"],
                    input=block_dict["input"]
                )
                content_blocks.append(tool_block)
            elif block_type == "ToolResultBlock":
                content_blocks.append(ToolResultBlock(
                    tool_use_id=block_dict["tool_use_id"],
                    content=block_dict["content"],
                    is_error=block_dict["is_error"]
                ))

        return AssistantMessage(content=content_blocks)

    elif msg_type == "UserMessage":
        # Reconstruct content blocks
        content_blocks = []
        for block_dict in msg_dict.get("content", []):
            block_type = block_dict.get("block_type")

            if block_type == "TextBlock":
                content_blocks.append(TextBlock(text=block_dict["text"]))
            elif block_type == "ToolResultBlock":
                content_blocks.append(ToolResultBlock(
                    tool_use_id=block_dict["tool_use_id"],
                    content=block_dict["content"],
                    is_error=block_dict["is_error"]
                ))

        return UserMessage(content=content_blocks)

    elif msg_type == "SystemMessage":
        # Create a SystemMessage-like object
        msg = SystemMessage()
        msg.subtype = msg_dict.get("subtype")
        msg.data = msg_dict.get("data")
        return msg

    elif msg_type == "ResultMessage":
        # Create a ResultMessage-like object
        msg = ResultMessage(
            subtype=msg_dict["subtype"],
            is_error=msg_dict["is_error"]
        )
        msg.duration_ms = msg_dict.get("duration_ms", 0)
        msg.duration_api_ms = msg_dict.get("duration_api_ms", 0)
        msg.num_turns = msg_dict.get("num_turns", 0)
        msg.total_cost_usd = msg_dict.get("total_cost_usd", 0.0)
        msg.usage = msg_dict.get("usage")
        return msg

    else:
        logger.warning(f"Unknown message type for deserialization: {msg_type}")
        return None


class LoggingClaudeSDKClient:
    """Wrapper around ClaudeSDKClient that logs all query calls and responses to JSONL files.

    This enables recording interactions for later replay with FakeClaudeSDKClient.

    Args:
        options: ClaudeAgentOptions to pass to the underlying client
        log_dir: Directory to store log files (default: ./data/claude_logs)
        session_prefix: Prefix for log filenames (default: session)
    """

    def __init__(
        self,
        options: ClaudeAgentOptions,
        log_dir: str = "./data/claude_logs",
        session_prefix: str = "session"
    ):
        """Initialize the logging client wrapper."""
        self.client = ClaudeSDKClient(options=options)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create unique session log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{session_prefix}_{timestamp}.jsonl"

        logger.info(f"LoggingClaudeSDKClient initialized. Logging to: {self.log_file}")

    async def connect(self):
        """Connect the underlying client."""
        await self.client.connect()
        logger.info("LoggingClaudeSDKClient connected")

    async def disconnect(self):
        """Disconnect the underlying client."""
        await self.client.disconnect()
        logger.info("LoggingClaudeSDKClient disconnected")

    async def query(self, message: str):
        """Send a query and log it to the JSONL file.

        Args:
            message: User message to send
        """
        # Log the query
        query_entry = {
            "direction": "outgoing",
            "entry_type": "query",
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(query_entry) + '\n')

        logger.debug(f"Logged query to {self.log_file}")

        # Forward to actual client
        await self.client.query(message)

    async def receive_response(self) -> AsyncGenerator[Any, None]:
        """Receive response from the client and log each message to JSONL.

        Yields:
            Message objects from the Claude SDK
        """
        async for msg in self.client.receive_response():
            # Serialize and log the message
            msg_dict = _serialize_message(msg)
            response_entry = {
                "direction": "incoming",
                "entry_type": "response",
                "message": msg_dict,
                "timestamp": datetime.utcnow().isoformat()
            }

            with open(self.log_file, 'a') as f:
                f.write(json.dumps(response_entry) + '\n')

            logger.debug(f"Logged response message type={msg_dict['type']} to {self.log_file}")

            # Yield the original message
            yield msg


class FakeClaudeSDKClient:
    """Fake client that replays recorded interactions from a JSONL log file.

    This allows testing and debugging without making actual API calls.

    Args:
        log_file: Path to the JSONL log file to replay
    """

    def __init__(self, log_file: str):
        """Initialize the fake client with a log file."""
        self.log_file = Path(log_file)
        if not self.log_file.exists():
            raise FileNotFoundError(f"Log file not found: {self.log_file}")

        # Load all entries from the log file
        self.entries = []
        with open(self.log_file, 'r') as f:
            for line in f:
                if line.strip():
                    self.entries.append(json.loads(line))

        # Track current position
        self.current_query_idx = 0
        self.current_response_idx = 0

        logger.info(f"FakeClaudeSDKClient initialized with {len(self.entries)} log entries from {self.log_file}")

    async def connect(self):
        """Mock connect - does nothing."""
        logger.info("FakeClaudeSDKClient connected (mock)")

    async def disconnect(self):
        """Mock disconnect - does nothing."""
        logger.info("FakeClaudeSDKClient disconnected (mock)")

    async def query(self, message: str):
        """Mock query - validates that message matches the recorded query.

        Args:
            message: User message (should match logged query)
        """
        # Find the next query entry
        while self.current_query_idx < len(self.entries):
            entry = self.entries[self.current_query_idx]
            if entry.get("direction") == "outgoing" and entry.get("entry_type") == "query":
                logged_message = entry.get("message")

                if logged_message != message:
                    logger.warning(
                        f"Query mismatch! Expected: {logged_message[:100]}, "
                        f"Got: {message[:100]}"
                    )

                self.current_query_idx += 1
                logger.debug(f"FakeClaudeSDKClient processed query #{self.current_query_idx}")
                return

            self.current_query_idx += 1

        logger.error("No matching query found in log file")
        raise RuntimeError("FakeClaudeSDKClient: No matching query found in log")

    async def receive_response(self) -> AsyncGenerator[Any, None]:
        """Replay response messages from the log file.

        Yields:
            Deserialized message objects
        """
        # Find all response entries for the current query
        responses_yielded = 0

        while self.current_response_idx < len(self.entries):
            entry = self.entries[self.current_response_idx]

            # Stop when we hit the next query
            if entry.get("direction") == "outgoing" and entry.get("entry_type") == "query":
                break

            if entry.get("direction") == "incoming" and entry.get("entry_type") == "response":
                msg_dict = entry.get("message")

                # Deserialize the message
                msg = _deserialize_message(msg_dict)

                if msg:
                    responses_yielded += 1
                    logger.debug(
                        f"FakeClaudeSDKClient yielding response #{responses_yielded}, "
                        f"type={msg_dict.get('type')}"
                    )
                    yield msg

            self.current_response_idx += 1

        logger.info(f"FakeClaudeSDKClient completed response stream with {responses_yielded} messages")
