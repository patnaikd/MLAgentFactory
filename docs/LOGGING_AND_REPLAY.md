# Claude Agent Logging and Replay

This document describes how to record and replay ChatAgent interactions using JSONL log files.

## Overview

The MLAgentFactory includes two special client wrappers for the Claude SDK:

1. **LoggingClaudeSDKClient** - Wraps the standard client and logs all `query()` calls and `receive_response()` messages to JSONL files
2. **FakeClaudeSDKClient** - Replays interactions from JSONL log files without making actual API calls

## Use Cases

### Recording (LoggingClaudeSDKClient)

- **Debugging**: Capture exact API interactions for debugging issues
- **Testing**: Record real interactions to create test fixtures
- **Auditing**: Keep records of agent conversations and tool usage
- **Analysis**: Analyze conversation patterns and token usage
- **Compliance**: Maintain logs for regulatory requirements

### Replay (FakeClaudeSDKClient)

- **Unit Testing**: Test UI and business logic without API calls
- **Development**: Work on features without consuming API credits
- **Demos**: Show consistent behavior in demos and presentations
- **Debugging**: Reproduce exact sequences of events
- **Performance Testing**: Test UI performance without API latency

## JSONL Log Format

Each line in the log file is a JSON object with the following structure:

### Query Entry (Outgoing)
```json
{
  "direction": "outgoing",
  "entry_type": "query",
  "message": "What is 2 + 2?",
  "timestamp": "2025-11-01T12:34:56.789012"
}
```

### Response Entry (Incoming)
```json
{
  "direction": "incoming",
  "entry_type": "response",
  "message": {
    "type": "AssistantMessage",
    "timestamp": "2025-11-01T12:34:57.123456",
    "content": [
      {
        "block_type": "TextBlock",
        "text": "2 + 2 equals 4."
      }
    ]
  },
  "timestamp": "2025-11-01T12:34:57.123456"
}
```

### Message Types

The `message.type` field in response entries can be:

- **AssistantMessage**: Text responses and tool usage from the agent
- **UserMessage**: Tool results and user inputs
- **SystemMessage**: System initialization and metadata
- **ResultMessage**: Final results including cost and usage statistics

## Usage

### Recording Interactions

```python
from mlagentfactory.agents.chat_agent import ChatAgent

# Create agent with logging enabled
agent = ChatAgent(
    enable_logging=True,           # Enable JSONL logging
    log_dir="./data/claude_logs",  # Directory for log files
    session_prefix="my_session"    # Prefix for log filenames
)

await agent.initialize()

# Use agent normally - all interactions are logged
async for chunk in agent.send_message("Hello!"):
    if chunk["type"] == "text":
        print(chunk["content"])

await agent.cleanup()

# Log file created: ./data/claude_logs/my_session_20251101_123456.jsonl
```

### Replaying Interactions

```python
from mlagentfactory.agents.chat_agent import ChatAgent

# Create agent with replay enabled
agent = ChatAgent(
    replay_log_file="./data/claude_logs/my_session_20251101_123456.jsonl"
)

await agent.initialize()

# Send the SAME queries as in the recording
# Responses will be replayed from the log file (no API calls)
async for chunk in agent.send_message("Hello!"):
    if chunk["type"] == "text":
        print(chunk["content"])

await agent.cleanup()
```

### Important Notes for Replay

1. **Query Matching**: You must send the exact same queries in the same order as recorded
2. **No API Calls**: The FakeClaudeSDKClient makes no network requests
3. **Mismatch Warnings**: If queries don't match, warnings are logged but replay continues
4. **Cost Tracking**: Total cost from the original session is replayed

## Example Script

See [examples/logging_example.py](../examples/logging_example.py) for a complete demo:

```bash
# Run the example
uv run python examples/logging_example.py
```

This script demonstrates:
1. Recording a chat session to a JSONL log
2. Replaying the session from the log file
3. Displaying log file contents

## Advanced Usage

### Custom Serialization

The logging system automatically serializes Claude SDK message objects. If you need to customize serialization, modify:

- `_serialize_message()` - Converts messages to dictionaries
- `_deserialize_message()` - Reconstructs messages from dictionaries

Both functions are in `src/mlagentfactory/agents/logging_client.py`.

### Log File Management

Log files are stored in JSONL format (newline-delimited JSON) with timestamps:

```
data/claude_logs/
├── session_20251101_123456.jsonl
├── session_20251101_134512.jsonl
└── session_20251101_145623.jsonl
```

You can:
- Archive old logs periodically
- Analyze logs with standard JSON tools (`jq`, Python's `json` module, etc.)
- Search logs with `grep` or text search tools
- Compress logs with `gzip` for long-term storage

### Analyzing Logs

```python
import json
from pathlib import Path

# Load a log file
log_file = Path("data/claude_logs/session_20251101_123456.jsonl")

queries = []
responses = []
total_tokens = 0

with open(log_file) as f:
    for line in f:
        entry = json.loads(line)

        if entry["entry_type"] == "query":
            queries.append(entry["message"])

        elif entry["entry_type"] == "response":
            msg = entry["message"]

            if msg["type"] == "ResultMessage":
                total_tokens = msg.get("usage", {}).get("total_tokens", 0)

            responses.append(msg)

print(f"Total queries: {len(queries)}")
print(f"Total response messages: {len(responses)}")
print(f"Total tokens used: {total_tokens}")
```

## Integration with UIs

### Gradio UI

To enable logging in the Gradio UI, modify `src/mlagentfactory/ui/gradio_ui.py`:

```python
# In create_demo() function
agent = ChatAgent(
    enable_logging=True,
    log_dir="./data/claude_logs",
    session_prefix="gradio_session"
)
```

### Streamlit UI

To enable logging in the Streamlit UI, modify `src/mlagentfactory/ui/streamlit_ui.py`:

```python
# In initialize_agent() function
agent = ChatAgent(
    enable_logging=True,
    log_dir="./data/claude_logs",
    session_prefix="streamlit_session"
)
```

## Testing with Replay

### Unit Testing Example

```python
import pytest
from mlagentfactory.agents.chat_agent import ChatAgent

@pytest.mark.asyncio
async def test_agent_response():
    """Test agent using a recorded log file."""
    agent = ChatAgent(
        replay_log_file="tests/fixtures/test_session.jsonl"
    )

    await agent.initialize()

    response_parts = []
    async for chunk in agent.send_message("What is 2 + 2?"):
        if chunk["type"] == "text":
            response_parts.append(chunk["content"])

    response = "".join(response_parts)
    assert "4" in response

    await agent.cleanup()
```

### Creating Test Fixtures

1. Run your agent with `enable_logging=True`
2. Copy the generated log file to `tests/fixtures/`
3. Use the log file in your tests with `replay_log_file`

## Performance

### Recording Impact

- **Overhead**: ~1-5ms per message for serialization and file I/O
- **File Size**: ~1-10 KB per message depending on content
- **Disk I/O**: Synchronous writes to ensure data integrity

### Replay Performance

- **Speed**: Instant - no network latency
- **Memory**: Entire log loaded into memory at initialization
- **Scalability**: Suitable for logs up to ~100 MB

## Security Considerations

### Log File Security

Log files may contain:
- User queries and inputs
- Agent responses
- Tool usage details (file paths, commands, etc.)
- API usage statistics

**Recommendations:**
- Store logs in secure directories with appropriate permissions
- Exclude log directories from version control (add to `.gitignore`)
- Sanitize logs before sharing or archiving
- Consider encrypting logs containing sensitive data
- Implement log rotation and retention policies

### Example `.gitignore` Entry

```
# Claude agent logs
data/claude_logs/
*.jsonl
```

## Troubleshooting

### Query Mismatch Warning

```
Query mismatch! Expected: What is 2 + 2?, Got: What is 3 + 3?
```

**Cause**: Replayed queries don't match recorded queries
**Solution**: Ensure you send the exact same queries in the same order

### Log File Not Found

```
FileNotFoundError: Log file not found: ./data/claude_logs/session.jsonl
```

**Cause**: Specified log file doesn't exist
**Solution**: Check the file path and ensure the file exists

### Deserialization Error

```
Unknown message type for deserialization: CustomMessage
```

**Cause**: Log contains message types not supported by deserializer
**Solution**: Update `_deserialize_message()` to handle the new type

## Future Enhancements

Potential improvements to the logging system:

- **Async File I/O**: Use `aiofiles` for non-blocking writes
- **Compression**: Automatic gzip compression for large logs
- **Streaming Replay**: Load logs incrementally instead of all at once
- **Partial Replay**: Skip to specific points in a log file
- **Log Merging**: Combine multiple log files
- **Diff Tool**: Compare two log files to find differences
- **Export Formats**: Convert logs to CSV, HTML, or other formats

## API Reference

### LoggingClaudeSDKClient

```python
class LoggingClaudeSDKClient:
    """Wrapper around ClaudeSDKClient that logs interactions to JSONL."""

    def __init__(
        self,
        options: ClaudeAgentOptions,
        log_dir: str = "./data/claude_logs",
        session_prefix: str = "session"
    ):
        """
        Args:
            options: Claude agent configuration
            log_dir: Directory for log files
            session_prefix: Prefix for log filenames
        """

    async def connect(self) -> None:
        """Connect the underlying client."""

    async def disconnect(self) -> None:
        """Disconnect the underlying client."""

    async def query(self, message: str) -> None:
        """Send a query and log it."""

    async def receive_response(self) -> AsyncGenerator:
        """Receive and log response messages."""
```

### FakeClaudeSDKClient

```python
class FakeClaudeSDKClient:
    """Replays recorded interactions from JSONL log files."""

    def __init__(self, log_file: str):
        """
        Args:
            log_file: Path to JSONL log file to replay

        Raises:
            FileNotFoundError: If log file doesn't exist
        """

    async def connect(self) -> None:
        """Mock connect - does nothing."""

    async def disconnect(self) -> None:
        """Mock disconnect - does nothing."""

    async def query(self, message: str) -> None:
        """Validate query matches logged query."""

    async def receive_response(self) -> AsyncGenerator:
        """Replay response messages from log."""
```

### ChatAgent Parameters

```python
class ChatAgent:
    def __init__(
        self,
        enable_logging: bool = False,
        log_dir: str = "./data/claude_logs",
        session_prefix: str = "session",
        replay_log_file: Optional[str] = None
    ):
        """
        Args:
            enable_logging: Enable JSONL logging
            log_dir: Directory for log files (if logging enabled)
            session_prefix: Prefix for log filenames (if logging enabled)
            replay_log_file: Path to log file for replay mode
        """
```

## Support

For issues or questions:
- Check this documentation first
- Review [examples/logging_example.py](../examples/logging_example.py)
- Check the source code in `src/mlagentfactory/agents/logging_client.py`
- Open an issue on GitHub
