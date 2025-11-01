# Examples

This directory contains example scripts demonstrating various features of MLAgentFactory.

## Available Examples

### [logging_example.py](logging_example.py)

Demonstrates how to record and replay ChatAgent interactions using JSONL log files.

**What it does:**
1. Creates a ChatAgent with logging enabled
2. Sends a simple query and logs the interaction to a JSONL file
3. Creates a new ChatAgent in replay mode
4. Replays the same interaction from the log file (no API calls)
5. Displays the log file contents

**Run it:**
```bash
uv run python examples/logging_example.py
```

**Expected output:**
```
Claude Agent Recording & Replay Demo
============================================================
RECORDING SESSION
============================================================

User: What is 2 + 2?

Agent: 2 + 2 equals 4.

✓ Interaction logged to: ./data/claude_logs/example_session_20251101_123456.jsonl
  Total cost: $0.0012
  Session ID: abc123...

============================================================
REPLAYING SESSION
============================================================
Log file: ./data/claude_logs/example_session_20251101_123456.jsonl

User: What is 2 + 2?

Agent (replayed): 2 + 2 equals 4.

✓ Successfully replayed session
  Total cost (from log): $0.0012
```

**Use cases:**
- Testing UI components without API calls
- Creating test fixtures for unit tests
- Debugging API interactions
- Working offline with recorded sessions
- Analyzing conversation patterns and costs

**See also:**
- [docs/LOGGING_AND_REPLAY.md](../docs/LOGGING_AND_REPLAY.md) - Detailed documentation
- [src/mlagentfactory/agents/logging_client.py](../src/mlagentfactory/agents/logging_client.py) - Implementation

## Running Examples

All examples can be run using the `uv` package manager:

```bash
# Run a specific example
uv run python examples/<example_name>.py

# Example
uv run python examples/logging_example.py
```

## Creating New Examples

When adding new examples:

1. Place them in this directory
2. Use clear, descriptive names (e.g., `feature_name_example.py`)
3. Add documentation at the top of the file
4. Update this README with a description
5. Ensure examples are self-contained and runnable

## Additional Resources

- **Main Examples** (in root directory):
  - [main.py](../main.py) - Basic Hello World
  - [file_creation_agent.py](../file_creation_agent.py) - File operations
  - [kaggle_agent_example.py](../kaggle_agent_example.py) - Kaggle integration
  - [uci_example.py](../uci_example.py) - UCI ML Repository

- **Documentation:**
  - [CLAUDE.md](../CLAUDE.md) - Project overview
  - [docs/LOGGING_AND_REPLAY.md](../docs/LOGGING_AND_REPLAY.md) - Logging feature docs
