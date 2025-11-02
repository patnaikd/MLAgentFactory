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

**Use cases:**
- Testing UI components without API calls
- Creating test fixtures for unit tests
- Debugging API interactions
- Working offline with recorded sessions
- Analyzing conversation patterns and costs

**See also:**
- [docs/LOGGING_AND_REPLAY.md](../docs/LOGGING_AND_REPLAY.md) - Detailed documentation
- [src/mlagentfactory/agents/logging_client.py](../src/mlagentfactory/agents/logging_client.py) - Implementation

---

### [markdown_to_pdf_example.py](markdown_to_pdf_example.py)

Demonstrates how to convert markdown files to professionally formatted PDF documents.

**What it does:**
1. Creates a sample markdown file with various formatting features
2. Converts the file to PDF using three different styles (default, github, minimal)
3. Uses the ChatAgent to perform a conversion via natural language
4. Displays file sizes and locations of generated PDFs

**Run it:**
```bash
uv run python examples/markdown_to_pdf_example.py
```

**Expected output:**
```
Markdown to PDF Conversion Demo
============================================================
DIRECT USAGE DEMO
============================================================

1. Creating sample markdown file: ./data/sample_document.md
   ✓ Sample file created (1234 bytes)

2. Converting with default styling...
   ✓ Successfully converted ./data/sample_document.md to ./data/sample_default.pdf

3. Converting with GitHub styling...
   ✓ Successfully converted ./data/sample_document.md to ./data/sample_github.pdf

4. Converting with minimal styling...
   ✓ Successfully converted ./data/sample_document.md to ./data/sample_minimal.pdf

Generated PDFs:
  - sample_default.pdf (45.3 KB)
  - sample_github.pdf (47.1 KB)
  - sample_minimal.pdf (42.8 KB)
```

**Use cases:**
- Converting documentation to shareable PDFs
- Creating reports from markdown files
- Generating styled documents from agent-created markdown
- Archiving markdown content in PDF format

**See also:**
- [src/mlagentfactory/tools/markdown_to_pdf_tools.py](../src/mlagentfactory/tools/markdown_to_pdf_tools.py) - Implementation
- [CLAUDE.md](../CLAUDE.md#markdown-to-pdf-setup) - Tool documentation

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
