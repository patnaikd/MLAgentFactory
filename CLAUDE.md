# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MLAgentFactory is a Python project (requires Python >=3.12) currently in early development stages. The project uses `uv` for Python package management and environment handling.

## Development Commands

### Running the Project
```bash
# Run the basic hello world
uv run main.py

# Run the file creation agent example
uv run file_creation_agent.py

# Run the Kaggle agent example
uv run kaggle_agent_example.py

# Run the UCI ML Repository example
uv run uci_example.py
```

### Python Environment
- Python version: 3.12 (specified in `.python-version`)
- Package manager: `uv`
- Dependencies are managed in `pyproject.toml`

### Installing Dependencies
```bash
uv sync
```

## Project Structure

- `main.py` - Entry point with basic Hello World implementation
- `file_creation_agent.py` - Example agent using claude-agent-sdk that demonstrates file creation
- `kaggle_agent_example.py` - Example agent demonstrating Kaggle CLI integration
- `uci_example.py` - Example agent demonstrating UCI ML Repository integration
- `pyproject.toml` - Project metadata and dependency configuration
- `.python-version` - Python version specification for tooling
- `src/mlagentfactory/agents/` - Agent implementations
  - `chat_agent.py` - Conversational agent with file I/O, web, Kaggle, and UCI ML Repository tools
- `src/mlagentfactory/tools/` - Tool modules for agents
  - `file_io_tools.py` - File and directory operations
  - `web_fetch_tools.py` - Web content fetching with Playwright
  - `kaggle_tools.py` - Kaggle CLI integration for datasets and competitions
  - `uci_tools.py` - UCI ML Repository integration for datasets

## Dependencies

- `claude-agent-sdk` - SDK for building Claude-powered agents with autonomous task execution
- `kaggle` - Official Kaggle API for downloading datasets and competition interaction
- `ucimlrepo` - Python library for accessing UCI Machine Learning Repository datasets
- `streamlit` - Web UI framework for interactive agent interfaces
- `playwright` - Browser automation for JavaScript-heavy web content
- `jupyter` / `ipykernel` - Jupyter notebook support

## Architecture Notes

This project uses the Claude Agent SDK to build autonomous agents. Agents are configured with tasks and can execute them independently using Claude's capabilities.

### Agent Development Pattern
- Agents are initialized with `AgentConfig` specifying name and description
- Tasks are provided as natural language instructions
- Agents run asynchronously using `asyncio`
- The SDK handles tool use and task execution automatically

### Available Tools

The ChatAgent includes the following tools:

**File I/O Tools:**
- `read_file` - Read content from files
- `write_file` - Write content to files (creates parent directories)
- `edit_file` - Edit files by replacing text
- `delete_file` - Delete files
- `list_directory` - List directory contents
- `create_directory` - Create directories
- `remove_directory` - Remove directories recursively

**Web Tools:**
- `fetch_webpage` - Fetch web content with Playwright (handles JavaScript)

**Kaggle Tools:**
- `kaggle_download_dataset` - Download Kaggle datasets
- `kaggle_list_competitions` - List Kaggle competitions with search
- `kaggle_download_competition_data` - Download competition data files
- `kaggle_submit_competition` - Submit solutions to competitions
- `kaggle_list_submissions` - List your competition submissions
- `kaggle_competition_leaderboard` - View competition leaderboards

**UCI ML Repository Tools:**
- `uci_list_datasets` - List available datasets from UCI ML Repository (with optional search/filter)
- `uci_fetch_dataset` - Fetch dataset by ID or name, optionally save to CSV files
- `uci_get_dataset_info` - Get detailed metadata about a dataset without downloading data

### Kaggle Setup

To use Kaggle tools, you need to:
1. Install Kaggle CLI: `uv add kaggle` (already done)
2. Set up Kaggle API credentials:
   - Go to https://www.kaggle.com/account
   - Click "Create New API Token"
   - Save the `kaggle.json` file to `~/.kaggle/kaggle.json`
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### UCI ML Repository Setup

The UCI ML Repository tools are ready to use out of the box:
1. Install ucimlrepo: `uv add ucimlrepo` (already done)
2. No API credentials required - datasets are publicly accessible
3. Popular dataset IDs:
   - Iris: 53
   - Wine: 109
   - Breast Cancer Wisconsin: 17
   - Adult (Census Income): 2
4. Full dataset list available at: https://archive.ics.uci.edu/

**Example Usage:**
```python
from ucimlrepo import fetch_ucirepo

# Fetch Iris dataset by ID
iris = fetch_ucirepo(id=53)
X = iris.data.features
y = iris.data.targets

# Or by name
iris = fetch_ucirepo(name='Iris')
```
