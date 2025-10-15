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
- `pyproject.toml` - Project metadata and dependency configuration
- `.python-version` - Python version specification for tooling
- `src/mlagentfactory/agents/` - Agent implementations
  - `chat_agent.py` - Conversational agent with file I/O, web, and Kaggle tools
- `src/mlagentfactory/tools/` - Tool modules for agents
  - `file_io_tools.py` - File and directory operations
  - `web_fetch_tools.py` - Web content fetching with Playwright
  - `kaggle_tools.py` - Kaggle CLI integration for datasets and competitions

## Dependencies

- `claude-agent-sdk` - SDK for building Claude-powered agents with autonomous task execution
- `kaggle` - Official Kaggle API for downloading datasets and competition interaction
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

### Kaggle Setup

To use Kaggle tools, you need to:
1. Install Kaggle CLI: `uv add kaggle` (already done)
2. Set up Kaggle API credentials:
   - Go to https://www.kaggle.com/account
   - Click "Create New API Token"
   - Save the `kaggle.json` file to `~/.kaggle/kaggle.json`
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`
