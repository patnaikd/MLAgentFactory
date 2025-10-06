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
- `pyproject.toml` - Project metadata and dependency configuration
- `.python-version` - Python version specification for tooling

## Dependencies

- `claude-agent-sdk` - SDK for building Claude-powered agents with autonomous task execution

## Architecture Notes

This project uses the Claude Agent SDK to build autonomous agents. Agents are configured with tasks and can execute them independently using Claude's capabilities.

### Agent Development Pattern
- Agents are initialized with `AgentConfig` specifying name and description
- Tasks are provided as natural language instructions
- Agents run asynchronously using `asyncio`
- The SDK handles tool use and task execution automatically
