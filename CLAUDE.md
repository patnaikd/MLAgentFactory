# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MLAgentFactory is a Python project (requires Python >=3.12) currently in early development stages. The project uses `uv` for Python package management and environment handling.

## Development Commands

### Running the Project

**Web UI Applications:**
```bash
# Run the Gradio web application (recommended)
./run_gradio_app.sh
# Or manually:
uv run python src/mlagentfactory/ui/gradio_ui.py

# Run the Streamlit web application (legacy)
./run_app.sh
# Or manually:
uv run streamlit run src/mlagentfactory/ui/streamlit_ui.py
```

**Command-line Examples:**
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
- `src/mlagentfactory/ui/` - Web UI implementations
  - `gradio_ui.py` - Gradio-based web interface (recommended)
  - `streamlit_ui.py` - Streamlit-based web interface (legacy)
- `src/mlagentfactory/tools/` - Tool modules for agents
  - `file_io_tools.py` - File and directory operations
  - `web_fetch_tools.py` - Web content fetching with Playwright
  - `kaggle_tools.py` - Kaggle CLI integration for datasets and competitions
  - `uci_tools.py` - UCI ML Repository integration for datasets

## Dependencies

- `claude-agent-sdk` - SDK for building Claude-powered agents with autonomous task execution
- `gradio` - Modern web UI framework for interactive agent interfaces (recommended)
- `streamlit` - Alternative web UI framework (legacy support)
- `kaggle` - Official Kaggle API for downloading datasets and competition interaction
- `ucimlrepo` - Python library for accessing UCI Machine Learning Repository datasets
- `playwright` - Browser automation for JavaScript-heavy web content
- `jupyter` / `ipykernel` - Jupyter notebook support

## Architecture Notes

This project uses the Claude Agent SDK to build autonomous agents. Agents are configured with tasks and can execute them independently using Claude's capabilities.

### Agent Development Pattern
- Agents are initialized with `AgentConfig` specifying name and description
- Tasks are provided as natural language instructions
- Agents run asynchronously using `asyncio`
- The SDK handles tool use and task execution automatically

### Logging and Replay

The ChatAgent supports recording and replaying interactions using JSONL log files:

**Recording Mode**: Logs all API calls and responses
```python
agent = ChatAgent(
    enable_logging=True,
    log_dir="./data/claude_logs",
    session_prefix="my_session"
)
```

**Replay Mode**: Replays from recorded log files (no API calls)
```python
agent = ChatAgent(
    replay_log_file="./data/claude_logs/my_session_20251101_123456.jsonl"
)
```

**Use Cases:**
- Debugging: Capture exact API interactions
- Testing: Replay sessions without API calls or credits
- Development: Work offline with recorded sessions
- Analysis: Examine conversation patterns and costs

See [docs/LOGGING_AND_REPLAY.md](docs/LOGGING_AND_REPLAY.md) for detailed documentation and [examples/logging_example.py](examples/logging_example.py) for a complete demo.

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

## Web UI Implementations

The project includes two web-based user interfaces for interacting with the ChatAgent:

### Gradio UI (Recommended)

**Location:** `src/mlagentfactory/ui/gradio_ui.py`

**Features:**
- Modern, responsive chat interface with message history
- Real-time streaming responses from the agent
- Task progress tracking with visual todo list
- Comprehensive logging viewer with filtering and auto-scroll
- Session management (session ID, cost tracking, log files)
- Special formatting for tool usage (Bash, file operations, etc.)
- Auto-generated REST API for programmatic access
- Better performance with concurrent users via built-in queuing
- Cleaner async/await implementation without manual threading

**Launch:**
```bash
# Using launch script (recommended)
./run_gradio_app.sh

# Or run directly
uv run python src/mlagentfactory/ui/gradio_ui.py

# Access at: http://localhost:7860
# API docs at: http://localhost:7860/?view=api
```

**Advantages over Streamlit:**
- Native async streaming support (no background threads needed)
- Simpler state management with `gr.State()`
- Better component composition and layout flexibility
- Auto-generated REST API
- Easier to embed in other applications
- Better concurrent user handling

### Streamlit UI (Legacy)

**Location:** `src/mlagentfactory/ui/streamlit_ui.py`

**Features:**
- Chat interface with message history
- Background thread processing for non-blocking UI
- Auto-refreshing todo list and logs using `@st.fragment`
- Custom HTML log viewer
- Session tracking and cost display

**Launch:**
```bash
./run_app.sh
# Access at: http://localhost:8501
```

**Note:** The Streamlit implementation is maintained for backward compatibility but the Gradio UI is recommended for new deployments.

### UI Feature Comparison

| Feature | Gradio UI | Streamlit UI |
|---------|-----------|--------------|
| Chat Interface | ✅ Modern chatbot | ✅ Chat messages |
| Streaming Responses | ✅ Native async | ✅ Background threads |
| Todo List | ✅ HTML display | ✅ Auto-refresh fragment |
| Logs Viewer | ✅ HTML + filters | ✅ Custom HTML component |
| Session Management | ✅ gr.State() | ✅ st.session_state |
| REST API | ✅ Auto-generated | ❌ Not available |
| Concurrent Users | ✅ Better handling | ⚠️ Limited |
| Code Complexity | ✅ Simpler async | ⚠️ Manual threading |

## Session Manager Architecture (NEW)

The Session Manager provides a standalone service for hosting long-running ChatAgent instances with pull-based message streaming.

### Architecture Overview

The Session Manager consists of:

1. **Message Store** (`src/mlagentfactory/services/message_store.py`)
   - SQLite-based persistent storage for sessions and messages
   - Cursor-based pagination for efficient message retrieval
   - Session lifecycle management (created, running, completed, failed, stopped)
   - Message types: text, tool_use, tool_result, todo_update, session_id, total_cost

2. **Process Manager** (`src/mlagentfactory/services/process_manager.py`)
   - Spawns isolated processes for each ChatAgent session
   - Handles process lifecycle (start, monitor, stop)
   - Captures agent output and stores in MessageStore
   - Handles crashes and restarts

3. **Session Manager** (`src/mlagentfactory/services/session_manager.py`)
   - High-level orchestration layer
   - Coordinates message store and process manager
   - Provides API for session creation, querying, and message retrieval

4. **FastAPI REST API** (`src/mlagentfactory/services/api.py`)
   - RESTful endpoints for session management
   - Auto-generated OpenAPI documentation
   - CORS-enabled for web clients

5. **Streamlit UI with Session Manager** (`src/mlagentfactory/ui/streamlit_ui_session_manager.py`)
   - Pull-based message polling via REST API
   - No direct agent instantiation
   - Stateless UI that polls for updates

### Key Features

- **Process Isolation**: Each agent runs in its own process for stability
- **Pull-based Messaging**: UI polls for messages instead of maintaining open connections
- **Persistent Storage**: All messages stored in SQLite for reliability
- **Cursor-based Pagination**: Efficient message retrieval using message IDs
- **Session Management**: Create, query, stop, and delete sessions via API

### Running the Session Manager

**Step 1: Start the Session Manager API**
```bash
# Using the startup script (recommended)
./start_session_manager.sh

# Or manually
uv run python -m mlagentfactory.cli.session_manager_cli start

# With auto-reload for development
./start_session_manager.sh --reload

# Custom host/port
uv run python -m mlagentfactory.cli.session_manager_cli start --host 0.0.0.0 --port 8000
```

The API will be available at:
- Base URL: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

**Step 2: Start the Streamlit UI**
```bash
# Using the startup script (recommended)
./run_streamlit_session_manager.sh

# Or manually
uv run streamlit run src/mlagentfactory/ui/streamlit_ui_session_manager.py
```

Access the UI at: `http://localhost:8501`

### Session Manager CLI

The CLI provides commands for managing sessions:

```bash
# Check service status
uv run python -m mlagentfactory.cli.session_manager_cli status

# List all sessions
uv run python -m mlagentfactory.cli.session_manager_cli sessions

# Get session info
uv run python -m mlagentfactory.cli.session_manager_cli info <session-id>

# Stop a session
uv run python -m mlagentfactory.cli.session_manager_cli stop-session <session-id>

# Stop the service
uv run python -m mlagentfactory.cli.session_manager_cli stop
```

### REST API Endpoints

**Session Management:**
- `POST /sessions` - Create new session
- `GET /sessions` - List all sessions
- `GET /sessions/{session_id}` - Get session info
- `GET /sessions/{session_id}/stats` - Get session statistics
- `DELETE /sessions/{session_id}` - Stop session
- `DELETE /sessions/{session_id}/data` - Delete session permanently

**Message Operations:**
- `POST /sessions/{session_id}/query` - Send query to agent
- `GET /sessions/{session_id}/messages` - Poll for new messages
  - Query params: `since_message_id` (cursor), `limit` (max messages)

**Health Check:**
- `GET /health` - Service health and statistics

### Message Polling Pattern

The UI implements cursor-based polling:

1. Initial request: `GET /sessions/{session_id}/messages?since_message_id=0`
2. Store `next_cursor` from response
3. Poll again: `GET /sessions/{session_id}/messages?since_message_id={next_cursor}`
4. Continue polling every 0.5-1 seconds while `is_processing`

### Database Schema

**Sessions Table:**
```sql
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    agent_session_id TEXT,
    total_cost REAL DEFAULT 0.0,
    metadata TEXT
);
```

**Messages Table:**
```sql
CREATE TABLE messages (
    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    message_type TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);
```

### Development Notes

- SQLite database stored in: `data/sessions.db`
- Process-safe: Uses multiprocessing with queues for IPC
- Thread-safe: SQLite with `check_same_thread=False` and context managers
- Graceful shutdown: Proper cleanup of processes and resources
- Error handling: Process failures don't crash the service

### Architecture Benefits

1. **Scalability**: Each agent runs in isolation
2. **Reliability**: Process crashes don't affect other sessions
3. **Persistence**: Messages survive service restarts
4. **Simplicity**: Pull-based polling is easier to debug than push/streaming
5. **Flexibility**: REST API allows any client (web, CLI, etc.)
