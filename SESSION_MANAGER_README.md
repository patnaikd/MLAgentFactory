# Session Manager - Long-Running Agent Architecture

This document describes the Session Manager architecture for MLAgentFactory, which provides a standalone service for hosting long-running ChatAgent instances with pull-based message streaming.

## Overview

The Session Manager solves the problem of running agents in isolated processes with reliable message delivery. Instead of running agents directly in the UI process, agents run in separate processes and communicate through a REST API with persistent message storage.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Layer                             │
│  ┌─────────────────────┐         ┌──────────────────────────┐   │
│  │  Streamlit UI       │         │   Any REST Client        │   │
│  │  (Pull-based poll)  │         │   (curl, Python, etc.)   │   │
│  └──────────┬──────────┘         └────────────┬─────────────┘   │
│             │                                   │                 │
└─────────────┼───────────────────────────────────┼─────────────────┘
              │                                   │
              │         HTTP/REST API             │
              │                                   │
┌─────────────▼───────────────────────────────────▼─────────────────┐
│                    FastAPI REST API Layer                         │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  Endpoints: /sessions, /sessions/{id}/query,             │    │
│  │            /sessions/{id}/messages                       │    │
│  └────────────────────┬─────────────────────────────────────┘    │
└───────────────────────┼──────────────────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────────────────┐
│                    Session Manager                                │
│  ┌────────────────────────────────────────────────────────┐      │
│  │  - Orchestrates sessions and processes                 │      │
│  │  - Coordinates Message Store and Process Manager       │      │
│  └─────────────┬──────────────────────────┬───────────────┘      │
│                │                           │                      │
│    ┌───────────▼────────────┐   ┌─────────▼─────────────┐        │
│    │   Message Store         │   │   Process Manager     │        │
│    │   (SQLite)              │   │   (Multiprocessing)   │        │
│    │                         │   │                       │        │
│    │  - sessions table       │   │  - AgentProcess 1     │        │
│    │  - messages table       │   │  - AgentProcess 2     │        │
│    │  - Cursor pagination    │   │  - AgentProcess N     │        │
│    └─────────────────────────┘   └─────────┬─────────────┘        │
└──────────────────────────────────────────────┼──────────────────┘
                                               │
                        ┌──────────────────────┴───────────────────┐
                        │                                          │
            ┌───────────▼────────────┐           ┌─────────────────▼─────────┐
            │  ChatAgent Process 1   │           │  ChatAgent Process N      │
            │  (Isolated Process)    │    ...    │  (Isolated Process)       │
            │                        │           │                           │
            │  - ClaudeSDKClient     │           │  - ClaudeSDKClient        │
            │  - Message streaming   │           │  - Message streaming      │
            │  - Tool execution      │           │  - Tool execution         │
            └────────────────────────┘           └───────────────────────────┘
```

## Components

### 1. Message Store (`src/mlagentfactory/services/message_store.py`)

- **Purpose**: Persistent storage for sessions and messages
- **Database**: SQLite (`data/sessions.db`)
- **Features**:
  - Session lifecycle management (created → running → stopped/failed)
  - Message storage with auto-incrementing IDs for cursor-based pagination
  - Thread-safe operations with context managers
  - JSON storage for flexible message content

**Key Methods**:
```python
# Session operations
create_session(session_id, metadata)
get_session(session_id)
update_session_status(session_id, status)
list_sessions(limit)
delete_session(session_id)

# Message operations
add_message(session_id, message_type, content)
get_messages(session_id, since_message_id, limit)
get_message_count(session_id)
```

### 2. Process Manager (`src/mlagentfactory/services/process_manager.py`)

- **Purpose**: Manage isolated agent processes
- **Technology**: Python multiprocessing with Queues
- **Features**:
  - Spawn agents in separate processes for isolation
  - IPC via multiprocessing Queues
  - Graceful shutdown with timeout
  - Process health monitoring

**Key Classes**:
```python
class AgentProcess:
    start()              # Start the agent process
    send_query(message)  # Send query to agent
    stop(timeout)        # Gracefully stop process
    is_alive()          # Check if process is running
    get_messages()       # Get messages from queue

class ProcessManager:
    create_process(session_id)
    get_process(session_id)
    stop_process(session_id)
    stop_all()
    cleanup_dead_processes()
```

### 3. Session Manager (`src/mlagentfactory/services/session_manager.py`)

- **Purpose**: High-level orchestration layer
- **Features**:
  - Unified API for session operations
  - Coordinates Message Store and Process Manager
  - Session lifecycle management
  - Health checks and statistics

**Key Methods**:
```python
create_session(metadata)
get_session(session_id)
send_query(session_id, message)
get_messages(session_id, since_message_id, limit)
stop_session(session_id)
delete_session(session_id)
get_session_stats(session_id)
health_check()
```

### 4. FastAPI REST API (`src/mlagentfactory/services/api.py`)

- **Purpose**: RESTful interface for session management
- **Features**:
  - Auto-generated OpenAPI/Swagger docs at `/docs`
  - CORS-enabled for web clients
  - Pydantic models for request/response validation
  - Proper error handling with HTTP status codes

**Endpoints**:
```
POST   /sessions                      # Create session
GET    /sessions                      # List sessions
GET    /sessions/{id}                 # Get session info
GET    /sessions/{id}/stats           # Get session stats
POST   /sessions/{id}/query           # Send query
GET    /sessions/{id}/messages        # Poll messages
DELETE /sessions/{id}                 # Stop session
DELETE /sessions/{id}/data            # Delete session data
GET    /health                        # Health check
```

### 5. Streamlit UI (`src/mlagentfactory/ui/streamlit_ui_session_manager.py`)

- **Purpose**: Web interface with API integration
- **Features**:
  - Pull-based message polling (no WebSockets)
  - Cursor-based pagination
  - Real-time todo list and logs
  - Session statistics display

## Message Flow

### Creating a Session and Sending a Query

```
1. Client → API: POST /sessions
   └─> SessionManager.create_session()
       └─> MessageStore.create_session()
       └─> ProcessManager.create_process()
           └─> Spawn AgentProcess (new process)
               └─> Initialize ChatAgent
               └─> Wait for commands on queue

2. Client → API: POST /sessions/{id}/query
   └─> SessionManager.send_query(message)
       └─> AgentProcess.send_query()
           └─> Put message in command_queue

3. AgentProcess (in separate process):
   └─> Read from command_queue
   └─> ChatAgent.send_message()
       └─> Stream chunks from Claude SDK
           └─> For each chunk:
               └─> MessageStore.add_message()
               └─> Put in message_queue

4. Client → API: GET /sessions/{id}/messages?since_message_id=N
   └─> SessionManager.get_messages()
       └─> MessageStore.get_messages(since_message_id)
           └─> Return messages with next_cursor

5. Client: Repeat step 4 with next_cursor until done
```

## Usage Guide

### Starting the Service

```bash
# Option 1: Using startup script
./start_session_manager.sh

# Option 2: Using CLI
uv run python -m mlagentfactory.cli.session_manager_cli start

# With auto-reload for development
./start_session_manager.sh --reload
```

### Starting the UI

```bash
# Option 1: Using startup script
./run_streamlit_session_manager.sh

# Option 2: Direct
uv run streamlit run src/mlagentfactory/ui/streamlit_ui_session_manager.py
```

### Using the CLI

```bash
# Check service status
uv run python -m mlagentfactory.cli.session_manager_cli status

# List sessions
uv run python -m mlagentfactory.cli.session_manager_cli sessions

# Get session details
uv run python -m mlagentfactory.cli.session_manager_cli info session-abc123

# Stop a session
uv run python -m mlagentfactory.cli.session_manager_cli stop-session session-abc123

# Stop the service
uv run python -m mlagentfactory.cli.session_manager_cli stop
```

### Using the API Directly

```bash
# Create a session
curl -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{"metadata": {"client": "curl"}}'

# Send a query
curl -X POST http://localhost:8000/sessions/session-abc123/query \
  -H "Content-Type: application/json" \
  -d '{"message": "What is 2+2?"}'

# Poll for messages
curl http://localhost:8000/sessions/session-abc123/messages?since_message_id=0

# Get session stats
curl http://localhost:8000/sessions/session-abc123/stats

# Stop session
curl -X DELETE http://localhost:8000/sessions/session-abc123
```

## Database Schema

### Sessions Table

```sql
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,           -- created, running, completed, failed, stopped
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    agent_session_id TEXT,          -- Agent's internal session ID
    total_cost REAL DEFAULT 0.0,    -- Total cost in USD
    metadata TEXT                   -- JSON metadata
);
```

### Messages Table

```sql
CREATE TABLE messages (
    message_id INTEGER PRIMARY KEY AUTOINCREMENT,  -- Auto-incrementing cursor
    session_id TEXT NOT NULL,
    message_type TEXT NOT NULL,  -- text, tool_use, tool_result, etc.
    content TEXT NOT NULL,       -- JSON content
    created_at TIMESTAMP NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX idx_messages_session_id ON messages(session_id, message_id);
```

## Message Types

Messages stored in the database have the following types:

- **text**: Assistant's text response
- **tool_use**: Tool being invoked (includes tool name and input)
- **tool_result**: Result from tool execution
- **todo_update**: Updated task list
- **session_id**: Agent's internal session ID
- **total_cost**: Updated total cost

Each message's `content` field contains a JSON object with the full message data.

## Cursor-based Pagination

The API uses cursor-based pagination for efficient message retrieval:

1. Client starts with `since_message_id=0`
2. Server returns messages with IDs > 0 (up to limit)
3. Server includes `next_cursor` in response (last message_id returned)
4. Client stores `next_cursor` and uses it for next poll
5. Repeat until `has_more=false` or no new messages

**Advantages**:
- No offset drift (new messages don't affect pagination)
- Efficient database queries with indexed `message_id`
- Client always gets consistent results

## Configuration

### Environment Variables

```bash
# API Configuration (optional, defaults shown)
API_BASE_URL=http://localhost:8000
API_PORT=8000

# Database Configuration
DB_PATH=data/sessions.db

# Polling Configuration
POLL_INTERVAL=0.5  # seconds
```

### File Locations

```
MLAgentFactory/
├── data/
│   └── sessions.db              # SQLite database
├── logs/
│   └── session-*.log            # Session logs (if enabled)
├── src/mlagentfactory/
│   ├── services/
│   │   ├── message_store.py     # Message storage
│   │   ├── process_manager.py   # Process management
│   │   ├── session_manager.py   # Session orchestration
│   │   └── api.py               # REST API
│   ├── cli/
│   │   └── session_manager_cli.py  # CLI tool
│   └── ui/
│       └── streamlit_ui_session_manager.py  # Streamlit UI
├── start_session_manager.sh     # Service startup script
└── run_streamlit_session_manager.sh  # UI startup script
```

## Development

### Adding New Features

1. **New message types**: Add to `MessageType` enum in `message_store.py`
2. **New API endpoints**: Add to `api.py` with proper Pydantic models
3. **UI enhancements**: Modify `streamlit_ui_session_manager.py`

### Testing

```bash
# Start the service in development mode (auto-reload)
./start_session_manager.sh --reload

# In another terminal, test the API
curl http://localhost:8000/health

# View API docs
open http://localhost:8000/docs
```

### Debugging

- **API logs**: Check FastAPI console output
- **Agent logs**: Messages include logging context
- **Database inspection**: Use SQLite browser on `data/sessions.db`
- **Process monitoring**: Use CLI `status` and `sessions` commands

## Design Decisions

### Why SQLite?

- Simple, no external dependencies
- File-based, easy backup
- Good performance for < 1000 concurrent sessions
- Can be upgraded to PostgreSQL if needed

### Why Multiprocessing?

- True process isolation (crash in one agent doesn't affect others)
- Better resource management
- Easier to reason about than threading
- Can leverage multiple CPU cores

### Why Pull-based Polling?

- Simpler than WebSockets or SSE
- Works through firewalls/proxies
- Easier to debug and test
- Client controls polling rate
- Stateless API design

### Why Cursor-based Pagination?

- No offset drift issues
- Efficient database queries
- Consistent results under concurrent writes
- Standard pattern for real-time systems

## Future Enhancements

Potential improvements:

1. **Redis Backend**: For better multi-instance scaling
2. **WebSocket Support**: For lower latency (optional alternative to polling)
3. **Session Replay**: Store and replay entire conversations
4. **Rate Limiting**: Protect API from abuse
5. **Authentication**: API keys or OAuth for production
6. **Monitoring**: Prometheus metrics, Grafana dashboards
7. **Docker**: Containerization for easier deployment

## Troubleshooting

### Service won't start

```bash
# Check if port 8000 is in use
lsof -i :8000

# Check logs for errors
uv run python -m mlagentfactory.cli.session_manager_cli start
```

### Messages not appearing

```bash
# Check session status
uv run python -m mlagentfactory.cli.session_manager_cli info <session-id>

# Check if process is alive
uv run python -m mlagentfactory.cli.session_manager_cli sessions

# Check database directly
sqlite3 data/sessions.db "SELECT * FROM messages WHERE session_id='<session-id>';"
```

### Process crashes

- Check agent logs for errors
- Verify environment variables (API keys, etc.)
- Check disk space and memory
- Review error messages in database

## License

Part of MLAgentFactory project.
