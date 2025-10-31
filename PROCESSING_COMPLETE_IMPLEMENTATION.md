# Processing Complete Signal Implementation

## Overview

This document describes the implementation of a `processing_complete` signal that allows the chat agent to notify the Streamlit UI when it has finished processing a query, enabling the UI to stop polling for messages and re-enable user input.

## Architecture

The implementation follows the existing message flow pattern:

```
ChatAgent → ProcessManager → MessageStore → API → Streamlit UI
```

## Changes Made

### 1. Message Type Addition (`src/mlagentfactory/services/message_store.py`)

Added new message type to the `MessageType` enum:

```python
class MessageType(str, Enum):
    """Message type enum."""
    TEXT = "text"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    TODO_UPDATE = "todo_update"
    SESSION_ID = "session_id"
    TOTAL_COST = "total_cost"
    PROCESSING_COMPLETE = "processing_complete"  # NEW
```

### 2. ChatAgent Update (`src/mlagentfactory/agents/chat_agent.py`)

Modified the `send_message()` method to emit a `processing_complete` signal after the response stream finishes:

```python
async def send_message(self, message: str) -> AsyncGenerator[Dict, None]:
    # ... existing code ...

    # Stream all messages from agent
    async for msg in self.client.receive_response():
        # ... process messages ...

    # Emit processing_complete signal
    logger.info("[INCOMING] Emitting processing_complete signal")
    yield {
        "type": "processing_complete",
        "content": "Agent has finished processing this query"
    }
```

**Updated docstring** to document the new message type:
- Added `"processing_complete"` to the list of yielded message types

### 3. Process Manager Update (`src/mlagentfactory/services/process_manager.py`)

Added logging for the `processing_complete` message (storage is automatic):

```python
# Handle special message types
if message_type == MessageType.SESSION_ID:
    logger.info(f"[PROCESS-{session_id[:8]}] Captured agent session ID: {content}")
    message_store.update_session_agent_id(session_id, content)
elif message_type == MessageType.TOTAL_COST:
    logger.info(f"[PROCESS-{session_id[:8]}] Updated total cost: ${content:.4f}")
    message_store.update_session_cost(session_id, content)
elif message_type == MessageType.PROCESSING_COMPLETE:  # NEW
    logger.info(f"[PROCESS-{session_id[:8]}] Processing completed for query")
```

### 4. Streamlit UI Update (`src/mlagentfactory/ui/streamlit_ui_session_manager.py`)

#### 4.1 Updated `process_message_chunk()` function:

Added handling for `processing_complete` message type to set `is_processing` flag to `False`:

```python
elif message_type == "processing_complete":
    # Signal that agent is done processing
    logging.info("Received processing_complete signal - agent finished processing query")
    st.session_state.is_processing = False
    return None
```

#### 4.2 Updated `poll_and_display_messages()` function:

Added logic to finalize the response and trigger UI update when processing completes:

```python
# Check if processing was completed in this batch
if not st.session_state.is_processing:
    # Finalize current response and add to message history
    if st.session_state.current_response:
        st.session_state.messages.append({
            "role": "assistant",
            "content": st.session_state.current_response,
            "timestamp": datetime.now()
        })
        st.session_state.current_response = ""

    # Trigger rerun to update UI and enable input
    st.rerun()
    return
```

## Flow Diagram

```
User sends query
    ↓
Streamlit UI: POST /sessions/{id}/query
    ↓
API: Accepts query, returns immediately
    ↓
ProcessManager: Sends query to ChatAgent process
    ↓
ChatAgent: Processes query and streams messages
    ├─→ text messages
    ├─→ tool_use messages
    ├─→ tool_result messages
    ├─→ todo_update messages
    ├─→ session_id message
    ├─→ total_cost message
    └─→ processing_complete message (NEW!)
    ↓
MessageStore: Stores all messages in SQLite
    ↓
Streamlit UI: Polls GET /sessions/{id}/messages
    ├─→ Receives messages in batches
    ├─→ Displays messages incrementally
    ├─→ Detects processing_complete
    ├─→ Sets is_processing = False
    ├─→ Finalizes response
    ├─→ Triggers st.rerun()
    └─→ Chat input becomes enabled again
```

## Benefits

1. **Automatic Input Re-enabling**: The chat input automatically becomes enabled when the agent finishes processing, without manual intervention or the "Stop Processing" button.

2. **Clean State Management**: The `is_processing` flag is automatically set to `False` by the agent itself, ensuring consistency.

3. **Better UX**: Users can see exactly when the agent is done and ready for the next query.

4. **Backward Compatible**: The change doesn't break existing functionality; the "Stop Processing" button still works for manual interruption.

## Testing

A test script (`test_processing_complete.py`) was created to verify the complete flow:

```bash
uv run python test_processing_complete.py
```

Test results:
- ✅ Session created successfully
- ✅ Query sent to agent
- ✅ Messages polled from API
- ✅ `processing_complete` signal received
- ✅ Total messages: 4 (session_id, text, total_cost, processing_complete)

## Usage

No changes are required for users. The feature works automatically:

1. Start the Session Manager API:
   ```bash
   ./start_session_manager.sh
   ```

2. Start the Streamlit UI:
   ```bash
   ./run_streamlit_session_manager.sh
   ```

3. Send a query in the UI
4. The chat input will automatically become enabled when the agent finishes processing

## Implementation Notes

- The `processing_complete` message is always emitted as the **last message** in the response stream
- It's stored in the database like any other message for consistency
- The Streamlit UI checks for this message during polling and immediately stops polling and re-enables input
- The message includes both `type` and `content` fields for consistency with other message types

## Future Enhancements

Potential improvements:
- Add a visual indicator (e.g., "✅ Response complete") when processing finishes
- Add typing indicators while the agent is processing
- Support for partial/incremental responses with multiple completion signals
- Add completion timestamps for performance tracking
