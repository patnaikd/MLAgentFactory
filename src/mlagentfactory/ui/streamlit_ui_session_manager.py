"""Streamlit UI for MLAgentFactory with Session Manager API integration.

This UI uses a pull-based approach to fetch messages from the Session Manager API,
replacing direct agent instantiation with REST API calls.
"""

import streamlit as st
import streamlit.components.v1 as components
import requests
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

# Handle both relative and absolute imports for flexibility
try:
    from ..utils.logging_config import initialize_observability
except ImportError:
    from mlagentfactory.utils.logging_config import initialize_observability


# Configuration
API_BASE_URL = "http://localhost:8000"  # Session Manager API URL
POLL_INTERVAL = 0.5  # Polling interval in seconds


# Custom logging handler that captures logs for display
class StreamlitLogHandler(logging.Handler):
    """Custom logging handler that stores logs in Streamlit session state."""

    def __init__(self):
        super().__init__()

    def emit(self, record):
        if 'log_messages' in st.session_state:
            log_entry = {
                'time': datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                'level': record.levelname,
                'name': record.name,
                'message': self.format(record)
            }
            st.session_state.log_messages.append(log_entry)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "log_messages" not in st.session_state:
        st.session_state.log_messages = []

    if "session_id" not in st.session_state:
        st.session_state.session_id = None

    if "agent_session_id" not in st.session_state:
        st.session_state.agent_session_id = None

    if "total_cost" not in st.session_state:
        st.session_state.total_cost = 0.0

    if "current_todos" not in st.session_state:
        st.session_state.current_todos = []

    if "last_message_id" not in st.session_state:
        st.session_state.last_message_id = 0

    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False

    if "current_response" not in st.session_state:
        st.session_state.current_response = ""


def setup_logging():
    """Configure logging with custom handler for Streamlit."""
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Check if StreamlitLogHandler is already added
    has_streamlit_handler = any(
        isinstance(handler, StreamlitLogHandler) for handler in root_logger.handlers
    )

    if not has_streamlit_handler:
        streamlit_handler = StreamlitLogHandler()
        streamlit_handler.setFormatter(formatter)
        streamlit_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(streamlit_handler)


# ===========================
# API Client Functions
# ===========================

def create_session(metadata: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Create a new session via API.

    Args:
        metadata: Optional metadata for the session

    Returns:
        Session information or None on error
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/sessions",
            json={"metadata": metadata},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"Failed to create session: {e}")
        st.error(f"Failed to create session: {e}")
        return None


def send_query(session_id: str, message: str) -> bool:
    """Send a query to the session via API.

    Args:
        session_id: Session identifier
        message: User message

    Returns:
        True on success, False on error
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/sessions/{session_id}/query",
            json={"message": message},
            timeout=10
        )
        response.raise_for_status()
        return True
    except Exception as e:
        logging.error(f"Failed to send query: {e}")
        st.error(f"Failed to send query: {e}")
        return False


def get_messages(session_id: str, since_message_id: int = 0, limit: int = 100) -> Optional[Dict[str, Any]]:
    """Get messages from the session via API.

    Args:
        session_id: Session identifier
        since_message_id: Get messages after this ID
        limit: Maximum number of messages

    Returns:
        Messages response or None on error
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/sessions/{session_id}/messages",
            params={"since_message_id": since_message_id, "limit": limit},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"Failed to get messages: {e}")
        return None


def get_session_stats(session_id: str) -> Optional[Dict[str, Any]]:
    """Get session statistics via API.

    Args:
        session_id: Session identifier

    Returns:
        Session statistics or None on error
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/sessions/{session_id}/stats",
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"Failed to get session stats: {e}")
        return None


def stop_session(session_id: str) -> bool:
    """Stop a session via API.

    Args:
        session_id: Session identifier

    Returns:
        True on success, False on error
    """
    try:
        response = requests.delete(
            f"{API_BASE_URL}/sessions/{session_id}",
            timeout=10
        )
        response.raise_for_status()
        return True
    except Exception as e:
        logging.error(f"Failed to stop session: {e}")
        st.error(f"Failed to stop session: {e}")
        return False


# ===========================
# Message Processing
# ===========================

def process_message_chunk(chunk: Dict[str, Any]) -> Optional[str]:
    """Process a message chunk and return display text.

    Args:
        chunk: Message chunk from API

    Returns:
        Display text or None
    """
    content = chunk.get("content")
    message_type = chunk.get("message_type")

    if message_type == "text":
        return content

    elif message_type == "tool_use":
        tool_name = content.get("tool_name", "")
        tool_input = content.get("tool_input", {})

        # Special display for Bash commands
        if tool_name == "Bash" and tool_input:
            command = tool_input.get("command", "")
            description = tool_input.get("description", "")

            if description:
                return f"\n\n**🔨 {description}**\n```bash\n{command}\n```\n\n"
            else:
                return f"\n\n**🔨 Running bash command:**\n```bash\n{command}\n```\n\n"

        # Special display for file operation tools
        elif tool_name in ["Write", "Read", "Edit", "Delete"] and tool_input:
            file_path = tool_input.get("file_path", "")

            tool_emoji = {
                "Write": "📝",
                "Read": "👁️",
                "Edit": "✏️",
                "Delete": "🗑️"
            }
            emoji = tool_emoji.get(tool_name, "📄")

            if file_path:
                return f"\n\n**{emoji} {tool_name}: `{file_path}`**\n\n"
            else:
                return f"\n\n**{emoji} {tool_name}**\n\n"
        else:
            # Default tool display
            tool_display = content.get("content", f"Using tool: {tool_name}")
            return f"\n\n*{tool_display}*\n\n"

    elif message_type == "tool_result":
        result_content = content.get("content", "")
        is_error = content.get("is_error", False)

        if is_error:
            return f"\n\n**⚠️ Tool Execution Error:**\n```\n{result_content}\n```\n\n"
        else:
            # Format tool result
            if isinstance(result_content, str):
                max_display_length = 2000
                if len(result_content) > max_display_length:
                    display_content = result_content[:max_display_length] + "\n... (truncated)"
                else:
                    display_content = result_content

                # Detect content type
                if display_content.strip().startswith('{') or display_content.strip().startswith('['):
                    return f"\n\n**🔧 Tool Result:**\n```json\n{display_content}\n```\n\n"
                elif display_content.strip().startswith('<'):
                    return f"\n\n**🔧 Tool Result:**\n```html\n{display_content}\n```\n\n"
                else:
                    return f"\n\n**🔧 Tool Result:**\n```\n{display_content}\n```\n\n"
            elif isinstance(result_content, list):
                return f"\n\n**🔧 Tool Result:** ({len(result_content)} items)\n```json\n{result_content}\n```\n\n"
            else:
                return f"\n\n**🔧 Tool Result:**\n```\n{str(result_content)}\n```\n\n"

    elif message_type == "session_id":
        st.session_state.agent_session_id = content
        return None

    elif message_type == "total_cost":
        st.session_state.total_cost = content
        return None

    elif message_type == "todo_update":
        st.session_state.current_todos = content
        logging.debug(f"Updated todos: {len(content) if content else 0} items")
        return None

    return None


def render_chat_message(role: str, content: str):
    """Render a chat message with appropriate styling."""
    with st.chat_message(role):
        st.markdown(content)


@st.fragment(run_every=f"{POLL_INTERVAL}s")
def poll_and_display_messages():
    """Poll for new messages from the API and display them.

    This fragment runs every POLL_INTERVAL seconds to fetch and display new messages.
    """
    if not st.session_state.session_id:
        return

    if not st.session_state.is_processing:
        return

    # Get new messages since last_message_id
    result = get_messages(
        st.session_state.session_id,
        since_message_id=st.session_state.last_message_id,
        limit=100
    )

    if not result:
        # Still show current response even if no new messages
        if st.session_state.current_response:
            with st.chat_message("assistant"):
                st.markdown(st.session_state.current_response)
        return

    messages = result.get("messages", [])

    # Process new messages if any
    if messages:
        for msg in messages:
            message_id = msg["message_id"]
            chunk = msg["content"]

            # Process the chunk and add to current response
            display_text = process_message_chunk(chunk)
            if display_text:
                st.session_state.current_response += display_text

            # Update last_message_id
            st.session_state.last_message_id = message_id

    # Display the current response (updated or not)
    if st.session_state.current_response:
        with st.chat_message("assistant"):
            st.markdown(st.session_state.current_response)

    # Show status with stop button
    with st.status("🤖 Agent is thinking...", expanded=False, state="running"):
        st.caption(f"Polling for responses... (last message ID: {st.session_state.last_message_id})")
        st.caption(f"Messages in response: {len(st.session_state.current_response)} characters")

        # Stop processing button
        if st.button("⏹️ Stop Processing", key=f"stop_{st.session_state.last_message_id}"):
            # Finalize current response
            if st.session_state.current_response:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": st.session_state.current_response,
                    "timestamp": datetime.now()
                })

            st.session_state.is_processing = False
            st.session_state.current_response = ""
            st.rerun()


@st.fragment(run_every="1s")
def render_todo_list():
    """Render the todo list in the sidebar with auto-refresh."""
    if not st.session_state.current_todos:
        return

    st.markdown("### 📋 Task Progress")

    # Calculate progress
    total_tasks = len(st.session_state.current_todos)
    completed_tasks = sum(1 for todo in st.session_state.current_todos if todo.get("status") == "completed")
    in_progress_tasks = sum(1 for todo in st.session_state.current_todos if todo.get("status") == "in_progress")

    # Progress bar
    progress = completed_tasks / total_tasks if total_tasks > 0 else 0
    st.progress(progress, text=f"{completed_tasks}/{total_tasks} tasks completed")

    # Display todos
    for todo in st.session_state.current_todos:
        status = todo.get("status", "pending")
        content = todo.get("content", "")

        # Status emoji and color
        if status == "completed":
            emoji = "✅"
            style = "color: #28a745;"
        elif status == "in_progress":
            emoji = "🔄"
            style = "color: #ffc107;"
        else:  # pending
            emoji = "⏳"
            style = "color: #6c757d;"

        st.markdown(f"{emoji} <span style='{style}'>{content}</span>", unsafe_allow_html=True)

    # Summary
    st.caption(f"✅ {completed_tasks} completed | 🔄 {in_progress_tasks} in progress | ⏳ {total_tasks - completed_tasks - in_progress_tasks} pending")


@st.fragment(run_every="1s")
def render_sidebar_stats():
    """Render sidebar statistics that update in real-time."""
    if st.session_state.session_id:
        stats = get_session_stats(st.session_state.session_id)
        if stats:
            st.markdown(f"**Status:** {stats['status']}")
            st.markdown(f"**Process:** {'🟢 Running' if stats['process_alive'] else '🔴 Stopped'}")
            st.markdown(f"**Messages:** {stats['message_count']}")
            if stats.get('agent_session_id'):
                st.markdown(f"**Agent Session:** `{stats['agent_session_id'][:16]}...`")


@st.fragment(run_every="1s")
def render_logs_tab():
    """Render the logs display tab with auto-refresh."""
    st.header("📋 Application Logs")

    # Control bar
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        log_levels = st.multiselect(
            "Filter by level:",
            options=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            default=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            key="log_level_filter"
        )
    with col2:
        if st.button("Clear Logs", key="clear_logs_btn"):
            st.session_state.log_messages = []
            st.rerun()
    with col3:
        auto_scroll = st.checkbox("Auto-scroll", value=True, key="auto_scroll_checkbox")

    # Display logs
    if st.session_state.log_messages:
        filtered_logs = [
            log for log in st.session_state.log_messages
            if log['level'] in log_levels
        ]

        # Create scrollable log container with HTML (similar to original)
        log_html = """
        <!DOCTYPE html>
        <html>
        <head>
        <style>
        body { margin: 0; padding: 0; font-family: 'Courier New', monospace; }
        .log-container {
            font-family: 'Courier New', monospace;
            font-size: 12px;
            background-color: #1e1e1e;
            color: #d4d4d4;
            padding: 15px;
            border-radius: 5px;
            height: 600px;
            overflow-y: auto;
            border: 1px solid #3e3e3e;
        }
        .log-entry { margin-bottom: 5px; line-height: 1.5; word-wrap: break-word; }
        .log-time { color: #858585; }
        .log-level-DEBUG { color: #808080; font-weight: bold; }
        .log-level-INFO { color: #4ec9b0; font-weight: bold; }
        .log-level-WARNING { color: #dcdcaa; font-weight: bold; }
        .log-level-ERROR { color: #f48771; font-weight: bold; }
        .log-level-CRITICAL { color: #ff00ff; font-weight: bold; }
        .log-name { color: #9cdcfe; }
        .log-message { color: #d4d4d4; }
        </style>
        </head>
        <body>
        <div class="log-container" id="log-container">
        """

        for log in filtered_logs:
            message = log['message'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            log_html += f"""
            <div class="log-entry">
                <span class="log-time">{log['time']}</span>
                <span class="log-level-{log['level']}"> [{log['level']}]</span>
                <span class="log-name"> {log['name']}</span> -
                <span class="log-message"> {message}</span>
            </div>
            """

        log_html += "</div>"

        if auto_scroll:
            log_html += """
            <script>
                function scrollToBottom() {
                    var logContainer = document.getElementById('log-container');
                    if (logContainer) { logContainer.scrollTop = logContainer.scrollHeight; }
                }
                scrollToBottom();
                window.addEventListener('load', scrollToBottom);
                setTimeout(scrollToBottom, 50);
                setTimeout(scrollToBottom, 200);
            </script>
            """

        log_html += "</body></html>"

        components.html(log_html, height=650, scrolling=True)
        st.caption(f"📊 Showing {len(filtered_logs)} of {len(st.session_state.log_messages)} log entries")
    else:
        st.info("ℹ️ No logs yet. Start chatting to see activity!")


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="MLAgentFactory - Session Manager",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": "MLAgentFactory with Session Manager API integration.",
        }
    )

    # Initialize observability
    initialize_observability(log_level="DEBUG", enable_tracing=False)

    # Initialize session state
    initialize_session_state()

    # Setup logging
    setup_logging()

    # Header
    st.title("🤖 MLAgentFactory - Session Manager")

    # Display session info
    if st.session_state.session_id:
        cost_display = f" | 💰 Total Cost: ${st.session_state.total_cost:.4f}" if st.session_state.total_cost > 0 else ""
        st.caption(f"🔗 Session ID: `{st.session_state.session_id}`{cost_display}")

    # Sidebar
    with st.sidebar:
        st.markdown("## 💬 Chat Assistant")
        st.markdown("""
        This assistant uses the Session Manager API for long-running agent sessions.

        Features:
        - Isolated agent processes
        - Pull-based message streaming
        - Persistent message storage
        - Session management
        """)

        st.markdown("---")

        # Create new session button
        if st.button("🔄 Start New Session"):
            # Stop old session if exists
            if st.session_state.session_id:
                stop_session(st.session_state.session_id)

            # Reset state
            st.session_state.messages = []
            st.session_state.session_id = None
            st.session_state.agent_session_id = None
            st.session_state.total_cost = 0.0
            st.session_state.current_todos = []
            st.session_state.last_message_id = 0
            st.session_state.is_processing = False
            st.session_state.current_response = ""

            # Create new session
            session = create_session(metadata={"client": "streamlit"})
            if session:
                st.session_state.session_id = session["session_id"]
                st.success(f"New session created: {session['session_id']}")
                st.rerun()

        # Display real-time statistics
        render_sidebar_stats()

        st.markdown("---")

        # Display todo list
        render_todo_list()

    # Create tabs
    tab1, tab2 = st.tabs(["💬 Chat", "📋 Logs"])

    with tab1:
        # Ensure we have a session
        if not st.session_state.session_id:
            st.info("👈 Click 'Start New Session' in the sidebar to begin")
        else:
            # Display chat history
            for message in st.session_state.messages:
                render_chat_message(message["role"], message["content"])

            # Chat input
            if not st.session_state.is_processing:
                if prompt := st.chat_input("Type your message here..."):
                    # Add user message
                    st.session_state.messages.append({
                        "role": "user",
                        "content": prompt,
                        "timestamp": datetime.now()
                    })

                    # Send query to API
                    if send_query(st.session_state.session_id, prompt):
                        st.session_state.is_processing = True
                        st.session_state.current_response = ""
                        st.rerun()
            else:
                st.chat_input("Agent is processing...", disabled=True)

            # Poll for messages and display if processing
            if st.session_state.is_processing:
                poll_and_display_messages()

    with tab2:
        render_logs_tab()


if __name__ == "__main__":
    main()
