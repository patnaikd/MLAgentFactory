"""Streamlit UI for MLAgentFactory with Session Manager API integration.

This UI uses a pull-based approach to fetch messages from the Session Manager API,
replacing direct agent instantiation with REST API calls.
"""

import streamlit as st
import streamlit.components.v1 as components
import requests
import logging
import time
import base64
from io import BytesIO
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple, Union

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

def process_message_chunk(chunk: Dict[str, Any]) -> Union[Tuple[str, List[bytes]], str, None]:
    """Process a message chunk and return display content.

    Args:
        chunk: Message chunk from API (the 'content' field from the message)

    Returns:
        Either:
        - A tuple of (text, list_of_image_bytes) for messages with images
        - A string for text-only messages
        - None for metadata updates
    """
    # chunk is already the 'content' object from the API message
    # It has structure: {"type": "text", "content": "...", ...}
    content = chunk.get("content")
    message_type = chunk.get("type")  # Fixed: was "message_type", should be "type"

    logging.debug(f"Processing chunk: type={message_type}, has_content={content is not None}")

    if message_type == "text":
        return content

    elif message_type == "tool_use":
        tool_name = chunk.get("tool_name", "")  # Fixed: get from chunk, not content
        tool_input = chunk.get("tool_input", {})  # Fixed: get from chunk, not content

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
            # Default tool display - use content variable which is chunk.get("content")
            tool_display = content if content else f"Using tool: {tool_name}"
            return f"\n\n*{tool_display}*\n\n"

    elif message_type == "tool_result":
        result_content = content  # content is already chunk.get("content")
        is_error = chunk.get("is_error", False)  # Fixed: get is_error from chunk

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
                    return f"\n\n**🔧 Tool Result (json):**\n```json\n{display_content}\n```\n\n"
                elif display_content.strip().startswith('<'):
                    return f"\n\n**🔧 Tool Result (html):**\n```html\n{display_content}\n```\n\n"
                else:
                    return f"\n\n**🔧 Tool Result: (text)**\n```\n{display_content}\n```\n\n"
            elif isinstance(result_content, list):
                # Check if it's a structured list with 'type' fields (text/image)
                if result_content and isinstance(result_content[0], dict) and 'type' in result_content[0]:
                    formatted_parts = []
                    images = []  # Collect decoded images
                    formatted_parts.append(f"\n\n**🔧 Tool Result:** ({len(result_content)} items)\n")

                    for idx, item in enumerate(result_content, 1):
                        item_type = item.get('type', 'unknown')

                        if item_type == 'text':
                            text_content = item.get('text', '')
                            # Truncate long text content
                            max_display_length = 2000
                            if len(text_content) > max_display_length:
                                text_content = text_content[:max_display_length] + "\n... (truncated)"
                            formatted_parts.append(f"\n**Item {idx} (text):**\n```\n{text_content}\n```\n")

                        elif item_type == 'image':
                            source = item.get('source', {})
                            source_type = source.get('type', 'unknown')

                            if source_type == 'base64':
                                # Get image data and media type
                                image_data = source.get('data', '')
                                media_type = source.get('media_type', 'image/png')

                                # Try to decode and display the image
                                try:
                                    image_bytes = base64.b64decode(image_data)
                                    images.append(image_bytes)
                                    formatted_parts.append(f"\n**Item {idx} (image):**\n*[Image displayed below - {len(image_bytes)} bytes, {media_type}]*\n")
                                except Exception as e:
                                    formatted_parts.append(f"\n**Item {idx} (image):**\n*[Unable to decode image: {str(e)}]*\n")
                            else:
                                formatted_parts.append(f"\n**Item {idx} (image):**\n- Source type: {source_type}\n")

                        else:
                            # Unknown type, show as JSON
                            formatted_parts.append(f"\n**Item {idx} ({item_type}):**\n```json\n{item}\n```\n")

                    text_result = "".join(formatted_parts) + "\n"

                    # Return tuple of (text, images) if images exist, otherwise just text
                    if images:
                        return (text_result, images)
                    else:
                        return text_result
                else:
                    # Generic list, show as JSON
                    return f"\n\n**🔧 Tool Result (json-list):** ({len(result_content)} items)\n```json\n{result_content}\n```\n\n"
            else:
                return f"\n\n**🔧 Tool Result (unknown type):**\n```\n{str(result_content)}\n```\n\n"

    elif message_type == "session_id":
        st.session_state.agent_session_id = content
        return None

    elif message_type == "total_cost":
        st.session_state.total_cost = content
        return None

    elif message_type == "todo_update":
        # content should be an array of todo items
        if content and isinstance(content, list):
            st.session_state.current_todos = content
            logging.info(f"Updated todos: {len(content)} items - statuses: {[t.get('status') for t in content]}")
        else:
            logging.warning(f"Received invalid todo_update content: {type(content)}")
            st.session_state.current_todos = []
        return None

    elif message_type == "processing_complete":
        # Signal that agent is done processing
        logging.info("Received processing_complete signal - agent finished processing query")
        st.session_state.is_processing = False
        return None

    return None


def render_chat_message(role: str, content: Union[str, Dict[str, Any]]):
    """Render a chat message with appropriate styling.

    Args:
        role: Message role (user/assistant)
        content: Either a string (text only) or dict with 'text' and optional 'images' keys
    """
    with st.chat_message(role):
        # Handle both old string format and new dict format
        if isinstance(content, dict):
            # Render text content
            if content.get('text'):
                st.markdown(content['text'])

            # Render images
            if content.get('images'):
                for img_bytes in content['images']:
                    try:
                        st.image(img_bytes, use_container_width=True)
                    except Exception as e:
                        st.error(f"Failed to display image: {e}")
        else:
            # Legacy string format
            st.markdown(content)


@st.fragment(run_every=f"{POLL_INTERVAL}s")
def poll_and_display_messages():
    """Poll for new messages from the API and update state.

    This fragment runs every POLL_INTERVAL seconds to fetch messages and update state.
    The actual display is handled by the HTML component in the main chat container.
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
        return

    messages = result.get("messages", [])

    # Process new messages if any
    if messages:
        for msg in messages:
            message_id = msg["message_id"]
            chunk = msg["content"]

            # Process the chunk and add to current response
            result = process_message_chunk(chunk)
            if result:
                # Handle tuple (text, images) or just text
                if isinstance(result, tuple):
                    display_text, images = result
                    # Store text separately
                    if display_text:
                        if isinstance(st.session_state.current_response, str):
                            st.session_state.current_response = {'text': st.session_state.current_response, 'images': []}
                        st.session_state.current_response['text'] += display_text

                    # Store images separately
                    if images:
                        if isinstance(st.session_state.current_response, str):
                            st.session_state.current_response = {'text': st.session_state.current_response, 'images': []}
                        st.session_state.current_response['images'].extend(images)
                else:
                    # Just text
                    if isinstance(st.session_state.current_response, dict):
                        st.session_state.current_response['text'] += result
                    else:
                        st.session_state.current_response += result

            # Update last_message_id
            st.session_state.last_message_id = message_id

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

        # Trigger rerun to update the HTML display with new content
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
        page_title="MLAgentFactory",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": "MLAgentFactory is your Agentic AI for solving machine learning problems.",
        }
    )

    # Initialize observability
    initialize_observability(log_level="DEBUG", enable_tracing=False)

    # Initialize session state
    initialize_session_state()

    # Setup logging
    setup_logging()

    # Header
    st.title("🤖 MLAgentFactory")

    # Display session info
    if st.session_state.session_id:
        cost_display = f" | 💰 Total Cost: ${st.session_state.total_cost:.4f}" if st.session_state.total_cost > 0 else ""
        st.caption(f"🔗 Session ID: `{st.session_state.session_id}`{cost_display}")

    # Sidebar
    with st.sidebar:
        # st.markdown("## 💬 Chat Assistant")
        # st.markdown("""
        # This assistant uses the Session Manager API for long-running agent sessions.

        # Features:
        # - Isolated agent processes
        # - Pull-based message streaming
        # - Persistent message storage
        # - Session management
        # """)

        # st.markdown("---")

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
            # Create a scrollable container with fixed height
            chat_container = st.container(height=800)

            with chat_container:
                # Display all messages using Streamlit's native chat_message
                for message in st.session_state.messages:
                    render_chat_message(message["role"], message["content"])

                # Display current response if processing
                if st.session_state.is_processing and st.session_state.current_response:
                    with st.chat_message("assistant"):
                        # Handle both string and dict format
                        if isinstance(st.session_state.current_response, dict):
                            if st.session_state.current_response.get('text'):
                                st.markdown(st.session_state.current_response['text'])
                            if st.session_state.current_response.get('images'):
                                for img_bytes in st.session_state.current_response['images']:
                                    try:
                                        st.image(img_bytes, use_container_width=True)
                                    except Exception as e:
                                        st.error(f"Failed to display image: {e}")
                        else:
                            st.markdown(st.session_state.current_response)
                        st.caption("*typing...*")

                # Add a marker element at the bottom for auto-scroll
                st.markdown('<div id="bottom-marker"></div>', unsafe_allow_html=True)

            # Add auto-scroll JavaScript
            st.markdown("""
                <script>
                    function scrollChatToBottom() {
                        const marker = document.getElementById('bottom-marker');
                        if (marker) {
                            marker.scrollIntoView({ behavior: 'smooth', block: 'end' });
                        }
                    }
                    // Run multiple times to ensure content is loaded
                    setTimeout(scrollChatToBottom, 100);
                    setTimeout(scrollChatToBottom, 300);
                    setTimeout(scrollChatToBottom, 500);

                    // Set up observer to auto-scroll when content changes
                    const observer = new MutationObserver(scrollChatToBottom);
                    const container = document.querySelector('[data-testid="stVerticalBlock"]');
                    if (container) {
                        observer.observe(container, { childList: true, subtree: true });
                    }
                </script>
            """, unsafe_allow_html=True)

            # Fixed section at bottom for status and input
            bottom_container = st.container()

            with bottom_container:
                # Show thinking indicator if processing
                if st.session_state.is_processing:
                    with st.status("🤖 Agent is thinking...", expanded=False, state="running"):
                        st.caption(f"Polling for responses... (last message ID: {st.session_state.last_message_id})")

                        # Calculate response length
                        if isinstance(st.session_state.current_response, dict):
                            text_len = len(st.session_state.current_response.get('text', ''))
                            img_count = len(st.session_state.current_response.get('images', []))
                            st.caption(f"Messages in response: {text_len} characters, {img_count} images")
                        else:
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

                # Chat input (always visible at bottom)
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

            # Poll for messages if processing (hidden fragment)
            if st.session_state.is_processing:
                poll_and_display_messages()

    with tab2:
        render_logs_tab()


if __name__ == "__main__":
    main()
