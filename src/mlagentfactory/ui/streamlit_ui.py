"""Streamlit UI for MLAgentFactory chat interface"""

import streamlit as st
import streamlit.components.v1 as components
import asyncio
import logging
import sys
import os
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from threading import Thread
from queue import Queue, Empty

from ..agents.chat_agent import ChatAgent
from ..utils.logging_config import initialize_observability

# Load environment variables from .env file
load_dotenv()


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
    """Initialize Streamlit session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "log_messages" not in st.session_state:
        st.session_state.log_messages = []

    if "agent" not in st.session_state:
        st.session_state.agent = None

    if "session_id" not in st.session_state:
        st.session_state.session_id = None

    if "total_cost" not in st.session_state:
        st.session_state.total_cost = 0.0

    if "session_file_handler" not in st.session_state:
        st.session_state.session_file_handler = None

    if "live_log_container" not in st.session_state:
        st.session_state.live_log_container = None

    if "last_displayed_log_count" not in st.session_state:
        st.session_state.last_displayed_log_count = 0

    if "current_todos" not in st.session_state:
        st.session_state.current_todos = []

    if "event_loop" not in st.session_state:
        # Create a persistent event loop for the session
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        st.session_state.event_loop = loop

    if "background_thread" not in st.session_state:
        st.session_state.background_thread = None

    if "chunk_queue" not in st.session_state:
        st.session_state.chunk_queue = None

    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False

    if "current_response" not in st.session_state:
        st.session_state.current_response = ""

    if "processing_error" not in st.session_state:
        st.session_state.processing_error = None


def setup_logging():
    """Configure logging with custom handler for Streamlit.

    Uses the same formatter as console logging for consistency, which includes
    filename and line number for easier debugging.
    """
    # Create formatter - same as console formatter in logging_config.py
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Get root logger
    root_logger = logging.getLogger()

    # Set root logger level to DEBUG to capture all logs
    root_logger.setLevel(logging.DEBUG)

    # Check if StreamlitLogHandler is already added
    has_streamlit_handler = any(
        isinstance(handler, StreamlitLogHandler) for handler in root_logger.handlers
    )

    if not has_streamlit_handler:
        # Add Streamlit handler
        streamlit_handler = StreamlitLogHandler()
        streamlit_handler.setFormatter(formatter)
        streamlit_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(streamlit_handler)

    # Re-attach file handler if session exists (for Streamlit reruns)
    # This is necessary because Streamlit reruns the script on each interaction
    # and the root logger gets reset, but session_state persists
    if "session_file_handler" in st.session_state and st.session_state.session_file_handler is not None:
        # Check if this handler is already in the root logger
        has_file_handler = st.session_state.session_file_handler in root_logger.handlers
        if not has_file_handler:
            root_logger.addHandler(st.session_state.session_file_handler)
            # Don't log on every rerun - too noisy during processing loops


def setup_session_file_logger(session_id: str):
    """Set up a file logger for the current session.

    Args:
        session_id: The session ID to use for the log file name
    """
    # Get root logger
    root_logger = logging.getLogger()

    # Remove previous session file handler if it exists
    if st.session_state.session_file_handler is not None:
        logging.info(f"Closing previous session file handler")
        root_logger.removeHandler(st.session_state.session_file_handler)
        st.session_state.session_file_handler.close()
        st.session_state.session_file_handler = None

    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Create new file handler for this session
    log_file_path = logs_dir / f"session-{session_id}.log"
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')

    # Create formatter - same as console formatter in logging_config.py
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # Add to root logger
    root_logger.addHandler(file_handler)

    # Store in session state
    st.session_state.session_file_handler = file_handler

    # Log the session start
    logging.info(f"Session file logger initialized: {log_file_path}")


def render_chat_message(role: str, content: str):
    """Render a chat message with appropriate styling"""
    with st.chat_message(role):
        st.markdown(content)


def get_or_create_agent():
    """Get existing agent or create a new one."""
    if st.session_state.agent is None:
        st.session_state.agent = ChatAgent()
    return st.session_state.agent


async def process_message_streaming(message: str, agent, previous_session_id):
    """Process a message through the agent with streaming response.

    Yields response chunks as they arrive asynchronously.
    This function should NOT access st.session_state directly to avoid
    ScriptRunContext issues when running in a background thread.

    Args:
        message: The message to process
        agent: The ChatAgent instance
        previous_session_id: The previous session ID for tracking changes

    Yields:
        tuple: (chunk_type, content) where chunk_type is one of:
            - "text": Display text content
            - "metadata": Metadata updates (session_id, total_cost, todo_update)
    """
    # Initialize if not already initialized
    if not agent._initialized:
        await agent.initialize()

    # Stream chunks from the agent
    async for chunk in agent.send_message(message):
        if chunk["type"] == "text":
            yield ("text", chunk["content"])
        elif chunk["type"] == "tool_use":
            # Show tool usage in the stream
            tool_display = chunk["content"]
            tool_name = chunk.get("tool_name", "")
            tool_input = chunk.get("tool_input", {})

            # Special display for Bash commands
            if tool_name == "Bash" and tool_input:
                command = tool_input.get("command", "")
                description = tool_input.get("description", "")

                if description:
                    yield ("text", f"\n\n**🔨 {description}**\n```bash\n{command}\n```\n\n")
                else:
                    yield ("text", f"\n\n**🔨 Running bash command:**\n```bash\n{command}\n```\n\n")
            # Special display for file operation tools
            elif tool_name in ["Write", "Read", "Edit", "Delete"] and tool_input:
                file_path = tool_input.get("file_path", "")

                # Map tool names to emojis
                tool_emoji = {
                    "Write": "📝",
                    "Read": "👁️",
                    "Edit": "✏️",
                    "Delete": "🗑️"
                }
                emoji = tool_emoji.get(tool_name, "📄")

                if file_path:
                    yield ("text", f"\n\n**{emoji} {tool_name}: `{file_path}`**\n\n")
                else:
                    yield ("text", f"\n\n**{emoji} {tool_name}**\n\n")
            else:
                # Default tool display
                yield ("text", f"\n\n*{tool_display}*\n\n")
        elif chunk["type"] == "tool_result":
            # Show tool results in code blocks
            content = chunk["content"]
            is_error = chunk.get("is_error", False)

            if is_error:
                # Display errors in a code block with error prefix
                yield ("text", f"\n\n**⚠️ Tool Execution Error:**\n```\n{content}\n```\n\n")
            else:
                # Format tool result content in code blocks
                if isinstance(content, str):
                    # Limit display length for very long results
                    max_display_length = 2000
                    if len(content) > max_display_length:
                        display_content = content[:max_display_length] + "\n... (truncated)"
                    else:
                        display_content = content

                    # Try to detect content type for syntax highlighting
                    if display_content.strip().startswith('{') or display_content.strip().startswith('['):
                        # Looks like JSON
                        yield ("text", f"\n\n**🔧 Tool Result:**\n```json\n{display_content}\n```\n\n")
                    elif display_content.strip().startswith('<'):
                        # Looks like HTML/XML
                        yield ("text", f"\n\n**🔧 Tool Result:**\n```html\n{display_content}\n```\n\n")
                    else:
                        # Plain text
                        yield ("text", f"\n\n**🔧 Tool Result:**\n```\n{display_content}\n```\n\n")
                elif isinstance(content, list):
                    # Format list content
                    yield ("text", f"\n\n**🔧 Tool Result:** ({len(content)} items)\n```json\n{content}\n```\n\n")
                else:
                    # Other types
                    yield ("text", f"\n\n**🔧 Tool Result:**\n```\n{str(content)}\n```\n\n")

        elif chunk["type"] == "session_id":
            # Pass session_id as metadata to be handled by main thread
            new_session_id = chunk["content"]
            yield ("metadata", {
                "type": "session_id",
                "content": new_session_id,
                "previous_session_id": previous_session_id
            })

        elif chunk["type"] == "total_cost":
            # Pass total_cost as metadata to be handled by main thread
            yield ("metadata", {
                "type": "total_cost",
                "content": chunk["content"]
            })

        elif chunk["type"] == "todo_update":
            # Pass todo_update as metadata to be handled by main thread
            yield ("metadata", {
                "type": "todo_update",
                "content": chunk["content"]
            })


def start_background_processing(async_gen):
    """Start processing async generator in a background thread.

    This function starts the background processing and returns immediately,
    allowing the Streamlit UI to remain unblocked. Chunks are placed in a queue
    that can be polled by the UI.

    Args:
        async_gen: The async generator to process
    """
    # Create a queue to pass chunks from the thread to the main thread
    chunk_queue = Queue()
    exception_holder = [None]  # Use list to hold exception from thread

    def run_in_thread():
        """Run the async generator in a separate thread with its own event loop."""
        try:
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def consume_generator():
                """Consume the async generator and put chunks in the queue."""
                try:
                    async for chunk_type, content in async_gen:
                        chunk_queue.put((chunk_type, content))
                        # Small yield point to allow other async tasks
                        await asyncio.sleep(0.001)
                except Exception as e:
                    exception_holder[0] = e
                    chunk_queue.put(("error", str(e)))
                finally:
                    # Signal completion
                    chunk_queue.put(("done", None))

            # Run the async generator
            loop.run_until_complete(consume_generator())
        except Exception as e:
            exception_holder[0] = e
            chunk_queue.put(("error", str(e)))
            chunk_queue.put(("done", None))
        finally:
            loop.close()

    # Start the background thread
    thread = Thread(target=run_in_thread, daemon=True)
    thread.start()

    # Store in session state
    st.session_state.background_thread = thread
    st.session_state.chunk_queue = chunk_queue
    st.session_state.is_processing = True
    st.session_state.current_response = ""
    st.session_state.processing_error = None


async def cleanup_agent():
    """Cleanup the agent."""
    if st.session_state.agent:
        await st.session_state.agent.cleanup()
        st.session_state.agent = None


@st.fragment(run_every="0.5s")
def render_todo_list():
    """Render the todo list in the sidebar with auto-refresh.

    This fragment auto-refreshes every 500ms to show todo updates in real-time
    as the agent processes tasks.
    """
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

        # Display with markdown
        st.markdown(f"{emoji} <span style='{style}'>{content}</span>", unsafe_allow_html=True)

    # Summary
    st.caption(f"✅ {completed_tasks} completed | 🔄 {in_progress_tasks} in progress | ⏳ {total_tasks - completed_tasks - in_progress_tasks} pending")


def poll_and_update_chunks():
    """Poll the background queue and update session state.

    This is called from within a fragment to process chunks without blocking.
    Must be called from the main Streamlit thread.
    """
    if not st.session_state.is_processing:
        return False

    chunk_queue = st.session_state.chunk_queue
    if chunk_queue is None:
        return False

    # Process all available chunks without blocking
    chunks_processed = 0
    max_chunks_per_poll = 10  # Process up to 10 chunks per poll to avoid blocking too long
    any_updates = False

    while chunks_processed < max_chunks_per_poll:
        try:
            # Non-blocking get
            msg_type, content = chunk_queue.get_nowait()
            any_updates = True

            if msg_type == "text":
                # Append text chunk to current response
                st.session_state.current_response += content
                chunks_processed += 1

            elif msg_type == "metadata":
                # Handle metadata updates (session_id, total_cost, todo_update)
                metadata_type = content.get("type")

                if metadata_type == "session_id":
                    new_session_id = content.get("content")
                    previous_session_id = content.get("previous_session_id")
                    st.session_state.session_id = new_session_id

                    # Set up file logger if session ID changed
                    if new_session_id != previous_session_id:
                        setup_session_file_logger(new_session_id)

                elif metadata_type == "total_cost":
                    st.session_state.total_cost = content.get("content")

                elif metadata_type == "todo_update":
                    todos = content.get("content")
                    st.session_state.current_todos = todos
                    logging.debug(f"Updated current_todos in session state: {len(todos) if todos else 0} todos")

            elif msg_type == "error":
                # Store error
                st.session_state.processing_error = content
                st.session_state.is_processing = False
                break

            elif msg_type == "done":
                # Processing complete
                st.session_state.is_processing = False
                break

        except Empty:
            # No more chunks available right now
            break

    return any_updates


@st.fragment(run_every="1s")
def render_sidebar_stats():
    """Render sidebar statistics that update in real-time."""
    # Display message count
    if st.session_state.messages:
        st.markdown(f"**Messages:** {len(st.session_state.messages)}")

    # Display log count
    if st.session_state.log_messages:
        st.markdown(f"**Log Entries:** {len(st.session_state.log_messages)}")

    # Display session log file path
    if st.session_state.session_id:
        log_file_path = f"logs/session-{st.session_state.session_id}.log"
        st.markdown(f"**Log File:** `{log_file_path}`")


@st.fragment(run_every="1s")
def render_logs_tab():
    """Render the logs display tab.

    This fragment auto-refreshes every 1 second to show real-time logs
    even while the agent is processing messages.
    """
    st.header("📋 Agent Logs")

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
        # Filter logs by selected levels
        filtered_logs = [
            log for log in st.session_state.log_messages
            if log['level'] in log_levels
        ]

        # Create scrollable log container with HTML
        log_html = """
        <!DOCTYPE html>
        <html>
        <head>
        <style>
        body {
            margin: 0;
            padding: 0;
            font-family: 'Courier New', monospace;
        }
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
        .log-entry {
            margin-bottom: 5px;
            line-height: 1.5;
            word-wrap: break-word;
        }
        .log-time {
            color: #858585;
        }
        .log-level-DEBUG {
            color: #808080;
            font-weight: bold;
        }
        .log-level-INFO {
            color: #4ec9b0;
            font-weight: bold;
        }
        .log-level-WARNING {
            color: #dcdcaa;
            font-weight: bold;
        }
        .log-level-ERROR {
            color: #f48771;
            font-weight: bold;
        }
        .log-level-CRITICAL {
            color: #ff00ff;
            font-weight: bold;
        }
        .log-name {
            color: #9cdcfe;
        }
        .log-message {
            color: #d4d4d4;
        }
        </style>
        </head>
        <body>
        <div class="log-container" id="log-container">
        """

        for log in filtered_logs:
            # Escape HTML characters in log message
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

        # Add JavaScript for auto-scrolling
        if auto_scroll:
            log_html += """
            <script>
                // Scroll to bottom on load
                function scrollToBottom() {
                    var logContainer = document.getElementById('log-container');
                    if (logContainer) {
                        logContainer.scrollTop = logContainer.scrollHeight;
                    }
                }

                // Execute immediately and after DOM is ready
                scrollToBottom();
                window.addEventListener('load', scrollToBottom);

                // Also scroll after a short delay to ensure content is rendered
                setTimeout(scrollToBottom, 50);
                setTimeout(scrollToBottom, 200);
            </script>
            """

        log_html += """
        </body>
        </html>
        """

        # Use components.html instead of st.markdown
        components.html(log_html, height=650, scrolling=True)

        # Display log count
        st.caption(f"📊 Showing {len(filtered_logs)} of {len(st.session_state.log_messages)} log entries")
    else:
        st.info("ℹ️ No logs yet. Start chatting to see agent activity!")


def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="MLAgentFactory",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": "MLAgentFactory is an AI chat assistant for developing machine learning models.",
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

    # Display session ID and cost if available
    if st.session_state.session_id:
        cost_display = f" | 💰 Total Cost: ${st.session_state.total_cost:.2f}" if st.session_state.total_cost > 0 else ""
        st.caption(f"🔗 Session ID: `{st.session_state.session_id}`{cost_display}")
    # st.markdown("### AI Chat Assistant with Tool Support")
    # st.markdown("---")

    # Sidebar
    with st.sidebar:
        # st.markdown("## 💬 ML Agent")
        # st.markdown("""
        # This assistant can:
        # - Solve machine learning tasks
        # - Write and execute code
        # - Generate reports and documentation
        # """)

        # st.markdown("---")

        # Clear conversation button
        if st.button("🔄 Start New Conversation"):
            # Clean up old agent
            if st.session_state.agent:
                try:
                    st.session_state.event_loop.run_until_complete(cleanup_agent())
                except Exception as e:
                    st.warning(f"Cleanup warning: {e}")

            # Clean up session file handler
            if st.session_state.session_file_handler is not None:
                root_logger = logging.getLogger()
                root_logger.removeHandler(st.session_state.session_file_handler)
                st.session_state.session_file_handler.close()
                st.session_state.session_file_handler = None

            st.session_state.messages = []
            st.session_state.session_id = None
            st.session_state.total_cost = 0.0
            st.session_state.current_todos = []
            st.rerun()

        # Display real-time statistics
        render_sidebar_stats()

        st.markdown("---")

        # Display todo list if there are any todos
        render_todo_list()

    # Create tabs
    tab1, tab2 = st.tabs(["💬 Chat", "📋 Logs"])

    with tab1:
        # Display chat history
        for message in st.session_state.messages:
            render_chat_message(message["role"], message["content"])

        # Chat input - only accept input if not currently processing
        if not st.session_state.is_processing:
            if prompt := st.chat_input("Type your message here..."):
                # Add user message
                st.session_state.messages.append({
                    "role": "user",
                    "content": prompt,
                    "timestamp": datetime.now()
                })

                # Get agent and session ID from main thread before starting background processing
                agent = get_or_create_agent()
                previous_session_id = st.session_state.session_id

                # Start background processing (non-blocking)
                async_gen = process_message_streaming(prompt, agent, previous_session_id)
                start_background_processing(async_gen)

                # Trigger rerun to start polling
                st.rerun()
        else:
            # Show disabled input while processing
            st.chat_input("Agent is processing...", disabled=True)

        # If processing, show the streaming response
        if st.session_state.is_processing:
            # Poll for new chunks (non-blocking)
            poll_and_update_chunks()

            # Show the current response as it streams in
            if st.session_state.current_response:
                with st.chat_message("assistant"):
                    st.markdown(st.session_state.current_response)

            # Show status
            with st.status("🤖 Agent is thinking...", expanded=False, state="running"):
                st.caption("Processing your request...")

            # Auto-refresh to keep polling
            time.sleep(0.1)
            st.rerun()

        # If just completed processing, finalize the response
        elif st.session_state.current_response and not st.session_state.is_processing:
            # Check if we need to finalize (response not yet in messages)
            if not st.session_state.messages or st.session_state.messages[-1]["content"] != st.session_state.current_response:
                # Handle errors
                if st.session_state.processing_error:
                    response = f"❌ Error: {st.session_state.processing_error}"
                    st.error(response)
                else:
                    response = st.session_state.current_response

                # Add assistant response to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now()
                })

                # Clear processing state
                st.session_state.current_response = ""
                st.session_state.processing_error = None
                st.session_state.chunk_queue = None
                st.session_state.background_thread = None

                st.rerun()

    with tab2:
        render_logs_tab()


if __name__ == "__main__":
    main()
