"""Streamlit UI for MLAgentFactory chat interface"""

import streamlit as st
import streamlit.components.v1 as components
import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

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


def setup_logging():
    """Configure logging with custom handler for Streamlit."""
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
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
            logging.debug(f"Re-attached session file handler for session: {st.session_state.session_id}")


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

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
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


def process_message_streaming(message: str):
    """Process a message through the agent with streaming response.

    Yields response chunks as they arrive synchronously.
    """
    agent = get_or_create_agent()

    # Initialize if not already initialized
    if not agent._initialized:
        st.session_state.event_loop.run_until_complete(agent.initialize())

    # Track previous session ID to detect changes
    previous_session_id = st.session_state.session_id

    # Create the async generator
    async def _stream():
        async for chunk in agent.send_message(message):
            if chunk["type"] == "text":
                yield chunk["content"]
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
                        yield f"\n\n**🔨 {description}**\n```bash\n{command}\n```\n\n"
                    else:
                        yield f"\n\n**🔨 Running bash command:**\n```bash\n{command}\n```\n\n"
                else:
                    # Default tool display
                    yield f"\n\n*{tool_display}*\n\n"
            elif chunk["type"] == "tool_result":
                # Show tool results in code blocks
                content = chunk["content"]
                is_error = chunk.get("is_error", False)

                if is_error:
                    # Display errors in a code block with error prefix
                    yield f"\n\n**⚠️ Tool Execution Error:**\n```\n{content}\n```\n\n"
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
                            yield f"\n\n**🔧 Tool Result:**\n```json\n{display_content}\n```\n\n"
                        elif display_content.strip().startswith('<'):
                            # Looks like HTML/XML
                            yield f"\n\n**🔧 Tool Result:**\n```html\n{display_content}\n```\n\n"
                        else:
                            # Plain text
                            yield f"\n\n**🔧 Tool Result:**\n```\n{display_content}\n```\n\n"
                    elif isinstance(content, list):
                        # Format list content
                        yield f"\n\n**🔧 Tool Result:** ({len(content)} items)\n```json\n{content}\n```\n\n"
                    else:
                        # Other types
                        yield f"\n\n**🔧 Tool Result:**\n```\n{str(content)}\n```\n\n"

            elif chunk["type"] == "session_id":
                # Capture session_id in session state
                new_session_id = chunk["content"]
                st.session_state.session_id = new_session_id

                # Set up file logger if session ID changed
                if new_session_id != previous_session_id:
                    setup_session_file_logger(new_session_id)

            elif chunk["type"] == "total_cost":
                # Capture total_cost in session state
                st.session_state.total_cost = chunk["content"]

            elif chunk["type"] == "todo_update":
                # Update the todo list in session state
                st.session_state.current_todos = chunk["content"]

    # Convert async generator to sync for Streamlit
    gen = _stream()
    while True:
        try:
            chunk = st.session_state.event_loop.run_until_complete(gen.__anext__())
            yield chunk
        except StopAsyncIteration:
            break


async def cleanup_agent():
    """Cleanup the agent."""
    if st.session_state.agent:
        await st.session_state.agent.cleanup()
        st.session_state.agent = None


def render_todo_list():
    """Render the todo list in the sidebar."""
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
        st.markdown("## 💬 Chat Assistant")
        st.markdown("""
        This assistant can:
        - Answer your questions
        - Maintain conversation history
        - Use tools like file I/O, web fetch, and Kaggle
        - Remember context from previous messages
        """)

        st.markdown("---")

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

        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now()
            })

            # Display user message
            render_chat_message("user", prompt)

            # Create a status expander to show logs during processing
            with st.status("🤖 Agent is thinking...", expanded=True) as status:
                # Create a placeholder for live logs
                log_placeholder = st.empty()

                # Process with agent using streaming
                try:
                    # Track the last log count shown
                    last_shown_log_count = len(st.session_state.log_messages)

                    # Custom streaming with log updates
                    response_chunks = []
                    chunk_count = 0

                    for chunk in process_message_streaming(prompt):
                        response_chunks.append(chunk)
                        chunk_count += 1

                        # Update logs after every chunks
                        current_log_count = len(st.session_state.log_messages)
                        if current_log_count > last_shown_log_count:
                            new_logs = st.session_state.log_messages[last_shown_log_count:current_log_count]
                            # Show recent logs in the status
                            with log_placeholder.container():
                                st.caption("**Recent Activity:**")
                                for log in new_logs[-5:]:  # Show last 5 new logs
                                    level_emoji = {
                                        'DEBUG': '🔍',
                                        'INFO': 'ℹ️',
                                        'WARNING': '⚠️',
                                        'ERROR': '❌',
                                        'CRITICAL': '🚨'
                                    }.get(log['level'], '📝')
                                    st.caption(f"{level_emoji} {log['level']}: {log['message']}")
                            last_shown_log_count = current_log_count

                    response = "".join(response_chunks)

                    # Final log update
                    current_log_count = len(st.session_state.log_messages)
                    if current_log_count > last_shown_log_count:
                        new_logs = st.session_state.log_messages[last_shown_log_count:current_log_count]
                        with log_placeholder.container():
                            st.caption("**Recent Activity:**")
                            for log in new_logs[-5:]:
                                level_emoji = {
                                    'DEBUG': '🔍',
                                    'INFO': 'ℹ️',
                                    'WARNING': '⚠️',
                                    'ERROR': '❌',
                                    'CRITICAL': '🚨'
                                }.get(log['level'], '📝')
                                st.caption(f"{level_emoji} {log['level']}: {log['message']}")

                    status.update(label="✅ Agent completed!", state="complete", expanded=False)
                except Exception as e:
                    response = f"❌ Error: {str(e)}"
                    st.error(response)
                    status.update(label="❌ Error occurred", state="error")

            # Display the response
            render_chat_message("assistant", response)

            # Add assistant response
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now()
            })

            st.rerun()

    with tab2:
        render_logs_tab()


if __name__ == "__main__":
    main()
