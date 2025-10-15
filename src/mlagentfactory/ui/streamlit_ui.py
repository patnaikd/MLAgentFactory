"""Streamlit UI for MLAgentFactory chat interface"""

import streamlit as st
import streamlit.components.v1 as components
import asyncio
import logging
import sys
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


def render_chat_message(role: str, content: str):
    """Render a chat message with appropriate styling"""
    with st.chat_message(role):
        st.markdown(content)


def get_or_create_agent():
    """Get existing agent or create a new one."""
    if st.session_state.agent is None:
        st.session_state.agent = ChatAgent()
    return st.session_state.agent


async def process_message(message: str) -> str:
    """Process a message through the agent."""
    agent = get_or_create_agent()

    # Initialize if not already initialized
    if not agent._initialized:
        await agent.initialize()

    return await agent.chat(message)


async def cleanup_agent():
    """Cleanup the agent."""
    if st.session_state.agent:
        await st.session_state.agent.cleanup()
        st.session_state.agent = None


def render_logs_tab():
    """Render the logs display tab."""
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
        components.html(log_html, height=650, scrolling=False)

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
            st.session_state.messages = []
            st.rerun()

        # Display message count
        if st.session_state.messages:
            st.markdown(f"**Messages:** {len(st.session_state.messages)}")

        # Display log count
        if st.session_state.log_messages:
            st.markdown(f"**Log Entries:** {len(st.session_state.log_messages)}")

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

            # Process with agent
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.event_loop.run_until_complete(process_message(prompt))
                        st.markdown(response)
                    except Exception as e:
                        response = f"❌ Error: {str(e)}"
                        st.error(response)

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
