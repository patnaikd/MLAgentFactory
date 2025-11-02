"""Gradio UI for MLAgentFactory chat interface"""

import gradio as gr
import asyncio
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional, Any
import json

# Add src directory to path if running as script
if __name__ == "__main__":
    src_path = Path(__file__).parent.parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

# Import agent and utils
try:
    from ..agents.chat_agent import ChatAgent
    from ..utils.logging_config import initialize_observability
except ImportError:
    # Fallback for direct script execution
    from mlagentfactory.agents.chat_agent import ChatAgent
    from mlagentfactory.utils.logging_config import initialize_observability

# Load environment variables from .env file
load_dotenv()


class GradioLogHandler(logging.Handler):
    """Custom logging handler that stores logs for Gradio display."""

    def __init__(self):
        super().__init__()
        self.log_messages: List[Dict[str, str]] = []

    def emit(self, record):
        log_entry = {
            'time': datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'level': record.levelname,
            'name': record.name,
            'message': self.format(record)
        }
        self.log_messages.append(log_entry)

    def get_logs(self, max_entries: int = 1000) -> List[Dict[str, str]]:
        """Get recent log entries."""
        return self.log_messages[-max_entries:]

    def clear_logs(self):
        """Clear all log entries."""
        self.log_messages = []


# Global log handler instance
log_handler = GradioLogHandler()


def setup_logging():
    """Configure logging with custom handler for Gradio."""
    # Create formatter - same as console formatter in logging_config.py
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Check if GradioLogHandler is already added
    has_gradio_handler = any(
        isinstance(handler, GradioLogHandler) for handler in root_logger.handlers
    )

    if not has_gradio_handler:
        log_handler.setFormatter(formatter)
        log_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(log_handler)


def setup_session_file_logger(session_id: str) -> logging.FileHandler:
    """Set up a file logger for the current session."""
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Create new file handler for this session
    log_file_path = logs_dir / f"session-{session_id}.log"
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # Add to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    logging.info(f"Session file logger initialized: {log_file_path}")
    return file_handler


def format_tool_display(chunk: Dict[str, Any]) -> str:
    """Format tool usage for display in chat."""
    tool_name = chunk.get("tool_name", "")
    tool_input = chunk.get("tool_input", {})

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

        # Map tool names to emojis
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
        return f"\n\n*{chunk.get('content', 'Using tool: ' + tool_name)}*\n\n"


def format_tool_result(chunk: Dict[str, Any]) -> str:
    """Format tool result for display in chat."""
    content = chunk["content"]
    is_error = chunk.get("is_error", False)

    if is_error:
        # Display errors in a code block with error prefix
        return f"\n\n**⚠️ Tool Execution Error:**\n```\n{content}\n```\n\n"
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
                return f"\n\n**🔧 Tool Result:**\n```json\n{display_content}\n```\n\n"
            elif display_content.strip().startswith('<'):
                # Looks like HTML/XML
                return f"\n\n**🔧 Tool Result:**\n```html\n{display_content}\n```\n\n"
            else:
                # Plain text
                return f"\n\n**🔧 Tool Result:**\n```\n{display_content}\n```\n\n"
        elif isinstance(content, list):
            # Format list content
            return f"\n\n**🔧 Tool Result:** ({len(content)} items)\n```json\n{content}\n```\n\n"
        else:
            # Other types
            return f"\n\n**🔧 Tool Result:**\n```\n{str(content)}\n```\n\n"


def format_todos_html(todos: List[Dict[str, str]]) -> str:
    """Format todos as HTML for display."""
    if not todos:
        return "<p style='color: #888; font-style: italic;'>No active tasks</p>"

    # Calculate progress
    total_tasks = len(todos)
    completed_tasks = sum(1 for todo in todos if todo.get("status") == "completed")
    in_progress_tasks = sum(1 for todo in todos if todo.get("status") == "in_progress")
    pending_tasks = total_tasks - completed_tasks - in_progress_tasks

    # Progress bar
    progress = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0

    html = f"""
    <div style="font-family: system-ui, -apple-system, sans-serif;">
        <h3 style="margin-top: 0;">📋 Task Progress</h3>

        <!-- Progress bar -->
        <div style="background: #e0e0e0; border-radius: 10px; height: 20px; margin: 10px 0;">
            <div style="background: linear-gradient(90deg, #28a745, #20c997); width: {progress}%; height: 100%; border-radius: 10px; transition: width 0.3s;"></div>
        </div>
        <p style="text-align: center; color: #666; font-size: 0.9em; margin: 5px 0;">{completed_tasks}/{total_tasks} tasks completed</p>

        <!-- Task list -->
        <div style="margin-top: 15px;">
    """

    for todo in todos:
        status = todo.get("status", "pending")
        content = todo.get("content", "")

        # Status emoji and color
        if status == "completed":
            emoji = "✅"
            color = "#28a745"
        elif status == "in_progress":
            emoji = "🔄"
            color = "#ffc107"
        else:  # pending
            emoji = "⏳"
            color = "#6c757d"

        html += f"""
        <div style="margin: 8px 0; padding: 8px; border-left: 3px solid {color}; background: rgba(0,0,0,0.02); border-radius: 3px;">
            <span style="font-size: 1.2em;">{emoji}</span>
            <span style="color: {color}; margin-left: 8px;">{content}</span>
        </div>
        """

    # Summary
    html += f"""
        </div>
        <p style="color: #888; font-size: 0.85em; margin-top: 15px; border-top: 1px solid #ddd; padding-top: 10px;">
            ✅ {completed_tasks} completed | 🔄 {in_progress_tasks} in progress | ⏳ {pending_tasks} pending
        </p>
    </div>
    """

    return html


def format_logs_html(log_messages: List[Dict[str, str]], log_levels: List[str], auto_scroll: bool = True) -> str:
    """Format logs as HTML for display."""
    if not log_messages:
        return "<p style='color: #888; font-style: italic;'>ℹ️ No logs yet. Start chatting to see agent activity!</p>"

    # Filter logs by selected levels
    filtered_logs = [
        log for log in log_messages
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

    return log_html


async def process_message(
    message: str,
    history: List[List[str]],
    agent_state: Dict[str, Any],
    session_info: Dict[str, Any]
) -> Tuple[str, List[List[str]], Dict[str, Any], Dict[str, Any], str, str]:
    """Process a message through the agent with streaming response.

    Args:
        message: The user's message
        history: Chat history as list of [user_msg, bot_msg] pairs
        agent_state: Current agent state
        session_info: Session information (session_id, total_cost, etc.)

    Returns:
        Tuple of (response, updated_history, updated_agent_state, updated_session_info, todos_html, session_display)
    """
    # Get or create agent
    if agent_state.get("agent") is None:
        agent_state["agent"] = ChatAgent()

    agent = agent_state["agent"]

    # Initialize if needed
    if not agent._initialized:
        await agent.initialize()

    # Track response parts
    response_parts = []
    current_todos = session_info.get("current_todos", [])

    # Stream chunks from the agent
    async for chunk in agent.send_message(message):
        if chunk["type"] == "text":
            response_parts.append(chunk["content"])

        elif chunk["type"] == "tool_use":
            # Format tool usage
            tool_display = format_tool_display(chunk)
            response_parts.append(tool_display)

        elif chunk["type"] == "tool_result":
            # Format tool result
            result_display = format_tool_result(chunk)
            response_parts.append(result_display)

        elif chunk["type"] == "session_id":
            # Update session ID
            new_session_id = chunk["content"]
            if new_session_id != session_info.get("session_id"):
                session_info["session_id"] = new_session_id
                # Set up file logger
                if "file_handler" in session_info and session_info["file_handler"]:
                    root_logger = logging.getLogger()
                    root_logger.removeHandler(session_info["file_handler"])
                    session_info["file_handler"].close()
                session_info["file_handler"] = setup_session_file_logger(new_session_id)

        elif chunk["type"] == "total_cost":
            # Update total cost
            session_info["total_cost"] = chunk["content"]

        elif chunk["type"] == "todo_update":
            # Update todos
            current_todos = chunk["content"]
            session_info["current_todos"] = current_todos

    # Combine response
    full_response = "".join(response_parts)

    # Update history
    history.append([message, full_response])

    # Format todos HTML
    todos_html = format_todos_html(current_todos)

    # Format session display
    session_id = session_info.get("session_id", "Not initialized")
    total_cost = session_info.get("total_cost", 0.0)
    log_file = f"logs/session-{session_id}.log" if session_id != "Not initialized" else "N/A"

    session_display = f"""
### Session Information
- **Session ID:** `{session_id}`
- **Total Cost:** ${total_cost:.4f}
- **Messages:** {len(history)}
- **Log File:** `{log_file}`
"""

    return full_response, history, agent_state, session_info, todos_html, session_display


async def chat_handler(message: str, history: List[List[str]], agent_state: Dict, session_info: Dict):
    """Handle chat messages with streaming support."""
    if not message.strip():
        return history, agent_state, session_info, session_info.get("todos_html", ""), session_info.get("session_display", "")

    # Process message
    _, updated_history, updated_agent_state, updated_session_info, todos_html, session_display = await process_message(
        message, history, agent_state, session_info
    )

    # Store HTML in session info for persistence
    updated_session_info["todos_html"] = todos_html
    updated_session_info["session_display"] = session_display

    return updated_history, updated_agent_state, updated_session_info, todos_html, session_display


def clear_conversation(agent_state: Dict, session_info: Dict):
    """Clear conversation history and reset state."""
    # Clean up agent
    if agent_state.get("agent"):
        try:
            agent = agent_state["agent"]
            if agent._initialized and agent.client:
                # Schedule cleanup
                asyncio.create_task(agent.cleanup())
        except Exception as e:
            logging.warning(f"Cleanup warning: {e}")

    # Clean up file handler
    if session_info.get("file_handler"):
        root_logger = logging.getLogger()
        root_logger.removeHandler(session_info["file_handler"])
        session_info["file_handler"].close()

    # Reset states
    new_agent_state = {"agent": None}
    new_session_info = {
        "session_id": None,
        "total_cost": 0.0,
        "current_todos": [],
        "file_handler": None,
        "todos_html": "",
        "session_display": "### Session Information\n- **Session ID:** Not initialized"
    }

    return [], new_agent_state, new_session_info, "", new_session_info["session_display"]


def update_logs(log_levels: List[str], auto_scroll: bool) -> str:
    """Update logs display based on filters."""
    logs = log_handler.get_logs()
    return format_logs_html(logs, log_levels, auto_scroll)


def clear_logs():
    """Clear all logs."""
    log_handler.clear_logs()
    return "<p style='color: #888; font-style: italic;'>Logs cleared.</p>"


def create_ui():
    """Create the Gradio UI."""
    # Initialize observability
    initialize_observability(log_level="DEBUG", enable_tracing=False)

    # Setup logging
    setup_logging()

    # Create custom theme
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="green",
    )

    with gr.Blocks(theme=theme, title="🤖 MLAgentFactory", css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        .chat-container {
            height: 600px !important;
        }
    """) as demo:
        # State management
        agent_state = gr.State({"agent": None})
        session_info = gr.State({
            "session_id": None,
            "total_cost": 0.0,
            "current_todos": [],
            "file_handler": None,
            "todos_html": "",
            "session_display": "### Session Information\n- **Session ID:** Not initialized"
        })

        # Header
        gr.Markdown("# 🤖 MLAgentFactory")
        gr.Markdown("### AI Chat Assistant with Tool Support")

        # Main layout
        with gr.Row():
            # Left column - Chat
            with gr.Column(scale=3):
                with gr.Tabs() as tabs:
                    with gr.TabItem("💬 Chat"):
                        chatbot = gr.Chatbot(
                            label="Chat with MLAgentFactory",
                            height=600,
                            show_copy_button=True,
                            type="tuples"  # Use tuples format: [[user_msg, bot_msg], ...]
                        )

                        with gr.Row():
                            msg = gr.Textbox(
                                label="Type your message here...",
                                placeholder="Ask me anything about machine learning, data science, or coding...",
                                scale=4,
                                lines=2
                            )
                            submit_btn = gr.Button("Send", variant="primary", scale=1)

                        clear_btn = gr.Button("🔄 Start New Conversation")

                    with gr.TabItem("📋 Logs"):
                        with gr.Row():
                            log_level_filter = gr.CheckboxGroup(
                                choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                                value=["INFO", "WARNING", "ERROR", "CRITICAL"],
                                label="Filter by level",
                                scale=3
                            )
                            auto_scroll_checkbox = gr.Checkbox(label="Auto-scroll", value=True, scale=1)
                            clear_logs_btn = gr.Button("Clear Logs", scale=1)

                        logs_display = gr.HTML(
                            value="<p style='color: #888; font-style: italic;'>ℹ️ No logs yet. Start chatting to see agent activity!</p>",
                            label="Agent Logs"
                        )

                        refresh_logs_btn = gr.Button("🔄 Refresh Logs")

            # Right column - Sidebar
            with gr.Column(scale=1):
                gr.Markdown("## 💬 Chat Assistant")
                gr.Markdown("""
                This assistant can:
                - Answer your questions
                - Maintain conversation history
                - Use tools like file I/O, web fetch, and Kaggle
                - Remember context from previous messages
                """)

                gr.Markdown("---")

                # Session info
                session_display = gr.Markdown("### Session Information\n- **Session ID:** Not initialized")

                gr.Markdown("---")

                # Todo list
                todos_display = gr.HTML(
                    value="<p style='color: #888; font-style: italic;'>No active tasks</p>",
                    label="Task Progress"
                )

        # Event handlers
        async def submit_message(message, history, agent_st, sess_info):
            return await chat_handler(message, history, agent_st, sess_info)

        # Submit message
        msg.submit(
            submit_message,
            inputs=[msg, chatbot, agent_state, session_info],
            outputs=[chatbot, agent_state, session_info, todos_display, session_display]
        ).then(
            lambda: "",
            outputs=[msg]
        )

        submit_btn.click(
            submit_message,
            inputs=[msg, chatbot, agent_state, session_info],
            outputs=[chatbot, agent_state, session_info, todos_display, session_display]
        ).then(
            lambda: "",
            outputs=[msg]
        )

        # Clear conversation
        clear_btn.click(
            clear_conversation,
            inputs=[agent_state, session_info],
            outputs=[chatbot, agent_state, session_info, todos_display, session_display]
        )

        # Update logs
        refresh_logs_btn.click(
            update_logs,
            inputs=[log_level_filter, auto_scroll_checkbox],
            outputs=[logs_display]
        )

        log_level_filter.change(
            update_logs,
            inputs=[log_level_filter, auto_scroll_checkbox],
            outputs=[logs_display]
        )

        # Clear logs
        clear_logs_btn.click(
            clear_logs,
            outputs=[logs_display]
        )

        # Note: Auto-refresh of logs removed due to Gradio 5.0 compatibility
        # Users can manually refresh logs using the "Refresh Logs" button

    return demo


def main():
    """Main application entry point"""
    demo = create_ui()
    demo.queue()  # Enable queuing for better performance with async
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        favicon_path=None,
        show_api=True
    )


if __name__ == "__main__":
    main()
