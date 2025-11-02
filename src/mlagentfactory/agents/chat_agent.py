"""Simple conversational agent using Claude Agent SDK with MCP servers."""
import json
import asyncio
import logging
from typing import AsyncGenerator, Dict, List, Optional

from claude_agent_sdk import (
    ClaudeSDKClient,
    AssistantMessage,
    UserMessage,
    SystemMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    create_sdk_mcp_server,
    ClaudeAgentOptions
)

from ..utils.logging_config import initialize_observability
from ..tools import file_io_tools, web_fetch_tools, kaggle_tools, uci_tools, markdown_to_pdf_tools
from .logging_client import LoggingClaudeSDKClient, FakeClaudeSDKClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = '''
You are an expert machine learning engineer designed to help with coding tasks, data science projects,
and technical challenges. Use the available tools to answer user questions.
                
Always think step-by-step and explain your reasoning.
If you use a tool, explain why you are using it and what you expect to find.

Guidelines:
- Pick a name for the project with a unique name by adding a date-time suffix, format: <project_name>_yyyy-mm-dd_HH-MM-SS. and use it consistently.
- Ensure all project files are saved in the project directory and subdirectories under "workspace/<project_name>/" consistently. Always use absolute path when referring to files.
- First create a plan before executing any code.
- Make a running markdown paper with detailed report of every step of solving the machine learning problem including implementation details, analysis of results, detailed observations, next steps in paper.md in the project directory.
  Include any analysis images and prefer markdown tables over plain text tables. Write the markdown paper as you go along.
  Add extra new line before and after images and tables for better readability.
- Use Python for coding tasks. Use uv to manage Python dependencies, virtual environments, and execution.
- Write clear, maintainable code with comments. Include logging statements to trace execution.
- Use best practices for data science and machine learning.
- Move to the next TODO once the current is complete don't wait for user response. After a TODO is complete, always provide an update in the markdown paper.
- Use the TodoWrite tool to maintain a todo list in todo.md in the project directory. And keep updating it as you progress.
- Prefer using the TodoWrite tool to update the todo list instead of writing it out in text.
- Prefer using the FileWrite tool to create or update files instead of writing out file contents in text.
- Prefer using PyTorch for deep learning tasks. Use TensorFlow only if absolutely necessary.
- Use XGBoost or LightGBM for tabular data tasks.
- Use pandas for data manipulation and analysis.
- Use matplotlib or seaborn for visualizations.
- for preparing the submission file, always refer to sample_submission.csv file in the data directory for the correct format.
- When working on Kaggle competitions:
  - Use the Kaggle MCP server tools to interact with Kaggle.
  - Always download competition data using the kaggle_download_competition_data tool.
  - Complete and not sample data should be used for training and generating predictions.
  - Always submit predictions using the kaggle_submit_competition tool.
  - Use the kaggle_list_submissions and kaggle_competition_leaderboard tools to monitor submission status and leaderboard position.
- When working with UCI ML Repository datasets:
  - Use the UCI MCP server tools to interact with the UCI ML Repository.
  - Always fetch datasets using the uci_fetch_dataset tool.
  - Use the uci_get_dataset_info tool to understand dataset details.
- Include images and plots in the index.md for better understanding of results.
- Include images and plots in the resonse when relevant for better understanding.
- Generate pdf reports from markdown using the markdown_to_pdf MCP server tool.
- When solving ML problems, break down the problem into data preprocessing, exploratory data analysis, model selection, training, evaluation, and hyperparameter tuning.
Remember to think step-by-step and explain your reasoning at each step.
'''


class ChatAgent:
    """A simple conversational agent that maintains conversation history.

    Args:
        enable_logging: If True, wraps client with LoggingClaudeSDKClient to record interactions
        log_dir: Directory for log files (only used if enable_logging=True)
        session_prefix: Prefix for log filenames (only used if enable_logging=True)
        replay_log_file: If provided, uses FakeClaudeSDKClient to replay from this log file
    """

    def __init__(
        self,
        enable_logging: bool = False,
        log_dir: str = "./data/claude_logs",
        session_prefix: str = "session",
        replay_log_file: Optional[str] = None
    ):
        """Initialize the chat agent."""
        # Only initialize observability if not already configured
        # (Streamlit UI will handle initialization)
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            initialize_observability(log_level="DEBUG", enable_tracing=False)
        else:
            # Ensure DEBUG level is set even if handlers exist
            root_logger.setLevel(logging.DEBUG)

        # Store client configuration
        self.enable_logging = enable_logging
        self.log_dir = log_dir
        self.session_prefix = session_prefix
        self.replay_log_file = replay_log_file

        self.web_server = create_sdk_mcp_server(
            name="web_tools",
            version="1.0.0",
            tools=[web_fetch_tools.fetch_webpage]
        )

        self.kaggle_server = create_sdk_mcp_server(
            name="kaggle_tools",
            version="1.0.0",
            tools=[
                kaggle_tools.kaggle_download_dataset,
                kaggle_tools.kaggle_list_competitions,
                kaggle_tools.kaggle_download_competition_data,
                kaggle_tools.kaggle_submit_competition,
                kaggle_tools.kaggle_list_submissions,
                kaggle_tools.kaggle_competition_leaderboard
            ]
        )

        self.uci_server = create_sdk_mcp_server(
            name="uci_tools",
            version="1.0.0",
            tools=[
                uci_tools.uci_list_datasets,
                uci_tools.uci_fetch_dataset,
                uci_tools.uci_get_dataset_info
            ]
        )

        self.markdown_pdf_server = create_sdk_mcp_server(
            name="markdown_pdf_tools",
            version="1.0.0",
            tools=[
                markdown_to_pdf_tools.markdown_to_pdf
            ]
        )

        # Configure agent options

        self.options = ClaudeAgentOptions(
            system_prompt=SYSTEM_PROMPT,
            mcp_servers={
                "web": self.web_server,
                "kaggle": self.kaggle_server,
                "uci": self.uci_server,
                "markdown_pdf": self.markdown_pdf_server
            },
            allowed_tools=[
                "fetch_webpage",
                "kaggle_download_dataset",
                "kaggle_list_competitions",
                "kaggle_download_competition_data",
                "kaggle_submit_competition",
                "kaggle_list_submissions",
                "kaggle_competition_leaderboard",
                "uci_list_datasets",
                "uci_fetch_dataset",
                "uci_get_dataset_info",
                "markdown_to_pdf"
            ],
            permission_mode="bypassPermissions"
        )

        self.client: Optional[ClaudeSDKClient] = None
        self._initialized = False
        self.session_id: Optional[str] = None
        self.total_cost: float = 0.0
        logger.info("ChatAgent initialized with file I/O, web, Kaggle, UCI ML Repository, and Markdown-to-PDF tools")

    async def initialize(self):
        """Initialize the client. Call this once before using the agent."""
        if not self._initialized:
            # Choose client type based on configuration
            if self.replay_log_file:
                # Use fake client for replay
                self.client = FakeClaudeSDKClient(log_file=self.replay_log_file)
                logger.info(f"ChatAgent using FakeClaudeSDKClient with log: {self.replay_log_file}")
            elif self.enable_logging:
                # Use logging wrapper
                self.client = LoggingClaudeSDKClient(
                    options=self.options,
                    log_dir=self.log_dir,
                    session_prefix=self.session_prefix
                )
                logger.info(f"ChatAgent using LoggingClaudeSDKClient, logs in: {self.log_dir}")
            else:
                # Use standard client
                self.client = ClaudeSDKClient(options=self.options)
                logger.info("ChatAgent using standard ClaudeSDKClient")

            await self.client.connect()
            self._initialized = True
            logger.info("ChatAgent client initialized and connected")

    async def cleanup(self):
        """Cleanup the client. Call this when done with the agent."""
        if self._initialized and self.client:
            await self.client.disconnect()
            self._initialized = False
            self.client = None
            logger.info("ChatAgent client cleaned up and disconnected")


    async def send_message(self, message: str) -> AsyncGenerator[Dict, None]:
        """Send a message and stream the response.

        Args:
            message: The user's message

        Yields:
            Dictionary containing response chunks with the following types:
            - "text": Assistant's text response (content: str)
            - "tool_use": Tool being used (content: display text, tool_name: str, tool_input: dict, tool_use_id: str)
              - tool_input may contain: command/description (Bash), file_path (Write/Read/Edit/Delete)
            - "tool_result": Result from tool execution (content: str/list, tool_use_id: str, is_error: bool)
            - "todo_update": TodoWrite tool usage (content: list of todos, tool_use_id: str)
            - "session_id": Session identifier (content: str)
            - "total_cost": Total cost in USD (content: float)
            - "processing_complete": Signal that agent is done processing (content: str)
        """
        if not self.client:
            raise RuntimeError("ChatAgent must be used as an async context manager")

        # Log outgoing message
        logger.info(f"[OUTGOING] User message: {message[:100]}{'...' if len(message) > 100 else ''}")
        logger.debug(f"[OUTGOING] Full message content: {message!r}")

        # Send the query
        try:
            await self.client.query(message)
            logger.debug("[OUTGOING] Message sent to ClaudeSDKClient successfully")
        except Exception as e:
            logger.error(f"[OUTGOING] Failed to send message: {e}", exc_info=True)
            raise

        # Stream the response
        response_count = 0
        async for msg in self.client.receive_response():
            response_count += 1
            logger.debug(f"[INCOMING] Message #{response_count} received: {type(msg).__name__}: {msg!r}")

            if isinstance(msg, AssistantMessage):

                for idx, block in enumerate(msg.content):
                    if isinstance(block, TextBlock):
                        logger.info(f"[INCOMING] TextBlock #{idx+1}: {block.text[:100]}{'...' if len(block.text) > 100 else ''}")
                        yield {
                            "type": "text",
                            "content": block.text
                        }
                    elif isinstance(block, ToolUseBlock):
                        logger.info(f"[INCOMING] ToolUseBlock #{idx+1}: tool={block.name}, id={block.id}")

                        # Special handling for TodoWrite tool
                        if block.name == "TodoWrite":
                            logger.info(f"[INCOMING] TodoWrite tool detected with input: {block.input}")
                            yield {
                                "type": "todo_update",
                                "content": block.input.get("todos", []) if hasattr(block, 'input') and block.input else [],
                                "tool_use_id": block.id
                            }

                        # Extract tool details for display
                        tool_display = f"Using tool: {block.name}"
                        tool_input = {}

                        # Special handling for Bash tool - include command
                        if block.name == "Bash" and hasattr(block, 'input') and block.input:
                            command = block.input.get("command", "")
                            description = block.input.get("description", "")
                            tool_input = {"command": command, "description": description}

                            if description:
                                tool_display = f"Running: {description}"
                            else:
                                tool_display = f"Running bash command"

                            logger.info(f"[INCOMING] Bash command: {command}")

                        # Special handling for file operation tools - extract file_path
                        file_operation_tools = ["Write", "Read", "Edit", "Delete"]
                        if block.name in file_operation_tools and hasattr(block, 'input') and block.input:
                            file_path = block.input.get("file_path", "")
                            if file_path:
                                tool_input["file_path"] = file_path
                                logger.info(f"[INCOMING] {block.name} tool file_path: {file_path}")

                        yield {
                            "type": "tool_use",
                            "content": tool_display,
                            "tool_name": block.name,
                            "tool_input": tool_input,
                            "tool_use_id": block.id
                        }
                    else:
                        logger.warning(f"[INCOMING] Unknown block type #{idx+1}: {type(block).__name__}")

            elif isinstance(msg, UserMessage):
                logger.info(f"[INCOMING] UserMessage with {len(msg.content)} content blocks")

                for idx, block in enumerate(msg.content):
                    if isinstance(block, ToolResultBlock):
                        logger.info(f"[INCOMING] ToolResultBlock #{idx+1}: tool_use_id={block.tool_use_id}, is_error={block.is_error}")

                        # Get the content - it can be a string or list of content blocks
                        content = block.content

                        if isinstance(content, str):
                            logger.debug(f"[INCOMING] ToolResultBlock #{idx+1} content (string): {content[:200]}{'...' if len(content) > 200 else ''}")
                        elif isinstance(content, list):
                            logger.debug(f"[INCOMING] ToolResultBlock #{idx+1} content (list): {len(content)} blocks")
                        else:
                            logger.debug(f"[INCOMING] ToolResultBlock #{idx+1} content type: {type(content).__name__}")

                        # Yield tool result content to show tool execution results
                        yield {
                            "type": "tool_result",
                            "content": content,
                            "tool_use_id": block.tool_use_id,
                            "is_error": block.is_error
                        }

                    elif isinstance(block, TextBlock):
                        logger.info(f"[INCOMING] UserMessage TextBlock #{idx+1}: {block.text[:100]}{'...' if len(block.text) > 100 else ''}")
                        logger.debug(f"[INCOMING] UserMessage TextBlock #{idx+1} full content: {block.text!r}")
                        # User messages are typically echoes or system messages, can be yielded if needed
                    else:
                        logger.warning(f"[INCOMING] Unknown UserMessage block type #{idx+1}: {type(block).__name__}")

            elif isinstance(msg, SystemMessage):
                subtype = getattr(msg, 'subtype', 'unknown')
                logger.info(f"[INCOMING] SystemMessage: subtype={subtype}")

                # Log key system information at debug level
                if hasattr(msg, 'data'):
                    data = msg.data
                    if isinstance(data, dict):
                        # Capture session_id from init message
                        if subtype == 'init' and 'session_id' in data and not self.session_id:
                            self.session_id = data['session_id']
                            logger.info(f"[INCOMING] Captured session_id: {self.session_id}")
                            # Yield session_id so UI can display it
                            yield {
                                "type": "session_id",
                                "content": self.session_id
                            }

                        # Log interesting system info
                        if 'session_id' in data:
                            logger.debug(f"[INCOMING] SystemMessage session_id: {data['session_id']}")
                        if 'cwd' in data:
                            logger.debug(f"[INCOMING] SystemMessage cwd: {data['cwd']}")
                        if 'model' in data:
                            logger.debug(f"[INCOMING] SystemMessage model: {data['model']}")
                        if 'mcp_servers' in data:
                            logger.debug(f"[INCOMING] SystemMessage mcp_servers: {data['mcp_servers']}")
                        if 'tools' in data and isinstance(data['tools'], list):
                            logger.debug(f"[INCOMING] SystemMessage tools count: {len(data['tools'])}")

                        # Full data at debug level
                        logger.debug(f"[INCOMING] SystemMessage full data: {data!r}")
                    else:
                        logger.debug(f"[INCOMING] SystemMessage data: {data!r}")
                # Don't yield system messages to the user - they're internal initialization messages

            elif isinstance(msg, ResultMessage):
                logger.info(f"[INCOMING] ResultMessage: subtype={msg.subtype}, is_error={msg.is_error}")

                # Extract and store total cost
                if hasattr(msg, 'total_cost_usd'):
                    self.total_cost = msg.total_cost_usd
                    logger.info(f"[INCOMING] Total cost: ${self.total_cost:.4f}")

                    # Yield cost information to UI
                    yield {
                        "type": "total_cost",
                        "content": self.total_cost
                    }

                # Log detailed information at debug level
                logger.debug(f"[INCOMING] ResultMessage details: duration={msg.duration_ms}ms, "
                           f"api_duration={msg.duration_api_ms}ms, turns={msg.num_turns}")
                if hasattr(msg, 'usage'):
                    logger.debug(f"[INCOMING] Token usage: {msg.usage}")

            else:
                logger.warning(f"[INCOMING] Unexpected message type: {type(msg).__name__}, content: {msg!r}")

        logger.info(f"[INCOMING] Response stream completed. Total messages received: {response_count}")

        # Emit processing_complete signal to indicate agent is done processing
        logger.info("[INCOMING] Emitting processing_complete signal")
        yield {
            "type": "processing_complete",
            "content": "Agent has finished processing this query"
        }

    async def chat(self, message: str) -> str:
        """Send a message and return the complete response.

        Args:
            message: The user's message

        Returns:
            The complete response text
        """
        response_parts = []
        logger.info(f"[CHAT] Starting chat session for message: {message[:100]}{'...' if len(message) > 100 else ''}")

        try:
            async for chunk in self.send_message(message):
                if chunk["type"] == "text":
                    response_parts.append(chunk["content"])

            full_response = "".join(response_parts)
            logger.info(f"[CHAT] Completed chat session. Response length: {len(full_response)} chars")
            logger.debug(f"[CHAT] Full response: {full_response!r}")

            return full_response
        except Exception as e:
            logger.error(f"[CHAT] Chat session failed: {e}", exc_info=True)
            raise


async def demo():
    """Demo the chat agent."""
    async with ChatAgent() as agent:
        # First question
        print("User: What's the capital of France?")
        response = await agent.chat("What's the capital of France?")
        print(f"Agent: {response}\n")

        # Follow-up question - agent remembers context
        print("User: What's the population of that city?")
        response = await agent.chat("What's the population of that city?")
        print(f"Agent: {response}\n")

        # Another follow-up
        print("User: What are some famous landmarks there?")
        response = await agent.chat("What are some famous landmarks there?")
        print(f"Agent: {response}\n")


if __name__ == "__main__":
    asyncio.run(demo())
