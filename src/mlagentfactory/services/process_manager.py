"""Process manager for running ChatAgent instances in isolated processes."""

import asyncio
import logging
import multiprocessing
from multiprocessing import Process, Queue
from typing import Optional, Dict, Any
from queue import Empty
import traceback
import signal
import sys

from ..agents.chat_agent import ChatAgent
from .message_store import MessageStore, MessageType, SessionStatus


logger = logging.getLogger(__name__)


def _agent_process_worker(
    session_id: str,
    message_queue: Queue,
    command_queue: Queue,
    db_path: str
):
    """Worker function that runs in a separate process.

    This function:
    1. Creates a ChatAgent instance
    2. Listens for commands on command_queue (queries from user)
    3. Streams agent responses to message_queue
    4. Stores all messages in MessageStore

    Args:
        session_id: Session identifier
        message_queue: Queue to send messages back to main process
        command_queue: Queue to receive commands (user queries)
        db_path: Path to SQLite database for message storage
    """
    # Set up logging for this process
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Agent process started for session: {session_id}")

    # Create message store instance for this process
    message_store = MessageStore(db_path=db_path)

    # Update session status to running
    message_store.update_session_status(session_id, SessionStatus.RUNNING)

    async def run_agent():
        """Async function to run the agent."""
        agent = ChatAgent()
        await agent.initialize()

        try:
            while True:
                # Check for commands (non-blocking)
                try:
                    command = command_queue.get(timeout=0.1)

                    if command["type"] == "query":
                        user_message = command["message"]
                        logger.info(f"Processing query for session {session_id}: {user_message[:100]}")

                        # Stream agent response
                        try:
                            async for chunk in agent.send_message(user_message):
                                # Store message in database
                                message_type = MessageType(chunk["type"])
                                content = chunk.get("content")

                                # Store the message
                                message_store.add_message(
                                    session_id=session_id,
                                    message_type=message_type,
                                    content=chunk  # Store the entire chunk with all fields
                                )

                                # Also send to queue for immediate processing if needed
                                message_queue.put({
                                    "type": "message",
                                    "session_id": session_id,
                                    "chunk": chunk
                                })

                                # Handle special message types
                                if message_type == MessageType.SESSION_ID:
                                    message_store.update_session_agent_id(session_id, content)
                                elif message_type == MessageType.TOTAL_COST:
                                    message_store.update_session_cost(session_id, content)

                            logger.info(f"Completed query processing for session {session_id}")

                        except Exception as e:
                            logger.error(f"Error processing query: {e}", exc_info=True)
                            message_queue.put({
                                "type": "error",
                                "session_id": session_id,
                                "error": str(e),
                                "traceback": traceback.format_exc()
                            })

                    elif command["type"] == "stop":
                        logger.info(f"Stop command received for session {session_id}")
                        break

                except Empty:
                    # No command available, continue loop
                    await asyncio.sleep(0.1)
                    continue

        except Exception as e:
            logger.error(f"Fatal error in agent process: {e}", exc_info=True)
            message_store.update_session_status(session_id, SessionStatus.FAILED)
            message_queue.put({
                "type": "error",
                "session_id": session_id,
                "error": str(e),
                "traceback": traceback.format_exc()
            })
        finally:
            # Cleanup
            await agent.cleanup()
            message_store.update_session_status(session_id, SessionStatus.STOPPED)
            message_queue.put({
                "type": "stopped",
                "session_id": session_id
            })
            logger.info(f"Agent process stopped for session: {session_id}")

    # Run the async agent
    try:
        asyncio.run(run_agent())
    except KeyboardInterrupt:
        logger.info(f"Agent process interrupted for session: {session_id}")
        message_store.update_session_status(session_id, SessionStatus.STOPPED)
    except Exception as e:
        logger.error(f"Unhandled exception in agent process: {e}", exc_info=True)
        message_store.update_session_status(session_id, SessionStatus.FAILED)


class AgentProcess:
    """Manages a single agent process."""

    def __init__(self, session_id: str, db_path: str):
        """Initialize the agent process.

        Args:
            session_id: Session identifier
            db_path: Path to SQLite database
        """
        self.session_id = session_id
        self.db_path = db_path

        # Create queues for communication
        self.message_queue = Queue()
        self.command_queue = Queue()

        # Process handle
        self.process: Optional[Process] = None

        logger.info(f"AgentProcess initialized for session: {session_id}")

    def start(self) -> None:
        """Start the agent process."""
        if self.process is not None and self.process.is_alive():
            logger.warning(f"Process already running for session {self.session_id}")
            return

        logger.info(f"Starting agent process for session: {self.session_id}")

        self.process = Process(
            target=_agent_process_worker,
            args=(self.session_id, self.message_queue, self.command_queue, self.db_path),
            daemon=False  # Don't use daemon to allow proper cleanup
        )
        self.process.start()

        logger.info(f"Agent process started with PID {self.process.pid} for session: {self.session_id}")

    def send_query(self, message: str) -> None:
        """Send a query to the agent.

        Args:
            message: User message to send to the agent
        """
        if not self.is_alive():
            raise RuntimeError(f"Agent process not running for session {self.session_id}")

        logger.debug(f"Sending query to session {self.session_id}: {message[:100]}")
        self.command_queue.put({
            "type": "query",
            "message": message
        })

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the agent process gracefully.

        Args:
            timeout: Maximum time to wait for graceful shutdown (seconds)
        """
        if self.process is None or not self.process.is_alive():
            logger.info(f"Process not running for session {self.session_id}")
            return

        logger.info(f"Stopping agent process for session: {self.session_id}")

        # Send stop command
        self.command_queue.put({"type": "stop"})

        # Wait for graceful shutdown
        self.process.join(timeout=timeout)

        # Force terminate if still alive
        if self.process.is_alive():
            logger.warning(f"Force terminating process for session {self.session_id}")
            self.process.terminate()
            self.process.join(timeout=2.0)

            # Last resort: kill
            if self.process.is_alive():
                logger.error(f"Force killing process for session {self.session_id}")
                self.process.kill()
                self.process.join()

        logger.info(f"Agent process stopped for session: {self.session_id}")

    def is_alive(self) -> bool:
        """Check if the process is alive.

        Returns:
            bool: True if process is running
        """
        return self.process is not None and self.process.is_alive()

    def get_messages(self, timeout: float = 0.1) -> list:
        """Get messages from the message queue (non-blocking).

        Args:
            timeout: Maximum time to wait for a message (seconds)

        Returns:
            List of messages
        """
        messages = []
        while True:
            try:
                msg = self.message_queue.get(timeout=timeout)
                messages.append(msg)
            except Empty:
                break
        return messages


class ProcessManager:
    """Manages multiple agent processes."""

    def __init__(self, db_path: str = "data/sessions.db"):
        """Initialize the process manager.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.processes: Dict[str, AgentProcess] = {}
        logger.info("ProcessManager initialized")

    def create_process(self, session_id: str) -> AgentProcess:
        """Create and start a new agent process.

        Args:
            session_id: Session identifier

        Returns:
            AgentProcess instance
        """
        if session_id in self.processes:
            process = self.processes[session_id]
            if process.is_alive():
                logger.warning(f"Process already exists for session {session_id}")
                return process
            else:
                # Clean up dead process
                logger.info(f"Cleaning up dead process for session {session_id}")
                del self.processes[session_id]

        logger.info(f"Creating new process for session: {session_id}")
        process = AgentProcess(session_id, self.db_path)
        process.start()
        self.processes[session_id] = process

        return process

    def get_process(self, session_id: str) -> Optional[AgentProcess]:
        """Get an existing agent process.

        Args:
            session_id: Session identifier

        Returns:
            AgentProcess instance or None if not found
        """
        return self.processes.get(session_id)

    def stop_process(self, session_id: str) -> None:
        """Stop an agent process.

        Args:
            session_id: Session identifier
        """
        process = self.processes.get(session_id)
        if process:
            process.stop()
            del self.processes[session_id]
            logger.info(f"Stopped and removed process for session: {session_id}")
        else:
            logger.warning(f"No process found for session: {session_id}")

    def stop_all(self) -> None:
        """Stop all agent processes."""
        logger.info(f"Stopping all {len(self.processes)} agent processes")
        for session_id in list(self.processes.keys()):
            self.stop_process(session_id)
        logger.info("All agent processes stopped")

    def get_active_sessions(self) -> list:
        """Get list of active session IDs.

        Returns:
            List of session IDs with active processes
        """
        return [
            session_id
            for session_id, process in self.processes.items()
            if process.is_alive()
        ]

    def cleanup_dead_processes(self) -> None:
        """Clean up dead processes from the manager."""
        dead_sessions = [
            session_id
            for session_id, process in self.processes.items()
            if not process.is_alive()
        ]

        for session_id in dead_sessions:
            logger.info(f"Cleaning up dead process for session: {session_id}")
            del self.processes[session_id]

        if dead_sessions:
            logger.info(f"Cleaned up {len(dead_sessions)} dead processes")
