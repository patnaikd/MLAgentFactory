"""Session manager for orchestrating agent sessions and message flow."""

import logging
import uuid
from typing import Optional, Dict, List, Any
from datetime import datetime

from .message_store import MessageStore, SessionStatus, MessageType
from .process_manager import ProcessManager, AgentProcess


logger = logging.getLogger(__name__)


class SessionManager:
    """High-level session manager that coordinates message store and process manager.

    This class provides the main API for:
    - Creating new agent sessions
    - Sending queries to agents
    - Retrieving messages with cursor-based pagination
    - Managing session lifecycle
    """

    def __init__(self, db_path: str = "data/sessions.db"):
        """Initialize the session manager.

        Args:
            db_path: Path to SQLite database
        """
        self.message_store = MessageStore(db_path=db_path)
        self.process_manager = ProcessManager(db_path=db_path)
        logger.info("SessionManager initialized")

    def create_session(self, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new session with a unique ID.

        Args:
            metadata: Optional metadata to store with the session

        Returns:
            Dict containing session information including session_id
        """
        # Generate unique session ID
        session_id = f"session-{uuid.uuid4().hex[:16]}"

        logger.info(f"Creating new session: {session_id}")

        # Create session in message store
        session = self.message_store.create_session(session_id, metadata=metadata)

        # Create and start agent process
        try:
            process = self.process_manager.create_process(session_id)
            logger.info(f"Session {session_id} created successfully with process")
        except Exception as e:
            logger.error(f"Failed to create process for session {session_id}: {e}", exc_info=True)
            self.message_store.update_session_status(session_id, SessionStatus.FAILED)
            raise

        return session

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information.

        Args:
            session_id: Session identifier

        Returns:
            Dict containing session information with additional runtime status
        """
        session = self.message_store.get_session(session_id)
        if not session:
            return None

        # Add process status
        process = self.process_manager.get_process(session_id)
        session["process_alive"] = process.is_alive() if process else False

        return session

    def list_sessions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List all sessions.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session dictionaries
        """
        sessions = self.message_store.list_sessions(limit=limit)

        # Enrich with process status
        for session in sessions:
            process = self.process_manager.get_process(session["session_id"])
            session["process_alive"] = process.is_alive() if process else False

        return sessions

    def send_query(self, session_id: str, message: str) -> Dict[str, Any]:
        """Send a query to an agent session.

        Args:
            session_id: Session identifier
            message: User message to send to the agent

        Returns:
            Dict with status information

        Raises:
            ValueError: If session doesn't exist
            RuntimeError: If agent process is not running
        """
        # Verify session exists
        session = self.message_store.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        # Get or create process
        process = self.process_manager.get_process(session_id)
        if not process or not process.is_alive():
            logger.info(f"Process not running for session {session_id}, creating new process")
            process = self.process_manager.create_process(session_id)

        # Send query to process
        try:
            process.send_query(message)
            logger.info(f"Query sent to session {session_id}")
            return {
                "status": "success",
                "session_id": session_id,
                "message": "Query sent successfully"
            }
        except Exception as e:
            logger.error(f"Failed to send query to session {session_id}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to send query: {str(e)}")

    def get_messages(
        self,
        session_id: str,
        since_message_id: int = 0,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Get messages for a session with cursor-based pagination.

        Args:
            session_id: Session identifier
            since_message_id: Return messages with ID > this value
            limit: Maximum number of messages to return

        Returns:
            Dict containing:
                - messages: List of message dictionaries
                - next_cursor: The message_id to use for next poll (or None if no more messages)
                - has_more: Boolean indicating if more messages are available

        Raises:
            ValueError: If session doesn't exist
        """
        # Verify session exists
        session = self.message_store.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        # Get messages from store
        messages = self.message_store.get_messages(
            session_id=session_id,
            since_message_id=since_message_id,
            limit=limit
        )

        # Determine next cursor and has_more
        next_cursor = None
        has_more = False

        if messages:
            # Next cursor is the last message_id we returned
            next_cursor = messages[-1]["message_id"]

            # Check if there are more messages after this batch
            if len(messages) == limit:
                # There might be more messages
                has_more = True

        return {
            "messages": messages,
            "next_cursor": next_cursor,
            "has_more": has_more,
            "count": len(messages)
        }

    def stop_session(self, session_id: str, timeout: float = 5.0) -> Dict[str, Any]:
        """Stop a session and cleanup resources.

        Args:
            session_id: Session identifier
            timeout: Maximum time to wait for graceful shutdown

        Returns:
            Dict with status information

        Raises:
            ValueError: If session doesn't exist
        """
        # Verify session exists
        session = self.message_store.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        logger.info(f"Stopping session: {session_id}")

        # Stop the process
        self.process_manager.stop_process(session_id)

        # Update session status
        self.message_store.update_session_status(session_id, SessionStatus.STOPPED)

        logger.info(f"Session {session_id} stopped successfully")

        return {
            "status": "success",
            "session_id": session_id,
            "message": "Session stopped successfully"
        }

    def delete_session(self, session_id: str) -> Dict[str, Any]:
        """Delete a session and all its data.

        Args:
            session_id: Session identifier

        Returns:
            Dict with status information

        Raises:
            ValueError: If session doesn't exist
        """
        # Verify session exists
        session = self.message_store.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        logger.info(f"Deleting session: {session_id}")

        # Stop the process if running
        process = self.process_manager.get_process(session_id)
        if process and process.is_alive():
            self.process_manager.stop_process(session_id)

        # Delete from message store (cascades to messages)
        self.message_store.delete_session(session_id)

        logger.info(f"Session {session_id} deleted successfully")

        return {
            "status": "success",
            "session_id": session_id,
            "message": "Session deleted successfully"
        }

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session.

        Args:
            session_id: Session identifier

        Returns:
            Dict containing session statistics

        Raises:
            ValueError: If session doesn't exist
        """
        session = self.message_store.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        message_count = self.message_store.get_message_count(session_id)

        process = self.process_manager.get_process(session_id)
        process_alive = process.is_alive() if process else False

        return {
            "session_id": session_id,
            "status": session["status"],
            "created_at": session["created_at"],
            "updated_at": session["updated_at"],
            "agent_session_id": session["agent_session_id"],
            "total_cost": session["total_cost"],
            "message_count": message_count,
            "process_alive": process_alive
        }

    def cleanup(self) -> None:
        """Cleanup all resources and stop all processes."""
        logger.info("Starting SessionManager cleanup")
        self.process_manager.stop_all()
        logger.info("SessionManager cleanup completed")

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the session manager.

        Returns:
            Dict containing health status
        """
        active_sessions = self.process_manager.get_active_sessions()
        total_sessions = len(self.message_store.list_sessions())

        return {
            "status": "healthy",
            "active_sessions": len(active_sessions),
            "total_sessions": total_sessions,
            "active_session_ids": active_sessions
        }
