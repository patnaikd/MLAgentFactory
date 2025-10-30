"""Message store for persistent agent message storage using SQLite."""

import sqlite3
import json
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from enum import Enum


logger = logging.getLogger(__name__)


class SessionStatus(str, Enum):
    """Session status enum."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class MessageType(str, Enum):
    """Message type enum."""
    TEXT = "text"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    TODO_UPDATE = "todo_update"
    SESSION_ID = "session_id"
    TOTAL_COST = "total_cost"


class MessageStore:
    """SQLite-based message store for agent sessions.

    This store manages:
    - Sessions: Track agent session lifecycle and metadata
    - Messages: Store all messages from agent responses for pull-based retrieval

    Features:
    - Cursor-based pagination for efficient message polling
    - Thread-safe SQLite access with context managers
    - JSON storage for flexible message content
    """

    def __init__(self, db_path: str = "data/sessions.db"):
        """Initialize the message store.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
        logger.info(f"MessageStore initialized with database: {self.db_path}")

    @contextmanager
    def _get_connection(self):
        """Get a database connection with context manager.

        Yields:
            sqlite3.Connection: Database connection
        """
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()

    def _initialize_database(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    agent_session_id TEXT,
                    total_cost REAL DEFAULT 0.0,
                    metadata TEXT
                )
            """)

            # Messages table with auto-incrementing ID for cursor-based pagination
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
            """)

            # Index for efficient message retrieval
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session_id
                ON messages(session_id, message_id)
            """)

            conn.commit()
            logger.debug("Database schema initialized successfully")

    # ===========================
    # Session Management Methods
    # ===========================

    def create_session(self, session_id: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new session.

        Args:
            session_id: Unique session identifier
            metadata: Optional metadata to store with the session

        Returns:
            Dict containing session information
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()

            metadata_json = json.dumps(metadata) if metadata else None

            cursor.execute("""
                INSERT INTO sessions (session_id, status, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, SessionStatus.CREATED, now, now, metadata_json))

            conn.commit()

            logger.info(f"Created session: {session_id}")
            return self.get_session(session_id)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information.

        Args:
            session_id: Session identifier

        Returns:
            Dict containing session information, or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()

            if row:
                return {
                    "session_id": row["session_id"],
                    "status": row["status"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "agent_session_id": row["agent_session_id"],
                    "total_cost": row["total_cost"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else None
                }
            return None

    def update_session_status(self, session_id: str, status: SessionStatus) -> None:
        """Update session status.

        Args:
            session_id: Session identifier
            status: New status
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()

            cursor.execute("""
                UPDATE sessions
                SET status = ?, updated_at = ?
                WHERE session_id = ?
            """, (status, now, session_id))

            conn.commit()
            logger.info(f"Updated session {session_id} status to {status}")

    def update_session_agent_id(self, session_id: str, agent_session_id: str) -> None:
        """Update the agent's internal session ID.

        Args:
            session_id: Session identifier
            agent_session_id: Agent's internal session ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()

            cursor.execute("""
                UPDATE sessions
                SET agent_session_id = ?, updated_at = ?
                WHERE session_id = ?
            """, (agent_session_id, now, session_id))

            conn.commit()
            logger.debug(f"Updated session {session_id} agent_session_id to {agent_session_id}")

    def update_session_cost(self, session_id: str, total_cost: float) -> None:
        """Update session total cost.

        Args:
            session_id: Session identifier
            total_cost: Total cost in USD
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()

            cursor.execute("""
                UPDATE sessions
                SET total_cost = ?, updated_at = ?
                WHERE session_id = ?
            """, (total_cost, now, session_id))

            conn.commit()
            logger.debug(f"Updated session {session_id} total_cost to ${total_cost:.4f}")

    def list_sessions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List all sessions.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM sessions
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))

            rows = cursor.fetchall()
            return [
                {
                    "session_id": row["session_id"],
                    "status": row["status"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "agent_session_id": row["agent_session_id"],
                    "total_cost": row["total_cost"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else None
                }
                for row in rows
            ]

    def delete_session(self, session_id: str) -> None:
        """Delete a session and all its messages.

        Args:
            session_id: Session identifier
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
            logger.info(f"Deleted session: {session_id}")

    # ===========================
    # Message Management Methods
    # ===========================

    def add_message(
        self,
        session_id: str,
        message_type: MessageType,
        content: Any
    ) -> int:
        """Add a message to the store.

        Args:
            session_id: Session identifier
            message_type: Type of message
            content: Message content (will be JSON serialized)

        Returns:
            int: The message_id of the inserted message
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()

            # Serialize content to JSON
            content_json = json.dumps(content)

            cursor.execute("""
                INSERT INTO messages (session_id, message_type, content, created_at)
                VALUES (?, ?, ?, ?)
            """, (session_id, message_type, content_json, now))

            conn.commit()
            message_id = cursor.lastrowid

            logger.debug(f"Added message {message_id} to session {session_id}: type={message_type}")
            return message_id

    def get_messages(
        self,
        session_id: str,
        since_message_id: int = 0,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get messages for a session with cursor-based pagination.

        Args:
            session_id: Session identifier
            since_message_id: Return messages with ID > this value (0 for all messages)
            limit: Maximum number of messages to return

        Returns:
            List of message dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM messages
                WHERE session_id = ? AND message_id > ?
                ORDER BY message_id ASC
                LIMIT ?
            """, (session_id, since_message_id, limit))

            rows = cursor.fetchall()
            messages = [
                {
                    "message_id": row["message_id"],
                    "session_id": row["session_id"],
                    "message_type": row["message_type"],
                    "content": json.loads(row["content"]),
                    "created_at": row["created_at"]
                }
                for row in rows
            ]

            logger.debug(f"Retrieved {len(messages)} messages for session {session_id} since message_id {since_message_id}")
            return messages

    def get_message_count(self, session_id: str) -> int:
        """Get total message count for a session.

        Args:
            session_id: Session identifier

        Returns:
            Total number of messages
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) as count FROM messages WHERE session_id = ?
            """, (session_id,))

            row = cursor.fetchone()
            return row["count"] if row else 0

    def clear_messages(self, session_id: str) -> None:
        """Clear all messages for a session.

        Args:
            session_id: Session identifier
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.commit()
            logger.info(f"Cleared all messages for session: {session_id}")
