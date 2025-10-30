"""FastAPI REST API for Session Manager.

This API provides endpoints for:
- Creating and managing agent sessions
- Sending queries to agents
- Polling for new messages (pull-based)
- Session statistics and health checks
"""

import logging
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .session_manager import SessionManager
from ..utils.logging_config import initialize_observability


logger = logging.getLogger(__name__)

# Global session manager instance
session_manager: Optional[SessionManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app.

    Handles startup and shutdown events.
    """
    # Startup
    global session_manager

    # Initialize logging
    initialize_observability(log_level="INFO", enable_tracing=False)
    logger.info("Starting Session Manager API")

    # Create session manager
    session_manager = SessionManager(db_path="data/sessions.db")
    logger.info("Session Manager initialized")

    yield

    # Shutdown
    logger.info("Shutting down Session Manager API")
    if session_manager:
        session_manager.cleanup()
    logger.info("Session Manager API shut down complete")


# Create FastAPI app
app = FastAPI(
    title="MLAgentFactory Session Manager API",
    description="REST API for managing long-running Claude agent sessions",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================
# Request/Response Models
# ===========================

class CreateSessionRequest(BaseModel):
    """Request model for creating a session."""
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata for the session")


class CreateSessionResponse(BaseModel):
    """Response model for session creation."""
    session_id: str = Field(..., description="Unique session identifier")
    status: str = Field(..., description="Session status")
    created_at: str = Field(..., description="Session creation timestamp")
    message: str = Field(default="Session created successfully", description="Status message")


class SendQueryRequest(BaseModel):
    """Request model for sending a query to an agent."""
    message: str = Field(..., description="User message to send to the agent", min_length=1)


class SendQueryResponse(BaseModel):
    """Response model for query submission."""
    status: str = Field(..., description="Status of the operation")
    session_id: str = Field(..., description="Session identifier")
    message: str = Field(..., description="Status message")


class MessageResponse(BaseModel):
    """Response model for a single message."""
    message_id: int = Field(..., description="Unique message identifier")
    session_id: str = Field(..., description="Session identifier")
    message_type: str = Field(..., description="Type of message (text, tool_use, etc.)")
    content: Any = Field(..., description="Message content")
    created_at: str = Field(..., description="Message timestamp")


class GetMessagesResponse(BaseModel):
    """Response model for getting messages."""
    messages: List[MessageResponse] = Field(..., description="List of messages")
    next_cursor: Optional[int] = Field(None, description="Cursor for next batch of messages")
    has_more: bool = Field(..., description="Whether more messages are available")
    count: int = Field(..., description="Number of messages in this response")


class SessionInfoResponse(BaseModel):
    """Response model for session information."""
    session_id: str = Field(..., description="Session identifier")
    status: str = Field(..., description="Session status")
    created_at: str = Field(..., description="Session creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    agent_session_id: Optional[str] = Field(None, description="Agent's internal session ID")
    total_cost: float = Field(..., description="Total cost in USD")
    process_alive: bool = Field(..., description="Whether the agent process is running")


class SessionStatsResponse(BaseModel):
    """Response model for session statistics."""
    session_id: str = Field(..., description="Session identifier")
    status: str = Field(..., description="Session status")
    created_at: str = Field(..., description="Session creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    agent_session_id: Optional[str] = Field(None, description="Agent's internal session ID")
    total_cost: float = Field(..., description="Total cost in USD")
    message_count: int = Field(..., description="Total number of messages")
    process_alive: bool = Field(..., description="Whether the agent process is running")


class HealthCheckResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Health status")
    active_sessions: int = Field(..., description="Number of active sessions")
    total_sessions: int = Field(..., description="Total number of sessions")
    active_session_ids: List[str] = Field(..., description="List of active session IDs")


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")


# ===========================
# API Endpoints
# ===========================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "MLAgentFactory Session Manager API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint.

    Returns:
        Health status including active and total session counts
    """
    try:
        health = session_manager.health_check()
        return health
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions", response_model=CreateSessionResponse, status_code=201)
async def create_session(request: CreateSessionRequest):
    """Create a new agent session.

    Args:
        request: Session creation request with optional metadata

    Returns:
        Session information including session_id
    """
    try:
        session = session_manager.create_session(metadata=request.metadata)
        return {
            "session_id": session["session_id"],
            "status": session["status"],
            "created_at": session["created_at"],
            "message": "Session created successfully"
        }
    except Exception as e:
        logger.error(f"Failed to create session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions", response_model=List[SessionInfoResponse])
async def list_sessions(
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of sessions to return")
):
    """List all sessions.

    Args:
        limit: Maximum number of sessions to return (1-1000)

    Returns:
        List of session information
    """
    try:
        sessions = session_manager.list_sessions(limit=limit)
        return sessions
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}", response_model=SessionInfoResponse)
async def get_session(session_id: str):
    """Get information about a specific session.

    Args:
        session_id: Session identifier

    Returns:
        Session information
    """
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
        return session
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/stats", response_model=SessionStatsResponse)
async def get_session_stats(session_id: str):
    """Get statistics for a specific session.

    Args:
        session_id: Session identifier

    Returns:
        Session statistics including message count
    """
    try:
        stats = session_manager.get_session_stats(session_id)
        return stats
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get session stats {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{session_id}/query", response_model=SendQueryResponse)
async def send_query(session_id: str, request: SendQueryRequest):
    """Send a query to an agent session.

    The agent will process the query asynchronously. Use the GET /sessions/{session_id}/messages
    endpoint to poll for responses.

    Args:
        session_id: Session identifier
        request: Query request with user message

    Returns:
        Status response
    """
    try:
        result = session_manager.send_query(session_id, request.message)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to send query to session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/messages", response_model=GetMessagesResponse)
async def get_messages(
    session_id: str,
    since_message_id: int = Query(0, ge=0, description="Return messages after this ID (0 for all)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of messages to return")
):
    """Get messages from a session with cursor-based pagination.

    This endpoint implements a pull-based message retrieval system. The client should:
    1. Call with since_message_id=0 to get initial messages
    2. Store the next_cursor from the response
    3. Poll again using next_cursor as since_message_id to get new messages
    4. Continue polling as long as has_more is true or while waiting for new messages

    Args:
        session_id: Session identifier
        since_message_id: Return messages with ID > this value (0 for all messages)
        limit: Maximum number of messages to return (1-1000)

    Returns:
        Messages with pagination cursor
    """
    try:
        result = session_manager.get_messages(
            session_id=session_id,
            since_message_id=since_message_id,
            limit=limit
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get messages for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sessions/{session_id}", status_code=200)
async def stop_session(session_id: str):
    """Stop a session and cleanup resources.

    This will gracefully stop the agent process and update the session status.
    The session data is NOT deleted and can still be queried.

    Args:
        session_id: Session identifier

    Returns:
        Status response
    """
    try:
        result = session_manager.stop_session(session_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to stop session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sessions/{session_id}/data", status_code=200)
async def delete_session(session_id: str):
    """Delete a session and all its data permanently.

    This will stop the agent process and delete all session data including messages.
    This action cannot be undone.

    Args:
        session_id: Session identifier

    Returns:
        Status response
    """
    try:
        result = session_manager.delete_session(session_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to delete session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ===========================
# Error Handlers
# ===========================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return HTTPException(status_code=404, detail="Resource not found")


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    return HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn

    # Run the API server
    uvicorn.run(
        "mlagentfactory.services.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
