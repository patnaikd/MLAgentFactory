#!/usr/bin/env python3
"""CLI for managing the Session Manager service.

This CLI provides commands for:
- Starting/stopping the Session Manager API service
- Monitoring running sessions
- Health checks
- Session statistics
"""

import argparse
import subprocess
import sys
import time
import requests
from typing import Optional
from pathlib import Path


API_BASE_URL = "http://localhost:8000"
API_PORT = 8000


def check_service_health() -> bool:
    """Check if the Session Manager service is running.

    Returns:
        bool: True if service is healthy, False otherwise
    """
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def start_service(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Start the Session Manager API service.

    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload for development
    """
    print(f"Starting Session Manager API on {host}:{port}")

    # Check if already running
    if check_service_health():
        print("❌ Service is already running!")
        print(f"   Access it at: {API_BASE_URL}")
        sys.exit(1)

    # Start uvicorn
    cmd = [
        "uvicorn",
        "mlagentfactory.services.api:app",
        "--host", host,
        "--port", str(port),
    ]

    if reload:
        cmd.append("--reload")

    print(f"Command: {' '.join(cmd)}")
    print("Starting service...")

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\nService stopped by user")
    except Exception as e:
        print(f"❌ Failed to start service: {e}")
        sys.exit(1)


def stop_service() -> None:
    """Stop the Session Manager API service."""
    print("Stopping Session Manager API...")

    # Try to find and kill the process
    try:
        # On Unix systems
        result = subprocess.run(
            ["pkill", "-f", "mlagentfactory.services.api"],
            capture_output=True
        )

        if result.returncode == 0:
            print("✅ Service stopped successfully")
        else:
            print("⚠️  No running service found")

    except Exception as e:
        print(f"❌ Failed to stop service: {e}")
        print("You may need to stop the process manually")
        sys.exit(1)


def status() -> None:
    """Check the status of the Session Manager service."""
    print("Checking Session Manager API status...")

    if check_service_health():
        print("✅ Service is running")
        print(f"   URL: {API_BASE_URL}")

        # Get health info
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            health = response.json()

            print(f"\n📊 Health Status:")
            print(f"   Status: {health.get('status', 'unknown')}")
            print(f"   Active Sessions: {health.get('active_sessions', 0)}")
            print(f"   Total Sessions: {health.get('total_sessions', 0)}")

            if health.get('active_session_ids'):
                print(f"   Active Session IDs:")
                for session_id in health['active_session_ids']:
                    print(f"      - {session_id}")

        except Exception as e:
            print(f"⚠️  Could not get detailed health info: {e}")

    else:
        print("❌ Service is not running")
        print(f"   Start it with: python -m mlagentfactory.cli.session_manager_cli start")


def list_sessions() -> None:
    """List all sessions."""
    print("Fetching sessions...")

    if not check_service_health():
        print("❌ Service is not running")
        sys.exit(1)

    try:
        response = requests.get(f"{API_BASE_URL}/sessions", timeout=10)
        response.raise_for_status()
        sessions = response.json()

        if not sessions:
            print("No sessions found")
            return

        print(f"\n📋 Sessions ({len(sessions)} total):\n")

        for session in sessions:
            session_id = session['session_id']
            status = session['status']
            cost = session['total_cost']
            created_at = session['created_at']
            process_alive = session.get('process_alive', False)

            process_status = "🟢 Running" if process_alive else "🔴 Stopped"

            print(f"  Session: {session_id}")
            print(f"    Status: {status}")
            print(f"    Process: {process_status}")
            print(f"    Cost: ${cost:.4f}")
            print(f"    Created: {created_at}")
            print()

    except Exception as e:
        print(f"❌ Failed to list sessions: {e}")
        sys.exit(1)


def session_info(session_id: str) -> None:
    """Get detailed information about a session.

    Args:
        session_id: Session identifier
    """
    print(f"Fetching info for session: {session_id}")

    if not check_service_health():
        print("❌ Service is not running")
        sys.exit(1)

    try:
        response = requests.get(f"{API_BASE_URL}/sessions/{session_id}/stats", timeout=10)
        response.raise_for_status()
        stats = response.json()

        print(f"\n📊 Session Statistics:\n")
        print(f"  Session ID: {stats['session_id']}")
        print(f"  Status: {stats['status']}")
        print(f"  Process: {'🟢 Running' if stats['process_alive'] else '🔴 Stopped'}")
        print(f"  Created: {stats['created_at']}")
        print(f"  Updated: {stats['updated_at']}")
        print(f"  Total Cost: ${stats['total_cost']:.4f}")
        print(f"  Message Count: {stats['message_count']}")

        if stats.get('agent_session_id'):
            print(f"  Agent Session ID: {stats['agent_session_id']}")

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"❌ Session not found: {session_id}")
        else:
            print(f"❌ Failed to get session info: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to get session info: {e}")
        sys.exit(1)


def stop_session(session_id: str) -> None:
    """Stop a specific session.

    Args:
        session_id: Session identifier
    """
    print(f"Stopping session: {session_id}")

    if not check_service_health():
        print("❌ Service is not running")
        sys.exit(1)

    try:
        response = requests.delete(f"{API_BASE_URL}/sessions/{session_id}", timeout=10)
        response.raise_for_status()

        print(f"✅ Session {session_id} stopped successfully")

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"❌ Session not found: {session_id}")
        else:
            print(f"❌ Failed to stop session: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to stop session: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CLI for managing the MLAgentFactory Session Manager service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start the service
  python -m mlagentfactory.cli.session_manager_cli start

  # Start with auto-reload for development
  python -m mlagentfactory.cli.session_manager_cli start --reload

  # Check service status
  python -m mlagentfactory.cli.session_manager_cli status

  # List all sessions
  python -m mlagentfactory.cli.session_manager_cli sessions

  # Get session info
  python -m mlagentfactory.cli.session_manager_cli info <session-id>

  # Stop a session
  python -m mlagentfactory.cli.session_manager_cli stop-session <session-id>

  # Stop the service
  python -m mlagentfactory.cli.session_manager_cli stop
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start the Session Manager API service")
    start_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    start_parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    start_parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    # Stop command
    subparsers.add_parser("stop", help="Stop the Session Manager API service")

    # Status command
    subparsers.add_parser("status", help="Check service status")

    # Sessions command
    subparsers.add_parser("sessions", help="List all sessions")

    # Info command
    info_parser = subparsers.add_parser("info", help="Get detailed session information")
    info_parser.add_argument("session_id", help="Session ID")

    # Stop session command
    stop_session_parser = subparsers.add_parser("stop-session", help="Stop a specific session")
    stop_session_parser.add_argument("session_id", help="Session ID")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    if args.command == "start":
        start_service(host=args.host, port=args.port, reload=args.reload)
    elif args.command == "stop":
        stop_service()
    elif args.command == "status":
        status()
    elif args.command == "sessions":
        list_sessions()
    elif args.command == "info":
        session_info(args.session_id)
    elif args.command == "stop-session":
        stop_session(args.session_id)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
