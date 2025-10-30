#!/bin/bash
# Start the Session Manager API service

echo "Starting MLAgentFactory Session Manager API..."
echo ""

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' is not installed. Please install it first."
    exit 1
fi

# Start the service
uv run python -m mlagentfactory.cli.session_manager_cli start "$@"
