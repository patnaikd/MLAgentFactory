#!/bin/bash
# Start the Session Manager API service

echo "Starting MLAgentFactory Session Manager API..."
echo ""

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' is not installed. Please install it first."
    exit 1
fi

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set PYTHONPATH to include src directory
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"

# Start the service
uv run python -m mlagentfactory.cli.session_manager_cli start "$@"
