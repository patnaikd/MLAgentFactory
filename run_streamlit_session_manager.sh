#!/bin/bash
# Start the Streamlit UI with Session Manager integration

echo "Starting MLAgentFactory Streamlit UI (Session Manager version)..."
echo ""
echo "NOTE: Make sure the Session Manager API is running first!"
echo "      Start it with: ./start_session_manager.sh"
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

# Check if Session Manager API is running
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "⚠️  Warning: Session Manager API is not running!"
    echo "   Start it with: ./start_session_manager.sh"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Start Streamlit using module syntax
cd "${SCRIPT_DIR}"
uv run python -m streamlit run src/mlagentfactory/ui/streamlit_ui_session_manager.py
