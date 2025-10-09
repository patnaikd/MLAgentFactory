"""
Main application entry point for MLAgentFactory.

This module provides a simple entry point to run the Streamlit UI.
"""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import and run the Streamlit UI
from mlagentfactory.ui.streamlit_ui import main

if __name__ == "__main__":
    main()
