"""Main Streamlit application for MLAgentFactory chat interface"""

import streamlit as st
from typing import Dict, List, Optional
import asyncio
from datetime import datetime
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from mlagentfactory.agents.orchestrator import OrchestratorAgent
from mlagentfactory.utils.logging_config import initialize_observability


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "experiment_id" not in st.session_state:
        st.session_state.experiment_id = None

    if "problem_details" not in st.session_state:
        st.session_state.problem_details = None

    if "conversation_started" not in st.session_state:
        st.session_state.conversation_started = False

    if "orchestrator" not in st.session_state:
        # Initialize orchestrator with API key from environment
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            st.error("⚠️ ANTHROPIC_API_KEY not found in environment variables. Please set it to use the application.")
            st.stop()
        st.session_state.orchestrator = OrchestratorAgent(api_key)


def render_chat_message(role: str, content: str):
    """Render a chat message with appropriate styling"""
    with st.chat_message(role):
        st.markdown(content)


def render_problem_details(problem_details: Optional[Dict]):
    """Render extracted problem details in sidebar"""
    if problem_details:
        st.sidebar.markdown("### 📊 Problem Details")
        st.sidebar.markdown(f"**Type:** {problem_details.get('problem_type', 'N/A')}")
        st.sidebar.markdown(f"**Target:** {problem_details.get('target_variable', 'N/A')}")
        st.sidebar.markdown(f"**Metric:** {problem_details.get('evaluation_metric', 'N/A')}")

        if problem_details.get('dataset_info'):
            st.sidebar.markdown("**Dataset Info:**")
            st.sidebar.json(problem_details['dataset_info'])


def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="MLAgentFactory - Agentic ML Assistant",
        page_icon="🤖",
        layout="wide"
    )

    # Initialize observability
    initialize_observability(log_level="DEBUG", enable_tracing=False)

    # Initialize session state
    initialize_session_state()

    # Header
    st.title("🤖 MLAgentFactory")
    st.markdown("### Agentic ML Experimentation System")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.markdown("## 🚀 Getting Started")
        st.markdown("""
        1. Enter the URL to your ML problem definition
        2. Chat with the AI agent to refine requirements
        3. The system will guide you through the process
        """)

        st.markdown("---")

        # Display problem details if available
        render_problem_details(st.session_state.problem_details)

        if st.session_state.experiment_id:
            st.markdown("---")
            st.markdown(f"**Experiment ID:** `{st.session_state.experiment_id}`")

    # Main chat interface
    if not st.session_state.conversation_started:
        st.info("👋 Welcome! Please provide the URL to your ML problem definition (e.g., Kaggle competition page) to get started.")

        # URL input
        url_input = st.text_input(
            "ML Problem URL",
            placeholder="https://www.kaggle.com/competitions/...",
            help="Enter the URL where your ML problem is defined"
        )

        if st.button("Start Analysis", type="primary"):
            if url_input:
                st.session_state.conversation_started = True

                # Add user message
                user_msg = f"Analyze this ML problem: {url_input}"
                st.session_state.messages.append({
                    "role": "user",
                    "content": user_msg,
                    "timestamp": datetime.now()
                })

                # Process URL with orchestrator
                with st.spinner("🔍 Analyzing the ML problem..."):
                    result = asyncio.run(
                        st.session_state.orchestrator.process_initial_url(url_input)
                    )

                    if result["success"]:
                        st.session_state.experiment_id = result.get("experiment_id")
                        st.session_state.problem_details = result.get("problem_details")

                        # Add assistant response
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["message"],
                            "timestamp": datetime.now()
                        })
                    else:
                        # Add error message
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"❌ {result['message']}",
                            "timestamp": datetime.now()
                        })

                st.rerun()
            else:
                st.warning("Please enter a valid URL")

    else:
        # Display chat history
        for message in st.session_state.messages:
            render_chat_message(message["role"], message["content"])

        # Chat input
        if prompt := st.chat_input("Ask about your ML problem..."):
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now()
            })

            # Display user message
            render_chat_message("user", prompt)

            # Process with orchestrator
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = asyncio.run(
                        st.session_state.orchestrator.handle_user_message(prompt)
                    )

                    if result["success"]:
                        response = result["message"]

                        # Update problem details if they changed
                        if "problem_details" in result:
                            st.session_state.problem_details = result["problem_details"]
                    else:
                        response = f"❌ {result['message']}"

                    st.markdown(response)

            # Add assistant response
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now()
            })

            st.rerun()

        # Clear conversation button
        if st.sidebar.button("🔄 Start New Conversation"):
            st.session_state.messages = []
            st.session_state.conversation_started = False
            st.session_state.problem_details = None
            st.session_state.experiment_id = None
            st.rerun()


if __name__ == "__main__":
    main()
