"""Streamlit UI for MLAgentFactory chat interface"""

import streamlit as st
import asyncio
from datetime import datetime
from dotenv import load_dotenv

from ..agents.chat_agent import ChatAgent
from ..utils.logging_config import initialize_observability

# Load environment variables from .env file
load_dotenv()


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "agent" not in st.session_state:
        st.session_state.agent = None


def render_chat_message(role: str, content: str):
    """Render a chat message with appropriate styling"""
    with st.chat_message(role):
        st.markdown(content)


def get_or_create_agent():
    """Get existing agent or create a new one."""
    if st.session_state.agent is None:
        st.session_state.agent = ChatAgent()
    return st.session_state.agent


async def process_message(message: str) -> str:
    """Process a message through the agent."""
    agent = get_or_create_agent()

    # Initialize if not already initialized
    if not agent._initialized:
        await agent.initialize()

    return await agent.chat(message)


async def cleanup_agent():
    """Cleanup the agent."""
    if st.session_state.agent:
        await st.session_state.agent.cleanup()
        st.session_state.agent = None


def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="MLAgentFactory - AI Chat Assistant",
        page_icon="🤖",
        layout="wide"
    )

    # Initialize observability
    initialize_observability(log_level="INFO", enable_tracing=False)

    # Initialize session state
    initialize_session_state()

    # Header
    st.title("🤖 MLAgentFactory")
    st.markdown("### AI Chat Assistant with Tool Support")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.markdown("## 💬 Chat Assistant")
        st.markdown("""
        This assistant can:
        - Answer your questions
        - Maintain conversation history
        - Use tools like file creation
        - Remember context from previous messages
        """)

        st.markdown("---")

        # Clear conversation button
        if st.button("🔄 Start New Conversation"):
            # Clean up old agent
            if st.session_state.agent:
                try:
                    asyncio.run(cleanup_agent())
                except Exception as e:
                    st.warning(f"Cleanup warning: {e}")
            st.session_state.messages = []
            st.rerun()

        # Display message count
        if st.session_state.messages:
            st.markdown(f"**Messages:** {len(st.session_state.messages)}")

    # Display chat history
    for message in st.session_state.messages:
        render_chat_message(message["role"], message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now()
        })

        # Display user message
        render_chat_message("user", prompt)

        # Process with agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = asyncio.run(process_message(prompt))
                    st.markdown(response)
                except Exception as e:
                    response = f"❌ Error: {str(e)}"
                    st.error(response)

        # Add assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now()
        })

        st.rerun()


if __name__ == "__main__":
    main()
