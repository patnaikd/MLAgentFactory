"""Orchestrator Agent for managing conversation and coordinating sub-agents"""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """States in the conversation workflow"""
    INITIALIZING = "initializing"
    ANALYZING_URL = "analyzing_url"
    REFINING_PROBLEM = "refining_problem"
    GATHERING_DATASET = "gathering_dataset"
    READY_FOR_ANALYSIS = "ready_for_analysis"
    ERROR = "error"
    COMPLETED = "completed"


class OrchestratorAgent:
    """Central coordinator agent managing the conversation flow and sub-agents"""

    def __init__(self, anthropic_api_key: str):
        """Initialize the orchestrator agent

        Args:
            anthropic_api_key: API key for Anthropic Claude
        """
        self.anthropic_api_key = anthropic_api_key
        self.state = ConversationState.INITIALIZING
        self.context: Dict = {}
        self.conversation_history: List[Dict] = []
        self.experiment_id = str(uuid.uuid4())

        # Import sub-agents lazily to avoid circular dependencies
        from .web_fetcher import WebFetcherAgent

        self.web_fetcher = WebFetcherAgent(anthropic_api_key)

        logger.info(f"Orchestrator initialized with experiment_id: {self.experiment_id}")

    def add_to_history(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to conversation history

        Args:
            role: Role of the speaker (user/assistant/system)
            content: Message content
            metadata: Optional metadata
        """
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        })

    async def process_initial_url(self, url: str) -> Dict:
        """Process the initial URL provided by the user

        Args:
            url: URL to the ML problem definition

        Returns:
            Response dictionary with analysis results
        """
        try:
            self.state = ConversationState.ANALYZING_URL
            logger.info(f"Processing initial URL: {url}")

            # Use web fetcher to analyze the URL
            analysis_result = await self.web_fetcher.analyze_url(url)

            if "error" in analysis_result:
                self.state = ConversationState.ERROR
                return {
                    "success": False,
                    "message": f"Failed to analyze URL: {analysis_result['error']}",
                    "state": self.state.value
                }

            # Store the analysis in context
            self.context["problem_definition"] = analysis_result
            self.context["source_url"] = url
            self.state = ConversationState.REFINING_PROBLEM

            # Generate a friendly summary
            summary = self._generate_problem_summary(analysis_result)

            return {
                "success": True,
                "message": summary,
                "problem_details": analysis_result,
                "state": self.state.value,
                "experiment_id": self.experiment_id
            }

        except Exception as e:
            logger.error(f"Error processing URL: {e}")
            self.state = ConversationState.ERROR
            return {
                "success": False,
                "message": f"An error occurred: {str(e)}",
                "state": self.state.value
            }

    def _generate_problem_summary(self, analysis: Dict) -> str:
        """Generate a human-friendly summary of the problem analysis

        Args:
            analysis: Problem analysis dictionary

        Returns:
            Formatted summary string
        """
        summary = "## 🎯 Problem Analysis Complete\n\n"
        summary += f"I've analyzed the ML problem. Here's what I found:\n\n"

        if analysis.get("problem_type"):
            summary += f"**Problem Type:** {analysis['problem_type']}\n\n"

        if analysis.get("target_variable"):
            summary += f"**Target Variable:** {analysis['target_variable']}\n\n"

        if analysis.get("evaluation_metric"):
            summary += f"**Evaluation Metric:** {analysis['evaluation_metric']}\n\n"

        if analysis.get("key_challenges"):
            summary += f"**Key Challenges:** {analysis['key_challenges']}\n\n"

        summary += "\n---\n\n"
        summary += "What would you like to do next?\n"
        summary += "- Ask me questions about the problem\n"
        summary += "- Provide the dataset URL\n"
        summary += "- Discuss specific aspects of the problem\n"

        return summary

    async def handle_user_message(self, message: str) -> Dict:
        """Handle a user message in the conversation

        Args:
            message: User's message

        Returns:
            Response dictionary
        """
        try:
            self.add_to_history("user", message)

            # Check if message contains a URL pattern for dataset
            if self._is_url(message) and self.state == ConversationState.REFINING_PROBLEM:
                return await self._handle_dataset_url(message)

            # Use Claude to generate contextual response
            response = await self._generate_contextual_response(message)

            self.add_to_history("assistant", response)

            return {
                "success": True,
                "message": response,
                "state": self.state.value,
                "context": self._get_context_summary()
            }

        except Exception as e:
            logger.error(f"Error handling message: {e}")
            return {
                "success": False,
                "message": f"Sorry, I encountered an error: {str(e)}",
                "state": self.state.value
            }

    def _is_url(self, text: str) -> bool:
        """Check if text contains a URL

        Args:
            text: Text to check

        Returns:
            True if URL detected
        """
        import re
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return bool(re.search(url_pattern, text))

    async def _handle_dataset_url(self, url: str) -> Dict:
        """Handle dataset URL provided by user

        Args:
            url: Dataset URL

        Returns:
            Response dictionary
        """
        self.state = ConversationState.GATHERING_DATASET
        self.context["dataset_url"] = url

        response = f"✅ Great! I've noted the dataset URL: `{url}`\n\n"
        response += "I'm now ready to help you with:\n"
        response += "- Understanding the dataset structure\n"
        response += "- Planning the analysis approach\n"
        response += "- Discussing preprocessing strategies\n"
        response += "- Model selection recommendations\n\n"
        response += "What would you like to explore?"

        self.state = ConversationState.READY_FOR_ANALYSIS

        return {
            "success": True,
            "message": response,
            "state": self.state.value
        }

    async def _generate_contextual_response(self, message: str) -> str:
        """Generate a contextual response using Claude

        Args:
            message: User's message

        Returns:
            Generated response
        """
        try:
            from anthropic import Anthropic

            client = Anthropic(api_key=self.anthropic_api_key)

            # Build context for Claude
            context_str = self._build_context_for_llm()

            system_prompt = f"""You are an AI assistant helping with ML experimentation.

Current Context:
{context_str}

Your role is to:
1. Answer questions about the ML problem
2. Guide the user through problem formulation
3. Suggest data analysis approaches
4. Recommend modeling strategies
5. Help clarify requirements

Be helpful, concise, and focus on practical ML advice."""

            # Build message history for Claude
            messages = []
            for msg in self.conversation_history[-5:]:  # Last 5 messages for context
                if msg["role"] in ["user", "assistant"]:
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

            messages.append({"role": "user", "content": message})

            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1500,
                system=system_prompt,
                messages=messages
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error generating a response. Could you rephrase your question?"

    def _build_context_for_llm(self) -> str:
        """Build context string for LLM

        Returns:
            Formatted context string
        """
        context_parts = []

        context_parts.append(f"Experiment ID: {self.experiment_id}")
        context_parts.append(f"Current State: {self.state.value}")

        if "problem_definition" in self.context:
            prob = self.context["problem_definition"]
            context_parts.append(f"\nProblem Definition:")
            context_parts.append(f"- Type: {prob.get('problem_type', 'Unknown')}")
            context_parts.append(f"- Target: {prob.get('target_variable', 'Unknown')}")
            context_parts.append(f"- Metric: {prob.get('evaluation_metric', 'Unknown')}")

        if "dataset_url" in self.context:
            context_parts.append(f"\nDataset URL: {self.context['dataset_url']}")

        return "\n".join(context_parts)

    def _get_context_summary(self) -> Dict:
        """Get a summary of current context

        Returns:
            Context summary dictionary
        """
        return {
            "experiment_id": self.experiment_id,
            "state": self.state.value,
            "has_problem_definition": "problem_definition" in self.context,
            "has_dataset_url": "dataset_url" in self.context,
            "message_count": len(self.conversation_history)
        }

    def get_experiment_context(self) -> Dict:
        """Get the full experiment context

        Returns:
            Full context dictionary
        """
        return {
            "experiment_id": self.experiment_id,
            "state": self.state.value,
            "context": self.context,
            "conversation_history": self.conversation_history
        }
