"""Web Fetcher Agent for extracting ML problem definitions from URLs"""

import requests
from bs4 import BeautifulSoup
from typing import Dict, Optional
import re
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class WebFetcherAgent:
    """Agent responsible for fetching and parsing ML problem definitions from URLs"""

    def __init__(self, anthropic_api_key: str):
        """Initialize the WebFetcher agent

        Args:
            anthropic_api_key: API key for Anthropic Claude
        """
        self.anthropic_api_key = anthropic_api_key
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }

    def fetch_page_content(self, url: str) -> Optional[str]:
        """Fetch HTML content from URL

        Args:
            url: URL to fetch

        Returns:
            HTML content as string, or None if failed
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"Failed to fetch URL {url}: {e}")
            return None

    def extract_text_content(self, html: str) -> str:
        """Extract text content from HTML

        Args:
            html: Raw HTML content

        Returns:
            Cleaned text content
        """
        soup = BeautifulSoup(html, 'lxml')

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # Get text
        text = soup.get_text()

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)

        return text

    def parse_with_llm(self, content: str, url: str) -> Dict:
        """Use Claude to parse the problem definition

        Args:
            content: Text content from the page
            url: Original URL for context

        Returns:
            Dictionary containing parsed problem details
        """
        try:
            from anthropic import Anthropic

            client = Anthropic(api_key=self.anthropic_api_key)

            prompt = f"""Analyze this ML problem definition from {url} and extract the following information:

Content:
{content[:8000]}  # Limit content size

Please provide a structured analysis with:
1. Problem Type (classification, regression, time-series, etc.)
2. Target Variable (what needs to be predicted)
3. Dataset Description (features, size, domain)
4. Evaluation Metric (accuracy, RMSE, F1, etc.)
5. Key Challenges (class imbalance, missing data, etc.)
6. Dataset URL (if mentioned)
7. Submission Format (if applicable)

Format your response as a structured summary that can be easily parsed."""

            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            response_text = message.content[0].text

            # Parse the response into structured format
            return self._structure_llm_response(response_text, url)

        except Exception as e:
            logger.error(f"Failed to parse with LLM: {e}")
            return {
                "error": str(e),
                "raw_content": content[:1000]
            }

    def _structure_llm_response(self, response: str, url: str) -> Dict:
        """Convert LLM response into structured format

        Args:
            response: LLM response text
            url: Original URL

        Returns:
            Structured dictionary
        """
        # Basic parsing - can be enhanced with more sophisticated extraction
        return {
            "url": url,
            "problem_type": self._extract_field(response, "Problem Type"),
            "target_variable": self._extract_field(response, "Target Variable"),
            "dataset_description": self._extract_field(response, "Dataset Description"),
            "evaluation_metric": self._extract_field(response, "Evaluation Metric"),
            "key_challenges": self._extract_field(response, "Key Challenges"),
            "dataset_url": self._extract_field(response, "Dataset URL"),
            "submission_format": self._extract_field(response, "Submission Format"),
            "full_analysis": response
        }

    def _extract_field(self, text: str, field_name: str) -> str:
        """Extract a field value from LLM response

        Args:
            text: Full text to search
            field_name: Name of field to extract

        Returns:
            Extracted value or "Not specified"
        """
        # Look for patterns like "Field Name: value" or "Field Name\nvalue"
        patterns = [
            rf"{field_name}:\s*(.+?)(?:\n\n|\n\d+\.|\Z)",
            rf"{field_name}\s*\n\s*(.+?)(?:\n\n|\n\d+\.|\Z)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                value = match.group(1).strip()
                # Clean up the value
                value = re.sub(r'\n+', ' ', value)
                return value[:500]  # Limit length

        return "Not specified"

    async def analyze_url(self, url: str) -> Dict:
        """Main method to analyze an ML problem URL

        Args:
            url: URL to analyze

        Returns:
            Structured problem definition
        """
        logger.info(f"Analyzing URL: {url}")

        # Validate URL
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            return {"error": "Invalid URL format"}

        # Fetch content
        html = self.fetch_page_content(url)
        if not html:
            return {"error": "Failed to fetch page content"}

        # Extract text
        text = self.extract_text_content(html)
        if not text or len(text) < 100:
            return {"error": "Insufficient content extracted from page"}

        # Parse with LLM
        result = self.parse_with_llm(text, url)

        logger.info("URL analysis completed")
        return result
