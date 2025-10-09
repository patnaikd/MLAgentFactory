"""
Web fetching tools for MLAgentFactory agents.

This module contains tools for fetching and extracting content from web pages.
"""

from pathlib import Path
from typing import Optional
import logging

from claude_agent_sdk import tool
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


async def fetch_with_playwright(url: str) -> Optional[str]:
    """Fetch HTML content from URL using Playwright.

    Args:
        url: URL to fetch

    Returns:
        HTML content as string, or None if failed
    """
    try:
        logger.info(f"Fetching content with Playwright from {url}")
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            # Set user agent
            await page.set_extra_http_headers({
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            })

            # Navigate to the page and wait for network to be idle
            await page.goto(url, wait_until="networkidle", timeout=30000)

            # Wait a bit more for any dynamic content
            await page.wait_for_timeout(2000)

            # Get the rendered HTML
            html = await page.content()
            await browser.close()

            logger.info(f"Successfully fetched {len(html)} bytes from {url}")
            return html
    except Exception as e:
        logger.error(f"Failed to fetch URL {url} with Playwright: {e}")
        return None


def extract_text_content(html: str) -> str:
    """Extract clean text content from HTML.

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


@tool(
    "fetch_webpage",
    "Fetch and extract text content from a webpage URL using Playwright (supports JavaScript-heavy sites)",
    {"url": str}
)
async def fetch_webpage(args):
    """Fetch and extract text content from a webpage."""
    try:
        url = args["url"]

        # Fetch HTML with Playwright
        html = await fetch_with_playwright(url)

        if not html:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Failed to fetch content from {url}"
                }],
                "is_error": True
            }

        # Extract text content
        text = extract_text_content(html)

        if not text or len(text) < 100:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Insufficient content extracted from {url}. Only {len(text)} characters found."
                }],
                "is_error": True
            }

        # Limit text length to avoid token limits
        max_length = 8000
        if len(text) > max_length:
            text = text[:max_length] + f"\n\n[Content truncated. Total length: {len(text)} characters]"

        return {
            "content": [{
                "type": "text",
                "text": f"Successfully fetched content from {url}\n\n{text}"
            }]
        }

    except Exception as e:
        logger.error(f"Error in fetch_webpage tool: {e}")
        return {
            "content": [{
                "type": "text",
                "text": f"Error fetching webpage: {str(e)}"
            }],
            "is_error": True
        }
