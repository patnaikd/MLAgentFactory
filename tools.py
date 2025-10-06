"""
Custom tools for MLAgentFactory agents.

This module contains reusable tools that can be used with Claude agents.
"""

from pathlib import Path

from claude_agent_sdk import tool


@tool(
    "write_file",
    "Write content to a file at the specified path",
    {"path": str, "content": str}
)
async def write_file(args):
    """Write content to a file."""
    try:
        file_path = Path(args["path"])
        file_path.write_text(args["content"])
        return {
            "content": [{
                "type": "text",
                "text": f"Successfully wrote to {args['path']}"
            }]
        }
    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"Error writing file: {str(e)}"
            }],
            "is_error": True
        }
