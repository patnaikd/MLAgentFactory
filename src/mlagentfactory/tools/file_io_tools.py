"""
File I/O tools for MLAgentFactory agents.

This module contains file and directory operation tools for Claude agents.
"""

from pathlib import Path
import shutil
import logging

from claude_agent_sdk import tool

logger = logging.getLogger(__name__)


@tool(
    "read_file",
    "Read content from a file at the specified path",
    {"path": str}
)
async def read_file(args):
    """Read content from a file."""
    try:
        file_path = Path(args["path"])
        if not file_path.exists():
            return {
                "content": [{
                    "type": "text",
                    "text": f"File not found: {args['path']}"
                }],
                "is_error": True
            }

        if not file_path.is_file():
            return {
                "content": [{
                    "type": "text",
                    "text": f"Path is not a file: {args['path']}"
                }],
                "is_error": True
            }

        content = file_path.read_text()
        return {
            "content": [{
                "type": "text",
                "text": f"Content of {args['path']}:\n\n{content}"
            }]
        }
    except Exception as e:
        logger.error(f"Error reading file {args.get('path')}: {e}")
        return {
            "content": [{
                "type": "text",
                "text": f"Error reading file: {str(e)}"
            }],
            "is_error": True
        }


@tool(
    "write_file",
    "Write content to a file at the specified path (creates parent directories if needed)",
    {"path": str, "content": str}
)
async def write_file(args):
    """Write content to a file."""
    try:
        file_path = Path(args["path"])
        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(args["content"])
        return {
            "content": [{
                "type": "text",
                "text": f"Successfully wrote to {args['path']}"
            }]
        }
    except Exception as e:
        logger.error(f"Error writing file {args.get('path')}: {e}")
        return {
            "content": [{
                "type": "text",
                "text": f"Error writing file: {str(e)}"
            }],
            "is_error": True
        }


@tool(
    "edit_file",
    "Edit a file by replacing old text with new text",
    {"path": str, "old_text": str, "new_text": str}
)
async def edit_file(args):
    """Edit a file by replacing old text with new text."""
    try:
        file_path = Path(args["path"])
        if not file_path.exists():
            return {
                "content": [{
                    "type": "text",
                    "text": f"File not found: {args['path']}"
                }],
                "is_error": True
            }

        content = file_path.read_text()
        old_text = args["old_text"]
        new_text = args["new_text"]

        if old_text not in content:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Text to replace not found in {args['path']}"
                }],
                "is_error": True
            }

        new_content = content.replace(old_text, new_text)
        file_path.write_text(new_content)

        return {
            "content": [{
                "type": "text",
                "text": f"Successfully edited {args['path']}"
            }]
        }
    except Exception as e:
        logger.error(f"Error editing file {args.get('path')}: {e}")
        return {
            "content": [{
                "type": "text",
                "text": f"Error editing file: {str(e)}"
            }],
            "is_error": True
        }


@tool(
    "delete_file",
    "Delete a file at the specified path",
    {"path": str}
)
async def delete_file(args):
    """Delete a file."""
    try:
        file_path = Path(args["path"])
        if not file_path.exists():
            return {
                "content": [{
                    "type": "text",
                    "text": f"File not found: {args['path']}"
                }],
                "is_error": True
            }

        if not file_path.is_file():
            return {
                "content": [{
                    "type": "text",
                    "text": f"Path is not a file: {args['path']}"
                }],
                "is_error": True
            }

        file_path.unlink()
        return {
            "content": [{
                "type": "text",
                "text": f"Successfully deleted {args['path']}"
            }]
        }
    except Exception as e:
        logger.error(f"Error deleting file {args.get('path')}: {e}")
        return {
            "content": [{
                "type": "text",
                "text": f"Error deleting file: {str(e)}"
            }],
            "is_error": True
        }


@tool(
    "list_directory",
    "List files and directories in the specified path",
    {"path": str}
)
async def list_directory(args):
    """List files and directories."""
    try:
        dir_path = Path(args["path"])
        if not dir_path.exists():
            return {
                "content": [{
                    "type": "text",
                    "text": f"Directory not found: {args['path']}"
                }],
                "is_error": True
            }

        if not dir_path.is_dir():
            return {
                "content": [{
                    "type": "text",
                    "text": f"Path is not a directory: {args['path']}"
                }],
                "is_error": True
            }

        items = []
        for item in sorted(dir_path.iterdir()):
            item_type = "DIR" if item.is_dir() else "FILE"
            items.append(f"{item_type}: {item.name}")

        result = f"Contents of {args['path']}:\n\n" + "\n".join(items)
        return {
            "content": [{
                "type": "text",
                "text": result
            }]
        }
    except Exception as e:
        logger.error(f"Error listing directory {args.get('path')}: {e}")
        return {
            "content": [{
                "type": "text",
                "text": f"Error listing directory: {str(e)}"
            }],
            "is_error": True
        }


@tool(
    "create_directory",
    "Create a new directory at the specified path (creates parent directories if needed)",
    {"path": str}
)
async def create_directory(args):
    """Create a new directory."""
    try:
        dir_path = Path(args["path"])
        dir_path.mkdir(parents=True, exist_ok=True)
        return {
            "content": [{
                "type": "text",
                "text": f"Successfully created directory {args['path']}"
            }]
        }
    except Exception as e:
        logger.error(f"Error creating directory {args.get('path')}: {e}")
        return {
            "content": [{
                "type": "text",
                "text": f"Error creating directory: {str(e)}"
            }],
            "is_error": True
        }


@tool(
    "remove_directory",
    "Remove a directory and all its contents at the specified path",
    {"path": str}
)
async def remove_directory(args):
    """Remove a directory and all its contents."""
    try:
        dir_path = Path(args["path"])
        if not dir_path.exists():
            return {
                "content": [{
                    "type": "text",
                    "text": f"Directory not found: {args['path']}"
                }],
                "is_error": True
            }

        if not dir_path.is_dir():
            return {
                "content": [{
                    "type": "text",
                    "text": f"Path is not a directory: {args['path']}"
                }],
                "is_error": True
            }

        shutil.rmtree(dir_path)
        return {
            "content": [{
                "type": "text",
                "text": f"Successfully removed directory {args['path']}"
            }]
        }
    except Exception as e:
        logger.error(f"Error removing directory {args.get('path')}: {e}")
        return {
            "content": [{
                "type": "text",
                "text": f"Error removing directory: {str(e)}"
            }],
            "is_error": True
        }
