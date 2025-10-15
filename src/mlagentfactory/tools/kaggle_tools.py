"""
Kaggle CLI tools for MLAgentFactory agents.

This module provides tools for interacting with Kaggle via the CLI,
including dataset downloads, competition interactions, and submissions.
"""

import subprocess
import logging
from pathlib import Path

from claude_agent_sdk import tool

logger = logging.getLogger(__name__)


@tool(
    "kaggle_download_dataset",
    "Download a Kaggle dataset by name (e.g., 'username/dataset-name')",
    {"dataset": str, "path": str}
)
async def kaggle_download_dataset(args):
    """Download a Kaggle dataset to the specified path."""
    try:
        dataset = args["dataset"]
        download_path = Path(args["path"])

        # Create directory if it doesn't exist
        download_path.mkdir(parents=True, exist_ok=True)

        # Run kaggle datasets download command
        cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(download_path), "--unzip"]

        logger.info(f"Downloading Kaggle dataset: {dataset} to {download_path}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Successfully downloaded dataset '{dataset}' to {download_path}\n\nOutput:\n{result.stdout}"
                }]
            }
        else:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error downloading dataset '{dataset}':\n{result.stderr}"
                }],
                "is_error": True
            }

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout downloading dataset {args.get('dataset')}")
        return {
            "content": [{
                "type": "text",
                "text": "Dataset download timed out after 5 minutes"
            }],
            "is_error": True
        }
    except Exception as e:
        logger.error(f"Error downloading dataset {args.get('dataset')}: {e}")
        return {
            "content": [{
                "type": "text",
                "text": f"Error downloading dataset: {str(e)}"
            }],
            "is_error": True
        }


@tool(
    "kaggle_list_competitions",
    "List Kaggle competitions with optional filtering",
    {"search": str}
)
async def kaggle_list_competitions(args):
    """List Kaggle competitions, optionally filtered by search term."""
    try:
        cmd = ["kaggle", "competitions", "list"]

        search_term = args.get("search", "").strip()
        if search_term:
            cmd.extend(["--search", search_term])

        logger.info(f"Listing Kaggle competitions{' with search: ' + search_term if search_term else ''}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Kaggle Competitions:\n\n{result.stdout}"
                }]
            }
        else:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error listing competitions:\n{result.stderr}"
                }],
                "is_error": True
            }

    except Exception as e:
        logger.error(f"Error listing competitions: {e}")
        return {
            "content": [{
                "type": "text",
                "text": f"Error listing competitions: {str(e)}"
            }],
            "is_error": True
        }


@tool(
    "kaggle_download_competition_data",
    "Download all data files for a Kaggle competition",
    {"competition": str, "path": str}
)
async def kaggle_download_competition_data(args):
    """Download all data files for a Kaggle competition."""
    try:
        competition = args["competition"]
        download_path = Path(args["path"])

        # Create directory if it doesn't exist
        download_path.mkdir(parents=True, exist_ok=True)

        # Run kaggle competitions download command
        cmd = ["kaggle", "competitions", "download", "-c", competition, "-p", str(download_path)]

        logger.info(f"Downloading competition data: {competition} to {download_path}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Successfully downloaded competition data for '{competition}' to {download_path}\n\nOutput:\n{result.stdout}"
                }]
            }
        else:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error downloading competition data for '{competition}':\n{result.stderr}"
                }],
                "is_error": True
            }

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout downloading competition data {args.get('competition')}")
        return {
            "content": [{
                "type": "text",
                "text": "Competition data download timed out after 5 minutes"
            }],
            "is_error": True
        }
    except Exception as e:
        logger.error(f"Error downloading competition data {args.get('competition')}: {e}")
        return {
            "content": [{
                "type": "text",
                "text": f"Error downloading competition data: {str(e)}"
            }],
            "is_error": True
        }


@tool(
    "kaggle_submit_competition",
    "Submit a file to a Kaggle competition",
    {"competition": str, "file_path": str, "message": str}
)
async def kaggle_submit_competition(args):
    """Submit a file to a Kaggle competition with a message."""
    try:
        competition = args["competition"]
        file_path = Path(args["file_path"])
        message = args["message"]

        if not file_path.exists():
            return {
                "content": [{
                    "type": "text",
                    "text": f"Submission file not found: {file_path}"
                }],
                "is_error": True
            }

        # Run kaggle competitions submit command
        cmd = ["kaggle", "competitions", "submit", "-c", competition, "-f", str(file_path), "-m", message]

        logger.info(f"Submitting to competition: {competition} with file {file_path}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Successfully submitted to '{competition}'\n\nOutput:\n{result.stdout}"
                }]
            }
        else:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error submitting to '{competition}':\n{result.stderr}"
                }],
                "is_error": True
            }

    except Exception as e:
        logger.error(f"Error submitting to competition {args.get('competition')}: {e}")
        return {
            "content": [{
                "type": "text",
                "text": f"Error submitting to competition: {str(e)}"
            }],
            "is_error": True
        }


@tool(
    "kaggle_list_submissions",
    "List your submissions for a Kaggle competition",
    {"competition": str}
)
async def kaggle_list_submissions(args):
    """List submissions for a Kaggle competition."""
    try:
        competition = args["competition"]

        # Run kaggle competitions submissions command
        cmd = ["kaggle", "competitions", "submissions", "-c", competition]

        logger.info(f"Listing submissions for competition: {competition}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Submissions for '{competition}':\n\n{result.stdout}"
                }]
            }
        else:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error listing submissions for '{competition}':\n{result.stderr}"
                }],
                "is_error": True
            }

    except Exception as e:
        logger.error(f"Error listing submissions for competition {args.get('competition')}: {e}")
        return {
            "content": [{
                "type": "text",
                "text": f"Error listing submissions: {str(e)}"
            }],
            "is_error": True
        }


@tool(
    "kaggle_competition_leaderboard",
    "View the leaderboard for a Kaggle competition",
    {"competition": str}
)
async def kaggle_competition_leaderboard(args):
    """View the leaderboard for a Kaggle competition."""
    try:
        competition = args["competition"]

        # Run kaggle competitions leaderboard command
        cmd = ["kaggle", "competitions", "leaderboard", "-c", competition, "--show"]

        logger.info(f"Fetching leaderboard for competition: {competition}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Leaderboard for '{competition}':\n\n{result.stdout}"
                }]
            }
        else:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error fetching leaderboard for '{competition}':\n{result.stderr}"
                }],
                "is_error": True
            }

    except Exception as e:
        logger.error(f"Error fetching leaderboard for competition {args.get('competition')}: {e}")
        return {
            "content": [{
                "type": "text",
                "text": f"Error fetching leaderboard: {str(e)}"
            }],
            "is_error": True
        }
