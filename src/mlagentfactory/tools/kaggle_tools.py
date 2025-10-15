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
    dataset = args.get("dataset", "unknown")
    download_path_str = args.get("path", "unknown")

    logger.info(f"[KAGGLE_TOOL] kaggle_download_dataset called with dataset='{dataset}', path='{download_path_str}'")
    logger.debug(f"[KAGGLE_TOOL] Full args: {args!r}")

    try:
        download_path = Path(download_path_str)
        logger.debug(f"[KAGGLE_TOOL] Creating directory: {download_path}")

        # Create directory if it doesn't exist
        download_path.mkdir(parents=True, exist_ok=True)

        # Run kaggle datasets download command
        cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(download_path), "--unzip"]
        logger.debug(f"[KAGGLE_TOOL] Executing command: {' '.join(cmd)}")

        logger.info(f"[KAGGLE_TOOL] Starting download of dataset '{dataset}' to {download_path}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        logger.debug(f"[KAGGLE_TOOL] Command exit code: {result.returncode}")
        logger.debug(f"[KAGGLE_TOOL] stdout: {result.stdout}")
        if result.stderr:
            logger.debug(f"[KAGGLE_TOOL] stderr: {result.stderr}")

        if result.returncode == 0:
            logger.info(f"[KAGGLE_TOOL] Successfully downloaded dataset '{dataset}'")
            return {
                "content": [{
                    "type": "text",
                    "text": f"Successfully downloaded dataset '{dataset}' to {download_path}\n\nOutput:\n{result.stdout}"
                }]
            }
        else:
            logger.error(f"[KAGGLE_TOOL] Failed to download dataset '{dataset}': {result.stderr}")
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error downloading dataset '{dataset}':\n{result.stderr}"
                }],
                "is_error": True
            }

    except subprocess.TimeoutExpired:
        logger.error(f"[KAGGLE_TOOL] Timeout downloading dataset '{dataset}' after 5 minutes")
        return {
            "content": [{
                "type": "text",
                "text": "Dataset download timed out after 5 minutes"
            }],
            "is_error": True
        }
    except Exception as e:
        logger.error(f"[KAGGLE_TOOL] Exception downloading dataset '{dataset}': {e}", exc_info=True)
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
    search_term = args.get("search", "").strip()

    logger.info(f"[KAGGLE_TOOL] kaggle_list_competitions called{' with search=' + repr(search_term) if search_term else ''}")
    logger.debug(f"[KAGGLE_TOOL] Full args: {args!r}")

    try:
        cmd = ["kaggle", "competitions", "list"]

        if search_term:
            cmd.extend(["--search", search_term])

        logger.debug(f"[KAGGLE_TOOL] Executing command: {' '.join(cmd)}")
        logger.info(f"[KAGGLE_TOOL] Fetching competitions list{' (filtered by: ' + search_term + ')' if search_term else ''}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        logger.debug(f"[KAGGLE_TOOL] Command exit code: {result.returncode}")
        logger.debug(f"[KAGGLE_TOOL] stdout length: {len(result.stdout)} chars")
        if result.stderr:
            logger.debug(f"[KAGGLE_TOOL] stderr: {result.stderr}")

        if result.returncode == 0:
            num_lines = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
            logger.info(f"[KAGGLE_TOOL] Successfully listed competitions ({num_lines} lines)")
            return {
                "content": [{
                    "type": "text",
                    "text": f"Kaggle Competitions:\n\n{result.stdout}"
                }]
            }
        else:
            logger.error(f"[KAGGLE_TOOL] Failed to list competitions: {result.stderr}")
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error listing competitions:\n{result.stderr}"
                }],
                "is_error": True
            }

    except subprocess.TimeoutExpired:
        logger.error(f"[KAGGLE_TOOL] Timeout listing competitions after 30 seconds")
        return {
            "content": [{
                "type": "text",
                "text": "Listing competitions timed out after 30 seconds"
            }],
            "is_error": True
        }
    except Exception as e:
        logger.error(f"[KAGGLE_TOOL] Exception listing competitions: {e}", exc_info=True)
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
    competition = args.get("competition", "unknown")
    download_path_str = args.get("path", "unknown")

    logger.info(f"[KAGGLE_TOOL] kaggle_download_competition_data called with competition='{competition}', path='{download_path_str}'")
    logger.debug(f"[KAGGLE_TOOL] Full args: {args!r}")

    try:
        download_path = Path(download_path_str)
        logger.debug(f"[KAGGLE_TOOL] Creating directory: {download_path}")

        # Create directory if it doesn't exist
        download_path.mkdir(parents=True, exist_ok=True)

        # Run kaggle competitions download command
        cmd = ["kaggle", "competitions", "download", "-c", competition, "-p", str(download_path)]
        logger.debug(f"[KAGGLE_TOOL] Executing command: {' '.join(cmd)}")

        logger.info(f"[KAGGLE_TOOL] Starting download of competition data '{competition}' to {download_path}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        logger.debug(f"[KAGGLE_TOOL] Command exit code: {result.returncode}")
        logger.debug(f"[KAGGLE_TOOL] stdout: {result.stdout}")
        if result.stderr:
            logger.debug(f"[KAGGLE_TOOL] stderr: {result.stderr}")

        if result.returncode == 0:
            logger.info(f"[KAGGLE_TOOL] Successfully downloaded competition data for '{competition}'")
            return {
                "content": [{
                    "type": "text",
                    "text": f"Successfully downloaded competition data for '{competition}' to {download_path}\n\nOutput:\n{result.stdout}"
                }]
            }
        else:
            logger.error(f"[KAGGLE_TOOL] Failed to download competition data for '{competition}': {result.stderr}")
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error downloading competition data for '{competition}':\n{result.stderr}"
                }],
                "is_error": True
            }

    except subprocess.TimeoutExpired:
        logger.error(f"[KAGGLE_TOOL] Timeout downloading competition data '{competition}' after 5 minutes")
        return {
            "content": [{
                "type": "text",
                "text": "Competition data download timed out after 5 minutes"
            }],
            "is_error": True
        }
    except Exception as e:
        logger.error(f"[KAGGLE_TOOL] Exception downloading competition data '{competition}': {e}", exc_info=True)
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
    competition = args.get("competition", "unknown")
    file_path_str = args.get("file_path", "unknown")
    message = args.get("message", "")

    logger.info(f"[KAGGLE_TOOL] kaggle_submit_competition called with competition='{competition}', file='{file_path_str}', message='{message[:50]}...'")
    logger.debug(f"[KAGGLE_TOOL] Full args: {args!r}")

    try:
        file_path = Path(file_path_str)
        logger.debug(f"[KAGGLE_TOOL] Checking if file exists: {file_path}")

        if not file_path.exists():
            logger.error(f"[KAGGLE_TOOL] Submission file not found: {file_path}")
            return {
                "content": [{
                    "type": "text",
                    "text": f"Submission file not found: {file_path}"
                }],
                "is_error": True
            }

        logger.debug(f"[KAGGLE_TOOL] File exists, size: {file_path.stat().st_size} bytes")

        # Run kaggle competitions submit command
        cmd = ["kaggle", "competitions", "submit", "-c", competition, "-f", str(file_path), "-m", message]
        logger.debug(f"[KAGGLE_TOOL] Executing command: kaggle competitions submit -c {competition} -f {file_path} -m '<message>'")

        logger.info(f"[KAGGLE_TOOL] Submitting to competition '{competition}' with file {file_path}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        logger.debug(f"[KAGGLE_TOOL] Command exit code: {result.returncode}")
        logger.debug(f"[KAGGLE_TOOL] stdout: {result.stdout}")
        if result.stderr:
            logger.debug(f"[KAGGLE_TOOL] stderr: {result.stderr}")

        if result.returncode == 0:
            logger.info(f"[KAGGLE_TOOL] Successfully submitted to competition '{competition}'")
            return {
                "content": [{
                    "type": "text",
                    "text": f"Successfully submitted to '{competition}'\n\nOutput:\n{result.stdout}"
                }]
            }
        else:
            logger.error(f"[KAGGLE_TOOL] Failed to submit to competition '{competition}': {result.stderr}")
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error submitting to '{competition}':\n{result.stderr}"
                }],
                "is_error": True
            }

    except subprocess.TimeoutExpired:
        logger.error(f"[KAGGLE_TOOL] Timeout submitting to competition '{competition}' after 60 seconds")
        return {
            "content": [{
                "type": "text",
                "text": "Submission timed out after 60 seconds"
            }],
            "is_error": True
        }
    except Exception as e:
        logger.error(f"[KAGGLE_TOOL] Exception submitting to competition '{competition}': {e}", exc_info=True)
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
    competition = args.get("competition", "unknown")

    logger.info(f"[KAGGLE_TOOL] kaggle_list_submissions called with competition='{competition}'")
    logger.debug(f"[KAGGLE_TOOL] Full args: {args!r}")

    try:
        # Run kaggle competitions submissions command
        cmd = ["kaggle", "competitions", "submissions", "-c", competition]
        logger.debug(f"[KAGGLE_TOOL] Executing command: {' '.join(cmd)}")

        logger.info(f"[KAGGLE_TOOL] Fetching submissions for competition '{competition}'")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        logger.debug(f"[KAGGLE_TOOL] Command exit code: {result.returncode}")
        logger.debug(f"[KAGGLE_TOOL] stdout length: {len(result.stdout)} chars")
        if result.stderr:
            logger.debug(f"[KAGGLE_TOOL] stderr: {result.stderr}")

        if result.returncode == 0:
            num_lines = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
            logger.info(f"[KAGGLE_TOOL] Successfully listed submissions for '{competition}' ({num_lines} lines)")
            return {
                "content": [{
                    "type": "text",
                    "text": f"Submissions for '{competition}':\n\n{result.stdout}"
                }]
            }
        else:
            logger.error(f"[KAGGLE_TOOL] Failed to list submissions for '{competition}': {result.stderr}")
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error listing submissions for '{competition}':\n{result.stderr}"
                }],
                "is_error": True
            }

    except subprocess.TimeoutExpired:
        logger.error(f"[KAGGLE_TOOL] Timeout listing submissions for '{competition}' after 30 seconds")
        return {
            "content": [{
                "type": "text",
                "text": "Listing submissions timed out after 30 seconds"
            }],
            "is_error": True
        }
    except Exception as e:
        logger.error(f"[KAGGLE_TOOL] Exception listing submissions for '{competition}': {e}", exc_info=True)
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
    competition = args.get("competition", "unknown")

    logger.info(f"[KAGGLE_TOOL] kaggle_competition_leaderboard called with competition='{competition}'")
    logger.debug(f"[KAGGLE_TOOL] Full args: {args!r}")

    try:
        # Run kaggle competitions leaderboard command
        cmd = ["kaggle", "competitions", "leaderboard", "-c", competition, "--show"]
        logger.debug(f"[KAGGLE_TOOL] Executing command: {' '.join(cmd)}")

        logger.info(f"[KAGGLE_TOOL] Fetching leaderboard for competition '{competition}'")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        logger.debug(f"[KAGGLE_TOOL] Command exit code: {result.returncode}")
        logger.debug(f"[KAGGLE_TOOL] stdout length: {len(result.stdout)} chars")
        if result.stderr:
            logger.debug(f"[KAGGLE_TOOL] stderr: {result.stderr}")

        if result.returncode == 0:
            num_lines = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
            logger.info(f"[KAGGLE_TOOL] Successfully fetched leaderboard for '{competition}' ({num_lines} lines)")
            return {
                "content": [{
                    "type": "text",
                    "text": f"Leaderboard for '{competition}':\n\n{result.stdout}"
                }]
            }
        else:
            logger.error(f"[KAGGLE_TOOL] Failed to fetch leaderboard for '{competition}': {result.stderr}")
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error fetching leaderboard for '{competition}':\n{result.stderr}"
                }],
                "is_error": True
            }

    except subprocess.TimeoutExpired:
        logger.error(f"[KAGGLE_TOOL] Timeout fetching leaderboard for '{competition}' after 30 seconds")
        return {
            "content": [{
                "type": "text",
                "text": "Fetching leaderboard timed out after 30 seconds"
            }],
            "is_error": True
        }
    except Exception as e:
        logger.error(f"[KAGGLE_TOOL] Exception fetching leaderboard for '{competition}': {e}", exc_info=True)
        return {
            "content": [{
                "type": "text",
                "text": f"Error fetching leaderboard: {str(e)}"
            }],
            "is_error": True
        }
