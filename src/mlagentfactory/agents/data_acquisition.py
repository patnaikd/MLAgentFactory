"""Data Acquisition Agent for handling dataset downloads and validation"""

import logging
import requests
from pathlib import Path
from typing import Dict, Optional
import hashlib
from urllib.parse import urlparse
import os

logger = logging.getLogger(__name__)


class DataAcquisitionAgent:
    """Agent responsible for acquiring datasets from various sources"""

    def __init__(self, download_dir: str = "./data"):
        """Initialize the Data Acquisition agent

        Args:
            download_dir: Directory to store downloaded datasets
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.headers = {
            'User-Agent': 'MLAgentFactory/0.1.0'
        }

    def get_file_hash(self, filepath: Path, algorithm: str = "sha256") -> str:
        """Calculate hash of a file

        Args:
            filepath: Path to file
            algorithm: Hash algorithm to use

        Returns:
            Hex digest of hash
        """
        hash_obj = hashlib.new(algorithm)
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()

    def validate_url(self, url: str) -> Dict:
        """Validate that URL is accessible

        Args:
            url: URL to validate

        Returns:
            Validation result dictionary
        """
        try:
            parsed = urlparse(url)
            if not all([parsed.scheme, parsed.netloc]):
                return {
                    "valid": False,
                    "error": "Invalid URL format"
                }

            # HEAD request to check accessibility
            response = requests.head(url, headers=self.headers, timeout=10, allow_redirects=True)

            return {
                "valid": True,
                "content_type": response.headers.get('Content-Type', 'unknown'),
                "content_length": response.headers.get('Content-Length', 'unknown'),
                "final_url": response.url
            }

        except requests.RequestException as e:
            return {
                "valid": False,
                "error": str(e)
            }

    async def download_dataset(self, url: str, filename: Optional[str] = None) -> Dict:
        """Download dataset from URL

        Args:
            url: URL to download from
            filename: Optional filename to save as

        Returns:
            Download result dictionary
        """
        try:
            logger.info(f"Starting download from {url}")

            # Validate URL first
            validation = self.validate_url(url)
            if not validation.get("valid"):
                return {
                    "success": False,
                    "error": validation.get("error", "URL validation failed")
                }

            # Determine filename
            if not filename:
                parsed = urlparse(url)
                filename = Path(parsed.path).name or "dataset"

            filepath = self.download_dir / filename

            # Download with progress tracking
            response = requests.get(url, headers=self.headers, stream=True, timeout=300)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

            # Calculate file hash
            file_hash = self.get_file_hash(filepath)

            logger.info(f"Download completed: {filepath}")

            return {
                "success": True,
                "filepath": str(filepath),
                "filename": filename,
                "size_bytes": filepath.stat().st_size,
                "hash": file_hash,
                "url": url
            }

        except requests.RequestException as e:
            logger.error(f"Download failed: {e}")
            return {
                "success": False,
                "error": f"Download failed: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Unexpected error during download: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }

    def get_dataset_info(self, filepath: str) -> Dict:
        """Get basic information about a downloaded dataset

        Args:
            filepath: Path to dataset file

        Returns:
            Dataset information dictionary
        """
        try:
            path = Path(filepath)
            if not path.exists():
                return {
                    "error": "File not found"
                }

            info = {
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                "extension": path.suffix,
                "hash": self.get_file_hash(path)
            }

            # Try to detect format
            if path.suffix.lower() in ['.csv', '.tsv']:
                info["format"] = "CSV"
                info["delimiter"] = ',' if path.suffix.lower() == '.csv' else '\t'
            elif path.suffix.lower() in ['.json', '.jsonl']:
                info["format"] = "JSON"
            elif path.suffix.lower() in ['.parquet', '.pq']:
                info["format"] = "Parquet"
            elif path.suffix.lower() in ['.xlsx', '.xls']:
                info["format"] = "Excel"
            else:
                info["format"] = "Unknown"

            return info

        except Exception as e:
            logger.error(f"Error getting dataset info: {e}")
            return {
                "error": str(e)
            }

    def check_kaggle_credentials(self) -> bool:
        """Check if Kaggle credentials are available

        Returns:
            True if credentials found
        """
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        return kaggle_json.exists() and os.getenv("KAGGLE_USERNAME") is not None

    async def download_from_kaggle(self, dataset_id: str) -> Dict:
        """Download dataset from Kaggle

        Args:
            dataset_id: Kaggle dataset identifier (e.g., "username/dataset-name")

        Returns:
            Download result dictionary
        """
        try:
            if not self.check_kaggle_credentials():
                return {
                    "success": False,
                    "error": "Kaggle credentials not found. Please set up kaggle.json"
                }

            # Import kaggle API
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi
            except ImportError:
                return {
                    "success": False,
                    "error": "Kaggle API not installed. Run: pip install kaggle"
                }

            api = KaggleApi()
            api.authenticate()

            # Download dataset
            download_path = str(self.download_dir)
            api.dataset_download_files(dataset_id, path=download_path, unzip=True)

            logger.info(f"Kaggle dataset {dataset_id} downloaded to {download_path}")

            return {
                "success": True,
                "dataset_id": dataset_id,
                "download_path": download_path,
                "source": "kaggle"
            }

        except Exception as e:
            logger.error(f"Kaggle download failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
