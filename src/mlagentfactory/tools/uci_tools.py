"""Tools for interacting with UCI Machine Learning Repository datasets."""

import json
import os
import logging
from io import StringIO
from typing import Any

from claude_agent_sdk import tool
from ucimlrepo import fetch_ucirepo, list_available_datasets

logger = logging.getLogger(__name__)


@tool(
    "uci_list_datasets",
    "List available datasets from the UCI Machine Learning Repository. You can optionally filter by category or search by name.",
    {
        "filter_category": {"type": "string", "description": "Optional category filter (e.g., 'aim-ahead')"},
        "search": {"type": "string", "description": "Optional search term to filter datasets by name"}
    }
)
async def uci_list_datasets(args):
    """List available datasets from the UCI Machine Learning Repository."""
    try:
        filter_category = args.get("filter_category")
        search = args.get("search")

        # Capture the output since list_available_datasets() prints to stdout
        import sys
        from io import StringIO

        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        if filter_category:
            list_available_datasets(filter=filter_category)
        elif search:
            list_available_datasets(search=search)
        else:
            list_available_datasets()

        sys.stdout = old_stdout
        output = captured_output.getvalue()

        if not output.strip():
            output = "No datasets found matching the criteria."

        return {
            "content": [{
                "type": "text",
                "text": f"UCI ML Repository Datasets:\n\n{output}"
            }]
        }

    except Exception as e:
        logger.error(f"Error in uci_list_datasets tool: {e}")
        return {
            "content": [{
                "type": "text",
                "text": f"Error listing datasets: {str(e)}"
            }],
            "is_error": True
        }


@tool(
    "uci_fetch_dataset",
    "Fetch a dataset from the UCI Machine Learning Repository by ID or name. Returns dataset information and optionally saves to CSV files. Popular dataset IDs: Iris (53), Wine (109), Breast Cancer Wisconsin (17), Adult (2).",
    {
        "dataset_id": {"type": "integer", "description": "Dataset ID number (e.g., 53 for Iris). Provide either dataset_id or dataset_name, not both."},
        "dataset_name": {"type": "string", "description": "Dataset name (e.g., 'Iris'). Provide either dataset_id or dataset_name, not both."},
        "save_to_csv": {"type": "boolean", "description": "Whether to save the dataset to CSV files (default: false)"},
        "output_dir": {"type": "string", "description": "Directory to save CSV files. Defaults to ./uci_datasets/{dataset_name}"}
    }
)
async def uci_fetch_dataset(args):
    """Fetch a dataset from the UCI Machine Learning Repository."""
    try:
        dataset_id = args.get("dataset_id")
        dataset_name = args.get("dataset_name")
        save_to_csv = args.get("save_to_csv", False)
        output_dir = args.get("output_dir")

        # Treat empty strings as None and convert dataset_id to int if present
        if dataset_id == "" or dataset_id == "null":
            dataset_id = None
        elif dataset_id is not None:
            dataset_id = int(dataset_id)

        if dataset_name == "" or dataset_name == "null":
            dataset_name = None
        if output_dir == "":
            output_dir = None

        if dataset_id is None and dataset_name is None:
            return {
                "content": [{
                    "type": "text",
                    "text": "Error: Must provide either dataset_id or dataset_name"
                }],
                "is_error": True
            }

        if dataset_id is not None and dataset_name is not None:
            return {
                "content": [{
                    "type": "text",
                    "text": "Error: Provide either dataset_id or dataset_name, not both"
                }],
                "is_error": True
            }

        # Fetch the dataset
        if dataset_id is not None:
            dataset = fetch_ucirepo(id=dataset_id)
        else:
            dataset = fetch_ucirepo(name=dataset_name)

        # Extract basic information
        result = {
            "success": True,
            "metadata": {
                "uci_id": dataset.metadata.uci_id,
                "name": dataset.metadata.name,
                "abstract": dataset.metadata.abstract,
                "area": dataset.metadata.area,
                "task": dataset.metadata.task,
                "characteristics": dataset.metadata.characteristics,
                "num_instances": dataset.metadata.num_instances,
                "num_features": dataset.metadata.num_features,
                "feature_types": dataset.metadata.feature_types,
                "has_missing_values": dataset.metadata.has_missing_values,
                "year_of_dataset_creation": dataset.metadata.year_of_dataset_creation,
                "repository_url": dataset.metadata.repository_url,
            },
            "data_info": {
                "num_features": len(dataset.data.features.columns) if dataset.data.features is not None else 0,
                "num_targets": len(dataset.data.targets.columns) if dataset.data.targets is not None else 0,
                "num_instances": len(dataset.data.features) if dataset.data.features is not None else 0,
                "feature_columns": list(dataset.data.features.columns) if dataset.data.features is not None else [],
                "target_columns": list(dataset.data.targets.columns) if dataset.data.targets is not None else [],
            },
            "variables": dataset.variables.to_dict('records') if dataset.variables is not None else []
        }

        # Save to CSV if requested
        if save_to_csv:
            if output_dir is None:
                output_dir = os.path.join("uci_datasets", dataset.metadata.name.replace(" ", "_"))

            os.makedirs(output_dir, exist_ok=True)

            saved_files = []

            # Save features
            if dataset.data.features is not None:
                features_path = os.path.join(output_dir, "features.csv")
                dataset.data.features.to_csv(features_path, index=False)
                saved_files.append(features_path)

            # Save targets
            if dataset.data.targets is not None:
                targets_path = os.path.join(output_dir, "targets.csv")
                dataset.data.targets.to_csv(targets_path, index=False)
                saved_files.append(targets_path)

            # Save complete dataset
            if dataset.data.original is not None:
                original_path = os.path.join(output_dir, "complete_dataset.csv")
                dataset.data.original.to_csv(original_path, index=False)
                saved_files.append(original_path)

            # Save variables info
            if dataset.variables is not None:
                variables_path = os.path.join(output_dir, "variables_info.csv")
                dataset.variables.to_csv(variables_path, index=False)
                saved_files.append(variables_path)

            # Save metadata as JSON
            metadata_path = os.path.join(output_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(result["metadata"], f, indent=2, default=str)
            saved_files.append(metadata_path)

            result["saved_files"] = saved_files
            result["output_directory"] = output_dir

        result_text = json.dumps(result, indent=2, default=str)
        return {
            "content": [{
                "type": "text",
                "text": f"Successfully fetched dataset!\n\n{result_text}"
            }]
        }

    except Exception as e:
        logger.error(f"Error in uci_fetch_dataset tool: {e}")
        return {
            "content": [{
                "type": "text",
                "text": f"Error fetching dataset: {str(e)}"
            }],
            "is_error": True
        }


@tool(
    "uci_get_dataset_info",
    "Get detailed metadata and information about a UCI ML Repository dataset without downloading the full data. Useful for exploring dataset characteristics before fetching.",
    {
        "dataset_id": {"type": "integer", "description": "Dataset ID number (e.g., 53 for Iris). Provide either dataset_id or dataset_name, not both."},
        "dataset_name": {"type": "string", "description": "Dataset name (e.g., 'Iris'). Provide either dataset_id or dataset_name, not both."}
    }
)
async def uci_get_dataset_info(args):
    """Get detailed information about a UCI ML Repository dataset without downloading the full data."""
    try:
        dataset_id = args.get("dataset_id")
        dataset_name = args.get("dataset_name")

        # Treat empty strings as None and convert dataset_id to int if present
        if dataset_id == "" or dataset_id == "null":
            dataset_id = None
        elif dataset_id is not None:
            dataset_id = int(dataset_id)

        if dataset_name == "" or dataset_name == "null":
            dataset_name = None

        if dataset_id is None and dataset_name is None:
            return {
                "content": [{
                    "type": "text",
                    "text": "Error: Must provide either dataset_id or dataset_name"
                }],
                "is_error": True
            }

        if dataset_id is not None and dataset_name is not None:
            return {
                "content": [{
                    "type": "text",
                    "text": "Error: Provide either dataset_id or dataset_name, not both"
                }],
                "is_error": True
            }

        # Fetch the dataset
        if dataset_id is not None:
            dataset = fetch_ucirepo(id=dataset_id)
        else:
            dataset = fetch_ucirepo(name=dataset_name)

        # Extract comprehensive metadata
        metadata = dataset.metadata

        result = {
            "uci_id": metadata.uci_id,
            "name": metadata.name,
            "abstract": metadata.abstract,
            "area": metadata.area,
            "task": metadata.task,
            "characteristics": metadata.characteristics,
            "num_instances": metadata.num_instances,
            "num_features": metadata.num_features,
            "feature_types": metadata.feature_types,
            "target_col": metadata.target_col,
            "index_col": metadata.index_col,
            "has_missing_values": metadata.has_missing_values,
            "missing_values_symbol": metadata.missing_values_symbol,
            "year_of_dataset_creation": metadata.year_of_dataset_creation,
            "dataset_doi": metadata.dataset_doi,
            "creators": metadata.creators,
            "intro_paper": metadata.intro_paper,
            "repository_url": metadata.repository_url,
            "data_url": metadata.data_url,
        }

        # Add additional info if available
        if hasattr(metadata, 'additional_info') and metadata.additional_info:
            additional_info = {}
            if hasattr(metadata.additional_info, 'summary'):
                additional_info['summary'] = metadata.additional_info.summary
            if hasattr(metadata.additional_info, 'purpose'):
                additional_info['purpose'] = metadata.additional_info.purpose
            if hasattr(metadata.additional_info, 'funding'):
                additional_info['funding'] = metadata.additional_info.funding
            result['additional_info'] = additional_info

        # Add external URL if available
        if hasattr(metadata, 'external_url'):
            result['external_url'] = metadata.external_url

        # Add variable information
        if dataset.variables is not None:
            result['variables'] = dataset.variables.to_dict('records')

        result_text = json.dumps(result, indent=2, default=str)
        return {
            "content": [{
                "type": "text",
                "text": f"Dataset Information:\n\n{result_text}"
            }]
        }

    except Exception as e:
        logger.error(f"Error in uci_get_dataset_info tool: {e}")
        return {
            "content": [{
                "type": "text",
                "text": f"Error getting dataset info: {str(e)}"
            }],
            "is_error": True
        }
