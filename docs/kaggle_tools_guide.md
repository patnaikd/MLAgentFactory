# Kaggle Tools Guide

This guide explains how to use the Kaggle CLI tools integrated into MLAgentFactory.

## Overview

The Kaggle tools allow your Claude agents to interact with Kaggle directly, including:
- Downloading datasets
- Listing and exploring competitions
- Downloading competition data
- Submitting solutions
- Viewing leaderboards and submission history

## Setup

### 1. Install Kaggle CLI

The Kaggle package is already included in the project dependencies:

```bash
uv sync
```

### 2. Configure Kaggle API Credentials

To use the Kaggle API, you need to set up authentication:

1. Go to your Kaggle account settings: https://www.kaggle.com/account
2. Scroll down to the "API" section
3. Click "Create New API Token"
4. This will download a `kaggle.json` file
5. Move the file to `~/.kaggle/`:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

Your `kaggle.json` should look like this:
```json
{
  "username": "your-kaggle-username",
  "key": "your-api-key"
}
```

## Available Tools

### 1. Download Datasets

Download public Kaggle datasets to your local machine:

```python
# Example usage with agent
response = await agent.chat(
    "Download the 'username/dataset-name' dataset to './data/dataset' directory"
)
```

**Tool:** `kaggle_download_dataset`
**Parameters:**
- `dataset` (str): Dataset identifier (e.g., "zillow/zecon")
- `path` (str): Local directory path to download to

### 2. List Competitions

Search and list Kaggle competitions:

```python
# Example usage
response = await agent.chat("List Kaggle competitions related to 'NLP'")
```

**Tool:** `kaggle_list_competitions`
**Parameters:**
- `search` (str): Optional search term to filter competitions

### 3. Download Competition Data

Download all data files for a specific competition:

```python
# Example usage
response = await agent.chat(
    "Download data for the 'titanic' competition to './data/titanic'"
)
```

**Tool:** `kaggle_download_competition_data`
**Parameters:**
- `competition` (str): Competition name
- `path` (str): Local directory path to download to

### 4. Submit to Competition

Submit your solution to a Kaggle competition:

```python
# Example usage
response = await agent.chat(
    "Submit my predictions file './submission.csv' to the titanic competition with message 'Initial submission'"
)
```

**Tool:** `kaggle_submit_competition`
**Parameters:**
- `competition` (str): Competition name
- `file_path` (str): Path to your submission file
- `message` (str): Submission description/message

### 5. List Submissions

View your submission history for a competition:

```python
# Example usage
response = await agent.chat("Show my submissions for the titanic competition")
```

**Tool:** `kaggle_list_submissions`
**Parameters:**
- `competition` (str): Competition name

### 6. View Leaderboard

Check the competition leaderboard:

```python
# Example usage
response = await agent.chat("Show the leaderboard for the titanic competition")
```

**Tool:** `kaggle_competition_leaderboard`
**Parameters:**
- `competition` (str): Competition name

## Example Workflow

Here's a complete example of using Kaggle tools in an agent workflow:

```python
import asyncio
from src.mlagentfactory.agents.chat_agent import ChatAgent

async def kaggle_workflow():
    agent = ChatAgent()
    await agent.initialize()

    try:
        # 1. Search for competitions
        response = await agent.chat("List competitions about computer vision")
        print(response)

        # 2. Download competition data
        response = await agent.chat(
            "Download the digit-recognizer competition data to ./data/mnist"
        )
        print(response)

        # 3. After training your model and creating predictions...
        response = await agent.chat(
            "Submit ./predictions.csv to digit-recognizer competition with message 'CNN baseline model'"
        )
        print(response)

        # 4. Check your position
        response = await agent.chat("Show my submissions for digit-recognizer")
        print(response)

        response = await agent.chat("Show the leaderboard for digit-recognizer")
        print(response)

    finally:
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(kaggle_workflow())
```

## Error Handling

The tools include comprehensive error handling:

- **Authentication errors**: Ensure your `kaggle.json` is properly configured
- **Competition access**: Some competitions require acceptance of rules before downloading
- **File not found**: Submission files must exist before submitting
- **Timeouts**: Large downloads have a 5-minute timeout
- **Rate limits**: Kaggle API has rate limits; space out requests if needed

## Tips and Best Practices

1. **Competition Rules**: Always accept competition rules on the Kaggle website before downloading data
2. **Data Storage**: Download datasets to organized directories (e.g., `./data/competition-name/`)
3. **Submission Format**: Ensure your submission file matches the required format for each competition
4. **API Limits**: Be mindful of Kaggle's API rate limits; avoid excessive requests
5. **Large Files**: Competition datasets can be large; ensure you have sufficient disk space

## Troubleshooting

### "401 Unauthorized" Error
- Check that `~/.kaggle/kaggle.json` exists and has correct permissions (`chmod 600`)
- Verify your API credentials are correct

### "403 Forbidden" Error
- You may need to accept the competition rules on Kaggle's website
- Some competitions are private or have entry restrictions

### "Command not found: kaggle"
- Run `uv sync` to install dependencies
- Or manually install: `uv add kaggle`

### Downloads Fail or Time Out
- Check your internet connection
- Large datasets may take time; increase timeout if needed
- Ensure you have sufficient disk space

## Additional Resources

- [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api)
- [Kaggle Competitions](https://www.kaggle.com/competitions)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
