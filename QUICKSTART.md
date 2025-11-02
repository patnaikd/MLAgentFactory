# 🚀 Quick Start Guide - Gradio UI

## Prerequisites

1. **Python 3.12+** installed
2. **uv package manager** installed
3. **Anthropic API key**

## Setup (5 minutes)

### Step 1: Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 2: Clone and Navigate

```bash
cd /path/to/MLAgentFactory
```

### Step 3: Create Environment File

Create a `.env` file in the project root:

```bash
cat > .env << 'EOF'
ANTHROPIC_API_KEY=your_api_key_here
LOG_LEVEL=INFO
EOF
```

Replace `your_api_key_here` with your actual Anthropic API key.

### Step 4: Install Dependencies

```bash
uv sync
```

This will install all required packages including Gradio, Claude Agent SDK, and more.

## Running the Application

### Option 1: Using the Launch Script (Recommended)

```bash
./run_gradio_app.sh
```

### Option 2: Direct Execution

```bash
uv run python src/mlagentfactory/ui/gradio_ui.py
```

## Access the UI

Once started, open your browser to:

- **Application:** http://localhost:7860
- **API Documentation:** http://localhost:7860/?view=api

## First Steps

1. **Start a Conversation**
   - Type a message in the chat input
   - Example: "Hello! Can you help me download the Iris dataset from UCI?"

2. **Watch the Agent Work**
   - See tool usage in real-time
   - Track task progress in the sidebar
   - Monitor logs in the "Logs" tab

3. **Explore Features**
   - Try file operations
   - Download datasets from Kaggle or UCI
   - Fetch web content
   - Create Python scripts

## Example Queries

```
"Download the Iris dataset from UCI and show me the first few rows"

"Search for Kaggle competitions about natural language processing"

"Create a Python script to analyze the titanic dataset"

"Fetch content from https://example.com and summarize it"

"Write a function to train a simple neural network with PyTorch"
```

## Optional: Kaggle Setup

To use Kaggle tools:

1. Go to https://www.kaggle.com/account
2. Click "Create New API Token"
3. Save `kaggle.json` to `~/.kaggle/kaggle.json`
4. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

## Troubleshooting

### Port Already in Use

If port 7860 is already in use, you can modify the port in `gradio_ui.py`:

```python
demo.launch(
    ...
    server_port=7861,  # Change to your preferred port
    ...
)
```

### Import Errors

Make sure you're in the project root directory and have run `uv sync`.

### API Key Issues

Ensure your `.env` file has the correct `ANTHROPIC_API_KEY` and is in the project root.

## Stopping the Application

Press `Ctrl+C` in the terminal where the app is running.

## Next Steps

- Check out [README.md](README.md) for detailed documentation
- See [CLAUDE.md](CLAUDE.md) for development guidelines
- Read [MIGRATION_NOTES.md](MIGRATION_NOTES.md) for technical details

## Support

For issues or questions:
- Check the logs in the `logs/` directory
- View session logs: `logs/session-{session_id}.log`
- Open an issue on GitHub

---

**Enjoy using MLAgentFactory! 🤖**
