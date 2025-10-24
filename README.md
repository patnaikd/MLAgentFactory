# 🤖 MLAgentFactory

An AI-powered machine learning assistant built with Claude Agent SDK, featuring both Gradio and Streamlit web interfaces for interactive agent conversations.

## ✨ Features

- **🎯 Conversational AI Agent** - Powered by Claude with autonomous task execution
- **🌐 Modern Web UI** - Gradio interface with real-time streaming (recommended)
- **📊 Task Tracking** - Visual todo list with progress tracking
- **📝 Comprehensive Logging** - Real-time log viewer with filtering
- **🔧 Rich Tool Support** - File I/O, web scraping, Kaggle, UCI ML Repository
- **💰 Cost Tracking** - Monitor API usage costs per session
- **🔄 Session Management** - Persistent conversation history
- **🎨 Special Formatting** - Enhanced display for tool usage and results

## 🚀 Quick Start

### Prerequisites

- Python >= 3.12
- [uv](https://github.com/astral-sh/uv) package manager
- Anthropic API key

### Installation

1. **Install uv (if not already installed):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Set up environment variables:**
```bash
# Copy the example env file (if it exists)
cp .env.example .env

# Edit .env and add your ANTHROPIC_API_KEY
nano .env
```

Add to `.env`:
```bash
ANTHROPIC_API_KEY=your_api_key_here
```

3. **Install dependencies:**
```bash
uv sync
```

### Running the Application

#### Gradio UI (Recommended) ⭐

```bash
# Using the launch script (recommended)
./run_gradio_app.sh

# Or run directly
uv run python src/mlagentfactory/ui/gradio_ui.py
```

Then open your browser to:
- **App:** http://localhost:7860
- **API Docs:** http://localhost:7860/?view=api

#### Streamlit UI (Legacy)

```bash
./run_app.sh
```

Then open your browser to: http://localhost:8501

## 🖥️ Web Interfaces

### Gradio UI

The modern, recommended interface with:

- **Native Async Streaming** - Smooth real-time responses
- **Auto-generated REST API** - Programmatic access to the agent
- **Better Concurrency** - Built-in queuing for multiple users
- **Flexible Layout** - Responsive design with tabs and columns
- **Simpler Code** - Clean async/await without manual threading

**Key Features:**
- Chat interface with markdown rendering
- Real-time task progress with visual indicators
- Session info sidebar with cost tracking
- Filterable log viewer with syntax highlighting

### Streamlit UI

The legacy interface (maintained for compatibility) with:

- Fragment-based auto-refresh for todos and logs
- Background thread processing
- Custom HTML components

## 🛠️ Available Tools

The ChatAgent has access to the following tools:

### File Operations
- `read_file` - Read file contents
- `write_file` - Write to files (creates directories)
- `edit_file` - Edit files by text replacement
- `delete_file` - Delete files
- `list_directory` - List directory contents
- `create_directory` - Create directories
- `remove_directory` - Remove directories recursively

### Web Tools
- `fetch_webpage` - Fetch web content with JavaScript support (Playwright)

### Kaggle Integration
- `kaggle_download_dataset` - Download datasets from Kaggle
- `kaggle_list_competitions` - Search and list competitions
- `kaggle_download_competition_data` - Download competition files
- `kaggle_submit_competition` - Submit competition entries
- `kaggle_list_submissions` - View your submissions
- `kaggle_competition_leaderboard` - Check leaderboards

**Setup:** See [Kaggle Setup](#kaggle-setup) section below.

### UCI ML Repository
- `uci_list_datasets` - List available datasets
- `uci_fetch_dataset` - Download datasets by ID or name
- `uci_get_dataset_info` - Get dataset metadata

No credentials required! Ready to use out of the box.

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with:

```bash
# Required
ANTHROPIC_API_KEY=your_api_key_here

# Optional
LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Kaggle Setup

To use Kaggle tools:

1. Go to https://www.kaggle.com/account
2. Click "Create New API Token"
3. Save `kaggle.json` to `~/.kaggle/kaggle.json`
4. Set permissions:
```bash
chmod 600 ~/.kaggle/kaggle.json
```

## 📚 Usage Examples

### Via Web UI

1. Start the Gradio app: `./run_gradio_app.sh`
2. Navigate to http://localhost:7860
3. Type your query in the chat interface
4. Watch the agent work in real-time with tool usage displayed

**Example Queries:**
- "Download the Iris dataset from UCI and create a visualization"
- "Search for recent Kaggle competitions about computer vision"
- "Create a Python script that analyzes the titanic dataset"
- "Fetch the latest machine learning news from a website"

### Via Python API

```python
import asyncio
from mlagentfactory.agents.chat_agent import ChatAgent

async def main():
    agent = ChatAgent()
    await agent.initialize()

    response = await agent.chat("What are the top 3 Kaggle competitions right now?")
    print(response)

    await agent.cleanup()

asyncio.run(main())
```

### Via Gradio REST API

Once the Gradio app is running, you can use the auto-generated REST API:

```python
import requests

response = requests.post(
    "http://localhost:7860/api/predict",
    json={
        "data": [
            "Tell me about the Iris dataset",  # message
            [],  # history
            {},  # agent_state
            {}   # session_info
        ]
    }
)

print(response.json())
```

## 🏗️ Project Structure

```
MLAgentFactory/
├── src/mlagentfactory/
│   ├── agents/
│   │   └── chat_agent.py           # Main conversational agent
│   ├── ui/
│   │   ├── gradio_ui.py           # Gradio web interface (recommended)
│   │   └── streamlit_ui.py        # Streamlit web interface (legacy)
│   ├── tools/
│   │   ├── file_io_tools.py       # File operation tools
│   │   ├── web_fetch_tools.py     # Web scraping tools
│   │   ├── kaggle_tools.py        # Kaggle API integration
│   │   └── uci_tools.py           # UCI ML Repository tools
│   └── utils/
│       └── logging_config.py       # Logging and observability
├── logs/                           # Session logs (auto-created)
├── workspace/                      # Agent workspace (auto-created)
├── run_gradio_app.sh              # Gradio launcher
├── run_app.sh                     # Streamlit launcher
├── pyproject.toml                 # Dependencies
└── README.md                      # This file
```

## 🔧 Development

### Running Examples

```bash
# Simple hello world
uv run main.py

# File creation agent
uv run file_creation_agent.py

# Kaggle agent example
uv run kaggle_agent_example.py

# UCI ML Repository example
uv run uci_example.py
```

### Adding New Tools

See [CLAUDE.md](CLAUDE.md) for detailed development guidelines.

## 📊 Logging

Logs are automatically created in the `logs/` directory:

- **Session logs:** `logs/session-{session_id}.log`
- **Format:** Timestamp, logger name, level, file:line, message
- **Levels:** DEBUG, INFO, WARNING, ERROR, CRITICAL

View logs in real-time via the web UI's "Logs" tab.

## 📄 Documentation

For detailed setup instructions, tool documentation, and development guidelines, see [CLAUDE.md](CLAUDE.md).

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a Pull Request

## 📄 License

This project is in early development. License TBD.

## 🙏 Acknowledgments

- **Claude Agent SDK** - Anthropic's SDK for building autonomous agents
- **Gradio** - Easy-to-use web UI framework
- **Streamlit** - Python web framework
- **Kaggle API** - Dataset and competition access
- **UCI ML Repository** - Classic machine learning datasets
- **Playwright** - Robust web scraping

---

**Built with ❤️ using Claude Agent SDK**
