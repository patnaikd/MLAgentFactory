#!/bin/bash

# MLAgentFactory - Run Gradio App Script
# This script launches the Gradio application with proper configuration

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}🤖 MLAgentFactory - Starting Gradio Application${NC}"
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  No .env file found. Creating from template...${NC}"
    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "${YELLOW}Please edit .env and add your ANTHROPIC_API_KEY${NC}"
        exit 1
    else
        echo -e "${RED}❌ .env.example not found${NC}"
        exit 1
    fi
fi

# Check if ANTHROPIC_API_KEY is set in .env
if ! grep -q "ANTHROPIC_API_KEY=.*[^[:space:]]" .env; then
    echo -e "${RED}❌ ANTHROPIC_API_KEY not set in .env file${NC}"
    echo -e "${YELLOW}Please edit .env and add your API key${NC}"
    exit 1
fi

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${RED}❌ uv not found. Please install it first:${NC}"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo -e "${GREEN}✓ Environment configured${NC}"
echo -e "${GREEN}✓ Installing dependencies...${NC}"

# Install/sync dependencies
uv sync

echo -e "${GREEN}✓ Starting Gradio application...${NC}"
echo -e "${BLUE}📍 Application will be available at: http://localhost:7860${NC}"
echo -e "${BLUE}📍 API documentation: http://localhost:7860/?view=api${NC}"
echo ""

# Run the Gradio app directly
uv run python src/mlagentfactory/ui/gradio_ui.py

# Note: The script will keep running until Ctrl+C is pressed
