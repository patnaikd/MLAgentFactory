#!/bin/bash

# MLAgentFactory - Run Script
# This script launches the Streamlit application with proper configuration

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🤖 MLAgentFactory - Starting Application${NC}"
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
echo -e "${GREEN}✓ Starting Streamlit application...${NC}"
echo ""

# Run the Streamlit app
uv run streamlit run src/mlagentfactory/ui/app.py

# Note: The script will keep running until Ctrl+C is pressed
