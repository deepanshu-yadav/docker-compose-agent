#!/bin/bash
# This script sets up the environment for the Docker Compose Agent.
# It installs the required packages, clones necessary repositories, and runs the agent.
# Make sure to run this script in a GitHub Codespace environment.

set -e  # Exit immediately if a command exits with a non-zero status
set -o pipefail  # Exit if any command in a pipeline fails

# Function to handle errors
error_handler() {
    echo "Error occurred at line $1"
    exit 1
}

# Set up error handling
trap 'error_handler $LINENO' ERR

echo "===== Starting Docker Compose Agent Setup ====="

# Check if running in GitHub Codespace
if [ -z "$CODESPACES" ]; then
    echo "Warning: This script is optimized for GitHub Codespaces environments."
    read -p "Continue anyway? (y/n): " confirm
    if [[ $confirm != [yY] ]]; then
        echo "Setup aborted."
        exit 0
    fi
fi

echo "Installing required Python packages..."
if ! pip install -r requirements.txt; then
    echo "Failed to install Python requirements. Please check requirements.txt file."
    exit 1
fi

echo "Creating repositories directory..."
mkdir -p repos

echo "Cloning necessary repositories..."
cd repos || { echo "Failed to change directory to repos"; exit 1; }

# Clone repositories with error checking
repositories=(
    "https://github.com/Haxxnet/Compose-Examples.git"
    "https://github.com/docker/awesome-compose.git"
    "https://github.com/ruanbekker/awesome-docker-compose.git"
)

for repo in "${repositories[@]}"; do
    repo_name=$(basename "$repo" .git)
    echo "Cloning $repo_name..."
    if [ -d "$repo_name" ]; then
        echo "$repo_name already exists, updating..."
        (cd "$repo_name" && git pull) || echo "Warning: Failed to update $repo_name"
    else
        git clone "$repo" || echo "Warning: Failed to clone $repo_name"
    fi
done

cd .. || { echo "Failed to return to root directory"; exit 1; }

echo "Installing curl..."
sudo apt update && sudo apt install -y curl || {
    echo "Failed to install curl. Attempting to continue..."
}

echo "Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh || {
    echo "Failed to install Ollama. This might affect functionality."
    read -p "Continue with the setup? (y/n): " continue_setup
    if [[ $continue_setup != [yY] ]]; then
        echo "Setup aborted."
        exit 1
    fi
}

echo "Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!

# Give Ollama time to start
echo "Waiting for Ollama to start..."
sleep 5

echo "Pulling LLama3.2 model..."
if ! ollama pull llama3.2:3b; then
    echo "Failed to pull LLama3.2 model. Check your internet connection."
    echo "Continuing with setup, but agent might not function properly."
fi

echo "Starting Streamlit application..."
echo "Press Ctrl+C to stop the application."
python start.py

# Clean up Ollama process if script is terminated
trap 'kill $OLLAMA_PID 2>/dev/null' EXIT

echo "===== Setup Complete ====="
