import os

# Define constants
MAIN_WORKSPACE_DIR = os.path.join(os.getcwd(), "workspaces")
DEEPSEEK_FREE_KEY = os.environ.get("DEEPSEEK_FREE_KEY", "")
# Maximum response length before truncation (adjust based on LLM token limit)
MAX_RESPONSE_LENGTH = 6000  # Safe for ~8k token limit, leaving room for prompt
OLLAMA_URL = "http://localhost:11434"  # Ollama server URL
OPEN_ROUTER_URL = "https://openrouter.ai/api/v1"  # OpenRouter server URL

MCP_HOST="0.0.0.0"
MCP_PORT=8787
MCP_IP="localhost"

SAVE_DIR = os.path.join(os.getcwd(), "workspaces", "agent")
os.makedirs(SAVE_DIR, exist_ok=True)

# Unified Configuration
CONFIG = {
    "response_filename": "LLM_RESPONSE.json",
    # model settings
    "online_models": ["microsoft/mai-ds-r1:free", 
                      "meta-llama/llama-4-maverick:free",
                      "deepseek/deepseek-r1-zero:free",
                      ],
    "offline_models": ["llama3.2:3b", "deepcoder:14b", "llama3.2:1b", "gemma3:4b",
                       "gemma3:12b","gemma3:4b","deepseek-r1:32b","deepseek-r1:7b",
                       "deepseek-r1:1.5b", "deepcoder:1.5b"
                       ],
    # General settings
    "storage_dir": "unified_docker_compose_rag",
    "embedding_model": "all-MiniLM-L6-v2",
    
    # Local repository settings
    "repo_dirs": [
        os.path.join("repos", "awesome-compose"),
        os.path.join("repos", "awesome-docker-compose"),
        os.path.join("repos", "Compose-Examples", "examples")
    ],
    "priority_extensions": ['.yml', '.yaml', '.dockerfile', 'Dockerfile'],
    
    # Documentation settings
    "docs_base_url": "https://docs.docker.com/compose/",
    "docs_exclude_keywords": [
        "release notes", "install docker", "edit this page",
        "faq", "migrate", "changelog", "github.com"
    ],
    "min_chunk_length": 50,
    
    # Stack Overflow settings
    "so_api_key": None,  # Optional: Add your Stack Exchange API key here
    "so_min_upvotes": 5,
    "so_top_answers": 3,
    "so_questions_limit": 300,
    "so_time_window_years": 2,
    "so_tag": "docker-compose",
    
    # Retrieval settings
    "overall_top_k": 7,  # Total results to retrieve
    "source_weights": {
        "docs": 1.0,    # Weight for official documentation
        "stackoverflow": 0.4  # Weight for Stack Overflow
    }
}