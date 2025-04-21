import os

# Unified Configuration
CONFIG = {
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