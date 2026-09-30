"""
Application configuration.
Loads settings from the project root .env file.
"""

import os
from dotenv import load_dotenv

project_root_env = os.path.join(os.path.dirname(__file__), '../../.env')

if os.path.exists(project_root_env):
    load_dotenv(project_root_env, override=True)
else:
    load_dotenv(override=True)


class Config:
    """Flask application configuration."""

    SECRET_KEY = os.environ.get('SECRET_KEY', 'mirofish-secret-key')
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    JSON_AS_ASCII = False

    # LLM: Ollama by default, or any OpenAI-compatible server
    LLM_API_KEY = os.environ.get('LLM_API_KEY', 'ollama')
    LLM_BASE_URL = os.environ.get('LLM_BASE_URL', 'http://localhost:11434/v1')
    LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME', 'qwen3:8b')

    # Propagate resolved values into env so CAMEL-AI / simulation scripts inherit them
    os.environ.setdefault('LLM_API_KEY', LLM_API_KEY or '')
    os.environ.setdefault('LLM_BASE_URL', LLM_BASE_URL or '')
    os.environ.setdefault('LLM_MODEL_NAME', LLM_MODEL_NAME or '')
    if LLM_API_KEY:
        os.environ.setdefault('OPENAI_API_KEY', LLM_API_KEY)
    if LLM_BASE_URL:
        os.environ.setdefault('OPENAI_API_BASE_URL', LLM_BASE_URL)

    # Neo4j
    NEO4J_URI = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
    NEO4J_USER = os.environ.get('NEO4J_USER', 'neo4j')
    NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD', 'mirofish')

    # Embeddings
    EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'nomic-embed-text')
    EMBEDDING_BASE_URL = os.environ.get('EMBEDDING_BASE_URL', 'http://localhost:11434')

    # File uploads
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../uploads')
    ALLOWED_EXTENSIONS = {'pdf', 'md', 'txt', 'markdown'}

    # Text processing
    DEFAULT_CHUNK_SIZE = 500
    DEFAULT_CHUNK_OVERLAP = 50

    # OASIS simulation
    OASIS_SIMULATION_DATA_DIR = os.path.join(os.path.dirname(__file__), '../uploads/simulations')

    @classmethod
    def validate(cls):
        errors = []
        if not cls.LLM_API_KEY:
            errors.append("LLM_API_KEY is not configured (set to any non-empty value, e.g. 'ollama')")
        if not cls.NEO4J_URI:
            errors.append("NEO4J_URI is not configured")
        if not cls.NEO4J_PASSWORD:
            errors.append("NEO4J_PASSWORD is not configured")
        return errors
