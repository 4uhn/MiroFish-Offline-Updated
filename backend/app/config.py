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
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    JSON_AS_ASCII = False

    # ── LLM Provider Configuration ──
    # LLM_PROVIDER selects the backend: "ollama" (default) or "groq"
    # Provider-specific env vars are resolved into unified LLM_API_KEY / LLM_BASE_URL / LLM_MODEL_NAME
    LLM_PROVIDER = os.environ.get('LLM_PROVIDER', 'ollama').lower()

    @staticmethod
    def _resolve_llm_config():
        """Resolve provider-specific env vars into unified LLM config."""
        provider = os.environ.get('LLM_PROVIDER', 'ollama').lower()

        if provider == 'groq':
            api_key = os.environ.get('GROQ_API_KEY', '')
            base_url = 'https://api.groq.com/openai/v1'
            model = os.environ.get('GROQ_MODEL', 'llama-3.3-70b-versatile')
        else:
            # Default: ollama (or any OpenAI-compatible provider)
            api_key = os.environ.get('LLM_API_KEY', 'ollama')
            base_url = os.environ.get('LLM_BASE_URL', 'http://localhost:11434/v1')
            model = os.environ.get('LLM_MODEL_NAME', 'qwen3:8b')

        return api_key, base_url, model

    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME = _resolve_llm_config.__func__()

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
    OASIS_DEFAULT_MAX_ROUNDS = int(os.environ.get('OASIS_DEFAULT_MAX_ROUNDS', '10'))
    OASIS_SIMULATION_DATA_DIR = os.path.join(os.path.dirname(__file__), '../uploads/simulations')
    OASIS_TWITTER_ACTIONS = [
        'CREATE_POST', 'LIKE_POST', 'REPOST', 'FOLLOW', 'DO_NOTHING', 'QUOTE_POST'
    ]
    OASIS_REDDIT_ACTIONS = [
        'LIKE_POST', 'DISLIKE_POST', 'CREATE_POST', 'CREATE_COMMENT',
        'LIKE_COMMENT', 'DISLIKE_COMMENT', 'SEARCH_POSTS', 'SEARCH_USER',
        'TREND', 'REFRESH', 'DO_NOTHING', 'FOLLOW', 'MUTE'
    ]

    # Report agent
    REPORT_AGENT_MAX_TOOL_CALLS = int(os.environ.get('REPORT_AGENT_MAX_TOOL_CALLS', '5'))
    REPORT_AGENT_MAX_REFLECTION_ROUNDS = int(os.environ.get('REPORT_AGENT_MAX_REFLECTION_ROUNDS', '2'))
    REPORT_AGENT_TEMPERATURE = float(os.environ.get('REPORT_AGENT_TEMPERATURE', '0.5'))

    @classmethod
    def validate(cls):
        errors = []
        if cls.LLM_PROVIDER == 'groq':
            if not os.environ.get('GROQ_API_KEY'):
                errors.append("GROQ_API_KEY is required when LLM_PROVIDER=groq (get one at https://console.groq.com)")
        elif not cls.LLM_API_KEY:
            errors.append("LLM_API_KEY is not configured (set to any non-empty value, e.g. 'ollama')")
        if not cls.NEO4J_URI:
            errors.append("NEO4J_URI is not configured")
        if not cls.NEO4J_PASSWORD:
            errors.append("NEO4J_PASSWORD is not configured")
        return errors
