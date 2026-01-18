import os
from pathlib import Path

# Model families to identify
## I will need to enable deepseek when I will have enough samples
""" MODEL_FAMILIES = {
    "gpt": 0,
    "deepseek": 1,
    "llama": 2,
    "gemini": 3,
    "mistral": 4,
    "qwen": 5,
    "gemma": 6
} """

MODEL_FAMILIES = {
    "gpt": 0,
    "gemma": 1,
    "llama": 2,
    "gemini": 3,
    "mistral": 4,
    "qwen": 5,
}

# Paths
BASE_DIR = Path(__file__).parent.parent
FINGERPRINTS_DIR = BASE_DIR / "fingerprints"
MODEL_DIR = BASE_DIR / "model"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FINGERPRINTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Feature extraction
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384 

LINGUISTIC_DIM = 12

BEHAVIORAL_DIM = 6

TOTAL_FEATURE_DIM = EMBEDDING_DIM + LINGUISTIC_DIM + BEHAVIORAL_DIM  # 402

# PCA target dimension (should be <= min samples or total features)
PCA_DIM = 64

# Prompt suite
PROMPT_REPEATS = 2
TEMPERATURE = 0.7
MAX_TOKENS = 512
REQUEST_TIMEOUT = 60

# Data augmentation settings (for training with few samples)
AUGMENTATION_SAMPLES_PER_ORIGINAL = 5

# Logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Ollama client settings (local)
OLLAMA_DEFAULT_ENDPOINT = "http://localhost:11434/"
OLLAMA_TIMEOUT = 60
OLLAMA_MAX_RETRIES = 3

# Ollama Cloud client settings
OLLAMA_CLOUD_DEFAULT_ENDPOINT = "https://ollama.com/"
OLLAMA_CLOUD_TIMEOUT = 60
OLLAMA_CLOUD_MAX_RETRIES = 3

# OpenAI client settings
OPENAI_DEFAULT_ENDPOINT = "https://api.openai.com/v1"
OPENAI_TIMEOUT = 60
OPENAI_MAX_RETRIES = 3

# DeepSeek client settings
DEEPSEEK_DEFAULT_ENDPOINT = "https://api.deepseek.com"
DEEPSEEK_TIMEOUT = 60
DEEPSEEK_MAX_RETRIES = 3

# Gemini client settings
GEMINI_DEFAULT_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_TIMEOUT = 60
GEMINI_MAX_RETRIES = 3

# API Backend options: 'ollama', 'ollama-cloud', 'openai', 'deepseek', 'gemini', 'custom'
DEFAULT_BACKEND = "custom"

CUSTOM_DEFAULT_ENDPOINT = "http://localhost:8000/v1"
CUSTOM_TIMEOUT = 60
CUSTOM_MAX_RETRIES = 3

API_KEY_ENV_VARS = {
    "ollama-cloud": "OLLAMA_CLOUD_API_KEY", 
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "gemini": "GEMINI_API_KEY",
}