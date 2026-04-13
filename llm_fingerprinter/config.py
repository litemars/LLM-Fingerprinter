import os
import shutil
from pathlib import Path

# Package directory (where the code lives)
PACKAGE_DIR = Path(__file__).parent

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


def _get_data_dir() -> Path:
    """Get the data directory for runtime files.

    Uses LLM_FINGERPRINTER_DATA env var if set, otherwise:
    - If running from a git checkout (has .git or .gitignore in parent),
      uses the project root (backward compatible).
    - Otherwise uses ~/.llm-fingerprinter/ as the user data directory.
    """
    env_dir = os.environ.get("LLM_FINGERPRINTER_DATA")
    if env_dir:
        return Path(env_dir)

    # Check if we're in a development/git checkout
    project_root = PACKAGE_DIR.parent
    if (project_root / ".git").exists() or (project_root / ".gitignore").exists():
        return project_root

    # Installed via pip - use a user data directory
    return Path.home() / ".llm-fingerprinter"


# Paths
BASE_DIR = _get_data_dir()
FINGERPRINTS_DIR = BASE_DIR / "fingerprints"
TRAINING_DIR = FINGERPRINTS_DIR / "training"   # simulation data for training
RESULTS_DIR = FINGERPRINTS_DIR / "results"     # inference/identification results
MODEL_DIR = BASE_DIR / "model"
LOGS_DIR = BASE_DIR / "logs"

# Bundled pre-trained model (shipped with the package)
BUNDLED_MODEL_DIR = PACKAGE_DIR.parent / "model"

# Ensure directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FINGERPRINTS_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Copy bundled model to user data dir if not already present
_bundled_classifier = BUNDLED_MODEL_DIR / "classifier_model.joblib"
_user_classifier = MODEL_DIR / "classifier_model.joblib"
if _bundled_classifier.exists() and not _user_classifier.exists():
    shutil.copy2(_bundled_classifier, _user_classifier)

# Feature extraction
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384 

LINGUISTIC_DIM = 12

BEHAVIORAL_DIM = 6

PER_PROMPT_FEATURE_DIM = EMBEDDING_DIM + LINGUISTIC_DIM + BEHAVIORAL_DIM  # 402

# Per-layer aggregation (3 prompt layers: stylistic, behavioral, discriminative)
NUM_PROMPT_LAYERS = 3
LAYER_ORDER = ['stylistic', 'behavioral', 'discriminative']
RAW_FINGERPRINT_DIM = PER_PROMPT_FEATURE_DIM * NUM_PROMPT_LAYERS  # 1206

# Embedding rebalancing: compress 384-dim embeddings to this per layer
EMBEDDING_PCA_DIM = 64

# PCA target dimension (should be <= min samples or total features)
PCA_DIM = 64

# OOD detection thresholds
OOD_CONFIDENCE_THRESHOLD = 0.3
OOD_DISAGREEMENT_THRESHOLD = 0.15

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