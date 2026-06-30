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
    "qwen": 1,
    "llama": 2,
    "gemini": 3,    # reserved for real Gemini-API models (no data yet)
    "mistral": 4,
    "gemma": 5      # Google Gemma open models — distinct from Gemini
}


def _find_project_root() -> Path:
    """Walk up from CWD to find the project root (marked by setup.py or .git)."""
    for path in [Path.cwd(), *Path.cwd().parents]:
        if (path / "setup.py").exists() or (path / ".git").exists():
            return path
    return Path.cwd()


def _get_data_dir() -> Path:
    """Runtime data dir: LLM_FINGERPRINTER_DATA, else the project root."""
    env_dir = os.environ.get("LLM_FINGERPRINTER_DATA")
    if env_dir:
        return Path(env_dir)

    return _find_project_root()


def _running_from_source() -> bool:
    """True when imported from a source checkout, not an installed package."""
    parts = set(PACKAGE_DIR.parts)
    return "site-packages" not in parts and "dist-packages" not in parts


def _dir_writable(path: Path) -> bool:
    """Whether files can be created under `path` (probes nearest existing parent)."""
    probe = path
    while not probe.exists():
        probe = probe.parent
    return os.access(probe, os.W_OK)


def _get_model_dir(base_dir: Path) -> Path:
    """Where model artifacts are read/written: LLM_FINGERPRINTER_MODEL, else the
    in-package dir when run from a writable source checkout (so `train` updates the
    bundled model directly), else base_dir/model for installed packages."""
    env_dir = os.environ.get("LLM_FINGERPRINTER_MODEL")
    if env_dir:
        return Path(env_dir)
    pkg_model = PACKAGE_DIR / "model"
    if _running_from_source() and _dir_writable(pkg_model):
        return pkg_model
    return base_dir / "model"


# Paths
BASE_DIR = _get_data_dir()
FINGERPRINTS_DIR = BASE_DIR / "fingerprints"
TRAINING_DIR = FINGERPRINTS_DIR / "training"   # simulation data for training
RESULTS_DIR = FINGERPRINTS_DIR / "results"     # inference/identification results
MODEL_DIR = _get_model_dir(BASE_DIR)           # bundled package dir when run from source
TEMPLATES_PATH = MODEL_DIR / "templates.joblib"
MODEL_TEMPLATES_PATH = MODEL_DIR / "model_templates.joblib"
LOGS_DIR = BASE_DIR / "logs"

# Pre-trained model shipped inside the package so it ships in the wheel. Equals
# MODEL_DIR when run from source; for installed packages the bootstrap below seeds
# the separate runtime MODEL_DIR from it.
BUNDLED_MODEL_DIR = PACKAGE_DIR / "model"

# Ensure directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FINGERPRINTS_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Seed the runtime dir with any bundled artifacts it's missing, so a fresh
# install can identify out of the box.
if BUNDLED_MODEL_DIR.exists() and BUNDLED_MODEL_DIR.resolve() != MODEL_DIR.resolve():
    for _bundled in BUNDLED_MODEL_DIR.glob("*.joblib"):
        _dest = MODEL_DIR / _bundled.name
        if not _dest.exists():
            try:
                shutil.copy2(_bundled, _dest)
            except Exception:
                pass

# Feature extraction
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384 

LINGUISTIC_DIM = 12

BEHAVIORAL_DIM = 6

PER_PROMPT_FEATURE_DIM = EMBEDDING_DIM + LINGUISTIC_DIM + BEHAVIORAL_DIM  # 402

NUM_PROMPT_LAYERS = 3
LAYER_ORDER = ['discriminative', 'behavioral', 'stylistic']
RAW_FINGERPRINT_DIM = PER_PROMPT_FEATURE_DIM * NUM_PROMPT_LAYERS  # 1206

# Embedding rebalancing: compress 384-dim embeddings to this per layer
EMBEDDING_PCA_DIM = 64

# PCA target dimension (should be <= min samples or total features)
PCA_DIM = 64

# OOD detection thresholds
OOD_CONFIDENCE_THRESHOLD = 0.3
OOD_DISAGREEMENT_THRESHOLD = 0.15

# Prompt suite
PROMPT_REPEATS = 1
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