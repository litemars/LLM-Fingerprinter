# LLM Fingerprinting System

[![PyPI version](https://badge.fury.io/py/llm-fingerprinter.svg)](https://pypi.org/project/llm-fingerprinter/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A black-box fingerprinting system that identifies the underlying LLM model family (GPT, LLaMA, Mistral, etc.) by analyzing response patterns across 75 discriminative prompts. The system can identify fine-tuned models as well, tracing them back to their foundational base model.

**Note: Check config.py to see all identifiable model families**

A pre-trained classifier is bundled with the package in the `model` directory.

 <img src="img/gpt.png" width="400" height="400" alt="GPT">

## Supported Backends

| Backend | Description | API Key Required |
|---------|-------------|------------------|
| `ollama` | Local Ollama instance | No |
| `ollama-cloud` | Ollama Cloud API | `OLLAMA_CLOUD_API_KEY` |
| `openai` | OpenAI API (or compatible) | `OPENAI_API_KEY` |
| `gemini` | Gemini API (or compatible) | `GEMINI_API_KEY` |
| `deepseek` | Deepseek API (or compatible) | `DEEPSEEK_API_KEY` |
| `custom` | Custom HTTP request | `CUSTOM_API_KEY` |

## Installation

### From PyPI

```bash
# Core package (Ollama + custom backends)
pip install llm-fingerprinter

# With OpenAI support
pip install llm-fingerprinter[openai]

# With Gemini support
pip install llm-fingerprinter[gemini]

# With all backends
pip install llm-fingerprinter[all]
```

### From source (development)

```bash
git clone https://github.com/litemars/LLM-Fingerprinter.git
cd LLM-Fingerprinter
pip install -e ".[all,dev]"

# Optional: Download NLTK data for text processing
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"
```

## Quick Start

### Ollama

```bash
# Identify model and fine-tuning
llm-fingerprinter identify -b ollama --model some-model

# Train your own classifier
# Fingerprint the LLM
llm-fingerprinter simulate --model llama3.2 --family llama
# Train on the Fingerprints
llm-fingerprinter train
```

### Custom - Interact with any LLM via HTTP request

```bash
llm-fingerprinter identify -r ./custom_request.txt --api-key <API_KEY>
# Example of custom request inside the example folder
```

### Ollama Cloud

```bash
export OLLAMA_CLOUD_API_KEY="your-key"
llm-fingerprinter simulate -b ollama-cloud --model llama3.2 --family llama
```

### OpenAI

```bash
export OPENAI_API_KEY="your-key"
llm-fingerprinter simulate -b openai --model gpt-4 --family gpt
```

### Gemini

```bash
export GEMINI_API_KEY="your-key"
llm-fingerprinter simulate -b gemini --model gemini-2.5-pro --family gpt
```

### Deepseek

```bash
export DEEPSEEK_API_KEY="your-key"
llm-fingerprinter simulate -b deepseek --model deepseek-v3.2 --family deepseek
```

### Custom API

```bash
export CUSTOM_API_KEY="your-key"
llm-fingerprinter simulate -b custom -e http://your-api.com/v1 --model your-model --family llama
```

## Python API

You can also use the library programmatically:

```python
from llm_fingerprinter import LLMFingerprinter, EnsembleClassifier, FeatureExtractor, PromptSuite
from llm_fingerprinter.ollama_client import OllamaClient

# Setup components
client = OllamaClient(endpoint="http://localhost:11434")
suite = PromptSuite()
extractor = FeatureExtractor()
classifier = EnsembleClassifier()

# Create fingerprinter and identify a model
fingerprinter = LLMFingerprinter("http://localhost:11434", client, suite, extractor, classifier)
fingerprint = fingerprinter.fingerprint_model("llama3.2")
```

---

## Commands

### Backend Options (all LLM commands)

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--backend` | `-b` | `custom` | Backend: `ollama`, `ollama-cloud`, `openai`,`deepseek`,`gemini` ,`custom`|
| `--endpoint` | `-e` | auto | API endpoint URL |
| `--api-key` | `-k` | env var | API key |

### `simulate`

Run fingerprinting simulations for training data.

```bash
llm-fingerprinter simulate [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | *required* | Model name |
| `--family` | *required* | Family: `gpt`, `llama`, `gemini`, `mistral`, `qwen`, `gemma` |
| `--num-sims` | *optional* | Number of simulations |
| `--repeats` | *optional* | Prompt repeats per simulation |

**Examples:**
```bash
# Ollama local
llm-fingerprinter simulate --model llama3.2 --family llama

# Ollama Cloud
llm-fingerprinter simulate -b ollama-cloud --model llama3.2 --family llama

# OpenAI
llm-fingerprinter simulate -b openai --model gpt-4 --family gpt --num-sims 5

# Custom endpoint
llm-fingerprinter simulate -b openai -e https://api.groq.com/openai/v1 -k $GROQ_KEY --model llama-3.1-70b --family llama
```

### `train`

Train classifier from saved fingerprints.

```bash
llm-fingerprinter train [--augment/--no-augment] [--cross-validate]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--augment/--no-augment` | `--augment` | Data augmentation |
| `--use-pca` | off | Use PCA reduction |
| `--pca-components` | 64 | PCA components |
| `--cross-validate` / `-cv` | off | Run k-fold cross-validation |
| `--cv-folds` | 5 | Number of CV folds |

### `identify`

Identify model family using trained classifier.

```bash
llm-fingerprinter identify --model <model-name> [-b <backend>]
```

### `list-models`

List available models on the API.

```bash
llm-fingerprinter list-models [-b <backend>]
```

### `list-fingerprints`

List saved fingerprints by family.

```bash
llm-fingerprinter list-fingerprints
```

### `info`

Show configuration and status.

```bash
llm-fingerprinter info
```

---

## Environment Variables

| Variable | Backend | Description |
|----------|---------|-------------|
| `OLLAMA_CLOUD_API_KEY` | ollama-cloud | Ollama Cloud API key |
| `OPENAI_API_KEY` | openai | OpenAI API key |
| `GEMINI_API_KEY` | gemini | Gemini API key |
| `DEEPSEEK_API_KEY` | deepseek | DeepSeek API key |
| `CUSTOM_API_KEY` | custom | Custom API key |
| `LOG_LEVEL` | all | Logging level (DEBUG, INFO, etc.) |
| `LLM_FINGERPRINTER_DATA` | all | Custom data directory path |

## Data Storage

When installed via pip, runtime data (fingerprints, trained models, logs) is stored in `~/.llm-fingerprinter/`. You can override this with the `LLM_FINGERPRINTER_DATA` environment variable. When running from a git checkout, data is stored in the project directory (backward compatible).

## How It Works

1. **75 Prompts** across 3 layers:
   - *Stylistic*: Analyze writing style and formatting preferences
   - *Behavioral*: Assess response patterns and decision-making behavior
   - *Discriminative*: Identify model-specific characteristics and inconsistencies

2. **Feature Extraction**: 384-dim embeddings + 12 linguistic + 6 behavioral features
3. **PCA** reduction to 64 dimensions (Optional)
4. **Ensemble Classification**: Random Forest (45%) + SVM (45%) + MLP (10%)

---

## Contributing

Contributions are welcome! Whether you're adding support for new models, improving accuracy, or extending to additional clients, please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT License
