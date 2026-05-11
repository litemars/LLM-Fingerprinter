# LLM Fingerprinting System

[![PyPI version](https://badge.fury.io/py/llm-fingerprinter.svg)](https://pypi.org/project/llm-fingerprinter/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A black-box fingerprinting system that identifies the underlying LLM model family (GPT, LLaMA, Mistral, etc.) by analysing response patterns across 31 carefully selected prompts. The system can identify fine-tuned models as well, tracing them back to their foundational base model.

**Note: Check `config.py` to see all identifiable model families.**

A pre-trained classifier is bundled with the package in the `model/` directory.

<img src="img/gpt.png" width="400" height="400" alt="GPT">

---

## How It Works

Fingerprinting runs in three sequential layers:

1. **31 prompts** across 3 layers (discriminative → behavioral → stylistic):
   - *Discriminative* (11): Identity, knowledge cutoff, architecture, reasoning — most separating power
   - *Behavioral* (7): Safety boundaries, jailbreak resistance, honesty, policy handling
   - *Stylistic* (13): Formatting, creativity, constraint following, default voice

2. **Feature extraction** per response: 384-dim sentence embeddings + 12 linguistic features + 6 behavioral features = **402 dims per layer**, **1206 dims total**

3. **Embedding rebalancing**: Per-layer PCA compresses 384-dim embeddings to 64 dims → **246-dim working space**

4. **Ensemble classification**: Random Forest (45%) + SVM (45%) + MLP (10%)

5. **Two-stage identification**: Ensemble → model family, Template classifier → specific model version

6. **Early stopping**: After each layer the classifier checks confidence — if it exceeds the threshold (default 0.95) the remaining layers are skipped, saving API calls.

---

## Supported Backends

| Backend | Description | API Key Required |
|---------|-------------|------------------|
| `ollama` | Local Ollama instance | ❌ No |
| `ollama-cloud` | Ollama Cloud API | ✅ `OLLAMA_CLOUD_API_KEY` |
| `openai` | OpenAI API (or compatible) | ✅ `OPENAI_API_KEY` |
| `gemini` | Gemini API | ✅ `GEMINI_API_KEY` |
| `custom` | **Any HTTP-based LLM API** | ✅ Optional |

### About the Custom Backend

The **custom backend** is the most flexible option — use it with:
- Proprietary LLM APIs not natively supported
- Self-hosted LLMs behind HTTP endpoints
- API proxies and gateways
- Any HTTP-based LLM service

All you need is an HTTP request template file. See examples in `./example/`.

---

## Installation

### From PyPI

```bash
# Core package
pip install llm-fingerprinter

# With OpenAI support
pip install llm-fingerprinter[openai]

# With Gemini support
pip install llm-fingerprinter[gemini]

# With all backends
pip install llm-fingerprinter[all]
```

## Quick Start

### 1. Identify a Model (Pre-trained Classifier)

```bash
# Local Ollama
llm-fingerprinter identify -b ollama --model llama3.2

# OpenAI
export OPENAI_API_KEY="your-key"
llm-fingerprinter identify -b openai --model gpt-4o-mini

# Custom endpoint
llm-fingerprinter identify -b custom -r ./custom_request.txt
```

### 2. Train Your Own Classifier

```bash
# Step 1: Generate training fingerprints for each family
#         Temperature is automatically varied across simulations for diversity
llm-fingerprinter simulate -b ollama --model llama3.2 --family llama --num-sims 5
llm-fingerprinter simulate -b openai --model gpt-4o-mini --family gpt --num-sims 5

# Step 2: Train the ensemble classifier
llm-fingerprinter train

# Step 3: Build template classifiers (for two-stage identification)
llm-fingerprinter build-templates
llm-fingerprinter build-model-templates

# Step 4: Identify unknown models
llm-fingerprinter identify -b ollama --model some-unknown-model
```

---

### `build-templates` — Build Family Template Classifier

Compute per-family mean vectors from training fingerprints for the open-set template classifier. Run after `train`.

```bash
llm-fingerprinter build-templates
```

The template classifier uses cosine distance to nearest mean — it doesn't require retraining when adding new families.

---

### `build-model-templates` — Build Model-Level Templates

Build templates at the specific model version level (e.g. `gpt-4o-mini` vs `gpt-4.1`) for two-stage identification.

```bash
llm-fingerprinter build-model-templates
```

Requires fingerprints that contain `model_name` in their metadata (all fingerprints generated with `simulate` on this version do).

---

### `add-family` — Add a New Family Without Retraining

Add a new model family to the template classifier from a few fingerprint samples, without retraining the full ensemble.

```bash
llm-fingerprinter add-family --model deepseek-chat --family deepseek --num-sims 3 -b deepseek
```

Recommended minimum: 3 simulations for a reliable mean template.

---

## Environment Variables

| Variable | Backend | Description |
|----------|---------|-------------|
| `OLLAMA_CLOUD_API_KEY` | ollama-cloud | Ollama Cloud API key |
| `OPENAI_API_KEY` | openai | OpenAI API key |
| `GEMINI_API_KEY` | gemini | Gemini API key |
| `DEEPSEEK_API_KEY` | deepseek | DeepSeek API key |
| `LOG_LEVEL` | all | Logging level (`DEBUG`, `INFO`, `WARNING`) |
| `LLM_FINGERPRINTER_DATA` | all | Override data directory (fingerprints, model, logs) |


---

## License

MIT License
