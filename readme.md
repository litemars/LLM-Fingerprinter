# LLM Fingerprinting System

A black-box fingerprinting system that identifies the underlying LLM model family (GPT, LLaMA, Mistral, etc.) by analyzing response patterns across 75 discriminative prompts. The system can identify fine-tuned models as well, tracing them back to their foundational base model.

**Disclaimer: check the config.py to see all the model families that can be identify** 

You can find an *already* NLP trained model in the `model` directory.

<div style="gap: 70px;">
    <img src="img/gemma.png" width="300" height="300" style="margin-right: 100px;">
    <img src="img/gpt.png" width="300" height="300">
</div>

## Supported Backends

| Backend | Description | API Key Required |
|---------|-------------|------------------|
| `ollama` | Local Ollama instance | ❌ No |
| `ollama-cloud` | Ollama Cloud API | ✅ `OLLAMA_CLOUD_API_KEY` |
| `openai` | OpenAI API (or compatible) | ✅ `OPENAI_API_KEY` |
| `gemini` | Gemini API (or compatible) | ✅ `GEMINI_API_KEY` |
| `deepseek` | Deepseek API (or compatible) | ✅ `DEEPSEEK_API_KEY` |
| `custom` | Custom API (template-based) | ✅ `CUSTOM_API_KEY` |

## Installation

```bash
cd llm_fingerprint
pip install -r requirements.txt

# Optional
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"
```

## Quick Start

### Ollama (Default)

```bash
# Idenitfy model and finetuning

python3 cli.py identify --model some-model

# Train your own classifier
# Get Samples
python3 cli.py simulate --model llama3.2 --family llama
# Train on the samples
python3 cli.py train

```

### Ollama Cloud

```bash
export OLLAMA_CLOUD_API_KEY="your-key"
python3 cli.py simulate -b ollama-cloud --model llama3.2 --family llama
```

### OpenAI

```bash
export OPENAI_API_KEY="your-key"
python3 cli.py simulate -b openai --model gpt-4 --family gpt
```

### Gemini

```bash
export GEMINI_API_KEY="your-key"
python3 cli.py simulate -b gemini --model gemini-2.5-pro --family gpt
```

### Deepseek

```bash
export DEEPSEEK_API_KEY="your-key"
python3 cli.py simulate -b deepseek --model deepseek-v3.2 --family deepseek
```

### Custom API

```bash
export CUSTOM_API_KEY="your-key"
python3 cli.py simulate -b custom -e http://your-api.com/v1 --model your-model --family llama
```

---

## Commands

### Backend Options (all LLM commands)

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--backend` | `-b` | `ollama` | Backend: `ollama`, `ollama-cloud`, `openai`,`deepseek`,`gemini` ,`custom`|
| `--endpoint` | `-e` | auto | API endpoint URL |
| `--api-key` | `-k` | env var | API key |

### `simulate`

Run fingerprinting simulations for training data.

```bash
python -m llm_fingerprint.cli simulate [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | *required* | Model name |
| `--family` | *required* | Family: `gpt`, `claude`, `llama`, `gemini`, `mistral`, `qwen`, `gemma` |
| `--num-sims` | *optional* | Number of simulations |
| `--repeats` | *optional* | Prompt repeats per simulation |

**Examples:**
```bash
# Ollama local
python -m llm_fingerprint.cli simulate --model llama3.2 --family llama

# Ollama Cloud
python -m llm_fingerprint.cli simulate -b ollama-cloud --model llama3.2 --family llama

# OpenAI
python3 cli.py simulate -b openai --model gpt-4 --family gpt --num-sims 5

# Custom endpoint
python3 cli.py simulate -b openai -e https://api.groq.com/openai/v1 -k $GROQ_KEY --model llama-3.1-70b --family llama
```

### `train`

Train classifier from saved fingerprints.

```bash
python3 cli.py train [--augment/--no-augment]
```

### `identify`

Identify model family using trained classifier.

```bash
python3 cli.py identify --model <model-name> [-b <backend>]
```

# Other commands
### `list-models`

List available models on the API.

```bash
python3 cli.py list-models [-b <backend>]
```

### `list-fingerprints`

List saved fingerprints by family.

```bash
python3 cli.py list-fingerprints
```

### `info`

Show configuration and status.

```bash
python3 cli.py info
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


## How It Works

1. **75 Prompts** across 3 layers (stylistic, behavioral, discriminative)
2. **Feature Extraction**: 384-dim embeddings + 12 linguistic + 6 behavioral features
3. **PCA** reduction to 64 dimensions (Optional)
4. **Ensemble Classification**: Random Forest (45%) + SVM (45%) + MLP (10%)

---

## License

MIT License

