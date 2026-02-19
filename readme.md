# LLM Fingerprinting System

A black-box fingerprinting system that identifies the underlying LLM model family (GPT, LLaMA, Mistral, etc.) by analyzing response patterns across 75 discriminative prompts. The system can identify fine-tuned models as well, tracing them back to their foundational base model.

**Note: Check config.py to see all identifiable model families** 

You can find an *already* NLP trained model in the `model` directory.

 <img src="img/gpt.png" width="400" height="400" alt="GPT">

## Supported Backends

| Backend | Description | API Key Required |
|---------|-------------|------------------|
| `ollama` | Local Ollama instance | ❌ No |
| `ollama-cloud` | Ollama Cloud API | ✅ `OLLAMA_CLOUD_API_KEY` |
| `openai` | OpenAI API (or compatible) | ✅ `OPENAI_API_KEY` |
| `gemini` | Gemini API (or compatible) | ✅ `GEMINI_API_KEY` |
| `deepseek` | Deepseek API (or compatible) | ✅ `DEEPSEEK_API_KEY` |
| `custom` | **Any HTTP-based LLM API** | ✅ Optional |

### About the Custom Backend

The **custom backend** is the most flexible option - use it with:
- **Proprietary LLM APIs** not natively supported
- **Self-hosted LLMs** behind HTTP endpoints
- **API proxies** and gateways
- **Any HTTP-based LLM service**

All you need is an HTTP request template file! See examples in `./example/` directory.

## Installation

```bash
pip install -r requirements.txt

# Or install as a package
pip3 install -e .

# Optional: Download NLTK data for text processing
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"
```

## Quick Start

### 1. Identify a Model (Using Pre-trained Classifier)

```bash
# Custom endpoint
llm-fingerprinter identify -b custom -r ./custom_request.txt

# Local Ollama
llm-fingerprinter identify -b ollama --model llama3.2

# OpenAI
export OPENAI_API_KEY="your-key"
llm-fingerprinter identify -b openai --model gpt-4o-mini
```

### 2. Train Your Own Classifier

```bash
# Step 1: Generate fingerprints for a known model
llm-fingerprinter simulate -b ollama --model llama3.2 --family llama --num-sims 3

# Step 2: Train classifier from fingerprints
llm-fingerprinter train

# Step 3: Identify unknown models
llm-fingerprinter identify -b ollama --model some-other-model
```

### 3. Backend-Specific Examples

**Ollama (Local)**
```bash
# List available models
llm-fingerprinter list-models -b ollama

# Identify
llm-fingerprinter identify -b ollama --model llama3.2

# Generate fingerprints
llm-fingerprinter simulate -b ollama --model llama3.2 --family llama
```

**Ollama Cloud**
```bash
export OLLAMA_CLOUD_API_KEY="your-key"
llm-fingerprinter identify -b ollama-cloud --model llama3.2
llm-fingerprinter simulate -b ollama-cloud --model llama3.2 --family llama
```

**OpenAI**
```bash
export OPENAI_API_KEY="your-key"
llm-fingerprinter identify -b openai --model gpt-4o
llm-fingerprinter simulate -b openai --model gpt-4 --family gpt --num-sims 5
```

**Gemini**
```bash
export GEMINI_API_KEY="your-key"
llm-fingerprinter identify -b gemini --model gemini-2.5-pro
llm-fingerprinter simulate -b gemini --model gemini-2.5-pro --family gemini
```

**DeepSeek**
```bash
export DEEPSEEK_API_KEY="your-key"
llm-fingerprinter identify -b deepseek --model deepseek-chat
llm-fingerprinter simulate -b deepseek --model deepseek-chat --family deepseek
```

**Custom API (Any HTTP Endpoint)**

Works with **any** LLM API via HTTP request template. No native backend support needed!

```bash
export CUSTOM_API_KEY="your-api-key"
llm-fingerprinter identify -b custom -r ./custom_request.txt
llm-fingerprinter identify -b custom -r ./custom_request.txt -k my-api-key
llm-fingerprinter simulate -b custom -r ./custom_request.txt --family gpt
```

💡 **Tip:** See `./example/` directory for request template examples (openai_request.txt, ollama_cloud_request.txt, etc.)

---

## Commands Reference

### Global Options

| Option | Short | Description |
|--------|-------|-------------|
| `--verbose` | `-v` | Enable verbose output (debug logging) |

### Backend Options (Common to all LLM commands)

These options are available for: `identify`, `simulate`, `test`, `fingerprint`, and `list-models`

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--backend` | `-b` | `ollama` | Backend: `ollama`, `ollama-cloud`, `openai`, `deepseek`, `gemini`, `custom` |
| `--endpoint` | `-e` | auto | API endpoint URL (overrides default) |
| `--api-key` | `-k` | env var | API key (fallback to environment variable) |
| `--request-file` | `-r` | - | Request template file (required for `custom` backend) |

---

### `identify` - Identify Unknown Model

Classify an unknown model using the trained classifier. Works with any LLM backend including **custom HTTP endpoints**.

```bash
llm-fingerprinter identify [OPTIONS]
```

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model` | `-m` | - | Model name (optional, may be in request template for custom backend) |
| `--repeats` | - | 1 | Number of times to repeat each prompt (increases confidence) |
| `--backend` | `-b` | `ollama` | LLM backend |
| `--endpoint` | `-e` | auto | API endpoint |
| `--api-key` | `-k` | env var | API key |

**Examples:**

```bash
# Local Ollama (simplest)
llm-fingerprinter identify -b ollama --model llama3.2

# With multiple repeats for higher confidence
llm-fingerprinter identify -b ollama --model llama3.2 --repeats 3

# OpenAI
export OPENAI_API_KEY="sk-..."
llm-fingerprinter identify -b openai --model gpt-4o-mini

# ⭐ Custom endpoint (e.g., proprietary LLM, local instance, proxy)
llm-fingerprinter identify -b custom -r ./example/openai_request.txt

# ⭐ Custom with API key
llm-fingerprinter identify -b custom -r ./example/openai_request.txt -k "your-api-key"

# ⭐ Any HTTP-based LLM (examples in ./example/)
llm-fingerprinter identify -b custom -r ./example/ollama_cloud_request.txt
```

**Output:**
```
═══════════════════════════════════════════════════════════════
                 IDENTIFICATION REPORT
═══════════════════════════════════════════════════════════════

  Identified: GPT (or LLAMA, GEMINI, etc.)
  Confidence: 92.5%

  Probabilities:
    gpt        92.5% █████████████████████
    llama      5.2%  █
    gemini     1.8%
    mistral    0.5%

═══════════════════════════════════════════════════════════════
```

---

### `simulate` - Generate Training Fingerprints

Create fingerprints for known models to build/improve the classifier. Works with any backend including **custom HTTP endpoints**.

```bash
llm-fingerprinter simulate [OPTIONS]
```

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model` | `-m` | - | Model name (optional) |
| `--family` | `-f` | - | **Required.** Model family: `gpt`, `claude`, `llama`, `gemini`, `mistral`, `qwen`, `gemma`, `deepseek` |
| `--num-sims` | `-n` | 3 | Number of fingerprints to generate |
| `--repeats` | - | 2 | Prompts repeats per simulation |
| `--backend` | `-b` | `ollama` | LLM backend |
| `--endpoint` | `-e` | auto | API endpoint |
| `--api-key` | `-k` | env var | API key |

**Examples:**

```bash
# Basic simulation (3 fingerprints, 2 repeats each)
llm-fingerprinter simulate -b ollama --model llama3.2 --family llama

# More comprehensive (10 fingerprints, 5 repeats each)
llm-fingerprinter simulate -b ollama --model llama3.2 --family llama --num-sims 10 --repeats 5

# OpenAI
export OPENAI_API_KEY="sk-..."
llm-fingerprinter simulate -b openai --model gpt-4 --family gpt --num-sims 5

# Custom endpoint with specific API
llm-fingerprinter simulate -b openai -e https://api.groq.com/openai/v1 -k $GROQ_KEY \
  --model llama-3.1-70b --family llama
```

---

### `train` - Build Classifier

Train an ensemble classifier from saved fingerprints.

```bash
llm-fingerprinter train [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--augment / --no-augment` | `--augment` | Enable/disable data augmentation |
| `--use-pca` | false | Use PCA dimensionality reduction |
| `--pca-components` | 64 | Number of PCA components (if `--use-pca`) |
| `--cross-validate` / `-cv` | false | Run k-fold cross-validation |
| `--cv-folds` | 5 | Number of cross-validation folds |

**Examples:**

```bash
# Default: raw features (402-dim), with augmentation
llm-fingerprinter train

# With PCA reduction (faster, less accurate)
llm-fingerprinter train --use-pca

# Custom PCA components
llm-fingerprinter train --use-pca --pca-components 128

# With cross-validation
llm-fingerprinter train --cross-validate --cv-folds 5

# Disable augmentation
llm-fingerprinter train --no-augment
```

**Output:**
```
🧠 Training classifier (raw features (402-dim))...
📊 Training data:
    gpt: 15 samples (402 dims)
    llama: 12 samples (402 dims)
    gemini: 10 samples (402 dims)
    Total: 37

📈 Running 5-fold cross-validation...
   Mean accuracy: 94.6% (5 folds)
   Per-family metrics:
   Family       Prec     Recall      F1   Support
   ──────────────────────────────────────────────
   gpt         0.96      0.95    0.96        15
   llama       0.93      0.92    0.92        12
   gemini      0.92      0.90    0.91        10

   Fold accuracies: 93%, 95%, 94%, 96%, 95%

✅ Classifier trained and saved!
   Mode: raw features (402-dim)
   Input dim: 402
```

---

### `test` - Test Backend Connection

Verify connectivity and generation with a backend.

```bash
llm-fingerprinter test [OPTIONS]
```

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model` | `-m` | - | Model name (optional) |
| `--prompt` | `-p` | "Hello! How are you?" | Test prompt |
| `--backend` | `-b` | `ollama` | LLM backend |
| `--endpoint` | `-e` | auto | API endpoint |
| `--api-key` | `-k` | env var | API key |

**Examples:**

```bash
# Test local Ollama
llm-fingerprinter test -b ollama --model llama3.2

# Test OpenAI
export OPENAI_API_KEY="sk-..."
llm-fingerprinter test -b openai --model gpt-4o

# Test with custom prompt
llm-fingerprinter test -b ollama --model llama3.2 -p "What is 2+2?"

# Test custom backend
llm-fingerprinter test -b custom -r ./custom_request.txt
```

---

### `fingerprint` - Generate Standalone Fingerprint

Generate a fingerprint without using the classifier (useful for analysis).

```bash
llm-fingerprinter fingerprint [OPTIONS]
```

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model` | `-m` | - | Model name (optional) |
| `--repeats` | - | 1 | Prompt repeats |
| `--output` | - | `./fingerprints` | Output directory |
| `--backend` | `-b` | `ollama` | LLM backend |
| `--endpoint` | `-e` | auto | API endpoint |
| `--api-key` | `-k` | env var | API key |

**Examples:**

```bash
# Generate and save fingerprint
llm-fingerprinter fingerprint -b ollama --model llama3.2

# With custom output directory
llm-fingerprinter fingerprint -b ollama --model llama3.2 --output ./my_fingerprints

# Multiple repeats for better accuracy
llm-fingerprinter fingerprint -b openai --model gpt-4o --repeats 3
```

---

### `list-models` - List Available Models

Show all models available on the backend.

```bash
llm-fingerprinter list-models [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--backend` | `-b` | LLM backend |
| `--endpoint` | `-e` | API endpoint |
| `--api-key` | `-k` | API key |

**Examples:**

```bash
# List Ollama models
llm-fingerprinter list-models -b ollama

# List OpenAI models
export OPENAI_API_KEY="sk-..."
llm-fingerprinter list-models -b openai

# Custom endpoint
llm-fingerprinter list-models -b openai -e https://api.groq.com/openai/v1 -k $GROQ_KEY
```

---

### `list-fingerprints` - List Saved Fingerprints

Show count of fingerprints by model family.

```bash
llm-fingerprinter list-fingerprints
```

**Output:**
```
📚 Fingerprints:

  gpt          15 ████████████████████
  llama        12 ████████████████
  gemini       10 ██████████████
  mistral       8 ███████████
  
  Total: 45

✅ Classifier trained (raw features, 402 dims)
```

---

### `info` - Show System Information

Display configuration, installed backends, available families, and status.

```bash
llm-fingerprinter info
```

**Output:**
```
⚙️  Config:
  Fingerprints: /folder/ fingerprints
  Embedding:    all-MiniLM-L6-v2 (384d)
  Total dims:   402 (384 + 12 + 6)

🔌 Backends:
  ollama:       http://localhost:11434
  ollama-cloud: https://api.ollama.ai
  openai:       https://api.openai.com/v1
  deepseek:     https://api.deepseek.com
  gemini:       https://generativelanguage.googleapis.com
  custom:       Via request template file (-r)

📋 Families: claude, deepseek, gemini, gemma, gpt, llama, mistral, qwen

📊 Status:
  Fingerprints: 45
  Classifier:   ✅ trained (raw features, 402 dims)

💡 Training options:
  train              # Use raw 402-dim features (default)
  train --use-pca    # Use PCA reduction (64 dims)
```

---

## Usage Workflow

### Complete Training Workflow

```bash
# 1. Generate fingerprints for GPT models
llm-fingerprinter simulate -b openai --model gpt-4 --family gpt --num-sims 5 --repeats 3
llm-fingerprinter simulate -b openai --model gpt-4o --family gpt --num-sims 5 --repeats 3

# 2. Generate fingerprints for LLaMA models
llm-fingerprinter simulate -b ollama --model llama3.2 --family llama --num-sims 5 --repeats 3
llm-fingerprinter simulate -b ollama --model llama2 --family llama --num-sims 5 --repeats 3

# 3. List all fingerprints
llm-fingerprinter list-fingerprints

# 4. Train classifier with cross-validation
llm-fingerprinter train --cross-validate

# 5. Test on unknown models
llm-fingerprinter identify -b ollama --model some-unknown-model
llm-fingerprinter identify -b openai --model gpt-4o-mini --repeats 3
```

### Quick Identification Workflow

```bash
# 1. Test connection
llm-fingerprinter test -b ollama --model llama3.2

# 2. Identify model
llm-fingerprinter identify -b ollama --model llama3.2

# 3. View results
llm-fingerprinter list-fingerprints
```

---

## Common Patterns

### Using Environment Variables for API Keys

```bash
# Set once, use multiple times
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="AIza..."

# No need to pass -k flag each time
llm-fingerprinter simulate -b openai --model gpt-4 --family gpt
llm-fingerprinter identify -b openai --model gpt-4o
llm-fingerprinter test -b gemini --model gemini-2.5-pro
```

### ⭐ Custom Backend with Request Template (Universal LLM Support)

The **custom backend** lets you use fingerprinting with **any** HTTP-based LLM API by providing a request template file.

```bash
# Use a request template file for custom APIs
llm-fingerprinter identify -b custom -r ./example/openai_request.txt

# Can also pass API key
llm-fingerprinter identify -b custom -r ./example/openai_request.txt -k "api-key-here"

# Generate training fingerprints
llm-fingerprinter simulate -b custom -r ./example/openai_request.txt --family gpt --num-sims 5

# Test connection
llm-fingerprinter test -b custom -r ./example/openai_request.txt

# See example templates in ./example/ directory:
# - openai_request.txt (OpenAI-compatible APIs)
# - ollama_cloud_request.txt
# - ollama_local_request.txt
```

**Why use custom backend?**
- 🔓 Support for proprietary/closed LLMs not in native backends
- 🏠 Self-hosted LLM servers behind HTTP endpoints
- 🔀 API proxies, gateways, and load balancers
- 🌐 Any HTTP-based LLM service (local or remote)
- 🎯 Complete control over request format

### Multi-Endpoint Configuration

```bash
# Test same model on different endpoints
llm-fingerprinter test -b openai -e https://api.openai.com/v1 --model gpt-4
llm-fingerprinter test -b openai -e https://api.groq.com/openai/v1 --model llama-3.1-70b -k $GROQ_KEY

# Identify via different providers
llm-fingerprinter identify -b openai --model gpt-4o
llm-fingerprinter identify -b openai -e https://my-proxy.com/v1 --model gpt-4o -k "proxy-key"
```

### Improving Accuracy

```bash
# Use higher repeats for more confident predictions
llm-fingerprinter identify -b ollama --model llama3.2 --repeats 5

# Train with more simulations per model
llm-fingerprinter simulate -b ollama --model llama3.2 --family llama --num-sims 10 --repeats 5

# Use PCA for faster training with slight accuracy trade-off
llm-fingerprinter train --use-pca --pca-components 128

# Cross-validate before deployment
llm-fingerprinter train --cross-validate --cv-folds 10
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


## 🔧 Custom Backend Deep Dive

The custom backend is the most powerful feature - it allows fingerprinting of **any** LLM accessible via HTTP, regardless of whether a native backend exists.

### How It Works

1. Create an HTTP request template file (JSON format)
2. Include placeholders for `model` and `prompt`
3. Pass template to fingerprinter with `-b custom -r ./template.txt`
4. The system automatically sends requests and analyzes responses

### Example: Creating a Custom Template

```json
{
  "url": "https://api.example.com/v1/completions",
  "method": "POST",
  "headers": {
    "Content-Type": "application/json",
    "Authorization": "Bearer {api_key}"
  },
  "body": {
    "model": "{model}",
    "prompt": "{prompt}",
    "max_tokens": 200,
    "temperature": 0.7
  }
}
```

### Usage Examples

```bash
# Create your template file
cat > my_llm_template.txt << 'EOF'
{
  "url": "https://my-llm.com/api/generate",
  "method": "POST",
  "headers": {
    "Authorization": "Bearer your-key"
  },
  "body": {
    "model": "{model}",
    "prompt": "{prompt}",
    "max_tokens": 200
  }
}
EOF

# Identify models
llm-fingerprinter identify -b custom -r ./my_llm_template.txt

# Generate training fingerprints
llm-fingerprinter simulate -b custom -r ./my_llm_template.txt --family gpt --num-sims 5

# Test connectivity
llm-fingerprinter test -b custom -r ./my_llm_template.txt

# Pass API key via environment or CLI
export CUSTOM_API_KEY="your-secret-key"
llm-fingerprinter identify -b custom -r ./my_llm_template.txt

# Or pass directly
llm-fingerprinter identify -b custom -r ./my_llm_template.txt -k "your-secret-key"
```

### Supported Template Placeholders

| Placeholder | Description | Example |
|-------------|-------------|---------|
| `{model}` | Model name passed via CLI | `gpt-4`, `llama3.2` |
| `{prompt}` | The fingerprinting prompt | (automatically populated) |
| `{api_key}` | API key from environment or CLI | (injected automatically) |

### Pre-built Examples

See `./example/` directory for ready-to-use templates:
- **openai_request.txt** - OpenAI, Groq, and compatible APIs
- **ollama_cloud_request.txt** - Ollama Cloud
- **ollama_local_request.txt** - Local Ollama

Copy and adapt these for your use case!

---

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

