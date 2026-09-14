# LLM Fingerprinting System

[![PyPI version](https://badge.fury.io/py/llm-fingerprinter.svg)](https://pypi.org/project/llm-fingerprinter/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A black-box fingerprinting system that identifies the underlying LLM model family (GPT, LLaMA, Mistral, etc.) by analysing response patterns across 31 carefully selected prompts. The system can identify fine-tuned models as well, tracing them back to their foundational base model.

**Note: Check `config.py` to see all identifiable model families.**

A pre-trained classifier is bundled with the package in the `model/` directory.

**Connect your own endpoint.** Use the custom backend with a request template for your chatbot, self-hosted model, or API gateway:

```bash
llm-fingerprinter identify -b custom -r ./custom_request.txt
```

[Set up a custom integration](#custom-endpoint-integration) with your endpoint's URL and JSON request body.

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
| `custom` | **Your own HTTP endpoint accepting JSON POST requests** | Optional |

### Custom Endpoint Integration

**Use your existing API without writing a new backend client.** The custom integration sends the fingerprinting prompts through a JSON POST request that you define. It works with compatible proprietary APIs, self-hosted endpoints, chatbots, proxies and gateways.

**1. Create `custom_request.txt`.** Put the full endpoint URL on the first line, followed by the JSON body your API expects. Use `"$PROMPT$"` for the field that receives each fingerprinting prompt:

```text
https://your-endpoint.example/chat
{
  "message": "$PROMPT$"
}
```

Replace the example URL and adapt the JSON fields to your API. For example, an API may require `prompt` instead of `message`, a `messages` array, or a fixed deployment name. The file contains only the URL and JSON template; do not include comments, HTTP headers or a `curl` command.

**2. Run identification.** The tool substitutes each prompt, sends it to your endpoint, extracts the returned text and builds its fingerprint:

```bash
llm-fingerprinter identify -b custom -r ./custom_request.txt
```

`-b custom` selects the custom backend; `-r` selects the request file. The URL in that file determines the endpoint, so `--endpoint` does not override it. If your endpoint already selects its model, you can omit `--model`.

**3. Add authentication if required.** For an endpoint using bearer-token authentication:

```bash
export CUSTOM_API_KEY="your-api-key"
llm-fingerprinter identify -b custom -r ./custom_request.txt -k "$CUSTOM_API_KEY"
```

`-k` sends `Authorization: Bearer <key>`. Pass the variable explicitly as shown; the custom backend does not automatically read `CUSTOM_API_KEY`.

**Optional template placeholders** let you adapt the request without changing application code:

| Placeholder | Value |
|-------------|-------|
| `"$PROMPT$"` | Required: the current fingerprinting prompt |
| `"$MODEL$"` | Model/deployment name supplied with `--model` |
| `$TEMPERATURE$` | Requested sampling temperature; leave this placeholder unquoted |
| `$MAX_TOKENS$` | Requested output-token limit; leave this placeholder unquoted |
| `"$SYSTEM$"` | Optional system text; empty by default in the CLI |

Include only fields your endpoint supports. Hardcode a required system message in the JSON template. If you use `"$MODEL$"`, supply `--model your-deployment-name` when running the command.

**Supported responses:** common JSON text fields such as `response`, `text`, `answer` and `choices[0].message.content`, plus SSE, NDJSON and responses declared as `Content-Type: text/plain`. Streamed spaces and newlines are preserved; malformed streams and explicit error events are rejected instead of becoming answer text. Unusual JSON response paths, custom authentication headers and headerless plaintext require configuration through the Python `CustomClient` API.

Request-file examples: [local Ollama](example/ollama_local_request.txt), [Ollama Cloud](example/ollama_cloud_request.txt), and [OpenAI-compatible chat](example/openai_request.txt). The OpenAI-compatible example uses `"$MODEL$"`, so pass `--model` as well as any required API key.

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

**Custom endpoint — bring your own API:**

```bash
llm-fingerprinter identify -b custom -r ./custom_request.txt
```

See [Custom Endpoint Integration](#custom-endpoint-integration) to create the request file or add authentication.

**Built-in backends:**

```bash
# Local Ollama
llm-fingerprinter identify -b ollama --model llama3.2

# OpenAI
export OPENAI_API_KEY="your-key"
llm-fingerprinter identify -b openai --model gpt-4o-mini
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
llm-fingerprinter add-family --model deepseek-chat --family deepseek --num-fps 3 -b deepseek
```

Recommended minimum: 3 fingerprints for a reliable mean template.

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
