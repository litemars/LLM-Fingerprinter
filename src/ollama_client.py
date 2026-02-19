import requests
import logging
import time
from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.base_client import BaseClient, ClientError, ConnectionError as BaseConnectionError, GenerationError, AuthError

logger = logging.getLogger(__name__)


class OllamaError(ClientError):
    """Base exception for Ollama client errors."""
    pass


class OllamaConnectionError(OllamaError):
    """Raised when connection to Ollama fails."""
    pass


class OllamaGenerationError(OllamaError):
    """Raised when generation fails."""
    pass


class OllamaClient(BaseClient):

    def __init__(self, endpoint: str = "http://localhost:11434",
                 timeout: int = 60,
                 max_retries: int = 3):

        super().__init__(timeout=timeout, max_retries=max_retries)
        self.endpoint = endpoint.rstrip("/")

        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=0
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

        logger.info(f"Initialized OllamaClient for {endpoint}")

    def _perform_health_check(self) -> bool:
        try:
            resp = self.session.get(
                f"{self.endpoint}/api/tags",
                timeout=10
            )
            return resp.status_code == 200
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error to Ollama at {self.endpoint}: {e}")
            return False
        except requests.exceptions.Timeout:
            logger.error(f"Timeout connecting to Ollama at {self.endpoint}")
            return False
        except Exception as e:
            logger.error(f"Error checking Ollama connectivity: {e}")
            return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError))
    )
    def generate(self, model, prompt, temperature=0.7, max_tokens=512, system=None):

        url = f"{self.endpoint}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        if system:
            payload["system"] = system

        try:
            start = time.time()
            response = self.session.post(url, json=payload, timeout=self.timeout)
            elapsed = time.time() - start

            if response.status_code == 200:
                result = response.json()
                text = result.get("response", "").strip()

                eval_count = result.get("eval_count", 0)
                eval_duration = result.get("eval_duration", 0)
                tokens_per_sec = eval_count / (eval_duration / 1e9) if eval_duration > 0 else 0

                logger.debug(f"Generated {len(text)} chars, {eval_count} tokens "
                           f"in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)")
                return text
            elif response.status_code == 404:
                raise OllamaGenerationError(f"Model '{model}' not found on Ollama")
            else:
                error_msg = response.text[:200] if response.text else "Unknown error"
                raise OllamaGenerationError(
                    f"Ollama error {response.status_code}: {error_msg}"
                )

        except requests.Timeout:
            logger.warning(f"Timeout querying {model} after {self.timeout}s")
            raise
        except requests.ConnectionError as e:
            logger.error(f"Connection error to Ollama: {e}")
            raise OllamaConnectionError(f"Cannot connect to Ollama at {self.endpoint}")
        except OllamaError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error generating from {model}: {e}")
            raise OllamaGenerationError(f"Generation failed: {e}")

    def list_models(self) -> List[str]:
        try:
            url = f"{self.endpoint}/api/tags"
            response = self.session.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                logger.info(f"Found {len(models)} models on Ollama")
                return models
            else:
                logger.error(f"Failed to list models: HTTP {response.status_code}")
                return []
        except requests.ConnectionError:
            logger.error(f"Cannot connect to Ollama at {self.endpoint}")
            return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    def pull_model(self, model: str, stream_progress: bool = True):
        try:
            url = f"{self.endpoint}/api/pull"
            response = self.session.post(
                url,
                json={"name": model, "stream": stream_progress},
                timeout=None,
                stream=stream_progress
            )

            if stream_progress:
                for line in response.iter_lines():
                    if line:
                        import json
                        data = json.loads(line)
                        status = data.get("status", "")
                        if "pulling" in status or "downloading" in status:
                            completed = data.get("completed", 0)
                            total = data.get("total", 0)
                            if total > 0:
                                pct = completed / total * 100
                                logger.info(f"Pulling {model}: {pct:.1f}%")
                        elif status == "success":
                            logger.info(f"Successfully pulled {model}")
                            return True

            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error pulling model {model}: {e}")
            return False

    def close(self):
        """Close the HTTP session."""
        self.session.close()
        logger.debug("Closed OllamaClient session")
