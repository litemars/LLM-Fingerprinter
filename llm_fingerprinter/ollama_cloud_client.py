import requests
import logging
import time
from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from llm_fingerprinter.base_client import BaseClient, ClientError

logger = logging.getLogger(__name__)


class OllamaCloudError(ClientError):
    """Base exception for Ollama Cloud client errors."""
    pass


class OllamaCloudConnectionError(OllamaCloudError):
    """Raised when connection to Ollama Cloud fails."""
    pass


class OllamaCloudGenerationError(OllamaCloudError):
    """Raised when generation fails."""
    pass


class OllamaCloudAuthError(OllamaCloudError):
    """Raised when authentication fails."""
    pass


class OllamaCloudClient(BaseClient):

    def __init__(self,
                 api_key: str,
                 endpoint: str = "https://api.ollama.com/v1",
                 timeout: int = 60,
                 max_retries: int = 3):
        super().__init__(timeout=timeout, max_retries=max_retries)

        self.api_key = api_key
        self.endpoint = endpoint.rstrip("/")

        # Session with connection pooling
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=0
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

        logger.info(f"Initialized OllamaCloudClient for {endpoint}")

    def _perform_health_check(self) -> bool:
        try:
            resp = self.session.get(
                f"{self.endpoint}/api/tags",
                timeout=10
            )

            if resp.status_code == 401:
                logger.warning("API reachable but authentication failed - check API key")
            elif resp.status_code == 403:
                logger.warning("API reachable but access forbidden - check permissions")

            return resp.status_code == 200

        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error to Ollama Cloud at {self.endpoint}: {e}")
            return False
        except requests.exceptions.Timeout:
            logger.error(f"Timeout connecting to Ollama Cloud at {self.endpoint}")
            return False
        except Exception as e:
            logger.error(f"Error checking Ollama Cloud connectivity: {e}")
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
                "temperature": temperature
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

            elif response.status_code == 401:
                raise OllamaCloudAuthError("Invalid API key")
            elif response.status_code == 403:
                raise OllamaCloudAuthError("Access forbidden - check API key permissions")
            elif response.status_code == 404:
                raise OllamaCloudGenerationError(f"Model '{model}' not found")
            elif response.status_code == 429:
                raise OllamaCloudGenerationError("Rate limit exceeded - please wait and retry")
            else:
                error_msg = response.text[:200] if response.text else "Unknown error"
                try:
                    error_json = response.json()
                    error_msg = error_json.get("error", error_msg)
                except (ValueError, KeyError):
                    pass
                raise OllamaCloudGenerationError(
                    f"API error {response.status_code}: {error_msg}"
                )

        except requests.Timeout:
            logger.warning(f"Timeout querying {model} after {self.timeout}s")
            raise
        except requests.ConnectionError as e:
            logger.error(f"Connection error to Ollama Cloud: {e}")
            raise OllamaCloudConnectionError(f"Cannot connect to Ollama Cloud at {self.endpoint}")
        except OllamaCloudError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error generating from {model}: {e}")
            raise OllamaCloudGenerationError(f"Generation failed: {e}")

    def list_models(self) -> List[str]:
        try:
            url = f"{self.endpoint}/api/tags"
            response = self.session.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                logger.info(f"Found {len(models)} models on Ollama Cloud")
                return models
            elif response.status_code == 401:
                logger.error("Authentication failed - check API key")
                return []
            else:
                logger.error(f"Failed to list models: HTTP {response.status_code}")
                return []
        except requests.ConnectionError:
            logger.error(f"Cannot connect to Ollama Cloud at {self.endpoint}")
            return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    def close(self):
        """Close the HTTP session."""
        self.session.close()
        logger.debug("Closed OllamaCloudClient session")
