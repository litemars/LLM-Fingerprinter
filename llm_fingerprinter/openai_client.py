import logging
import time
from typing import List, Optional

from llm_fingerprinter.base_client import BaseClient, ClientError

logger = logging.getLogger(__name__)

# Reasoning models (o1/o3/o4, gpt-5 family) spend hidden reasoning tokens BEFORE
# the visible answer, and `max_completion_tokens` bounds reasoning + output
# combined. Capping at the visible target alone would let reasoning consume the
# whole budget and return an empty/truncated answer, so we add this headroom on
# top of the requested visible length for reasoning models only.
REASONING_TOKEN_BUFFER = 4096


class OpenAIError(ClientError):
    """Base exception for OpenAI client errors."""
    pass


class OpenAIConnectionError(OpenAIError):
    """Raised when connection to OpenAI API fails."""
    pass


class OpenAIGenerationError(OpenAIError):
    """Raised when generation fails."""
    pass


class OpenAIAuthError(OpenAIError):
    """Raised when authentication fails."""
    pass


class OpenAIClient(BaseClient):
    """Client for OpenAI API using official openai SDK.

    Works with:
    - OpenAI API (api.openai.com)
    - Azure OpenAI
    - Any OpenAI-compatible endpoint

    Install: pip install openai
    """

    def __init__(self,
                 api_key: str,
                 endpoint: str = "https://api.openai.com/v1",
                 timeout: int = 60,
                 max_retries: int = 3,
                 organization: Optional[str] = None):
        super().__init__(timeout=timeout, max_retries=max_retries)

        # Lazy import - only import when class is instantiated
        try:
            from openai import OpenAI
            import openai as openai_module
        except ImportError as e:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            ) from e

        self._openai_module = openai_module

        self.api_key = api_key
        self.endpoint = endpoint.rstrip("/")
        self.organization = organization

        self.client = OpenAI(
            api_key=api_key,
            base_url=endpoint,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
        )

        logger.info(f"Initialized OpenAIClient for {endpoint}")

    def _perform_health_check(self) -> bool:
        try:
            list(self.client.models.list())
            return True
        except self._openai_module.AuthenticationError:
            logger.warning("API reachable but authentication failed - check API key")
            return False
        except self._openai_module.PermissionDeniedError:
            logger.warning("API reachable but access forbidden - check permissions")
            return False
        except self._openai_module.APIConnectionError as e:
            logger.error(f"Connection error to API at {self.endpoint}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error checking API connectivity: {e}")
            return False

    @staticmethod
    def _is_reasoning_model(model: Optional[str]) -> bool:
        """Reasoning models (o1/o3/o4 and the gpt-5 family) have different rules:
        they reject the `temperature` parameter (only the default 1.0 is allowed)
        and require `max_completion_tokens` instead of the legacy `max_tokens`.
        """
        if not model:
            return False
        m = model.lower().lstrip()
        return m.startswith(("o1", "o3", "o4")) or m.startswith("gpt-5")

    def generate(self, model, prompt, temperature=0.7, max_tokens=512, system=None):
        start = time.time()

        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            # Forward sampling controls. Reasoning models need a different
            # parameter set, so branch on the model name.
            if self._is_reasoning_model(model):
                params = {
                    "model": model,
                    "messages": messages,
                    # Headroom so hidden reasoning tokens don't starve the visible
                    # answer (see REASONING_TOKEN_BUFFER note above).
                    "max_completion_tokens": max_tokens + REASONING_TOKEN_BUFFER,
                    # temperature intentionally omitted — only default is allowed
                }
            else:
                params = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }

            try:
                response = self.client.chat.completions.create(**params)
            except self._openai_module.BadRequestError as e:
                # Some models / OpenAI-compatible endpoints reject temperature
                # or max_tokens. Retry once with the most compatible param set
                # (drop temperature, use max_completion_tokens).
                msg = str(e).lower()
                if any(k in msg for k in ("temperature", "max_tokens", "max_completion_tokens")):
                    logger.warning(f"{model}: retrying without unsupported sampling "
                                   f"params ({e})")
                    # Apply the reasoning headroom here too so the fallback path
                    # doesn't starve reasoning models' visible output.
                    retry_budget = max_tokens + (
                        REASONING_TOKEN_BUFFER if self._is_reasoning_model(model) else 0
                    )
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_completion_tokens=retry_budget,
                    )
                else:
                    raise

            elapsed = time.time() - start

            if not response.choices:
                raise OpenAIGenerationError("No choices in response")

            text = response.choices[0].message.content
            text = text.strip() if text else ""

            usage = response.usage
            completion_tokens = usage.completion_tokens if usage else 0

            logger.debug(f"Generated {len(text)} chars, {completion_tokens} tokens in {elapsed:.2f}s")
            return text

        except self._openai_module.AuthenticationError:
            raise OpenAIAuthError("Invalid API key")
        except self._openai_module.PermissionDeniedError:
            raise OpenAIAuthError("Access forbidden - check API key permissions")
        except self._openai_module.NotFoundError:
            raise OpenAIGenerationError(f"Model '{model}' not found")
        except self._openai_module.RateLimitError:
            raise OpenAIGenerationError("Rate limit exceeded - please wait and retry")
        except self._openai_module.APIConnectionError as e:
            logger.error(f"Connection error to API: {e}")
            raise OpenAIConnectionError(f"Cannot connect to API at {self.endpoint}")
        except self._openai_module.APITimeoutError:
            logger.warning(f"Timeout querying {model} after {self.timeout}s")
            raise OpenAIConnectionError(f"Request timeout after {self.timeout}s")
        except OpenAIError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error generating from {model}: {e}")
            raise OpenAIGenerationError(f"Generation failed: {e}")

    def list_models(self) -> List[str]:
        try:
            models = [model.id for model in self.client.models.list()]
            models.sort(key=lambda x: (0 if 'gpt' in x.lower() else 1, x))
            logger.info(f"Found {len(models)} models on API")
            return models
        except self._openai_module.AuthenticationError:
            logger.error("Authentication failed - check API key")
            return []
        except self._openai_module.APIConnectionError:
            logger.error(f"Cannot connect to API at {self.endpoint}")
            return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    def model_info(self, model):
        try:
            model_obj = self.client.models.retrieve(model)
            return {
                "id": model_obj.id,
                "object": model_obj.object,
                "created": model_obj.created,
                "owned_by": model_obj.owned_by,
            }
        except self._openai_module.NotFoundError:
            logger.warning(f"Model '{model}' not found")
            return None
        except Exception as e:
            logger.error(f"Error getting model info for {model}: {e}")
            return None

    def close(self):
        self.client.close()
        logger.debug("Closed OpenAIClient")
