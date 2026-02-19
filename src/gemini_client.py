import logging
import time
from typing import List, Optional

from src.base_client import BaseClient, ClientError

logger = logging.getLogger(__name__)


class GeminiError(ClientError):
    """Base exception for Gemini client errors."""
    pass


class GeminiConnectionError(GeminiError):
    """Raised when connection to Gemini API fails."""
    pass


class GeminiGenerationError(GeminiError):
    """Raised when generation fails."""
    pass


class GeminiAuthError(GeminiError):
    """Raised when authentication fails."""
    pass


class GeminiClient(BaseClient):

    def __init__(self,
                 api_key: str,
                 endpoint: Optional[str] = None,  # Not used, kept for API compatibility
                 timeout: int = 60,
                 max_retries: int = 3):
        super().__init__(timeout=timeout, max_retries=max_retries)

        # Lazy import - only import when class is instantiated
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise ImportError(
                "google-genai package is required. Install with: pip install google-genai"
            ) from e

        self._genai = genai
        self._types = types

        self.api_key = api_key

        self.client = genai.Client(api_key=api_key)

        logger.info("Initialized GeminiClient with google-genai SDK")

    def _perform_health_check(self) -> bool:
        try:
            list(self.client.models.list())
            return True
        except Exception as e:
            error_str = str(e).lower()
            if "api key" in error_str or "authentication" in error_str or "401" in error_str:
                logger.warning("API reachable but authentication failed - check API key")
            elif "403" in error_str or "forbidden" in error_str:
                logger.warning("API reachable but access forbidden - check permissions")
            else:
                logger.error(f"Error checking Gemini API connectivity: {e}")
            return False

    def generate(self, model, prompt, temperature=0.7,
                 max_tokens=512, system=None):
        start = time.time()

        try:
            config = self._types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

            if system:
                config.system_instruction = system

            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )

            elapsed = time.time() - start

            if not response.candidates:
                if response.prompt_feedback:
                    block_reason = getattr(response.prompt_feedback, 'block_reason', None)
                    if block_reason:
                        raise GeminiGenerationError(f"Content blocked: {block_reason}")
                raise GeminiGenerationError("No candidates in response")

            text = response.text.strip() if response.text else ""

            usage = getattr(response, 'usage_metadata', None)
            output_tokens = getattr(usage, 'candidates_token_count', 0) if usage else 0

            logger.debug(f"Generated {len(text)} chars, {output_tokens} tokens in {elapsed:.2f}s")
            return text

        except GeminiError:
            raise
        except Exception as e:
            error_str = str(e).lower()

            if "api key" in error_str or "authentication" in error_str or "401" in error_str:
                raise GeminiAuthError("Invalid API key")
            elif "403" in error_str or "forbidden" in error_str:
                raise GeminiAuthError("Access forbidden - check API key permissions")
            elif "404" in error_str or "not found" in error_str:
                raise GeminiGenerationError(f"Model '{model}' not found")
            elif "429" in error_str or "rate limit" in error_str or "quota" in error_str:
                raise GeminiGenerationError("Rate limit exceeded - please wait and retry")
            elif "timeout" in error_str:
                raise GeminiConnectionError(f"Request timeout after {self.timeout}s")
            else:
                logger.error(f"Unexpected error generating from {model}: {e}")
                raise GeminiGenerationError(f"Generation failed: {e}")

    def list_models(self) -> List[str]:
        try:
            models = []
            for model in self.client.models.list():
                name = model.name

                if name.startswith("models/"):
                    name = name[7:]

                supported_methods = getattr(model, 'supported_generation_methods', [])
                if 'generateContent' in supported_methods:
                    models.append(name)

            logger.info(f"Found {len(models)} Gemini models")
            return sorted(models)

        except Exception as e:
            error_str = str(e).lower()
            if "api key" in error_str or "authentication" in error_str:
                logger.error("Authentication failed - check API key")
            else:
                logger.error(f"Error listing models: {e}")
            return []

    def model_info(self, model):
        try:
            if not model.startswith("models/"):
                model_path = f"models/{model}"
            else:
                model_path = model

            model_obj = self.client.models.get(model=model_path)

            return {
                "name": model_obj.name,
                "display_name": getattr(model_obj, 'display_name', None),
                "description": getattr(model_obj, 'description', None),
                "input_token_limit": getattr(model_obj, 'input_token_limit', None),
                "output_token_limit": getattr(model_obj, 'output_token_limit', None),
                "supported_generation_methods": getattr(model_obj, 'supported_generation_methods', []),
            }

        except Exception as e:
            error_str = str(e).lower()
            if "404" in error_str or "not found" in error_str:
                logger.warning(f"Model '{model}' not found")
            else:
                logger.error(f"Error getting model info for {model}: {e}")
            return None

    def close(self):
        logger.debug("Closed GeminiClient")
