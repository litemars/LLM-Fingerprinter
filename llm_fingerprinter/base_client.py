"""Abstract base class for all LLM API clients.

All backend clients (Ollama, OpenAI, DeepSeek, Gemini, etc.) must inherit
from BaseClient and implement its abstract methods. This enforces a
consistent interface across backends and enables type-safe usage in the
fingerprinting pipeline.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import List, Optional

logger = logging.getLogger(__name__)


class ClientError(Exception):
    """Base exception for all client errors."""
    pass


class ConnectionError(ClientError):
    """Raised when connection to the API fails."""
    pass


class GenerationError(ClientError):
    """Raised when text generation fails."""
    pass


class AuthError(ClientError):
    """Raised when authentication fails."""
    pass


class BaseClient(ABC):
    """Abstract base class that all LLM backend clients must implement.

    Provides:
    - A consistent interface for generate(), list_models(), _check_connectivity()
    - Built-in health check caching (configurable interval)
    - Context manager support (__enter__/__exit__)
    """

    def __init__(self, timeout: int = 60, max_retries: int = 3,
                 health_check_interval: int = 30):
        self.timeout = timeout
        self.max_retries = max_retries
        self._last_health_check: Optional[float] = None
        self._health_check_interval = health_check_interval
        self._is_healthy = False

    def _check_connectivity(self, force: bool = False) -> bool:
        """Check if API is reachable, with caching.

        Subclasses must implement _perform_health_check() which does
        the actual connectivity test. This method handles caching.

        Args:
            force: If True, bypass the cache and check now.

        Returns:
            True if the API is reachable, False otherwise.
        """
        now = time.time()

        if not force and self._last_health_check is not None:
            if now - self._last_health_check < self._health_check_interval:
                return self._is_healthy

        self._is_healthy = self._perform_health_check()
        self._last_health_check = now
        return self._is_healthy

    @abstractmethod
    def _perform_health_check(self) -> bool:
        """Perform the actual connectivity/health check.

        Returns:
            True if API is reachable and healthy.
        """
        ...

    @abstractmethod
    def generate(self, model: Optional[str], prompt: str,
                 temperature: float = 0.7, max_tokens: int = 512,
                 system: Optional[str] = None) -> str:
        """Generate a text completion from the model.

        Args:
            model: Model identifier (backend-specific).
            prompt: The user prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.
            system: Optional system prompt.

        Returns:
            The generated text response.

        Raises:
            AuthError: If authentication fails.
            ConnectionError: If the API is unreachable.
            GenerationError: If generation fails for any other reason.
        """
        ...

    @abstractmethod
    def list_models(self) -> List[str]:
        """List available models on this backend.

        Returns:
            A list of model identifiers, or an empty list on error.
        """
        ...

    def close(self) -> None:
        """Release resources held by this client.

        Override in subclasses that hold HTTP sessions or SDK clients.
        """
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
