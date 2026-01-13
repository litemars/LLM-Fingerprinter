import logging
import time
from openai import OpenAI
import openai as openai_module

logger = logging.getLogger(__name__)


class DeepSeekError(Exception):
    """Base exception for DeepSeek client errors."""
    pass


class DeepSeekConnectionError(DeepSeekError):
    """Raised when connection to DeepSeek API fails."""
    pass


class DeepSeekGenerationError(DeepSeekError):
    """Raised when generation fails."""
    pass


class DeepSeekAuthError(DeepSeekError):
    """Raised when authentication fails."""
    pass


class DeepSeekClient:    
    
    def __init__(self, 
                 api_key: str,
                 endpoint: str = None,
                 timeout: int = 60,
                 max_retries: int = 3):
        """
        Initialize DeepSeek client.
        
        Args:
            api_key: DeepSeek API key
            endpoint: API endpoint URL (default: DeepSeek API)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        
        self.api_key = api_key
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Initialize OpenAI client with DeepSeek endpoint
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.endpoint,
            timeout=timeout,
            max_retries=max_retries,
        )
        
        # Cache connectivity status
        self._last_health_check = None
        self._health_check_interval = 30
        self._is_healthy = False
        
        logger.info(f"Initialized DeepSeekClient for {self.endpoint}")
    
    def _check_connectivity(self, force = False):
        """
        Check if API is reachable.
        
        Args:
            force: Force check even if cached result is recent
            
        Returns:
            True if API is reachable, False otherwise
        """
        now = time.time()
        
        if not force and self._last_health_check is not None:
            if now - self._last_health_check < self._health_check_interval:
                return self._is_healthy
        
        try:
            # Try to list models as health check
            list(self.client.models.list())
            self._is_healthy = True
            self._last_health_check = now
            return True
            
        except openai_module.AuthenticationError:
            logger.warning("API reachable but authentication failed - check API key")
            self._is_healthy = False
            self._last_health_check = now
            return False
        except openai_module.PermissionDeniedError:
            logger.warning("API reachable but access forbidden - check permissions")
            self._is_healthy = False
            self._last_health_check = now
            return False
        except openai_module.APIConnectionError as e:
            logger.error(f"Connection error to API at {self.endpoint}: {e}")
            self._is_healthy = False
            self._last_health_check = now
            return False
        except Exception as e:
            logger.error(f"Error checking API connectivity: {e}")
            self._is_healthy = False
            self._last_health_check = now
            return False
    
    def generate(self, model, prompt, system = None):
        """
        Generate completion from model.
        
        Args:
            model: Model name (e.g., 'deepseek-chat', 'deepseek-coder', 'deepseek-reasoner')
            prompt: Input prompt
        
        Returns:
            Generated text
        """
        start = time.time()
        
        try:
            # Build messages
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            
            # Generate completion
            response = self.client.chat.completions.create(
                model=model,
                messages=messages
            )
            
            elapsed = time.time() - start
            
            # Extract text from response
            if not response.choices:
                raise DeepSeekGenerationError("No choices in response")
            
            text = response.choices[0].message.content
            text = text.strip() if text else ""
            
            # Log performance metrics
            usage = response.usage
            completion_tokens = usage.completion_tokens if usage else 0
            
            logger.debug(f"Generated {len(text)} chars, {completion_tokens} tokens in {elapsed:.2f}s")
            return text
            
        except openai_module.AuthenticationError:
            raise DeepSeekAuthError("Invalid API key")
        except openai_module.PermissionDeniedError:
            raise DeepSeekAuthError("Access forbidden - check API key permissions")
        except openai_module.NotFoundError:
            raise DeepSeekGenerationError(f"Model '{model}' not found")
        except openai_module.RateLimitError:
            raise DeepSeekGenerationError("Rate limit exceeded - please wait and retry")
        except openai_module.APIConnectionError as e:
            logger.error(f"Connection error to API: {e}")
            raise DeepSeekConnectionError(f"Cannot connect to API at {self.endpoint}")
        except openai_module.APITimeoutError:
            logger.warning(f"Timeout querying {model} after {self.timeout}s")
            raise DeepSeekConnectionError(f"Request timeout after {self.timeout}s")
        except DeepSeekError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error generating from {model}: {e}")
            raise DeepSeekGenerationError(f"Generation failed: {e}")
    
    def list_models(self):
        """
        List available models from the API.
        
        Returns:
            List of model IDs, empty list on error
        """
        try:
            models = [model.id for model in self.client.models.list()]
            # Sort by name
            models.sort()
            logger.info(f"Found {len(models)} models on DeepSeek API")
            return models
            
        except openai_module.AuthenticationError:
            logger.error("Authentication failed - check API key")
            return []
        except openai_module.APIConnectionError:
            logger.error(f"Cannot connect to API at {self.endpoint}")
            return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            # Return known models as fallback
            return ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"]
    
    def model_info(self, model):
        """
        Get model information from the API.
        
        Args:
            model: Model ID
            
        Returns:
            Model info dict or None on error
        """
        try:
            model_obj = self.client.models.retrieve(model)
            return {
                "id": model_obj.id,
                "object": model_obj.object,
                "created": getattr(model_obj, 'created', None),
                "owned_by": getattr(model_obj, 'owned_by', 'deepseek'),
            }
        except openai_module.NotFoundError:
            logger.warning(f"Model '{model}' not found")
            return None
        except Exception as e:
            logger.error(f"Error getting model info for {model}: {e}")
            return None
    
    def close(self):
        """Close the client."""
        self.client.close()
        logger.debug("Closed DeepSeekClient")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False