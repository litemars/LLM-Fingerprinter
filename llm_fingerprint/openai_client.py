import logging
import time
from openai import OpenAI
import openai as openai_module

logger = logging.getLogger(__name__)

class OpenAIError(Exception):
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


class OpenAIClient:
    """Client for OpenAI API using official openai SDK.
    
    Works with:
    - OpenAI API (api.openai.com)
    - Azure OpenAI
    - Any OpenAI-compatible endpoint
    
    Install: pip install openai
    """
    
    def __init__(self, 
                 api_key,
                 endpoint = "https://api.openai.com/v1",
                 timeout = 60,
                 max_retries = 3,
                 organization = None):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key
            endpoint: API endpoint URL (default: OpenAI)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            organization: Optional organization ID for OpenAI
        """
        
        self.api_key = api_key
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.organization = organization
        
        # Initialize the official OpenAI client
        self.client = OpenAI(
            api_key=api_key,
            base_url=endpoint,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
        )
        
        # Cache connectivity status
        self._last_health_check = None
        self._health_check_interval = 30
        self._is_healthy: bool = False
        
        logger.info(f"Initialized OpenAIClient for {endpoint}")
    
    def _check_connectivity(self, force: bool = False) -> bool:
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
    
    def generate(self, model, prompt, temperature = 0.7, max_tokens = 512, system = None) -> str:
        """
        Generate completion from model using chat completions API.
        
        Args:
            model: Model name (e.g., 'gpt-4o', 'gpt-4o-mini')
            prompt: Input prompt
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum response length in tokens
            system: Optional system prompt
        
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
                raise OpenAIGenerationError("No choices in response")
            
            text = response.choices[0].message.content
            text = text.strip() if text else ""
            
            # Log performance metrics
            usage = response.usage
            completion_tokens = usage.completion_tokens if usage else 0
            
            logger.debug(f"Generated {len(text)} chars, {completion_tokens} tokens in {elapsed:.2f}s")
            return text
            
        except openai_module.AuthenticationError:
            raise OpenAIAuthError("Invalid API key")
        except openai_module.PermissionDeniedError:
            raise OpenAIAuthError("Access forbidden - check API key permissions")
        except openai_module.NotFoundError:
            raise OpenAIGenerationError(f"Model '{model}' not found")
        except openai_module.RateLimitError:
            raise OpenAIGenerationError("Rate limit exceeded - please wait and retry")
        except openai_module.APIConnectionError as e:
            logger.error(f"Connection error to API: {e}")
            raise OpenAIConnectionError(f"Cannot connect to API at {self.endpoint}")
        except openai_module.APITimeoutError:
            logger.warning(f"Timeout querying {model} after {self.timeout}s")
            raise OpenAIConnectionError(f"Request timeout after {self.timeout}s")
        except OpenAIError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error generating from {model}: {e}")
            raise OpenAIGenerationError(f"Generation failed: {e}")
    
    def list_models(self):
        """
        List available models from the API.
        
        Returns:
            List of model IDs, empty list on error
        """
        try:
            models = [model.id for model in self.client.models.list()]
            # Sort to show GPT models first
            models.sort(key=lambda x: (0 if 'gpt' in x.lower() else 1, x))
            logger.info(f"Found {len(models)} models on API")
            return models
            
        except openai_module.AuthenticationError:
            logger.error("Authentication failed - check API key")
            return []
        except openai_module.APIConnectionError:
            logger.error(f"Cannot connect to API at {self.endpoint}")
            return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
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
                "created": model_obj.created,
                "owned_by": model_obj.owned_by,
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
        logger.debug("Closed OpenAIClient")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False