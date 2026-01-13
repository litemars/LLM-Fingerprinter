import logging
import time
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GeminiError(Exception):
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


class GeminiClient:
    
    def __init__(self, 
                 api_key,
                 endpoint = None,  # Not used, kept for API compatibility
                 timeout = 60,
                 max_retries = 3):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Google AI API key
            endpoint: Not used (kept for API compatibility)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Initialize the client
        self.client = genai.Client(api_key=api_key)
        
        # Cache connectivity status
        self._last_health_check = None
        self._health_check_interval = 30
        self._is_healthy = False
        
        logger.info("Initialized GeminiClient with google-genai SDK")
    
    def _check_connectivity(self, force = False):
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
            
        except Exception as e:
            error_str = str(e).lower()
            if "api key" in error_str or "authentication" in error_str or "401" in error_str:
                logger.warning("API reachable but authentication failed - check API key")
            elif "403" in error_str or "forbidden" in error_str:
                logger.warning("API reachable but access forbidden - check permissions")
            else:
                logger.error(f"Error checking Gemini API connectivity: {e}")
            
            self._is_healthy = False
            self._last_health_check = now
            return False
    
    def generate(self, model, prompt, temperature = 0.7,
                 max_tokens = 512, system = None):
        """
        Generate completion from model.
        
        Args:
            model: Model name (e.g., 'gemini-2.0-flash-exp', 'gemini-1.5-flash')
            prompt: Input prompt
        
        Returns:
            Generated text
        """
        start = time.time()
        
        try:
            # Build generation config
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            
            # Add system instruction if provided
            if system:
                config.system_instruction = system
            
            # Generate content
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
    
    def list_models(self):
        """
        List available Gemini models.
        
        Returns:
            List of model names, empty list on error
        """
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
        """
        Get model information from Gemini API.
        
        Args:
            model: Model name
            
        Returns:
            Model info dict or None on error
        """
        try:
            # Normalize model name
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
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False