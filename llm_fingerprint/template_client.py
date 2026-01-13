"""Template client for custom LLM API endpoints.

This is a template that you can copy and customize to support any LLM API.
Simply implement the required methods to match your API's specification.

To create a new client:
1. Copy this file and rename it (e.g., my_custom_client.py)
2. Rename the class (e.g., MyCustomClient)
3. Update the API endpoint and authentication method
4. Modify the generate() method to match your API's request/response format
5. Update list_models() if your API supports them
6. Add your client to cli.py and __init__.py files

Required methods for fingerprinting:
- generate(model, prompt, temperature, max_tokens, system) -> str
- _check_connectivity() -> bool

Optional but recommended:
- list_models() -> List[str]
"""

import requests
import logging
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


# =============================================================================
# TODO - Rename these for your client
# =============================================================================

class TemplateClientError(Exception):
    """Base exception for template client errors."""
    pass


class TemplateConnectionError(TemplateClientError):
    """Raised when connection to API fails."""
    pass


class TemplateGenerationError(TemplateClientError):
    """Raised when generation fails."""
    pass


class TemplateAuthError(TemplateClientError):
    """Raised when authentication fails."""
    pass

class TemplateClient:

    DEFAULT_ENDPOINT = "https://api.example.com/v1"
    DEFAULT_TIMEOUT = 60
    AUTH_HEADER_NAME = "Authorization" 
    AUTH_HEADER_PREFIX = "Bearer"
    
    def __init__(self, 
                 api_key: str,
                 endpoint: str = None,
                 timeout: int = None,
                 max_retries: int = 3,
                 **kwargs):
        """
        Initialize the client.
        
        Args:
            api_key: API key for authentication
            endpoint: API endpoint URL (uses DEFAULT_ENDPOINT if not provided)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            **kwargs: Additional configuration options
        """
        self.api_key = api_key
        self.endpoint = (endpoint or self.DEFAULT_ENDPOINT).rstrip("/")
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self.max_retries = max_retries
        self.extra_config = kwargs
        
        # Session with connection pooling
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=0
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        # =====================================================================
        # TODO: Update headers for your API's authentication method
        # =====================================================================
        self.session.headers.update({
            "Content-Type": "application/json",
        })
        
        # Set auth header
        if self.AUTH_HEADER_PREFIX:
            self.session.headers[self.AUTH_HEADER_NAME] = f"{self.AUTH_HEADER_PREFIX} {api_key}"
        else:
            self.session.headers[self.AUTH_HEADER_NAME] = api_key
        
        # Add any extra headers from kwargs
        if "extra_headers" in kwargs:
            self.session.headers.update(kwargs["extra_headers"])
        
        # Cache connectivity status
        self._last_health_check = None
        self._health_check_interval = 30
        self._is_healthy = False
        
        logger.info(f"Initialized TemplateClient for {self.endpoint}")
    
    def _check_connectivity(self, force: bool = False) -> bool:
        """
        Check if API is reachable.
        
        CUSTOMIZE: Update the health check endpoint and response parsing
        for your API.
        
        Args:
            force: Force check even if cached result is recent
            
        Returns:
            True if API is reachable and authenticated, False otherwise
        """
        now = time.time()
        
        if not force and self._last_health_check is not None:
            if now - self._last_health_check < self._health_check_interval:
                return self._is_healthy
        
        try:

            health_url = f"{self.endpoint}/models"
            
            resp = self.session.get(health_url, timeout=10)
            self._is_healthy = resp.status_code == 200
            self._last_health_check = now
            
            if resp.status_code == 401:
                logger.warning("API reachable but authentication failed - check API key")
            elif resp.status_code == 403:
                logger.warning("API reachable but access forbidden - check permissions")
                
            return resp.status_code == 200
            
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error to API at {self.endpoint}: {e}")
            self._is_healthy = False
            self._last_health_check = now
            return False
        except requests.exceptions.Timeout:
            logger.error(f"Timeout connecting to API at {self.endpoint}")
            self._is_healthy = False
            self._last_health_check = now
            return False
        except Exception as e:
            logger.error(f"Error checking API connectivity: {e}")
            self._is_healthy = False
            self._last_health_check = now
            return False
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError))
    )
    def generate(self, model, prompt, temperature = 0.7,
                 max_tokens = 512, system = None):
        """
        Generate completion from model.
        
        CUSTOMIZE: This is the main method to modify. Update the:
        - URL endpoint
        - Request payload structure
        - Response parsing
        
        Args:
            model: Model name/identifier
            prompt: Input prompt
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum response length in tokens
            system: Optional system prompt
        
        Returns:
            Generated text
            
        """
        url = f"{self.endpoint}/chat/completions"

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        try:
            start = time.time()
            response = self.session.post(url, json=payload, timeout=self.timeout)
            elapsed = time.time() - start
            
            if response.status_code == 200:
                result = response.json()

                choices = result.get("choices", [])
                if choices:
                    text = choices[0].get("message", {}).get("content", "").strip()
                else:
                    text = ""
                
                logger.debug(f"Generated {len(text)} chars in {elapsed:.2f}s")
                return text
                
            elif response.status_code == 401:
                raise TemplateAuthError("Invalid API key")
            elif response.status_code == 403:
                raise TemplateAuthError("Access forbidden - check API key permissions")
            elif response.status_code == 404:
                raise TemplateGenerationError(f"Model '{model}' not found")
            elif response.status_code == 429:
                raise TemplateGenerationError("Rate limit exceeded - please wait and retry")
            else:
                error_msg = response.text[:200] if response.text else "Unknown error"
                try:
                    error_json = response.json()
                    error_msg = error_json.get("error", {}).get("message", error_msg)
                except:
                    pass
                raise TemplateGenerationError(
                    f"API error {response.status_code}: {error_msg}"
                )
        
        except requests.Timeout:
            logger.warning(f"Timeout querying {model} after {self.timeout}s")
            raise
        except requests.ConnectionError as e:
            logger.error(f"Connection error to API: {e}")
            raise TemplateConnectionError(f"Cannot connect to API at {self.endpoint}")
        except TemplateClientError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error generating from {model}: {e}")
            raise TemplateGenerationError(f"Generation failed: {e}")
    
    def list_models(self):
        """
        List available models from the API.
        
        CUSTOMIZE: Update for your API's model listing endpoint and response format.
        
        Returns:
            List of model IDs, empty list on error
        """
        try:
            url = f"{self.endpoint}/models"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                
                models = [m["id"] for m in data.get("data", [])]
                
                
                logger.info(f"Found {len(models)} models on API")
                return models
            elif response.status_code == 401:
                logger.error("Authentication failed - check API key")
                return []
            else:
                logger.error(f"Failed to list models: HTTP {response.status_code}")
                return []
        except requests.ConnectionError:
            logger.error(f"Cannot connect to API at {self.endpoint}")
            return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def close(self):
        """Close the HTTP session."""
        self.session.close()
        logger.debug("Closed TemplateClient session")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

class ExampleCustomClient(TemplateClient):
    
    DEFAULT_ENDPOINT = "https://api.my-custom-llm.com/v1"
    AUTH_HEADER_NAME = "X-API-Key"
    AUTH_HEADER_PREFIX = ""  # No prefix, just the raw key
    
    def generate(self, model, prompt, temperature = 0.7,
                 max_tokens = 512, system = None) -> str:
        """Override generate with custom payload format."""
        url = f"{self.endpoint}/inference"
        
        payload = {
            "model_id": model,
            "input_text": prompt,
            "parameters": {
                "temp": temperature,
                "max_length": max_tokens,
            }
        }
        
        if system:
            payload["system_prompt"] = system
        
        try:
            response = self.session.post(url, json=payload, timeout=self.timeout)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("output_text", "").strip()
            else:
                raise TemplateGenerationError(f"API error: {response.status_code}")
                
        except TemplateClientError:
            raise
        except Exception as e:
            raise TemplateGenerationError(f"Generation failed: {e}")
