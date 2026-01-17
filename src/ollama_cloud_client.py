import requests
import logging
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


class OllamaCloudError(Exception):
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


class OllamaCloudClient:
    
    def __init__(self, 
                 api_key: str,
                 endpoint = "https://api.ollama.com/v1",
                 timeout: int = 60,
                 max_retries: int = 3):
        """
        Initialize Ollama Cloud client.
        
        Args:
            api_key: Ollama Cloud API key
            endpoint: API endpoint URL (default: Ollama Cloud)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.api_key = api_key
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Session with connection pooling
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=0
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        # Set default headers
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })
        
        # Cache connectivity status
        self._last_health_check = None
        self._health_check_interval = 30
        self._is_healthy = False
        
        logger.info(f"Initialized OllamaCloudClient for {endpoint}")
    
    def _check_connectivity(self, force = False):

        now = time.time()
        
        if not force and self._last_health_check is not None:
            if now - self._last_health_check < self._health_check_interval:
                return self._is_healthy
        
        try:
            # Try to list models as health check
            resp = self.session.get(
                f"{self.endpoint}/api/tags",
                timeout=10
            )
            self._is_healthy = resp.status_code in [200, 401, 403]
            self._last_health_check = now
            
            if resp.status_code == 401:
                logger.warning("API reachable but authentication failed - check API key")
            elif resp.status_code == 403:
                logger.warning("API reachable but access forbidden - check permissions")
                
            return resp.status_code == 200
            
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error to Ollama Cloud at {self.endpoint}: {e}")
            self._is_healthy = False
            self._last_health_check = now
            return False
        except requests.exceptions.Timeout:
            logger.error(f"Timeout connecting to Ollama Cloud at {self.endpoint}")
            self._is_healthy = False
            self._last_health_check = now
            return False
        except Exception as e:
            logger.error(f"Error checking Ollama Cloud connectivity: {e}")
            self._is_healthy = False
            self._last_health_check = now
            return False
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError))
    )
    def generate(self, model, prompt, temperature = 0.7, max_tokens = 512, system = None):
        """
        Generate completion from model.
        
        Args:
            model: Model name (e.g., 'llama3.2', 'mistral')
            prompt: Input prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum response length in tokens
            system: Optional system prompt
        
        Returns:
            Generated text
        """
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
                
                # Log performance metrics
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
                except:
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
    
    def list_models(self):
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
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
