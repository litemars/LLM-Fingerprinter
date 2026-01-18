"""Custom client that uses a request template file.

Request file format:
    Line 1: URL (required)
    Line 2+: JSON payload with $PROMPT$ placeholder
    
placeholders:
    $PROMPT$ - User prompt (required)
    $SYSTEM$ - System prompt
    $MODEL$ - Model name
    $TEMPERATURE$ - Temperature value
    $MAX_TOKENS$ - Max tokens value

Example request file (request.txt):
    http://localhost:11434/api/generate
    {
        "model": "llama3.2",
        "prompt": "$PROMPT$",
        "stream": false
    }

Usage:
    from src.custom_client import CustomClient
    
    client = CustomClient(request_file="request.txt")
    response = client.generate(prompt="Hello!")
"""

import requests
import logging
import time
import json
from pathlib import Path
from typing import Dict, List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


class CustomClientError(Exception):
    """Base exception for custom client errors."""
    pass


class CustomConnectionError(CustomClientError):
    """Raised when connection to API fails."""
    pass


class CustomGenerationError(CustomClientError):
    """Raised when generation fails."""
    pass


class CustomAuthError(CustomClientError):
    """Raised when authentication fails."""
    pass


class CustomClient:
    
    def __init__(self,
                 request_file: str = None,
                 api_key: str = None,
                 timeout: int = 120,
                 auth_header: str = "Authorization",
                 auth_prefix: str = "Bearer",
                 default_model: str = None,
                 default_temperature: float = 0.7,
                 default_max_tokens: int = 512,
                 default_system: str = None,
                 response_path: List = None):
        """
        Initialize custom client from request file.
        
        Args:
            request_file: Path to request template file
            api_key: API key for authentication (optional)
            timeout: Request timeout in seconds
            auth_header: Header name for API key (default: "Authorization")
            auth_prefix: Prefix for API key (default: "Bearer", use "" for none)
            default_model: Default model name for $MODEL$ placeholder
            default_temperature: Default temperature for $TEMPERATURE$ placeholder
            default_max_tokens: Default max_tokens for $MAX_TOKENS$ placeholder
            default_system: Default system prompt for $SYSTEM$ placeholder
            response_path: Path to extract text from response (auto-detected if None)
        """
        self.api_key = api_key
        self.timeout = timeout
        self.auth_header = auth_header
        self.auth_prefix = auth_prefix
        
        self.default_model = default_model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.default_system = default_system or ""
        
        self.response_path = response_path
        
        self.url = None
        self.payload_template = None
        
        if request_file:
            self._parse_request_file(request_file)
        
        # Setup session
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=0
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        self.session.headers.update({"Content-Type": "application/json"})
        
        if api_key:
            if auth_prefix:
                self.session.headers[auth_header] = f"{auth_prefix} {api_key}"
            else:
                self.session.headers[auth_header] = api_key
        
        self._last_health_check = None
        self._health_check_interval = 30
        self._is_healthy = False
        
        if self.url:
            logger.info(f"Initialized CustomClient for {self.url}")
    
    def _parse_request_file(self, request_file):

        path = Path(request_file)
        
        if not path.exists():
            raise CustomClientError(f"Request file not found: {request_file}")
        
        content = path.read_text().strip()
        lines = content.split('\n')
        
        if not lines:
            raise CustomClientError(f"Request file is empty: {request_file}")
        
        self.url = lines[0].strip()
        
        if not self.url.startswith(('http://', 'https://')):
            raise CustomClientError(f"Invalid URL in request file: {self.url}")
        
        if len(lines) > 1:
            json_content = '\n'.join(lines[1:]).strip()
            
            test_json = json_content
            test_json = test_json.replace('$PROMPT$', 'test')
            test_json = test_json.replace('$SYSTEM$', 'test')
            test_json = test_json.replace('$MODEL$', 'test')
            test_json = test_json.replace('$TEMPERATURE$', '0.7')
            test_json = test_json.replace('$MAX_TOKENS$', '512')
            
            try:
                json.loads(test_json)
            except json.JSONDecodeError as e:
                raise CustomClientError(f"Invalid JSON in request file: {e}")
            
            self.payload_template = json_content
        else:
            raise CustomClientError("Request file must contain JSON payload after URL")
        
        if '$PROMPT$' not in self.payload_template:
            raise CustomClientError("Request file must contain $PROMPT$ placeholder")
        
        logger.debug(f"Parsed request file: URL={self.url}")
    
    def _build_payload(self, prompt, model = None, 
                       temperature = None, max_tokens = None,
                       system = None):

        if not self.payload_template:
            raise CustomClientError("No payload template configured")
        
        payload_str = self.payload_template
        
        escaped_prompt = json.dumps(prompt)[1:-1]
        escaped_system = json.dumps(system or self.default_system)[1:-1]
        
        payload_str = payload_str.replace('$PROMPT$', escaped_prompt)
        payload_str = payload_str.replace('$SYSTEM$', escaped_system)
        payload_str = payload_str.replace('$MODEL$', model or self.default_model or '')
        payload_str = payload_str.replace('$TEMPERATURE$', str(temperature if temperature is not None else self.default_temperature))
        payload_str = payload_str.replace('$MAX_TOKENS$', str(max_tokens if max_tokens is not None else self.default_max_tokens))
        
        try:
            return json.loads(payload_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse payload: {payload_str[:200]}")
            raise CustomGenerationError(f"Invalid payload after substitution: {e}")
    
    def _extract_response_text(self, data):
        
        # Check for empty/loading responses
        if isinstance(data, dict):
            done_reason = data.get('done_reason', '')
            if done_reason == 'load':
                return ""
            
            if 'response' in data and data['response'] == '' and data.get('done'):
                return ""
            
            if 'error' in data:
                logger.error(f"API returned error: {data['error']}")
                return ""
        
        # Try configured path first
        if self.response_path:
            result = data
            path_valid = True
            for key in self.response_path:
                if result is None:
                    path_valid = False
                    break
                if isinstance(key, int):
                    if isinstance(result, (list, tuple)) and len(result) > key:
                        result = result[key]
                    else:
                        path_valid = False
                        break
                elif isinstance(result, dict):
                    result = result.get(key)
                else:
                    path_valid = False
                    break
            
            if path_valid and isinstance(result, str) and result.strip():
                return result.strip()
        
        # Try fallback paths
        fallback_paths = [
            ["choices", 0, "message", "content"],
            ["choices", 0, "text"],
            ["choices", 0, "delta", "content"],
            ["response"],
            ["content"],
            ["text"],
            ["output"],
            ["message"],
            ["result"],
            ["answer"],
            ["completion"],
            ["generated_text"],
            ["data", "content"],
            ["data", "text"],
            ["message", "content"],
            ["content", 0, "text"],
        ]
        
        for path in fallback_paths:
            result = data
            path_valid = True
            
            for key in path:
                if result is None:
                    path_valid = False
                    break
                if isinstance(key, int):
                    if isinstance(result, (list, tuple)) and len(result) > key:
                        result = result[key]
                    else:
                        path_valid = False
                        break
                elif isinstance(result, dict):
                    result = result.get(key)
                else:
                    path_valid = False
                    break
            
            if path_valid and isinstance(result, str) and result.strip():
                return result.strip()
        
        return ""
    
    def _parse_streaming_response(self, response_text):
        full_text = []
        
        # Try to split by newlines first
        lines = response_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Handle multiple JSON objects on same line (no delimiter)
            objects = self._split_json_objects(line)
            
            for obj_str in objects:
                try:
                    obj = json.loads(obj_str)
                    text = self._extract_response_text(obj)
                    if text:
                        full_text.append(text)
                except json.JSONDecodeError:
                    continue
        
        return ''.join(full_text)
    
    def _split_json_objects(self, text: str):

        objects = []
        depth = 0
        start = 0
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\' and in_string:
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if in_string:
                continue
            
            if char == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    objects.append(text[start:i+1])
        
        return objects
    
    def _check_connectivity(self, force: bool = False):

        now = time.time()
        
        if not force and self._last_health_check is not None:
            if now - self._last_health_check < self._health_check_interval:
                return self._is_healthy
        
        if not self.url:
            logger.error("No URL configured")
            return False
        
        try:
            base_url = '/'.join(self.url.split('/')[:3])
            resp = self.session.head(base_url, timeout=10)
            
            self._is_healthy = resp.status_code < 500
            self._last_health_check = now
            return self._is_healthy
            
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error: {e}")
            self._is_healthy = False
            self._last_health_check = now
            return False
        except requests.exceptions.Timeout:
            logger.error(f"Timeout connecting to {self.url}")
            self._is_healthy = False
            self._last_health_check = now
            return False
        except Exception as e:
            logger.error(f"Error checking connectivity: {e}")
            self._is_healthy = False
            self._last_health_check = now
            return False
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError))
    )
    def generate(self, prompt = "", model = None,
                 temperature = None, max_tokens = None,
                 system = None):
        """        
        Args:
            prompt: User prompt (replaces $PROMPT$ in template)
            model: Model name (replaces $MODEL$ in template)
            temperature: Temperature (replaces $TEMPERATURE$ in template)
            max_tokens: Max tokens (replaces $MAX_TOKENS$ in template)
            system: System prompt (replaces $SYSTEM$ in template)
        """
        logger.debug(f"Generating with prompt length {model}")
        if not self.url:
            raise CustomClientError("No URL configured. Provide request_file.")
        
        payload = self._build_payload(prompt, model, temperature, max_tokens, system)
        try:
            start = time.time()
            response = self.session.post(self.url, json=payload, timeout=self.timeout)
            elapsed = time.time() - start
            
            if response.status_code == 200:
                response_text = response.text
                
                # First, try to parse as single JSON object
                try:
                    result = response.json()
                    text = self._extract_response_text(result)
                    if text:
                        logger.debug(f"Generated {len(text)} chars in {elapsed:.2f}s")
                        return text
                except json.JSONDecodeError:
                    pass
                
                # If that fails, try to parse as streaming response
                text = self._parse_streaming_response(response_text)
                if text:
                    logger.debug(f"Generated {len(text)} chars (streaming) in {elapsed:.2f}s")
                    return text
                
                # Last resort: check if it's just plain text
                if response_text.strip() and not response_text.strip().startswith('{'):
                    return response_text.strip()
                
                logger.error(f"Could not extract text from response")
                logger.debug(f"Response: {response_text[:500]}")
                return ""
            
            elif response.status_code == 401:
                raise CustomAuthError("Invalid API key")
            elif response.status_code == 403:
                raise CustomAuthError("Access forbidden - check API key permissions")
            elif response.status_code == 404:
                raise CustomGenerationError(f"Endpoint not found: {self.url}")
            elif response.status_code == 429:
                raise CustomGenerationError("Rate limit exceeded - please wait and retry")
            else:
                error_msg = self._extract_error_message(response)
                raise CustomGenerationError(f"API error {response.status_code}: {error_msg}")
        
        except requests.Timeout:
            logger.warning(f"Timeout after {self.timeout}s")
            raise
        except requests.ConnectionError as e:
            logger.error(f"Connection error: {e}")
            raise CustomConnectionError(f"Cannot connect to {self.url}")
        except CustomClientError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise CustomGenerationError(f"Generation failed: {e}")
    
    def _extract_error_message(self, response: requests.Response):
        """Extract error message from response."""
        try:
            error_json = response.json()
            if "error" in error_json:
                err = error_json["error"]
                if isinstance(err, dict):
                    return err.get("message", str(err))
                return str(err)
            if "message" in error_json:
                return error_json["message"]
            if "detail" in error_json:
                return error_json["detail"]
            return str(error_json)[:200]
        except:
            return response.text[:200] if response.text else "Unknown error"
    
    def list_models(self):
        """List models (not supported for template-based client)."""
        logger.warning("list_models not supported for template-based custom client")
        return []
    
    def close(self):
        """Close the HTTP session."""
        self.session.close()
        logger.debug("Closed CustomClient session")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    def __repr__(self):
        if self.url:
            return f"CustomClient(url='{self.url}')"
        return "CustomClient(not configured)"