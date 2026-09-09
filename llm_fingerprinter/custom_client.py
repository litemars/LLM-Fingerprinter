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
    from llm_fingerprinter.custom_client import CustomClient

    client = CustomClient(request_file="request.txt")
    response = client.generate(prompt="Hello!")

Responses may be JSON, SSE, NDJSON, or text/plain. For a known plaintext
endpoint that omits Content-Type, pass allow_plain_text=True. Structured
errors and malformed or incomplete streams always raise CustomGenerationError.
"""

import requests
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from llm_fingerprinter.base_client import BaseClient, ClientError

logger = logging.getLogger(__name__)


class CustomClientError(ClientError):
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


class CustomTransientError(CustomClientError):
    """Raised when the backend produced no answer but is expected to succeed on
    retry (e.g. an Ollama-style model-load placeholder). Retried by generate()."""
    pass


class CustomClient(BaseClient):

    def __init__(self,
                 request_file: Optional[str] = None,
                 api_key: Optional[str] = None,
                 timeout: int = 120,
                 auth_header: str = "Authorization",
                 auth_prefix: str = "Bearer",
                 default_model: Optional[str] = None,
                 default_temperature: float = 0.7,
                 default_max_tokens: int = 512,
                 default_system: Optional[str] = None,
                 response_path: Optional[List] = None,
                 allow_plain_text: bool = False):
        super().__init__(timeout=timeout)

        self.api_key = api_key
        self.auth_header = auth_header
        self.auth_prefix = auth_prefix

        self.default_model = default_model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.default_system = default_system or ""

        self.response_path = response_path
        # Unframed text is accepted when the server declares text/plain, or
        # explicitly for a known plaintext endpoint lacking that content type.
        self.allow_plain_text = allow_plain_text

        self.url: Optional[str] = None
        self.payload_template: Optional[str] = None

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

    def _build_payload(self, prompt, model=None,
                       temperature=None, max_tokens=None,
                       system=None):

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
        self._raise_for_response_error(data)

        if isinstance(data, str):
            return data

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

            if path_valid and isinstance(result, str) and result != '':
                return result

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
            ["delta", "text"],
            ["delta"],
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

            if path_valid and isinstance(result, str) and result != '':
                return result

        return ""

    @staticmethod
    def _raise_for_response_error(data):
        # Arrays are ordinary JSON responses too; never return their serialized
        # error records via a plaintext fallback or an otherwise valid path.
        if isinstance(data, list):
            for item in data:
                CustomClient._raise_for_response_error(item)
        elif isinstance(data, dict):
            # Some compatible APIs include error:null on successful responses.
            if data.get('error') is not None or data.get('type') in ('error', 'response.failed', 'response.incomplete'):
                raise CustomGenerationError(
                    f"API returned an error payload: {data.get('error', data)}"
                )
            if data.get('done_reason') == 'load':
                raise CustomTransientError(
                    "Backend returned a model-load placeholder (no answer yet)"
                )

    @staticmethod
    def _looks_like_sse(text):
        first_line = text.lstrip().split('\n', 1)[0]
        return first_line.startswith(('data:', 'event:', 'id:', 'retry:', ':'))

    @staticmethod
    def _sse_records(response_text):
        """Decode SSE events, joining multiple data fields before JSON parsing."""
        data_lines = []
        event = ''
        # SSE recognizes CR/LF, not every Unicode line separator in JSON text.
        lines = response_text.replace('\r\n', '\n').replace('\r', '\n').split('\n')
        for line in lines + ['']:
            if not line:
                if event == 'error':
                    raise CustomGenerationError(
                        f"API returned an error event: {' '.join(data_lines)}"
                    )
                if data_lines:
                    data = '\n'.join(data_lines)
                    if data.strip():
                        yield data
                data_lines = []
                event = ''
                continue
            if line.startswith(':'):
                continue
            field, separator, value = line.partition(':')
            if not separator and field not in ('data', 'event', 'id', 'retry'):
                raise CustomGenerationError("Malformed SSE response frame")
            if value.startswith(' '):
                value = value[1:]
            if field == 'data':
                data_lines.append(value)
            elif field == 'event':
                event = value
            elif field not in ('id', 'retry'):
                raise CustomGenerationError(f"Malformed SSE response field: {field}")

    def _parse_streaming_response(self, response_text, framing=None):
        """Assemble a complete stream, rejecting errors and malformed tails."""
        if framing is None:
            framing = 'sse' if self._looks_like_sse(response_text) else 'json'
        records = (self._sse_records(response_text) if framing == 'sse'
                   else self._split_json_objects(response_text))
        full_text = []
        requires_completion = False
        complete = False
        sentinel_seen = False
        for record in records:
            if record.strip() == '[DONE]':
                complete = sentinel_seen = True
                continue
            try:
                obj = json.loads(record)
            except ValueError as exc:
                raise CustomGenerationError("Malformed JSON in streaming response") from exc
            self._raise_for_response_error(obj)
            if not isinstance(obj, dict):
                raise CustomGenerationError("Expected a JSON object in streaming response")
            if sentinel_seen:
                raise CustomGenerationError("Received data after stream completion")

            event_type = obj.get('type', '')
            # Responses API "done" events repeat the already streamed text.
            # Only token deltas contribute to the assembled answer.
            text = ('' if event_type == 'response.output_text.done'
                    else self._extract_response_text(obj))
            if complete and text:
                raise CustomGenerationError("Received answer text after stream completion")
            full_text.append(text)

            # These protocols define explicit completion. EOF after a valid
            # token record alone must not make a truncated answer successful.
            if 'choices' in obj:
                requires_completion = True
                choices = obj['choices']
                if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                    complete |= choices[0].get('finish_reason') is not None
            if 'done' in obj or 'response' in obj or isinstance(obj.get('message'), dict):
                requires_completion = True
                complete |= obj.get('done') is True
            if isinstance(event_type, str) and event_type.startswith(('message_', 'content_block_', 'response.')):
                requires_completion = True
                complete |= event_type in ('message_stop', 'response.completed')

        if requires_completion and not complete:
            raise CustomGenerationError("Incomplete streaming response: missing completion marker")
        return ''.join(full_text)

    @staticmethod
    def _split_json_objects(text: str):
        """Decode every JSON value; never skip junk or incomplete final records."""
        decoder = json.JSONDecoder()
        offset = 0
        while offset < len(text):
            if text[offset].isspace():
                offset += 1
                continue
            try:
                _, end = decoder.raw_decode(text, offset)
            except ValueError as exc:
                raise CustomGenerationError("Malformed or truncated JSON response") from exc
            yield text[offset:end]
            offset = end

    def _perform_health_check(self) -> bool:
        if not self.url:
            logger.error("No URL configured")
            return False

        try:
            base_url = '/'.join(self.url.split('/')[:3])
            resp = self.session.head(base_url, timeout=10)
            return resp.status_code < 500

        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error: {e}")
            return False
        except requests.exceptions.Timeout:
            logger.error(f"Timeout connecting to {self.url}")
            return False
        except Exception as e:
            logger.error(f"Error checking connectivity: {e}")
            return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(
            (requests.Timeout, requests.ConnectionError,
             CustomConnectionError, CustomTransientError)
        ),
        reraise=True,
    )
    def generate(self, prompt="", model=None,
                 temperature=None, max_tokens=None,
                 system=None):
        logger.debug(f"Generating with model {model}")
        if not self.url:
            raise CustomClientError("No URL configured. Provide request_file.")

        payload = self._build_payload(prompt, model, temperature, max_tokens, system)
        try:
            start = time.time()
            response = self.session.post(self.url, json=payload, timeout=self.timeout)
            elapsed = time.time() - start

            if response.status_code == 200:
                response_text = response.text
                content_type = getattr(response, 'headers', {}).get('Content-Type', '')
                content_type = content_type.split(';', 1)[0].strip().lower()
                text = ''

                # Select framing before extraction. A failed structured payload
                # is never reinterpreted as plaintext or mined for inner objects.
                if content_type == 'text/event-stream' or self._looks_like_sse(response_text):
                    text = self._parse_streaming_response(response_text, framing='sse')
                elif content_type in ('application/x-ndjson', 'application/ndjson',
                                       'application/jsonl', 'application/jsonlines'):
                    text = self._parse_streaming_response(response_text, framing='json')
                else:
                    try:
                        result = response.json()
                    except ValueError:
                        if (response_text.lstrip().startswith(('{', '['))
                                or content_type == 'application/json'
                                or content_type.endswith('+json')):
                            text = self._parse_streaming_response(response_text, framing='json')
                        elif content_type == 'text/plain' or (
                                self.allow_plain_text and not content_type):
                            text = response_text
                    else:
                        self._raise_for_response_error(result)
                        # A one-record Ollama/OpenAI stream can be a truncated
                        # request too. Ordinary nonstreaming JSON needs no marker.
                        is_stream_record = isinstance(result, dict) and (
                            any(key in result for key in ('done', 'response', 'choices'))
                            or isinstance(result.get('message'), dict)
                        )
                        if payload.get('stream') is True and is_stream_record:
                            text = self._parse_streaming_response(response_text, framing='json')
                        else:
                            text = self._extract_response_text(result)

                # Validate once, after assembly; whitespace inside and around
                # tokens is model output and must survive fingerprinting intact.
                if text.strip():
                    logger.debug(f"Generated {len(text)} chars in {elapsed:.2f}s")
                    return text

                logger.debug(f"Unparseable response: {response_text[:500]}")
                raise CustomGenerationError(
                    f"Could not extract any text from the 200 response "
                    f"({len(response_text)} bytes). Check the endpoint, or set "
                    f"response_path to point at the text field. Plaintext endpoints "
                    f"must return Content-Type: text/plain or use allow_plain_text=True."
                )

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
        except (ValueError, KeyError):
            return response.text[:200] if response.text else "Unknown error"

    def list_models(self) -> List[str]:
        """List models (not supported for template-based client)."""
        logger.warning("list_models not supported for template-based custom client")
        return []

    def close(self):
        """Close the HTTP session."""
        self.session.close()
        logger.debug("Closed CustomClient session")

    def __repr__(self):
        if self.url:
            return f"CustomClient(url='{self.url}')"
        return "CustomClient(not configured)"
