"""
LLM Client Wrapper
OpenAI-compatible client for Ollama (or any OpenAI-compatible server).
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from openai import OpenAI

from ..config import Config

logger = logging.getLogger('mirofish.llm_client')


# Ollama's OpenAI-compatible /v1 endpoint silently ignores the native `think`
# flag and `options` (e.g. num_ctx). `reasoning_effort: "none"` is the only
# per-request switch it honours for turning off qwen3 reasoning, which otherwise
# runs hidden on every call at ~10x the tokens of the actual answer.
OLLAMA_NO_THINK = {"reasoning_effort": "none"}


def is_ollama(base_url: Optional[str] = None) -> bool:
    """True when LLM_PROVIDER is ollama, or the base URL is Ollama's default port."""
    if os.environ.get('LLM_PROVIDER', 'ollama').lower() == 'ollama':
        return True
    return '11434' in (base_url or '')


def ollama_extra_body(base_url: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """extra_body for chat.completions.create: disables reasoning on Ollama."""
    return dict(OLLAMA_NO_THINK) if is_ollama(base_url) else None


@dataclass
class ChatResult:
    """A chat reply with the token counts the server reported for it."""
    content: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    finish_reason: Optional[str] = None


class LLMClient:
    """LLM client over the OpenAI SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 300.0,
        max_retries: int = 2,
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME

        if not self.api_key:
            raise ValueError("LLM_API_KEY is not configured")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

        # Ollama's /v1 endpoint ignores a per-request num_ctx, so the window is
        # the server's OLLAMA_CONTEXT_LENGTH; OLLAMA_NUM_CTX must match it.
        self._num_ctx = int(os.environ.get('OLLAMA_NUM_CTX', '8192'))

    def _is_ollama(self) -> bool:
        """Check if we're talking to an Ollama server."""
        return is_ollama(self.base_url)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        """Send a chat completion request and return the response text."""
        return self.chat_result(messages, temperature, max_tokens, response_format).content

    @property
    def num_ctx(self) -> int:
        """Context window of the server (OLLAMA_NUM_CTX, matching OLLAMA_CONTEXT_LENGTH)."""
        return self._num_ctx

    def chat_result(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> ChatResult:
        """Like chat(), but also returns the server's token counts.

        A prompt plus reply longer than the context window does not fail on
        Ollama: it shifts the context, silently dropping the start of the
        prompt (the system instructions) mid-reply. That is logged here.
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format:
            kwargs["response_format"] = response_format

        if self._is_ollama():
            kwargs["extra_body"] = dict(OLLAMA_NO_THINK)

        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        # Strip <think> tags (some models embed thinking in content)
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        usage = getattr(response, "usage", None)
        result = ChatResult(
            content=content,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            finish_reason=response.choices[0].finish_reason,
        )
        if result.prompt_tokens and result.completion_tokens and \
                result.prompt_tokens + result.completion_tokens > self._num_ctx:
            logger.warning(
                f"Context overflow: prompt {result.prompt_tokens} + reply {result.completion_tokens} "
                f"tokens > num_ctx {self._num_ctx}; the server dropped part of the prompt"
            )
        return result

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        schema: Optional[Dict[str, Any]] = None,
        schema_name: str = "response",
    ) -> Dict[str, Any]:
        """Send a chat request and parse the response as JSON.

        Pass ``schema`` (a JSON Schema) to constrain decoding to that shape.
        Plain json_object mode only guarantees valid JSON, and qwen3 can emit
        malformed objects such as {" ": "description"} inside arrays.
        """
        if schema:
            response_format = {"type": "json_schema",
                               "json_schema": {"name": schema_name, "strict": True, "schema": schema}}
        else:
            response_format = {"type": "json_object"}
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format
        )
        # Strip markdown code fences
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            raise ValueError(f"LLM returned invalid JSON: {cleaned_response[:200]}")
