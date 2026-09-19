"""
LLM Client Wrapper
Unified OpenAI-compatible interface for Ollama and Groq.
Supports Ollama num_ctx to control context window size.
"""

import json
import os
import re
from typing import Optional, Dict, Any, List
from openai import OpenAI

from ..config import Config


class LLMClient:
    """Unified LLM client supporting Ollama and Groq via OpenAI SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 300.0,
        num_ctx: Optional[int] = None
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
        )

        # Ollama context window size — smaller = faster inference.
        # 2048 is sufficient for social media post generation. Use 8192 for report agent.
        self._num_ctx = num_ctx or int(os.environ.get('OLLAMA_NUM_CTX', '2048'))

    def _is_ollama(self) -> bool:
        """Check if we're talking to an Ollama server."""
        provider = os.environ.get('LLM_PROVIDER', 'ollama').lower()
        if provider == 'ollama':
            return True
        # Fallback: detect by port for backwards compatibility
        return '11434' in (self.base_url or '')

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        """Send a chat completion request and return the response text."""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format:
            kwargs["response_format"] = response_format

        # For Ollama: pass num_ctx and disable thinking (qwen3 compatibility)
        if self._is_ollama():
            extra = kwargs.get("extra_body", {})
            if self._num_ctx:
                extra.setdefault("options", {})["num_ctx"] = self._num_ctx
            extra["think"] = False
            kwargs["extra_body"] = extra

        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        # Strip <think> tags (some models embed thinking in content)
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        return content

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """Send a chat request and parse the response as JSON."""
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
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
