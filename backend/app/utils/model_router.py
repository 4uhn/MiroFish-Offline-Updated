"""
Tiered Model Router — routes LLM tasks to the appropriate model tier.

Applies System One/System Two thinking to model selection:
- Fast tier (e.g. qwen3:0.6b): Constrained tasks — NER, classification, extraction
- Quality tier (e.g. qwen3:8b): Creative tasks — text generation, reasoning, analysis

Falls back to quality tier when fast model is unavailable or fails.
Configure via environment variables:
  LLM_FAST_MODEL_NAME=qwen3:0.6b
  LLM_FAST_NUM_CTX=2048
"""

import logging
import os
from enum import Enum
from typing import Optional

from .llm_client import LLMClient

logger = logging.getLogger('mirofish.model_router')


class TaskType(str, Enum):
    EXTRACTION = "extraction"
    GENERATION = "generation"
    ANALYSIS = "analysis"


_TASK_TO_TIER = {
    TaskType.EXTRACTION: "fast",
    TaskType.GENERATION: "quality",
    TaskType.ANALYSIS: "quality",
}


class ModelRouter:
    """Routes LLM calls to fast or quality model based on task type."""

    def __init__(self):
        self._fast_model = os.environ.get('LLM_FAST_MODEL_NAME', '').strip()
        self._fast_num_ctx = int(os.environ.get('LLM_FAST_NUM_CTX', '2048'))
        self._quality_model = os.environ.get('LLM_MODEL_NAME', 'qwen3:8b')

        if self._fast_model:
            logger.info(
                "Tiered routing enabled: fast=%s, quality=%s",
                self._fast_model, self._quality_model,
            )
        else:
            logger.info("Single-tier mode: all tasks use %s", self._quality_model)

    @property
    def fast_available(self) -> bool:
        return bool(self._fast_model)

    @property
    def fast_model_name(self) -> str:
        return self._fast_model

    @property
    def quality_model_name(self) -> str:
        return self._quality_model

    def get_client(
        self,
        task: TaskType,
        num_ctx: Optional[int] = None,
    ) -> LLMClient:
        """Return an LLMClient configured for the given task type."""
        tier = _TASK_TO_TIER.get(task, "quality")

        if tier == "fast" and self._fast_model:
            ctx = num_ctx or self._fast_num_ctx
            logger.debug("Routing %s -> %s (fast)", task.value, self._fast_model)
            return LLMClient(model=self._fast_model, num_ctx=ctx)

        logger.debug("Routing %s -> %s (quality)", task.value, self._quality_model)
        return LLMClient(num_ctx=num_ctx)

    def get_quality_client(self, num_ctx: Optional[int] = None) -> LLMClient:
        """Always return the quality-tier client."""
        return LLMClient(num_ctx=num_ctx)


_router: Optional[ModelRouter] = None


def get_router() -> ModelRouter:
    global _router
    if _router is None:
        _router = ModelRouter()
    return _router
