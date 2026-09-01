from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.0


@dataclass(frozen=True)
class Generation:
    text: str
    model: str
    metadata: dict[str, Any] = field(default_factory=dict)


class LocalGenerator(Protocol):
    """Provider boundary for generators that execute on the local machine."""

    @property
    def model_id(self) -> str: ...

    def generate(self, request: GenerationRequest) -> Generation: ...
