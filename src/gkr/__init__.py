"""Local-first governed knowledge runtime."""

from .runtime import GovernedKnowledgeRuntime, RuntimeResult
from .schemas import Actor, KnowledgeRecord, PolicyRule

__all__ = [
    "Actor",
    "GovernedKnowledgeRuntime",
    "KnowledgeRecord",
    "PolicyRule",
    "RuntimeResult",
]

__version__ = "0.1.0"
