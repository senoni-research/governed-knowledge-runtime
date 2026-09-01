from __future__ import annotations

import pytest

from gkr.ai import GenerationRequest
from gkr.ai.mlx import MLXGenerator


def test_mlx_provider_requires_local_model_unless_download_is_explicit() -> None:
    generator = MLXGenerator("mlx-community/not-a-local-directory")

    with pytest.raises(RuntimeError, match="must be a local path"):
        generator.generate(GenerationRequest(prompt="test"))
