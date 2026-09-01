from __future__ import annotations

from pathlib import Path
from typing import Any

from gkr.ai.protocol import Generation, GenerationRequest


class MLXGenerator:
    """Lazy, in-process MLX-LM generator; no remote inference API is used."""

    def __init__(
        self,
        model: str | Path,
        *,
        adapter_path: str | Path | None = None,
        allow_download: bool = False,
    ) -> None:
        candidate_path = Path(model).expanduser()
        self._model_reference = (
            str(candidate_path.resolve()) if candidate_path.exists() else str(model)
        )
        self._adapter_path = str(adapter_path) if adapter_path else None
        self._allow_download = allow_download
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._generate_function: Any | None = None
        self._make_sampler: Any | None = None

    @property
    def model_id(self) -> str:
        return self._model_reference

    def generate(self, request: GenerationRequest) -> Generation:
        self._ensure_loaded()
        messages = [{"role": "user", "content": request.prompt}]
        try:
            prompt = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        except (AttributeError, TypeError):
            prompt = request.prompt
        sampler = self._make_sampler(temp=request.temperature)
        text = self._generate_function(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=request.max_tokens,
            sampler=sampler,
            verbose=False,
        )
        return Generation(
            text=str(text).strip(),
            model=self.model_id,
            metadata={"provider": "mlx-lm", "execution": "local"},
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        model_path = Path(self._model_reference).expanduser()
        if not self._allow_download and not model_path.exists():
            raise RuntimeError(
                "The MLX model must be a local path. Download it explicitly first, "
                "or opt in with allow_download=True."
            )
        try:
            from mlx_lm import generate, load
            from mlx_lm.sample_utils import make_sampler
        except ImportError as exc:
            raise RuntimeError('MLX-LM is required. Install with: pip install -e ".[mac]"') from exc

        self._model, self._tokenizer = load(
            self._model_reference,
            adapter_path=self._adapter_path,
        )
        self._generate_function = generate
        self._make_sampler = make_sampler
