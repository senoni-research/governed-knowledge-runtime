from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from gkr.ai.protocol import Generation, GenerationRequest

_FALLBACK_THINKING_DELIMITERS = (
    ("<think>", "</think>"),
    ("<|channel>thought", "<channel|>"),
)
_FINAL_CHANNEL_MARKERS = (
    "<|channel>final<|message>",
    "<|channel>final<channel|>",
)
_TRAILING_CONTROL_TOKENS = ("<|end|>", "<|eot_id|>", "<|im_end|>")


class MLXGenerator:
    """Lazy, in-process MLX-LM generator; no remote inference API is used."""

    def __init__(
        self,
        model: str | Path,
        *,
        adapter_path: str | Path | None = None,
        allow_download: bool = False,
        enable_thinking: bool = False,
    ) -> None:
        candidate_path = Path(model).expanduser()
        self._model_reference = (
            str(candidate_path.resolve()) if candidate_path.exists() else str(model)
        )
        self._adapter_path = str(adapter_path) if adapter_path else None
        self._allow_download = allow_download
        self._enable_thinking = enable_thinking
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._stream_generate_function: Any | None = None
        self._make_sampler: Any | None = None

    @property
    def model_id(self) -> str:
        return self._model_reference

    def generate(self, request: GenerationRequest) -> Generation:
        self._ensure_loaded()
        messages = [{"role": "user", "content": request.prompt}]
        apply_chat_template = getattr(self._tokenizer, "apply_chat_template", None)
        if apply_chat_template is None:
            prompt = request.prompt
            thinking_control = "no-chat-template"
        else:
            try:
                prompt = apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=self._enable_thinking,
                )
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "MLX tokenizer rejected explicit thinking-mode control"
                ) from exc
            thinking_control = "explicit-chat-template"
        sampler = self._make_sampler(temp=request.temperature)
        responses = self._stream_generate_function(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=request.max_tokens,
            sampler=sampler,
        )
        chunks: list[str] = []
        final_response: Any | None = None
        for response in responses:
            chunks.append(str(response.text))
            if response.finish_reason is not None:
                final_response = response
        if final_response is None:
            raise RuntimeError("MLX-LM generation ended without a completion status")
        if final_response.finish_reason != "stop":
            if final_response.finish_reason == "length":
                raise RuntimeError(
                    f"MLX-LM generation reached the {request.max_tokens}-token limit"
                )
            raise RuntimeError(
                "MLX-LM generation finished with unexpected status "
                f"{final_response.finish_reason!r}"
            )

        raw_text = "".join(chunks).strip()
        text, extraction = _extract_final_response(
            raw_text,
            think_start=getattr(self._tokenizer, "think_start", None),
            think_end=getattr(self._tokenizer, "think_end", None),
        )
        return Generation(
            text=text,
            model=self.model_id,
            metadata={
                "provider": "mlx-lm",
                "execution": "local",
                "thinking_enabled": self._enable_thinking,
                "thinking_control": thinking_control,
                "output_extraction": extraction,
                "finish_reason": final_response.finish_reason,
                "prompt_tokens": final_response.prompt_tokens,
                "generation_tokens": final_response.generation_tokens,
                "prompt_tokens_per_second": round(final_response.prompt_tps, 3),
                "generation_tokens_per_second": round(final_response.generation_tps, 3),
                "peak_memory_gb": round(final_response.peak_memory, 3),
                "raw_output_sha256": hashlib.sha256(raw_text.encode()).hexdigest(),
            },
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
            from mlx_lm import load, stream_generate
            from mlx_lm.sample_utils import make_sampler
        except ImportError as exc:
            raise RuntimeError('MLX-LM is required. Install with: pip install -e ".[mac]"') from exc

        self._model, self._tokenizer = load(
            self._model_reference,
            adapter_path=self._adapter_path,
        )
        self._stream_generate_function = stream_generate
        self._make_sampler = make_sampler


def _extract_final_response(
    raw_text: str,
    *,
    think_start: str | None,
    think_end: str | None,
) -> tuple[str, str]:
    text = raw_text.strip()
    if not text:
        raise RuntimeError("MLX-LM returned an empty response")

    final_markers = [marker for marker in _FINAL_CHANNEL_MARKERS if marker in text]
    if final_markers:
        marker = max(final_markers, key=text.rfind)
        text = text[text.rfind(marker) + len(marker) :].strip()
        extraction = "final-channel"
    else:
        delimiters = list(_FALLBACK_THINKING_DELIMITERS)
        if think_start and think_end:
            delimiters.insert(0, (think_start, think_end))
        latest_end: tuple[int, str] | None = None
        for start, end in dict.fromkeys(delimiters):
            start_position = text.rfind(start)
            end_position = text.rfind(end)
            if start_position >= 0 and end_position < start_position:
                raise RuntimeError("MLX-LM returned an unclosed reasoning channel")
            if end_position >= 0 and (
                latest_end is None or end_position > latest_end[0]
            ):
                latest_end = (end_position, end)
        if latest_end is None:
            extraction = "direct"
        else:
            position, marker = latest_end
            text = text[position + len(marker) :].strip()
            extraction = "reasoning-delimited"

    for marker in _FINAL_CHANNEL_MARKERS:
        if text.startswith(marker):
            text = text[len(marker) :].strip()
            extraction = "final-channel"
    stripped_control = True
    while stripped_control:
        stripped_control = False
        for token in _TRAILING_CONTROL_TOKENS:
            if text.endswith(token):
                text = text[: -len(token)].rstrip()
                stripped_control = True

    leaked_markers = tuple(
        marker
        for pair in _FALLBACK_THINKING_DELIMITERS
        for marker in pair
        if marker in text
    )
    if leaked_markers:
        raise RuntimeError("MLX-LM final response still contains reasoning delimiters")
    if not text:
        raise RuntimeError("MLX-LM returned no final answer after reasoning extraction")
    return text, extraction
