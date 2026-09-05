from __future__ import annotations

from types import SimpleNamespace

import pytest

from gkr.ai import GenerationRequest
from gkr.ai.mlx import MLXGenerator
from gkr.ai.mlx.generator import _extract_final_response


def test_mlx_provider_requires_local_model_unless_download_is_explicit() -> None:
    generator = MLXGenerator("mlx-community/not-a-local-directory")

    with pytest.raises(RuntimeError, match="must be a local path"):
        generator.generate(GenerationRequest(prompt="test"))


def test_mlx_provider_disables_thinking_and_returns_only_final_response() -> None:
    generator, tokenizer = _loaded_generator(
        '<think>{"wrong":"reasoning"}</think>{"outcome":"abstain","claims":[]}'
    )

    result = generator.generate(GenerationRequest(prompt="test"))

    assert tokenizer.template_kwargs["enable_thinking"] is False
    assert result.text == '{"outcome":"abstain","claims":[]}'
    assert result.metadata["finish_reason"] == "stop"
    assert result.metadata["thinking_control"] == "explicit-chat-template"
    assert result.metadata["output_extraction"] == "reasoning-delimited"
    assert result.metadata["generation_tokens"] == 24
    assert result.metadata["peak_memory_gb"] == 12.5


def test_mlx_provider_treats_token_limit_as_execution_error() -> None:
    generator, _tokenizer = _loaded_generator(
        '{"verdict":"support',
        finish_reason="length",
    )

    with pytest.raises(RuntimeError, match="reached the 256-token limit"):
        generator.generate(GenerationRequest(prompt="test", max_tokens=256))


def test_mlx_provider_rejects_unknown_completion_status() -> None:
    generator, _tokenizer = _loaded_generator(
        '{"verdict":"supported"}',
        finish_reason="cancelled",
    )

    with pytest.raises(RuntimeError, match="unexpected status 'cancelled'"):
        generator.generate(GenerationRequest(prompt="test"))


def test_mlx_provider_does_not_drop_explicit_thinking_control() -> None:
    generator, _tokenizer = _loaded_generator('{"verdict":"supported"}')
    generator._tokenizer = _RejectingTokenizer()

    with pytest.raises(RuntimeError, match="rejected explicit thinking-mode control"):
        generator.generate(GenerationRequest(prompt="test"))


def test_final_response_extraction_handles_gemma_channel_format() -> None:
    result, extraction = _extract_final_response(
        (
            "<|channel>thought<|message>private reasoning<channel|>"
            '<|channel>final<|message>{"verdict":"supported"}<|end|>'
        ),
        think_start="<|channel>thought",
        think_end="<channel|>",
    )

    assert result == '{"verdict":"supported"}'
    assert extraction == "final-channel"


def test_final_response_extraction_rejects_unclosed_reasoning() -> None:
    with pytest.raises(RuntimeError, match="unclosed reasoning channel"):
        _extract_final_response(
            "<think>private reasoning",
            think_start="<think>",
            think_end="</think>",
        )


class _Tokenizer:
    think_start = "<think>"
    think_end = "</think>"

    def __init__(self) -> None:
        self.template_kwargs: dict[str, object] = {}

    def apply_chat_template(self, _messages: object, **kwargs: object) -> str:
        self.template_kwargs = kwargs
        return "formatted prompt"


class _RejectingTokenizer(_Tokenizer):
    def apply_chat_template(self, _messages: object, **kwargs: object) -> str:
        if "enable_thinking" in kwargs:
            raise TypeError("unsupported keyword")
        return "silently enabled thinking"


def _loaded_generator(
    text: str,
    *,
    finish_reason: str = "stop",
) -> tuple[MLXGenerator, _Tokenizer]:
    generator = MLXGenerator("unused")
    tokenizer = _Tokenizer()
    generator._model = object()
    generator._tokenizer = tokenizer
    generator._make_sampler = lambda **_kwargs: object()
    generator._stream_generate_function = lambda *_args, **_kwargs: iter(
        (
            SimpleNamespace(
                text=text,
                finish_reason=finish_reason,
                prompt_tokens=12,
                generation_tokens=24,
                prompt_tps=100.1234,
                generation_tps=20.5678,
                peak_memory=12.5,
            ),
        )
    )
    return generator, tokenizer
