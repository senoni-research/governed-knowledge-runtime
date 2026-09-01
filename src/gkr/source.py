from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from gkr.schemas import KnowledgeRecord, source_digest

HashScope = Literal["statement", "raw_bytes"]


@dataclass(frozen=True)
class SourceArtifact:
    """Locally read source material whose digests were computed by the runtime."""

    source_uri: str
    source_version: str
    content_type: str
    local_path: str
    raw_sha256: str
    raw_byte_length: int
    extracted_text_sha256: str | None = None

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        source_uri: str,
        source_version: str,
        content_type: str,
        extracted_text: str | None = None,
    ) -> SourceArtifact:
        normalized_uri = _required_text("source_uri", source_uri)
        normalized_version = _required_text("source_version", source_version)
        normalized_content_type = _required_text("content_type", content_type)
        resolved = Path(path).expanduser().resolve()
        raw_bytes = resolved.read_bytes()
        return cls(
            source_uri=normalized_uri,
            source_version=normalized_version,
            content_type=normalized_content_type,
            local_path=str(resolved),
            raw_sha256=hashlib.sha256(raw_bytes).hexdigest(),
            raw_byte_length=len(raw_bytes),
            extracted_text_sha256=source_digest(extracted_text) if extracted_text else None,
        )

    def verify_current_bytes(self) -> None:
        """Reject source material changed after this artifact was computed."""

        current_bytes = Path(self.local_path).expanduser().resolve().read_bytes()
        current_digest = hashlib.sha256(current_bytes).hexdigest()
        if current_digest != self.raw_sha256 or len(current_bytes) != self.raw_byte_length:
            raise ValueError("Source artifact bytes changed after digest computation")

    @classmethod
    def from_descriptor(
        cls,
        value: dict[str, Any],
        *,
        base_directory: Path,
    ) -> SourceArtifact:
        required = ("path", "source_uri", "source_version", "content_type")
        missing = [key for key in required if value.get(key) in (None, "")]
        if missing:
            raise ValueError(f"Source artifact is missing fields: {', '.join(missing)}")
        source_path = Path(str(value["path"]))
        if source_path.is_absolute():
            raise ValueError("Source artifact path must be relative to the import directory")
        resolved = (base_directory / source_path).expanduser().resolve()
        root = base_directory.expanduser().resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("Source artifact path must stay inside the import directory")
        extracted_text = value.get("extracted_text")
        return cls.from_file(
            resolved,
            source_uri=str(value["source_uri"]),
            source_version=str(value["source_version"]),
            content_type=str(value["content_type"]),
            extracted_text=str(extracted_text) if extracted_text is not None else None,
        )


def verify_record_source(
    record: KnowledgeRecord,
    artifact: SourceArtifact | None = None,
) -> None:
    """Verify a record digest against source material computed inside the runtime."""

    hash_scope = record.metadata.get("hash_scope")
    if hash_scope == "statement":
        expected = source_digest(record.statement)
        if record.source_hash != expected:
            raise ValueError(
                f"{record.reference}: source_hash does not match the canonical statement"
            )
        if artifact is not None:
            raise ValueError("A statement-scoped record must not include a raw source artifact")
        return
    if hash_scope != "raw_bytes":
        raise ValueError(f"{record.reference}: metadata.hash_scope must be statement or raw_bytes")
    if artifact is None:
        raise ValueError(f"{record.reference}: raw_bytes hash scope requires a SourceArtifact")
    _required_text("source_uri", artifact.source_uri)
    _required_text("source_version", artifact.source_version)
    _required_text("content_type", artifact.content_type)
    artifact.verify_current_bytes()
    if record.source_uri != artifact.source_uri:
        raise ValueError(f"{record.reference}: source_uri does not match the SourceArtifact")
    if record.source_hash != artifact.raw_sha256:
        raise ValueError(f"{record.reference}: supplied source_hash does not match source bytes")

    expected_metadata = {
        "source_version": artifact.source_version,
        "content_type": artifact.content_type,
        "raw_byte_length": artifact.raw_byte_length,
    }
    for key, expected in expected_metadata.items():
        if record.metadata.get(key) != expected:
            raise ValueError(f"{record.reference}: metadata.{key} does not match source material")
    supplied_text_hash = record.metadata.get("extracted_text_sha256")
    if supplied_text_hash != artifact.extracted_text_sha256:
        raise ValueError(
            f"{record.reference}: metadata.extracted_text_sha256 does not match extracted text"
        )


def _required_text(field_name: str, value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    if len(normalized) > 512 or any(ord(character) < 32 for character in normalized):
        raise ValueError(f"{field_name} contains invalid characters or is too long")
    return normalized
