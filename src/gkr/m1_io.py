"""Crash-safe text publication helpers for M1 authoring and suite tooling."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def assert_outside_repository(path: Path, repo_root: Path, label: str) -> None:
    resolved = path.resolve()
    root = repo_root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return
    raise ValueError(f"{path}: {label} must be written outside the repository")


def load_jsonl(path: str | Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            rows.append((line_number, value))
    if not rows:
        raise ValueError(f"{path}: JSONL file is empty")
    return rows


def jsonl_text(rows: list[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def canonical_json_text(value: Mapping[str, Any], *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ) + "\n"


def stage_text_file(directory: Path, filename: str, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    handle, raw_path = tempfile.mkstemp(prefix=f".{filename}.", suffix=".tmp", dir=directory)
    path = Path(raw_path)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def publish_text_files(pairs: list[tuple[Path, str]]) -> None:
    """Stage, fsync, then ``os.replace`` each destination.

    On any pre-publish failure, no destination is created by this call. Replacing
    pre-existing destinations is not a pair transaction.
    """

    staged: list[Path] = []
    destinations = [path for path, _text in pairs]
    try:
        for dest, text in pairs:
            dest.parent.mkdir(parents=True, exist_ok=True)
            staged.append(stage_text_file(dest.parent, dest.name, text))
        remaining = list(staged)
        for dest, tmp in zip(destinations, remaining, strict=True):
            os.replace(tmp, dest)
            staged.remove(tmp)
    finally:
        for leftover in staged:
            leftover.unlink(missing_ok=True)


def publish_new_directory(dest: Path, files: Mapping[str, str]) -> None:
    """Atomically publish a new directory of text files, or fail before replace.

    Refuses if ``dest`` already exists. All files are staged and fsynced in a
    sibling temporary directory, then the directory is ``os.replace``d. A
    prepublication failure leaves no new release directory.
    """

    dest = Path(dest)
    if dest.exists():
        raise ValueError(f"{dest}: refuse to overwrite an existing suite directory")
    parent = dest.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{dest.name}.", suffix=".tmp", dir=parent))
    published = False
    try:
        for name, text in files.items():
            staged = stage_text_file(tmp, name, text)
            os.replace(staged, tmp / name)
        fd = os.open(tmp, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, dest)
        published = True
    finally:
        if not published:
            shutil.rmtree(tmp, ignore_errors=True)
