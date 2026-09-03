#!/usr/bin/env python3
from __future__ import annotations

import argparse

from gkr.m1_authoring import hash_dedup_file, hash_prompt_file, hash_review_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Hash frozen M1 preimages. prompt: CRLF/CR to LF then UTF-8. "
            "review: exact raw bytes. dedup: one JSON object hashed as the "
            "canonical JSON-object preimage."
        )
    )
    parser.add_argument("mode", choices=("prompt", "review", "dedup"))
    parser.add_argument("path")
    args = parser.parse_args()
    if args.mode == "prompt":
        digest = hash_prompt_file(args.path)
    elif args.mode == "review":
        digest = hash_review_file(args.path)
    else:
        digest = hash_dedup_file(args.path)
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
