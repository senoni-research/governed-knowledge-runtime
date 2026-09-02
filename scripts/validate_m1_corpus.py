#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from gkr.m1_corpus import DEFAULT_CORPUS_DIR, validate_m1_corpus, write_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate or regenerate the M1 authority corpus")
    parser.add_argument(
        "--corpus-dir",
        default=str(DEFAULT_CORPUS_DIR),
        help="Directory containing authority.jsonl and corpus-manifest.json",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Regenerate authority.jsonl and corpus-manifest.json from the deterministic builder",
    )
    args = parser.parse_args()
    if args.write:
        write_corpus(args.corpus_dir)
    report = validate_m1_corpus(args.corpus_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
