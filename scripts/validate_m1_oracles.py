#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from gkr.m1_corpus import DEFAULT_CORPUS_DIR
from gkr.m1_oracle_validation import validate_m1_oracles


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate M1 v3 oracles against the frozen corpus. "
            "This checks deterministic invariants only and does not establish semantic support."
        )
    )
    parser.add_argument("cases")
    parser.add_argument(
        "--corpus-dir",
        default=str(DEFAULT_CORPUS_DIR),
        help="Directory containing the frozen authority.jsonl",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Require 120 independent scenarios and 15 per class in each split",
    )
    args = parser.parse_args()
    report = validate_m1_oracles(
        args.cases,
        corpus_dir=args.corpus_dir,
        allow_incomplete=not args.require_complete,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
