#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from gkr.m1_authoring import assemble_reviewed_cases
from gkr.m1_corpus import DEFAULT_CORPUS_DIR


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble role-separated question, oracle, and semantic-review packets "
            "into canonical v3 scoring JSONL. Question fields cannot be overwritten "
            "by the oracle packet. Review provenance is stamped only from the "
            "review artifact. This does not establish semantic support."
        )
    )
    parser.add_argument("--questions", required=True)
    parser.add_argument("--oracle-drafts", required=True)
    parser.add_argument(
        "--review-artifact",
        required=True,
        action="append",
        dest="review_artifacts",
        help=(
            "Exact raw v2 JSON review batch. Repeat as needed; one matching "
            "approved content binding must cover every current case."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS_DIR))
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Require 120 scenarios and 15 per class in every split.",
    )
    args = parser.parse_args()
    report = assemble_reviewed_cases(
        questions_path=args.questions,
        oracle_drafts_path=args.oracle_drafts,
        review_artifact_paths=args.review_artifacts,
        output_path=args.output,
        corpus_dir=args.corpus_dir,
        require_complete=args.require_complete,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
