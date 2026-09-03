#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from gkr.m1_corpus import DEFAULT_CORPUS_DIR
from gkr.m1_suite import prepare_m1_scoring_suite


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare salted test staging and a redacted public-test candidate "
            "outside the repository. Encrypt the exact staging bytes with local "
            "age before finalization. The preparation report may bind private "
            "plaintext and must not be published. This command emits no suite "
            "manifest and makes no Gate 1 pass claim."
        )
    )
    parser.add_argument("cases")
    parser.add_argument("--plaintext-staging", required=True)
    parser.add_argument("--public-test-candidate", required=True)
    parser.add_argument("--preparation-report", required=True)
    parser.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS_DIR))
    args = parser.parse_args()
    report = prepare_m1_scoring_suite(
        args.cases,
        plaintext_staging_path=args.plaintext_staging,
        public_test_candidate_path=args.public_test_candidate,
        preparation_report_path=args.preparation_report,
        corpus_dir=args.corpus_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
