#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from gkr.m1_corpus import DEFAULT_CORPUS_DIR
from gkr.m1_suite import finalize_m1_scoring_suite


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize a scoring suite from unchanged prepared staging/public "
            "bytes, a loaded semantic dedup report, and an external age "
            "ciphertext. This runtime does not decrypt age and does not prove "
            "ciphertext/plaintext correspondence. The destination must not "
            "already exist."
        )
    )
    parser.add_argument("cases")
    parser.add_argument("--preparation-report", required=True)
    parser.add_argument("--plaintext-staging", required=True)
    parser.add_argument("--public-test-candidate", required=True)
    parser.add_argument("--semantic-dedup-report", required=True)
    parser.add_argument("--age-recipient-file", required=True)
    parser.add_argument("--age-tool-version", required=True)
    parser.add_argument("--encrypted-artifact", required=True)
    parser.add_argument(
        "--output-dir",
        default="evaluation/m1/suites/gkr-m1-scoring-suite-v1",
    )
    parser.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS_DIR))
    args = parser.parse_args()
    report = finalize_m1_scoring_suite(
        args.cases,
        preparation_report_path=args.preparation_report,
        plaintext_staging_path=args.plaintext_staging,
        public_test_candidate_path=args.public_test_candidate,
        semantic_dedup_report_path=args.semantic_dedup_report,
        age_recipient_file=args.age_recipient_file,
        age_tool_version=args.age_tool_version,
        encrypted_artifact_path=args.encrypted_artifact,
        output_dir=args.output_dir,
        corpus_dir=args.corpus_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
