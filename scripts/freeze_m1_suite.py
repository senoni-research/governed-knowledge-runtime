#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from gkr.m1_corpus import DEFAULT_CORPUS_DIR
from gkr.m1_freeze import freeze_m1_suite


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate M1 v3 cases and write a plaintext staging bundle plus the redacted "
            "public form. Plaintext staging is not a sealed bundle and must live outside "
            "the repository. Scoring mode also requires an age-x25519-v1 externally "
            "encrypted artifact descriptor. Gate 1 binds no retrieval-configuration "
            "digest. This command does not encrypt, does not decrypt, and does not "
            "prove ciphertext decrypts to the staged bytes."
        )
    )
    parser.add_argument("cases")
    parser.add_argument(
        "--mode",
        required=True,
        choices=("scoring", "conformance"),
        help="Explicit freeze mode. Tags cannot select conformance or bypass scoring checks.",
    )
    parser.add_argument("--plaintext-staging", required=True)
    parser.add_argument("--public-test", required=True)
    parser.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS_DIR))
    parser.add_argument("--encryption-profile-id")
    parser.add_argument("--encrypted-tool-family")
    parser.add_argument("--encrypted-tool-version")
    parser.add_argument("--encrypted-recipient-key-fingerprint-sha256")
    parser.add_argument("--encrypted-artifact-path")
    parser.add_argument("--encrypted-artifact-sha256")
    parser.add_argument("--dedup-report-sha256")
    args = parser.parse_args()
    encrypted = None
    fields = {
        "encryption_profile_id": args.encryption_profile_id,
        "tool_family": args.encrypted_tool_family,
        "tool_version": args.encrypted_tool_version,
        "recipient_key_fingerprint_sha256": args.encrypted_recipient_key_fingerprint_sha256,
        "encrypted_artifact_path": args.encrypted_artifact_path,
        "encrypted_artifact_sha256": args.encrypted_artifact_sha256,
    }
    if any(fields.values()):
        encrypted = {key: value or "" for key, value in fields.items()}
    report = freeze_m1_suite(
        args.cases,
        mode=args.mode,
        plaintext_staging_path=args.plaintext_staging,
        public_test_path=args.public_test,
        encrypted_artifact=encrypted,
        dedup_report_sha256=args.dedup_report_sha256,
        corpus_dir=args.corpus_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
