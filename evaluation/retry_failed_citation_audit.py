"""Retry only failed Gemini citation-support judgments."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from evaluation.validate_citation_support import (
    ALLOWED_VERDICTS,
    judge_batch,
)

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_FILE = BASE_DIR / "evaluation" / "citation_support_results.json"
WORKING_FILE = BASE_DIR / "evaluation" / "citation_support_results.retry_working.json"
BATCH_SIZE = 2


def save_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    load_dotenv(BASE_DIR / ".env")

    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

    if not api_key:
        raise SystemExit("GEMINI_API_KEY is missing from your environment.")

    if not RESULTS_FILE.exists():
        raise SystemExit(f"Existing audit results not found: {RESULTS_FILE}")

    audit = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    results = audit.get("results", [])

    failed = [
        row for row in results
        if row.get("verdict") == "judge_error"
    ]

    if not failed:
        print("No judge_error records found. Nothing to retry.")
        return

    if not all(row.get("evidence") for row in failed):
        raise SystemExit(
            "Some failed records have no evidence. No changes were made."
        )

    print(f"Model: {model_name}")
    print(f"Failed claims to retry: {len(failed)}")
    print(f"Batch size: {BATCH_SIZE}")
    print("Existing successful verdicts will be preserved.")

    client = genai.Client(api_key=api_key)
    results_by_id = {
        row["claim_id"]: row for row in results if row.get("claim_id")
    }

    for offset in range(0, len(failed), BATCH_SIZE):
        batch = failed[offset:offset + BATCH_SIZE]
        ids = [row["claim_id"] for row in batch]
        print(f"\nRetrying {ids}...")

        try:
            judged = judge_batch(client, model_name, batch)
            judged_by_id = {
                row["claim_id"]: row
                for row in judged
                if row.get("claim_id")
            }

            for original in batch:
                claim_id = original["claim_id"]
                updated = judged_by_id.get(claim_id)

                if updated and updated.get("verdict") in ALLOWED_VERDICTS:
                    results_by_id[claim_id] = updated
                    print(f"  {claim_id}: {updated['verdict']}")
                else:
                    print(f"  {claim_id}: no valid verdict; preserving prior error")

        except Exception as exc:
            print(f"  Batch failed; preserving prior errors: {exc}")

        # Save checkpoint separately; the original audit remains untouched.
        audit["results"] = [
            results_by_id.get(row.get("claim_id"), row)
            for row in results
        ]
        save_json(WORKING_FILE, audit)

        if offset + BATCH_SIZE < len(failed):
            time.sleep(3)

    # Reload the checkpoint and replace only the failed records that
    # received valid verdicts. Preserve the existing output structure.
    working = json.loads(WORKING_FILE.read_text(encoding="utf-8"))
    final_results = working["results"]

    verdict_counts = {}
    for row in final_results:
        verdict = row.get("verdict", "unknown")
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

    working["claims_audited"] = len(final_results)
    working["verdict_counts"] = verdict_counts
    working["retry_note"] = (
        "Only prior judge_error claims were retried. Existing completed "
        "verdicts were preserved. Remaining judge_error records require "
        "manual review or a later retry."
    )

    save_json(RESULTS_FILE, working)

    print("\nUpdated audit summary:")
    print(json.dumps({
        "claims_audited": len(final_results),
        "verdict_counts": verdict_counts,
        "working_checkpoint": str(WORKING_FILE),
        "updated_results": str(RESULTS_FILE),
    }, indent=2))


if __name__ == "__main__":
    main()
