"""Optional Gemini-based claim-to-citation support audit.

Run after `python -m evaluation.evaluate_answers`.
This is an LLM-as-judge diagnostic, not ground truth. Review high-impact claims
against the original PDF and use human labels for formal quality reporting.
"""
from __future__ import annotations

import json
import time
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "evaluation" / "answer_evaluation_results.json"
OUTPUT_FILE = BASE_DIR / "evaluation" / "citation_support_results.json"
BATCH_SIZE = 8
ALLOWED_VERDICTS = {
    "supported",
    "partially_supported",
    "unsupported",
    "insufficient_evidence",
}


def extract_json(text: str) -> dict[str, Any]:
    """Parse JSON returned by the judge, tolerating optional code fences."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("The citation judge did not return a JSON object.")
    return json.loads(cleaned[start:end + 1])


def collect_claims(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten factual-sentence audits into stable claim IDs with evidence."""
    claims = []
    for q_index, record in enumerate(records, start=1):
        for c_index, item in enumerate(record.get("claim_support_audit", []), start=1):
            if not item.get("has_citation"):
                # Missing citations are already captured by the deterministic audit.
                continue
            claims.append({
                "claim_id": f"Q{q_index}-C{c_index}",
                "question": record.get("question", ""),
                "claim": item.get("sentence", ""),
                "citation_ids": item.get("citation_ids", []),
                "evidence": item.get("cited_evidence", []),
            })
    return claims


def judge_batch(client, model_name: str, claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from google.genai import types
    payload = json.dumps(claims, ensure_ascii=False, indent=2)
    prompt = f"""
You are auditing citations in an ESG research assistant. For each claim, inspect ONLY the evidence passages listed for that claim.

Return one JSON object with this exact shape:
{{"results": [{{"claim_id":"Q1-C1","verdict":"supported","reason":"short explanation"}}]}}

Allowed verdicts:
- supported: the cited evidence directly supports all material parts of the claim.
- partially_supported: the evidence supports only part of the claim or omits a material detail.
- unsupported: the evidence contradicts the claim or clearly does not support it.
- insufficient_evidence: the provided passage is too incomplete or ambiguous to decide.

Rules:
- Check numbers, units, scope, reporting year, baseline year, and whether a statement is an actual result or a future target.
- Do not use outside knowledge. A keyword match alone is not support.
- Do not assume that a citation supports a claim just because the source is a sustainability report.
- If no evidence is supplied for a citation, use insufficient_evidence.
- Keep reasons concise. Include every claim_id exactly once. Do not add claims.

CLAIMS AND THEIR CITED EVIDENCE:
{payload}
"""
    response = None
    max_attempts = 5

    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=3500,
                    response_mime_type="application/json",
                ),
            )
            break
        except Exception as exc:
            error_text = str(exc).casefold()
            transient = any(
                marker in error_text
                for marker in (
                    "503", "unavailable", "429", "resource_exhausted",
                    "500", "internal server error", "deadline exceeded",
                    "timed out", "timeout"
                )
            )

            if not transient or attempt == max_attempts - 1:
                raise

            delay = min(2 ** attempt * 2, 30)
            print(
                f"Transient Gemini error (attempt {attempt + 1}/"
                f"{max_attempts}): {exc}. Retrying in {delay}s..."
            )
            time.sleep(delay)

    if response is None:
        raise RuntimeError("Gemini returned no response after retries.")
    parsed = extract_json(response.text or "")
    rows = parsed.get("results", [])
    by_id = {str(row.get("claim_id")): row for row in rows if isinstance(row, dict)}
    output = []
    for claim in claims:
        row = by_id.get(claim["claim_id"], {})
        verdict = str(row.get("verdict", "insufficient_evidence")).casefold()
        if verdict not in ALLOWED_VERDICTS:
            verdict = "insufficient_evidence"
        output.append({
            **claim,
            "verdict": verdict,
            "judge_reason": str(row.get("reason", "No explanation returned.")),
        })
    return output


def main() -> None:
    from google import genai

    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is missing. Add it to .env first.")
    if not INPUT_FILE.exists():
        raise SystemExit(
            "Answer evaluation results are missing. Run `python -m evaluation.evaluate_answers` first."
        )

    records = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    claims = collect_claims(records)
    if not claims:
        raise SystemExit(
            "No cited factual claims were found. Re-run answer evaluation after updating it to include cited_evidence."
        )

    client = genai.Client(api_key=api_key)
    results = []
    for offset in range(0, len(claims), BATCH_SIZE):
        batch = claims[offset:offset + BATCH_SIZE]
        print(f"Auditing claims {offset + 1}-{offset + len(batch)} of {len(claims)}...")
        try:
            results.extend(judge_batch(client, model_name, batch))
        except Exception as exc:
            # API failures are operational errors, not evidence verdicts.
            for claim in batch:
                results.append({
                    **claim,
                    "verdict": "judge_error",
                    "judge_reason": f"Judge API call failed after retries; manual review required: {exc}",
                })

    report_verdicts = sorted(ALLOWED_VERDICTS | {"judge_error"})
    counts = {
        verdict: sum(row["verdict"] == verdict for row in results)
        for verdict in report_verdicts
    }
    output = {
        "model": model_name,
        "claims_audited": len(results),
        "verdict_counts": counts,
        "note": (
            "LLM-as-judge results are triage signals, not ground truth. Manually review unsupported, "
            "partially supported, insufficient-evidence, and high-impact numeric claims against the PDF."
        ),
        "results": results,
    }
    OUTPUT_FILE.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nCitation support audit summary:")
    print(json.dumps({"claims_audited": len(results), "verdict_counts": counts}, indent=2))
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()




