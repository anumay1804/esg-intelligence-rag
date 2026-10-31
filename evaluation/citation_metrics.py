"""Dependency-light citation and evidence-overlap diagnostics."""
import re


def split_sentences(text):
    """Split an answer into approximate sentences."""
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]



def extract_citation_ids(text):
    """Extract citations such as [S1], [S2], and [S3, S6]."""
    citation_groups = re.findall(r"\[([^\]]+)\]", text)

    citation_ids = set()
    for group in citation_groups:
        citation_ids.update(re.findall(r"\bS\d+\b", group))

    return citation_ids



def calculate_citation_metrics(answer, citations, passages):
    """Check citation validity and estimate evidence overlap."""

    valid_ids = {
        citation["id"]
        for citation in citations
        if citation.get("id")
    }

    used_ids = extract_citation_ids(answer)

    invalid_ids = used_ids - valid_ids

    sentences = split_sentences(answer)

    # Exclude sentences without citation IDs from the coverage numerator.
    # This is a simple heuristic, not a complete factual-claim detector.
    factual_sentences = [
        sentence
        for sentence in sentences
        if len(re.findall(r"\b\d+(?:\.\d+)?%?\b", sentence)) > 0
        or any(
            word in sentence.casefold()
            for word in [
                "target",
                "emissions",
                "reduction",
                "renewable",
                "scope",
                "reported",
                "increased",
                "decreased",
            ]
        )
    ]

    cited_sentences = [
        sentence
        for sentence in factual_sentences
        if extract_citation_ids(sentence)
    ]

    citation_coverage = (
        len(cited_sentences) / len(factual_sentences)
        if factual_sentences
        else 0.0
    )

    # Build a mapping from citation IDs to passage text.
    # The generator creates citation IDs in passage order.
    passage_by_id = {}
    passage_meta_by_id = {}

    for passage in passages:
        text = passage.get("text", "").strip()
        if not text:
            continue

        citation_id = f"S{len(passage_by_id) + 1}"
        passage_by_id[citation_id] = text
        passage_meta_by_id[citation_id] = {
            "source_file": passage.get("source_file", passage.get("source", "Unknown source")),
            "page": passage.get("page", passage.get("page_number", "Unknown")),
        }

    stopwords = {
        "the", "and", "for", "from", "that", "this", "with",
        "was", "were", "are", "has", "have", "had", "its",
        "into", "their", "they", "than", "then", "also", "by",
        "in", "on", "of", "to", "a", "an", "is", "as", "or",
        "it", "be", "at", "based", "about", "approximately",
    }

    overlaps = []
    claim_support_audit = []

    for sentence in factual_sentences:
        cited_ids = extract_citation_ids(sentence)
        claim_text = re.sub(r"\[[^\]]+\]", "", sentence).casefold()

        claim_terms = {
            term
            for term in re.findall(r"\b[a-z0-9.%-]+\b", claim_text)
            if len(term) > 2 and term not in stopwords
        }

        evidence_text = " ".join(
            passage_by_id.get(citation_id, "")
            for citation_id in cited_ids
        ).casefold()

        evidence_terms = {
            term
            for term in re.findall(r"\b[a-z0-9.%-]+\b", evidence_text)
            if len(term) > 2 and term not in stopwords
        }

        overlap = (
            len(claim_terms & evidence_terms) / len(claim_terms)
            if claim_terms else 0.0
        )
        if cited_ids and claim_terms:
            overlaps.append(overlap)
        cited_evidence = []
        for citation_id in sorted(cited_ids):
            if citation_id in passage_by_id:
                cited_evidence.append({
                    "citation_id": citation_id,
                    **passage_meta_by_id.get(citation_id, {}),
                    # Include a short excerpt so a reviewer can audit the claim
                    # without rerunning retrieval or searching the source manually.
                    "excerpt": passage_by_id[citation_id][:3000],
                })

        claim_support_audit.append({
            "sentence": sentence,
            "citation_ids": sorted(cited_ids),
            "has_citation": bool(cited_ids),
            "lexical_overlap_score": round(overlap, 4),
            "cited_evidence": cited_evidence,
            "review_flag": (
                "missing_citation" if not cited_ids else
                "invalid_citation_id" if not cited_ids.issubset(valid_ids) else
                "review_support" if overlap < 0.35 else
                "lexical_overlap_found_manual_support_check_required"
            ),
        })

    return {
        "used_citation_ids": sorted(used_ids),
        "invalid_citation_ids": sorted(invalid_ids),
        "citation_validity": (
            len(used_ids & valid_ids) / len(used_ids)
            if used_ids
            else 0.0
        ),
        "citation_coverage": citation_coverage,
        "mean_evidence_keyword_overlap": (
            sum(overlaps) / len(overlaps)
            if overlaps
            else 0.0
        ),
        "factual_sentence_count": len(factual_sentences),
        "cited_factual_sentence_count": len(cited_sentences),
        "claim_support_audit": claim_support_audit,
        "claims_flagged_for_review": sum(
            1 for item in claim_support_audit
            if item["review_flag"] in {
                "missing_citation", "invalid_citation_id", "review_support"
            }
        ),
        "support_metric_note": (
            "Lexical overlap is a triage heuristic, not entailment or proof of citation support. "
            "Manually inspect flagged and high-impact claims against the cited source passage."
        ),
    }

