import re
from typing import Any


def split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs and remove empty sections."""
    paragraphs = re.split(r"\n\s*\n+", text)

    return [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]


def split_long_text(text: str, max_chars: int) -> list[str]:
    """Split an oversized paragraph into smaller pieces."""
    pieces = []
    remaining = text.strip()

    while len(remaining) > max_chars:
        split_at = remaining.rfind(" ", 0, max_chars)

        if split_at <= 0:
            split_at = max_chars

        pieces.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()

    if remaining:
        pieces.append(remaining)

    return pieces


def chunk_page(
    text: str,
    max_chars: int = 1800,
    overlap_chars: int = 250,
) -> list[str]:
    """
    Create paragraph-aware chunks from one PDF page.

    Chunks remain within their original page so that page-level
    citations are straightforward.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero.")

    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError(
            "overlap_chars must be non-negative and smaller than max_chars."
        )

    paragraphs = split_into_paragraphs(text)

    if not paragraphs:
        return []

    # Split paragraphs that are larger than the chunk limit.
    units = []

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            units.extend(split_long_text(paragraph, max_chars))
        else:
            units.append(paragraph)

    chunks = []
    current = ""

    for unit in units:
        candidate = f"{current}\n\n{unit}" if current else unit

        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)

        # Retain a small amount of previous context.
        overlap = current[-overlap_chars:] if overlap_chars else ""
        current = f"{overlap}\n\n{unit}".strip() if overlap else unit

        # An overlap can make the combined text exceed the limit.
        if len(current) > max_chars:
            current = unit

    if current:
        chunks.append(current)

    return chunks


def chunk_document(
    document: dict[str, Any],
    max_chars: int = 1800,
    overlap_chars: int = 250,
) -> list[dict[str, Any]]:
    """
    Chunk a parsed document while preserving source and page metadata.
    """
    source_file = document.get("source_file", "unknown.pdf")
    source_path = document.get("source_path", "")
    document_metadata = document.get("metadata", {})

    all_chunks = []

    for page_data in document.get("pages", []):
        page_number = page_data["page"]
        page_text = page_data.get("text", "").strip()

        if not page_text:
            continue

        page_chunks = chunk_page(
            text=page_text,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )

        for chunk_number, chunk_text in enumerate(page_chunks, start=1):
            all_chunks.append(
                {
                    "chunk_id": (
                        f"{source_file}:page_{page_number}:"
                        f"chunk_{chunk_number}"
                    ),
                    "source_file": source_file,
                    "source_path": source_path,
                    "page": page_number,
                    "chunk_number": chunk_number,
                    "text": chunk_text,
                    "char_count": len(chunk_text),
                    "metadata": document_metadata,
                }
            )

    return all_chunks