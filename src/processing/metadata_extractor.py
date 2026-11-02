import re
from typing import Any


YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")

# Recognizes Scope 1, Scope 2, Scope 3 and Scope 3.1.
# Prevents Scope 3.11, Scope 3.12, etc. from being misclassified.
SCOPE_PATTERN = re.compile(
    r"\bscope\s*([123](?:\.1)?)"
    r"(?![\d.])"
    r"(?:\s+emissions)?"
    r"(?:\s+greenhouse\s+gas\s+emissions)?\b",
    re.IGNORECASE,
)

METRIC_PATTERNS = {
    "greenhouse_gas_emissions": (
        r"\bgreenhouse gas emissions?\b",
        r"\bghg emissions?\b",
        r"\bcarbon emissions?\b",
        r"\bco2 emissions?\b",
        r"\bcarbon footprint\b",
    ),
    "energy": (
        r"\benergy consumption\b",
        r"\brenewable energy\b",
        r"\benergy use\b",
    ),
    "water": (
        r"\bwater consumption\b",
        r"\bwater withdrawal\b",
        r"\bwater discharge\b",
    ),
    "waste": (
        r"\bwaste generated\b",
        r"\bwaste recycled\b",
        r"\brecycling rate\b",
    ),
    "safety": (
        r"\binjury rate\b",
        r"\bwork-related injuries\b",
        r"\blost time injury\b",
        r"\bltir\b",
    ),
    "targets": (
        r"\bemissions reduction target\b",
        r"\bnet[\s-]?zero\b",
        r"\bcarbon neutral(?:ity)?\b",
        r"\bdecarboni[sz]ation target\b",
    ),
}

# Candidate numeric values with commonly used ESG units.
VALUE_PATTERN = re.compile(
    r"(?<!\w)"
    r"(?P<value>-?\d[\d,]*(?:\.\d+)?)"
    r"\s*"
    r"(?P<unit>"
    r"%|percent(?:age)?|"
    r"tonnes?(?:\s+of\s+CO2(?:e|eq)?|\s+of\s+carbon)?|"
    r"metric\s+tons?(?:\s+of\s+CO2(?:e|eq)?)?|"
    r"tCO2e|tCOâ‚‚e|CO2e|COâ‚‚e|"
    r"kg|kilograms?|"
    r"Mt|million\s+tonnes?|"
    r"kWh|MWh|GWh|TWh|"
    r"m3|mÂ³|cubic\s+met(?:er|re)s?|"
    r"lit(?:er|re)s?|"
    r"hectares?|"
    r"employees?|"
    r"incidents?"
    r")(?![A-Za-z])",
    re.IGNORECASE,
)


def identify_company(source_file: str) -> str | None:
    """Infer a likely company name from the report filename."""
    filename = source_file.lower()

    known_companies = {
        "basf": "BASF",
        "eastman": "Eastman",
        "dow": "Dow",
        "covestro": "Covestro",
        "lyondellbasell": "LyondellBasell",
        "sabic": "SABIC",
    }

    for keyword, company_name in known_companies.items():
        if keyword in filename:
            return company_name

    return None


def extract_years(text: str) -> list[int]:
    """Return unique years mentioned in the text."""
    return sorted({int(year) for year in YEAR_PATTERN.findall(text)})


def extract_scopes(text: str) -> list[str]:
    """Identify explicitly mentioned GHG scopes, including Scope 3.1."""
    matches = SCOPE_PATTERN.finditer(text)

    scopes = {
        f"Scope {match.group(1)}"
        for match in matches
    }

    return sorted(scopes)


def extract_metric_categories(text: str) -> list[str]:
    """Identify possible ESG metric categories."""
    categories = []

    for category, patterns in METRIC_PATTERNS.items():
        if any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in patterns
        ):
            categories.append(category)

    return categories


def extract_value_mentions(text: str) -> list[dict[str, Any]]:
    """
    Extract numeric values with recognized units and nearby context.

    These are candidate mentions, not verified metric facts.
    """
    mentions = []

    for match in VALUE_PATTERN.finditer(text):
        start, end = match.span()

        mentions.append(
            {
                "value": match.group("value").replace(",", ""),
                "unit": match.group("unit"),
                "context": text[
                    max(0, start - 100):min(len(text), end + 100)
                ],
            }
        )

    return mentions


def classify_statement(text: str) -> str:
    """
    Classify likely ESG statements using heuristic language patterns.

    Possible results:
    - potential_target
    - potential_actual
    - mixed_or_uncertain
    - unclassified

    These labels are heuristic and must not be treated as verified facts.
    """
    lowered = re.sub(r"\s+", " ", text.lower()).strip()

    if not lowered:
        return "unclassified"

    # Avoid classifying obvious navigation text as ESG evidence.
    navigation_patterns = (
        r"\btable of contents\b",
        r"\bcontents\s+(?:at a glance|combined management|"
        r"consolidated financial statements)\b",
        r"\bpage\s+\d+\s+(?:of\s+\d+)?\b",
    )

    if any(
        re.search(pattern, lowered)
        for pattern in navigation_patterns
    ):
        return "unclassified"

    target_patterns = (
        r"\btargets?\b",
        r"\btargeted\b",
        r"\baim(?:s)? to\b",
        r"\bambition(?:s)? to\b",
        r"\bcommitted to\b",
        r"\bcommitment to\b",
        r"\bplan(?:s|ned)? to\b",
        r"\bwill reduce\b",
        r"\bby\s+(?:20[2-9]\d|21\d{2})\b",
        r"\bnet[\s-]?zero\b",
        r"\bcarbon neutral(?:ity)?\b",
        r"\bdecarboni[sz]ation target\b",
    )

    actual_patterns = (
        r"\breported\b",
        r"\bachieved\b",
        r"\brecorded\b",
        r"\bmeasured\b",
        r"\bactual(?:ly)?\b",
        r"\b(?:increased|decreased|reduced|rose|fell|declined)\b",
        r"\b(?:emissions|energy consumption|water consumption|"
        r"water withdrawal|waste generated|injury rate)\b"
        r".{0,100}\b(?:was|were|reached|totalled|totaled|amounted to)\b",
        r"\b(?:was|were|reached|totalled|totaled|amounted to)\b"
        r".{0,100}\b(?:tonnes?|tco2e|co2e|kwh|mwh|gwh|twh|"
        r"percent|%|kilograms?|kg|cubic metres?|cubic meters?)\b",
    )

    has_target = any(
        re.search(pattern, lowered)
        for pattern in target_patterns
    )

    has_actual = any(
        re.search(pattern, lowered)
        for pattern in actual_patterns
    )

    if has_target and has_actual:
        return "mixed_or_uncertain"

    if has_target:
        return "potential_target"

    if has_actual:
        return "potential_actual"

    return "unclassified"


def extract_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    """Add searchable ESG metadata while preserving original chunk fields."""
    enriched = dict(chunk)

    text = chunk.get("text", "")
    source_file = chunk.get("source_file", "")

    enriched["esg_metadata"] = {
        "company": identify_company(source_file),
        "years_mentioned": extract_years(text),
        "scopes_mentioned": extract_scopes(text),
        "metric_categories": extract_metric_categories(text),
        "value_mentions": extract_value_mentions(text),
        "statement_classification": classify_statement(text),
    }

    return enriched


