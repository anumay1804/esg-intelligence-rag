"""User-safe diagnostics for Gemini API failures."""


def format_gemini_api_error(exc: Exception, model_name: str) -> str:
    detail = str(exc).strip() or type(exc).__name__
    lowered = detail.casefold()
    if any(term in lowered for term in ("quota", "resource_exhausted", "429", "rate limit")):
        guidance = "Gemini API quota or rate limit was reached. Check your Google AI Studio quota or retry later."
    elif any(term in lowered for term in ("api key", "unauthenticated", "permission_denied", "401", "403")):
        guidance = "Gemini authentication failed. Check that GEMINI_API_KEY is valid and enabled for this model."
    elif any(term in lowered for term in ("not found", "404", "model")):
        guidance = f"The configured Gemini model may be unavailable: {model_name}. Check GEMINI_MODEL."
    else:
        guidance = "Gemini request failed. Check your network connection, model availability, and API status."
    return f"{guidance} Details: {detail}"
