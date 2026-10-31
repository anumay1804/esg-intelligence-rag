import sys
import types as py_types
from src.generation.api_errors import format_gemini_api_error
from src.generation.gemini_generator import GeminiAnswerGenerator


def test_unsupported_question_with_no_evidence_abstains_without_api_call():
    generator = GeminiAnswerGenerator.__new__(GeminiAnswerGenerator)
    generator.model_name = "gemini-3.1-flash-lite"
    result = generator.generate_answer("Who won the 2022 World Cup?", passages=[])
    assert "couldn't find relevant evidence" in result["answer"]
    assert result["citations"] == []


def test_empty_text_passages_abstain_without_api_call():
    generator = GeminiAnswerGenerator.__new__(GeminiAnswerGenerator)
    generator.model_name = "gemini-3.1-flash-lite"
    result = generator.generate_answer("What is BASF's target?", passages=[{"text": "  "}])
    assert "no usable text" in result["answer"]
    assert result["citations"] == []


def test_gemini_quota_failure_has_actionable_message():
    message = format_gemini_api_error(Exception("429 RESOURCE_EXHAUSTED quota exceeded"), "gemini-3.1-flash-lite")
    assert "quota or rate limit" in message
    assert "Details:" in message


def test_gemini_auth_failure_has_actionable_message():
    message = format_gemini_api_error(Exception("401 invalid API key"), "gemini-3.1-flash-lite")
    assert "authentication failed" in message
    assert "GEMINI_API_KEY" in message


def test_gemini_model_failure_names_configured_model():
    message = format_gemini_api_error(Exception("404 model not found"), "gemini-example")
    assert "gemini-example" in message
    assert "GEMINI_MODEL" in message


def test_generate_answer_wraps_api_failure(monkeypatch):
    # Stub the tiny SDK surface used by generate_answer; no real API call is made.
    fake_google_genai = py_types.ModuleType("google.genai")
    fake_types = py_types.SimpleNamespace(
        GenerateContentConfig=lambda **kwargs: kwargs
    )
    fake_google_genai.types = fake_types
    monkeypatch.setitem(sys.modules, "google.genai", fake_google_genai)

    class FakeModels:
        def generate_content(self, **kwargs):
            raise Exception("429 RESOURCE_EXHAUSTED quota exceeded")

    generator = GeminiAnswerGenerator.__new__(GeminiAnswerGenerator)
    generator.model_name = "gemini-3.1-flash-lite"
    generator.client = py_types.SimpleNamespace(models=FakeModels())

    try:
        generator.generate_answer("What is BASF's target?", [{"text": "BASF target is 25% by 2030."}])
    except RuntimeError as exc:
        assert "quota or rate limit" in str(exc)
    else:
        raise AssertionError("Expected API failure to be converted to a RuntimeError")


def test_prompt_instructs_model_to_abstain_when_evidence_is_irrelevant(monkeypatch):
    fake_google_genai = py_types.ModuleType("google.genai")
    fake_google_genai.types = py_types.SimpleNamespace(
        GenerateContentConfig=lambda **kwargs: kwargs
    )
    monkeypatch.setitem(sys.modules, "google.genai", fake_google_genai)

    class FakeModels:
        prompt = ""
        def generate_content(self, **kwargs):
            self.prompt = kwargs["contents"]
            return py_types.SimpleNamespace(text="I couldn't find sufficient evidence in the indexed reports to answer this question.", candidates=[])

    models = FakeModels()
    generator = GeminiAnswerGenerator.__new__(GeminiAnswerGenerator)
    generator.model_name = "gemini-3.1-flash-lite"
    generator.client = py_types.SimpleNamespace(models=models)
    result = generator.generate_answer(
        "Who won the 2022 World Cup?",
        [{"text": "BASF reported renewable electricity share of 26% in 2024.", "source_file": "basf.pdf", "page": 189}],
    )
    assert "do not answer from general knowledge" in models.prompt.casefold()
    assert "do not directly contain evidence relevant to the question" in models.prompt.casefold()
    assert "sufficient evidence" in result["answer"]
