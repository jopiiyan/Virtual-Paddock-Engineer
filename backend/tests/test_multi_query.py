"""Unit tests for query expansion (no LLM / network — a fake chat model is injected)."""

from backend.retrieval.multi_query import expand_query


class _FakeLLM:
    def __init__(self, text):
        self._text = text

    def invoke(self, _prompt):
        class _Msg:
            content = self._text
        return _Msg()


class _BrokenLLM:
    def invoke(self, _prompt):
        raise RuntimeError("model down")


def test_expand_keeps_original_first_and_parses_lines():
    llm = _FakeLLM("HAM final stint pace\nVER final stint pace")
    out = expand_query("compare HAM and VER", n=3, mode="decompose", llm=llm)
    assert out[0] == "compare HAM and VER"                 # original always first
    assert "HAM final stint pace" in out and "VER final stint pace" in out


def test_expand_strips_numbering_and_bullets():
    llm = _FakeLLM("1. first query\n- second query\n2) third query")
    out = expand_query("q", n=3, mode="paraphrase", llm=llm)
    assert out[1:] == ["first query", "second query", "third query"]


def test_expand_dedupes_case_insensitively():
    llm = _FakeLLM("Compare HAM and VER\nsomething else")
    out = expand_query("compare HAM and VER", n=3, mode="paraphrase", llm=llm)
    # the echo of the original (different case) must not appear twice
    assert sum(1 for q in out if q.lower() == "compare ham and ver") == 1


def test_expand_caps_at_n():
    llm = _FakeLLM("a\nb\nc\nd\ne")
    out = expand_query("q", n=2, mode="paraphrase", llm=llm)
    assert len(out) == 3                                   # original + at most n=2


def test_expand_degrades_gracefully_on_llm_error():
    out = expand_query("q", n=3, mode="decompose", llm=_BrokenLLM())
    assert out == ["q"]                                    # falls back to original only
