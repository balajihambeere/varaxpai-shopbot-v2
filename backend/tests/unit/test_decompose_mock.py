from src.providers.llm import MockLLMProvider


def test_single_intent_query_is_not_split():
    llm = MockLLMProvider()
    assert llm.decompose("cotton kurta in mint") == ["cotton kurta in mint"]


def test_compound_query_with_and_is_split_into_two():
    llm = MockLLMProvider()
    result = llm.decompose("something warm and something festive")
    assert result == ["something warm", "something festive"]


def test_trailing_fragment_too_short_is_not_treated_as_compound():
    llm = MockLLMProvider()
    # "and" present but the right-hand side is not a real second clause.
    result = llm.decompose("black and ")
    assert len(result) == 1
