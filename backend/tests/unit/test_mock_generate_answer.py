from src.providers.llm import REFUSAL_TEXT, EvidenceItem, MockLLMProvider


def test_empty_evidence_always_refuses():
    llm = MockLLMProvider()
    answer = llm.generate_answer("anything", [], ["anything"], "", [])
    assert answer == REFUSAL_TEXT


def test_answer_only_contains_facts_present_in_evidence_metadata():
    llm = MockLLMProvider()
    evidence = [
        EvidenceItem(
            item_id="p999",
            chunk_type="overview",
            text="irrelevant raw text",
            metadata={"name": "Test Kurta", "price": 1234, "colors": ["red"], "sizes": ["M"]},
        )
    ]
    answer = llm.generate_answer("test kurta", evidence, ["test kurta"], "", [])
    assert "Test Kurta" in answer
    assert "1234" in answer
    assert "red" in answer
    # The grounding contract: no price/color should appear that wasn't in the evidence.
    assert "999999" not in answer


def test_two_items_from_two_sub_queries_both_appear_with_two_parts_prefix():
    llm = MockLLMProvider()
    evidence = [
        EvidenceItem(item_id="p1", chunk_type="overview", text="", metadata={"name": "Shawl"}),
        EvidenceItem(item_id="p2", chunk_type="overview", text="", metadata={"name": "Saree"}),
    ]
    answer = llm.generate_answer("warm and festive", evidence, ["warm", "festive"], "", [])
    assert "Shawl" in answer
    assert "Saree" in answer
    assert answer.startswith("Your question has two parts.")
