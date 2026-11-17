"""Unit test suite for the Akshaya Service Desk Agent, BM25 Retriever, and Number Guard."""

from pathlib import Path
import pytest

from src.agent.retriever import BM25Retriever
from src.agent.guard import NumberGuard
from src.agent.prompt_builder import AgentPromptBuilder

KB_PATH = Path(__file__).resolve().parent.parent / "data" / "services_kb.json"


@pytest.fixture(scope="module")
def retriever():
    return BM25Retriever(KB_PATH)


# ---------------------------------------------------------
# 1. BM25 RETRIEVAL TESTS (ENGLISH & HINDI)
# ---------------------------------------------------------
class TestBM25Retriever:
    def test_english_retrieval_all_services(self, retriever):
        queries = [
            ("How to apply for an Income Certificate?", "income_certificate"),
            ("I need to link my mobile number to my Aadhaar card", "aadhaar_mobile_update"),
            ("What is the process for driving license renewal?", "driving_license_renewal"),
            ("How can I add a newborn baby to my family ration card?", "ration_card_member_addition"),
            ("Which documents do I need for a caste certificate?", "caste_certificate"),
        ]
        for query, expected_id in queries:
            results = retriever.retrieve(query, top_k=1)
            assert len(results) == 1
            doc, score = results[0]
            assert doc.service_id == expected_id, f"Query '{query}' returned '{doc.service_id}', expected '{expected_id}'"
            assert score > 0.0

    def test_hindi_retrieval_all_services(self, retriever):
        queries = [
            ("आय प्रमाण पत्र बनवाने की क्या प्रक्रिया है?", "income_certificate"),
            ("आधार कार्ड में मोबाइल नंबर कैसे जोड़ें?", "aadhaar_mobile_update"),
            ("ड्राइविंग लाइसेंस का नवीनीकरण कैसे कराएं?", "driving_license_renewal"),
            ("राशन कार्ड में नए सदस्य का नाम कैसे जोड़ें?", "ration_card_member_addition"),
            ("जाति प्रमाण पत्र के लिए कौन से दस्तावेज चाहिए?", "caste_certificate"),
        ]
        for query, expected_id in queries:
            results = retriever.retrieve(query, top_k=1)
            assert len(results) == 1
            doc, score = results[0]
            assert doc.service_id == expected_id, f"Hindi query '{query}' returned '{doc.service_id}', expected '{expected_id}'"
            assert score > 0.0


# ---------------------------------------------------------
# 2. NUMBER GUARD TESTS
# ---------------------------------------------------------
class TestNumberGuard:
    def test_grounded_response_passes(self, retriever):
        results = retriever.retrieve("Income Certificate", top_k=1)
        doc = results[0][0]  # processing_time_days = 7, fee = null

        # Legitimate response quoting verified days
        legit_text = "To apply for an Income Certificate, submit your salary certificate. It takes 7 days to process."
        audit = NumberGuard.audit(legit_text, doc)
        assert audit.passed is True
        assert len(audit.hallucinated_numbers) == 0

    def test_grounded_hindi_response_passes(self, retriever):
        results = retriever.retrieve("आय प्रमाण पत्र", top_k=1)
        doc = results[0][0]

        legit_text_hi = "आय प्रमाण पत्र के लिए वेतन पर्ची आवश्यक है। यह 7 कार्य दिवसों में बन जाता है।"
        audit = NumberGuard.audit(legit_text_hi, doc)
        assert audit.passed is True
        assert len(audit.hallucinated_numbers) == 0

    def test_hallucinated_fee_is_flagged(self, retriever):
        results = retriever.retrieve("Income Certificate", top_k=1)
        doc = results[0][0]  # fee is null

        # Fabricated fee of ₹500
        hallucinated_text = "The application fee is ₹500 and the processing time is 7 days."
        audit = NumberGuard.audit(hallucinated_text, doc)
        assert audit.passed is False
        assert "500" in audit.hallucinated_numbers
        assert "Number Guard Violation" in audit.reason

    def test_hallucinated_hindi_fee_is_flagged(self, retriever):
        results = retriever.retrieve("आय प्रमाण पत्र", top_k=1)
        doc = results[0][0]

        hallucinated_text_hi = "आय प्रमाण पत्र का आवेदन शुल्क ₹250 है।"
        audit = NumberGuard.audit(hallucinated_text_hi, doc)
        assert audit.passed is False
        assert "250" in audit.hallucinated_numbers
        assert "Number Guard Violation" in audit.reason


# ---------------------------------------------------------
# 3. PROMPT BUILDER TESTS
# ---------------------------------------------------------
class TestAgentPromptBuilder:
    def test_prompt_construction(self, retriever):
        doc = retriever.retrieve("Income Certificate")[0][0]
        sys_p, usr_p, lang = AgentPromptBuilder.build_prompt("Income Certificate query", doc)
        
        assert lang == "en"
        assert "Akshaya service desk voice assistant" in sys_p
        assert "Income Certificate" in sys_p
        assert "Do NOT invent or fabricate any fees" in sys_p
        assert usr_p == "Income Certificate query"

    def test_general_query_prompt_construction(self):
        """When doc is None, system prompt should allow general helpful answers."""
        sys_p, usr_p, lang = AgentPromptBuilder.build_prompt("Who is the Prime Minister of India?", doc=None)
        assert lang == "en"
        assert "helpful and knowledgeable voice assistant" in sys_p
        assert "Income Certificate" not in sys_p
        assert usr_p == "Who is the Prime Minister of India?"
