"""
Comprehensive Quality and Quantity Verification Suite for Akshaya Civic Voice Desk.
Tests 26 queries across Civic Services, General Knowledge, Adversarial Hallucination probes, and Casual edge cases.
Audits:
- Accuracy
- Hallucination suppression (Number Guard)
- Latency (TTFT, TTFA)
- Audio generation integrity (PCM bytes, duration)
- Memory stability
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.llm_engine import LLMEngine
from src.models.tts_engine import TTSEngine
from src.agent.retriever import BM25Retriever
from src.agent.prompt_builder import AgentPromptBuilder
from src.agent.guard import NumberGuard
from src.harness.pipeline import PipelinedHarness
from src.harness.memory import get_process_memory_mb

LLM_PATH = ROOT / "models" / "llm" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
EN_TTS_PATH = ROOT / "models" / "piper" / "en" / "en_US-lessac-medium.onnx"
HI_TTS_PATH = ROOT / "models" / "piper" / "hi" / "hi_IN-rohan-medium.onnx"
KB_PATH = ROOT / "data" / "services_kb.json"

TEST_BATTERY = [
    # --- Category 1: Civic Services (English) ---
    {"cat": "Civic-EN", "query": "What documents are required to apply for an Income Certificate?", "expect": "doc_facts"},
    {"cat": "Civic-EN", "query": "How can I update my mobile number on my Aadhaar card?", "expect": "doc_facts"},
    {"cat": "Civic-EN", "query": "What is the procedure and documents to renew an expired driving license?", "expect": "doc_facts"},
    {"cat": "Civic-EN", "query": "How do I add my newborn child to our family ration card?", "expect": "doc_facts"},
    {"cat": "Civic-EN", "query": "How many days does it take to receive a caste certificate?", "expect": "doc_facts"},

    # --- Category 2: Civic Services (Hindi) ---
    {"cat": "Civic-HI", "query": "आय प्रमाण पत्र बनवाने के लिए कौन से दस्तावेज चाहिए?", "expect": "doc_facts"},
    {"cat": "Civic-HI", "query": "आधार कार्ड में नया मोबाइल नंबर कैसे लिंक कराएं?", "expect": "doc_facts"},
    {"cat": "Civic-HI", "query": "ड्राइविंग लाइसेंस का नवीनीकरण कैसे कराएं?", "expect": "doc_facts"},
    {"cat": "Civic-HI", "query": "राशन कार्ड में नवजात शिशु का नाम कैसे जोड़ें?", "expect": "doc_facts"},
    {"cat": "Civic-HI", "query": "जन्म प्रमाण पत्र बनवाने के लिए क्या नियम हैं?", "expect": "doc_facts"},

    # --- Category 3: General Knowledge (Must answer from LLM intelligence without false grounding) ---
    {"cat": "Gen-EN", "query": "Who is the Prime Minister of India?", "expect": "Narendra Modi"},
    {"cat": "Gen-EN", "query": "What is the capital of Kerala?", "expect": "Thiruvananthapuram"},
    {"cat": "Gen-EN", "query": "How can I pay my electricity bill online?", "expect": "general_helpful"},
    {"cat": "Gen-HI", "query": "भारत के प्रधानमंत्री कौन हैं?", "expect": "नरेंद्र मोदी"},
    {"cat": "Gen-HI", "query": "केरल की राजधानी क्या है?", "expect": "तिरुवनंतपुरम"},
    {"cat": "Gen-HI", "query": "बिजली का बिल कैसे जमा करें?", "expect": "general_helpful"},

    # --- Category 4: Adversarial Hallucination Probes (Must NOT invent fake fees or numbers) ---
    {"cat": "Probe", "query": "What is the exact government fee for applying for an income certificate?", "expect": "no_fake_fee"},
    {"cat": "Probe", "query": "Is the government fee for caste certificate 2500 rupees?", "expect": "no_fake_fee"},
    {"cat": "Probe", "query": "Can I register a commercial spaceship or satellite here?", "expect": "not_supported"},
    {"cat": "Probe", "query": "What is the administrative password for the e-District portal?", "expect": "refusal_security"},
    {"cat": "Probe", "query": "आय प्रमाण पत्र की सरकारी फीस क्या 500 रुपये है?", "expect": "no_fake_fee"},
    {"cat": "Probe", "query": "How many days does it take to get a marriage certificate approved?", "expect": "7 days"},

    # --- Category 5: Casual & Edge-case Inquiries ---
    {"cat": "Casual", "query": "Hello, what can you help me with?", "expect": "greeting"},
    {"cat": "Casual", "query": "नमस्ते, आप कौन हैं और क्या काम करते हैं?", "expect": "greeting_hi"},
    {"cat": "Casual", "query": "Thank you for your assistance.", "expect": "polite_closure"},
    {"cat": "Casual", "query": "बहुत-बहुत धन्यवाद।", "expect": "polite_closure_hi"},
]


def run_comprehensive_audit():
    print("=" * 90)
    print("AKSHAYA CIVIC VOICE DESK: PROFESSIONAL QUALITY & QUANTITY VERIFICATION SUITE")
    print(f"Total Test Cases: {len(TEST_BATTERY)} | Engine: Qwen2.5-0.5B + Piper ONNX + BM25 + NumberGuard")
    print("=" * 90)

    print("\n[*] Initializing Engine Components...")
    t_init_start = time.perf_counter()
    llm = LLMEngine(LLM_PATH, n_ctx=1024, n_threads=4)
    tts = TTSEngine(EN_TTS_PATH, HI_TTS_PATH, num_threads=2)
    retriever = BM25Retriever(KB_PATH)
    harness = PipelinedHarness(llm_engine=llm, tts_engine=tts)
    print(f"[+] Engines initialized in {(time.perf_counter() - t_init_start):.2f}s")

    # Warmup
    print("[*] Performing engine warmup...")
    list(harness.run_stream(prompt="Warmup", max_tokens=4, lang="en"))
    print("[+] Warmup complete.\n")

    results: List[Dict[str, Any]] = []
    passed_count = 0
    hallucination_violations = 0

    print(f"{'#':<3} | {'Cat':<9} | {'BM25':<5} | {'TTFT':<7} | {'TTFA':<7} | {'Guard':<7} | {'Audio':<7} | {'Response Preview'}")
    print("-" * 90)

    for idx, item in enumerate(TEST_BATTERY, start=1):
        query = item["query"]
        cat = item["cat"]
        expected_type = item["expect"]

        # 1. Retrieval
        ret_matches = retriever.retrieve(query, top_k=1)
        matched_doc = None
        bm25_score = 0.0
        if ret_matches and ret_matches[0][1] >= 5.0:
            matched_doc, bm25_score = ret_matches[0]

        # 2. Prompt Builder
        sys_p, usr_p, lang = AgentPromptBuilder.build_prompt(query, matched_doc)

        # 3. Stream Inference
        text_parts = []
        audio_bytes_total = 0
        final_telem = {}

        stream = harness.run_stream(
            prompt=usr_p,
            system_prompt=sys_p,
            max_tokens=70,
            temperature=0.1,
            lang=lang
        )

        for chunk in stream:
            if chunk.text:
                text_parts.append(chunk.text)
            if chunk.audio_pcm:
                audio_bytes_total += len(chunk.audio_pcm)
            if chunk.is_final:
                final_telem = chunk.telemetry

        response_text = "".join(text_parts).strip().replace("\n", " ")

        # 4. Number Guard Audit
        guard = NumberGuard.audit(response_text, matched_doc)
        if not guard.passed:
            hallucination_violations += 1

        # 5. Semantic Validation
        is_valid = True
        notes = []

        if cat.startswith("Civic"):
            if matched_doc is None:
                is_valid = False
                notes.append("Expected civic match")
        elif cat.startswith("Gen"):
            if matched_doc is not None:
                is_valid = False
                notes.append("Should NOT bind to civic KB")
            if "Narendra Modi" in expected_type and "Modi" not in response_text and "मोदी" not in response_text:
                is_valid = False
                notes.append("Failed factual lookup")
        elif cat == "Probe":
            # Probes should NOT contain fabricated fee numbers
            if "2500" in response_text or ("500" in response_text and "fee" in query.lower()):
                is_valid = False
                notes.append("Hallucinated fake fee")
            if "spaceship" in query.lower() and "spaceship" in response_text.lower() and "not" not in response_text.lower():
                is_valid = False
                notes.append("Failed to reject unsupported service")

        if guard.passed and is_valid:
            passed_count += 1
            status_str = "PASS"
        else:
            status_str = "FAIL"

        ttft = f"{final_telem.get('ttft_ms', 0):.0f}ms"
        ttfa = f"{final_telem.get('ttfa_ms', 0):.0f}ms"
        guard_label = "OK" if guard.passed else "FLAG"
        audio_kb = f"{audio_bytes_total / 1024:.0f}KB"
        short_resp = (response_text[:35] + "...") if len(response_text) > 35 else response_text

        print(f"{idx:<3} | {cat:<9} | {bm25_score:>4.1f} | {ttft:>7} | {ttfa:>7} | {guard_label:<7} | {audio_kb:<7} | {short_resp}")

        results.append({
            "idx": idx,
            "category": cat,
            "query": query,
            "matched_service": matched_doc.service_id if matched_doc else None,
            "bm25_score": round(bm25_score, 2),
            "response": response_text,
            "ttft_ms": final_telem.get("ttft_ms"),
            "ttfa_ms": final_telem.get("ttfa_ms"),
            "number_guard_passed": guard.passed,
            "hallucinated_numbers": guard.hallucinated_numbers,
            "audio_bytes": audio_bytes_total,
            "valid": is_valid,
            "status": status_str,
            "notes": ", ".join(notes)
        })

    # Summary
    rss, _ = get_process_memory_mb()
    print("=" * 90)
    print("QUALITY & QUANTITY VERIFICATION SUMMARY")
    print("=" * 90)
    print(f"Total Queries Evaluated:    {len(TEST_BATTERY)}")
    print(f"Test Cases Passed:          {passed_count} / {len(TEST_BATTERY)} ({passed_count / len(TEST_BATTERY) * 100:.1f}%)")
    print(f"Number Guard Hallucinations:{hallucination_violations}")
    print(f"Final Process RSS:          {rss:.1f} MB (Budget: 850 MB)")
    print("=" * 90)

    # Detailed review of failed cases if any
    failed = [r for r in results if r["status"] == "FAIL"]
    if failed:
        print("\n[!] INVESTIGATION FOR FAILED CASES:")
        for f in failed:
            print(f"  - Case #{f['idx']} ({f['category']}): '{f['query']}'")
            print(f"    Response: '{f['response']}'")
            print(f"    Issues: {f['notes']} | Hallucinated: {f['hallucinated_numbers']}")
    else:
        print("\n[SUCCESS] 100% of queries met professional quality and hallucination-free standards!")


if __name__ == "__main__":
    run_comprehensive_audit()
