"""Number Guard auditing LLM generation against retrieved factual context."""

import re
from dataclasses import dataclass, field
from typing import Set, List, Optional, Any
from src.agent.retriever import ServiceDocument

# Common English number words mapped to integers
EN_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "fifteen": 15, "twenty": 20, "forty": 40, "fifty": 50,
    "hundred": 100, "thousand": 1000
}

# Common Hindi number words mapped to integers
HI_NUMBER_WORDS = {
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5,
    "पांच": 5, "छह": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10,
    "पंद्रह": 15, "बीस": 20, "चालीस": 40, "पचास": 50, "सौ": 100, "हज़ार": 1000
}


@dataclass
class GuardResult:
    """Result of Number Guard verification."""
    passed: bool
    allowed_numbers: List[str] = field(default_factory=list)
    found_numbers: List[str] = field(default_factory=list)
    hallucinated_numbers: List[str] = field(default_factory=list)
    reason: Optional[str] = None


class NumberGuard:
    """Detects and flags unauthorized numbers or fabricated fees in model responses.
    
    Ensures that any numerical claim made by the assistant (processing days, fees, 
    document counts) is strictly present in the retrieved knowledge base document.
    """

    @classmethod
    def extract_numbers_from_text(cls, text: str) -> Set[str]:
        """Extract all numeric representations (digits and currency strings) from text."""
        numbers: Set[str] = set()

        # 1. Match numeric digits: e.g. "500", "7", "3.14", "1A"
        digit_matches = re.findall(r'\b\d+(?:[.,]\d+)*\b', text)
        for m in digit_matches:
            # Clean commas
            clean_num = m.replace(',', '').rstrip('.')
            if clean_num:
                numbers.add(clean_num)

        # 2. Match currency expressions explicitly: ₹500, Rs 500, 500 rupees
        curr_matches = re.findall(r'(?:₹|Rs\.?|INR|\$)\s*(\d+)', text, flags=re.IGNORECASE)
        for c in curr_matches:
            numbers.add(c)

        # 3. Match English number words
        for word, val in EN_NUMBER_WORDS.items():
            if re.search(rf'\b{word}\b', text, flags=re.IGNORECASE):
                numbers.add(str(val))

        # 4. Match Hindi number words
        for word, val in HI_NUMBER_WORDS.items():
            if re.search(rf'(?:^|\s){word}(?:$|\s|[।,\.])', text):
                numbers.add(str(val))

        return numbers

    @classmethod
    def get_allowed_numbers(cls, doc: ServiceDocument) -> Set[str]:
        """Collect all legitimately allowed numbers from the service document."""
        context_text = f"{doc.get_search_text()} {doc.get_context_summary('en')} {doc.get_context_summary('hi')}"
        allowed = cls.extract_numbers_from_text(context_text)

        # Always explicitly allow the exact processing days
        allowed.add(str(doc.processing_time_days))
        
        # If fee is specified, allow it; if null, NO fee number is added
        if doc.fee_inr is not None:
            allowed.add(str(doc.fee_inr))

        # Common structural numbers (like list numbering 1-10 for document items)
        for i in range(1, 11):
            allowed.add(str(i))

        return allowed

    @classmethod
    def audit(cls, output_text: str, doc: Optional[ServiceDocument] = None) -> GuardResult:
        """Audit the model's generated response against the retrieved document."""
        if doc is None:
            return GuardResult(passed=True, reason="General inquiry - unconstrained by civic KB.")

        allowed = cls.get_allowed_numbers(doc)
        found = cls.extract_numbers_from_text(output_text)

        # Check for unauthorized / hallucinated numbers
        hallucinated = set()
        for num in found:
            # Check direct match
            if num not in allowed:
                # Also check integer equivalence (e.g. "07" vs "7")
                try:
                    num_int = str(int(float(num)))
                    if num_int not in allowed:
                        hallucinated.add(num)
                except ValueError:
                    hallucinated.add(num)

        passed = len(hallucinated) == 0
        reason = None
        if not passed:
            sorted_hal = sorted(list(hallucinated))
            reason = f"Number Guard Violation: Output contains unverified number(s) {sorted_hal} not present in service KB."

        return GuardResult(
            passed=passed,
            allowed_numbers=sorted(list(allowed)),
            found_numbers=sorted(list(found)),
            hallucinated_numbers=sorted(list(hallucinated)),
            reason=reason
        )
