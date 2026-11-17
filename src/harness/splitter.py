"""Streaming sentence splitter with support for English punctuation and Hindi danda (।)."""

import re
from typing import List

# Common abbreviations that should NOT trigger a sentence split
ABBREVIATIONS = {
    "dr.", "mr.", "mrs.", "ms.", "prof.", "sr.", "jr.", "vs.",
    "e.g.", "i.e.", "etc.", "no.", "st.", "approx.", "rs.", "inr.",
    "डॉ.", "श्री.", "श्रीमती."
}

# Regex to detect sentence boundaries:
# English: [.?!] followed by whitespace or quote
# Hindi: [।॥] (Devanagari danda / double danda) followed by whitespace, quote, or next character
SPLIT_PATTERN = re.compile(
    r'(?<=[.?!।॥])(?:\s+|\n+|$)|(?<=[.?!।॥]["\'])(?:\s+|\n+|$)'
)


class SentenceSplitter:
    """Incrementally splits a stream of tokens into complete sentences.
    
    Handles:
    - Standard English terminators: '.', '?', '!'
    - Hindi Devanagari danda: '।' and '॥'
    - Abbreviations (e.g., 'Dr.', 'Rs.', 'e.g.')
    - Decimals (e.g., '3.14', '₹500.00')
    """

    def __init__(self, min_sentence_chars: int = 5):
        self.min_sentence_chars = min_sentence_chars
        self.buffer = ""

    def _is_valid_sentence_end(self, text: str, split_idx: int) -> bool:
        """Verify that a candidate split position is a genuine sentence boundary."""
        left = text[:split_idx].rstrip()
        
        # Check for empty or too short text
        if len(left) < self.min_sentence_chars:
            return False

        # Check for decimal numbers: e.g. "3." followed by "14"
        if split_idx < len(text):
            if left and left[-1] == '.' and text[split_idx:split_idx+1].isdigit():
                # Preceding char is also a digit
                if len(left) >= 2 and left[-2].isdigit():
                    return False

        # Check for abbreviations at end of candidate
        words = left.split()
        if words:
            last_word = words[-1].lower()
            if last_word in ABBREVIATIONS:
                return False
            # Check single-letter abbreviations like "A."
            if re.match(r'^[a-zA-Z]\.$', last_word):
                return False

        # Check for ellipsis: "..."
        if left.endswith("..") or left.endswith("..."):
            return False

        return True

    def feed(self, token: str) -> List[str]:
        """Feed a new token into the splitter and return any completed sentences."""
        self.buffer += token
        sentences = []

        # Find all candidate split boundaries
        while True:
            found = False
            # Look for boundary matching English or Hindi terminators
            for match in re.finditer(r'([.?!।॥])(\s+|\n+|(?=[A-Z\u0900-\u097F]))', self.buffer):
                end_pos = match.end()
                cand = self.buffer[:match.start() + 1].strip()

                if self._is_valid_sentence_end(cand, len(cand)):
                    sentences.append(cand)
                    self.buffer = self.buffer[end_pos:].lstrip()
                    found = True
                    break  # Found valid boundary; restart search on remaining buffer
                # Otherwise, skip rejected boundary and continue scanning further in self.buffer

            if found:
                continue

            # Fallback for Hindi danda followed directly by characters or end of string if buffer is long
            match_danda = re.search(r'([।॥])', self.buffer)
            if match_danda:
                end_pos = match_danda.end()
                if end_pos < len(self.buffer):
                    cand = self.buffer[:end_pos].strip()
                    if len(cand) >= self.min_sentence_chars and self._is_valid_sentence_end(cand, len(cand)):
                        sentences.append(cand)
                        self.buffer = self.buffer[end_pos:].lstrip()
                        continue

            break

        return sentences

    def flush(self) -> List[str]:
        """Flush any remaining text in the buffer as the final sentence."""
        remaining = self.buffer.strip()
        self.buffer = ""
        if remaining:
            return [remaining]
        return []
