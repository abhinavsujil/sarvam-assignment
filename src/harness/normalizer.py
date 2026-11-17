"""Text normalizer for TTS preprocessing in English and Hindi."""

import re
from typing import Dict

# English number maps
ONES_EN = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
    10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen"
}
TENS_EN = {
    20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
    60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"
}

# Hindi number maps: Full 0 to 99 lookup table + 100
HINDI_NUMS: Dict[int, str] = {
    0: "शून्य", 1: "एक", 2: "दो", 3: "तीन", 4: "चार", 5: "पाँच",
    6: "छह", 7: "सात", 8: "आठ", 9: "नौ", 10: "दस",
    11: "ग्यारह", 12: "बारह", 13: "तेरह", 14: "चौदह", 15: "पंद्रह",
    16: "सोलह", 17: "सत्रह", 18: "अठारह", 19: "उन्नीस", 20: "बीस",
    21: "इक्कीस", 22: "बाईस", 23: "तेईस", 24: "चौबीस", 25: "पच्चीस",
    26: "छब्बीस", 27: "सत्ताईस", 28: "अट्ठाईस", 29: "उनतीस", 30: "तीस",
    31: "इकतीस", 32: "बत्तीस", 33: "तैंतीस", 34: "चौंतीस", 35: "पैंतीस",
    36: "छत्तीस", 37: "सैंतीस", 38: "अड़तीस", 39: "उनतालीस", 40: "चालीस",
    41: "इकतालीस", 42: "बयालीस", 43: "तैंतालीस", 44: "चवालीस", 45: "पैंतालीस",
    46: "छियालीस", 47: "सैंतालीस", 48: "अड़तालीस", 49: "उनचास", 50: "पचास",
    51: "इक्यावन", 52: "बावन", 53: "तिरपन", 54: "चौवन", 55: "पचपन",
    56: "छप्पन", 57: "सत्तावन", 58: "अट्ठावन", 59: "उनसठ", 60: "साठ",
    61: "इकसठ", 62: "बासठ", 63: "तिरसठ", 64: "चौंसठ", 65: "पैंसठ",
    66: "छियासठ", 67: "सरसठ", 68: "अड़सठ", 69: "उनहत्तर", 70: "सत्तर",
    71: "इकहत्तर", 72: "बहत्तर", 73: "तिहत्तर", 74: "चौहत्तर", 75: "पचहत्तर",
    76: "छिहत्तर", 77: "सतहत्तर", 78: "अठहत्तर", 79: "उनासी", 80: "अस्सी",
    81: "इक्यासी", 82: "बयासी", 83: "तिरासी", 84: "चौरासी", 85: "पचासी",
    86: "छियासी", 87: "सत्तासी", 88: "अट्ठासी", 89: "नवासी", 90: "नब्बे",
    91: "इक्यानवे", 92: "बानवे", 93: "तिरानवे", 94: "चौरानवे", 95: "पंचानवे",
    96: "छियानवे", 97: "सत्तानवे", 98: "अट्ठानवे", 99: "निन्यानवे", 100: "एक सौ"
}


def num_to_en(n: int) -> str:
    """Convert an integer up to 999,999 to English words."""
    if n in ONES_EN:
        return ONES_EN[n]
    if n < 100:
        ten, rem = divmod(n, 10)
        return TENS_EN[ten * 10] + (" " + ONES_EN[rem] if rem else "")
    if n < 1000:
        hundred, rem = divmod(n, 100)
        res = ONES_EN[hundred] + " hundred"
        if rem:
            res += " " + num_to_en(rem)
        return res
    if n < 100000:
        thousand, rem = divmod(n, 1000)
        res = num_to_en(thousand) + " thousand"
        if rem:
            res += " " + num_to_en(rem)
        return res
    if n < 10000000:
        lakh, rem = divmod(n, 100000)
        res = num_to_en(lakh) + " lakh"
        if rem:
            res += " " + num_to_en(rem)
        return res
    return str(n)


def num_to_hi(n: int) -> str:
    """Convert an integer to Hindi Devanagari words."""
    if n in HINDI_NUMS:
        return HINDI_NUMS[n]
    if n < 1000:
        hundred, rem = divmod(n, 100)
        h_word = (HINDI_NUMS.get(hundred, str(hundred)) + " सौ") if hundred > 1 else "एक सौ"
        if rem == 0:
            return h_word
        rem_word = HINDI_NUMS.get(rem, str(rem))
        return f"{h_word} {rem_word}"
    if n < 100000:
        thousand, rem = divmod(n, 1000)
        t_word = (HINDI_NUMS.get(thousand, str(thousand)) + " हज़ार") if thousand in HINDI_NUMS else (num_to_hi(thousand) + " हज़ार")
        if rem == 0:
            return t_word
        return f"{t_word} {num_to_hi(rem)}"
    if n < 10000000:
        lakh, rem = divmod(n, 100000)
        l_word = (HINDI_NUMS.get(lakh, str(lakh)) + " लाख") if lakh in HINDI_NUMS else (num_to_hi(lakh) + " लाख")
        if rem == 0:
            return l_word
        return f"{l_word} {num_to_hi(rem)}"
    return str(n)


class TextNormalizer:
    """Sanitizes and normalizes text for Piper TTS in English and Hindi."""

    @staticmethod
    def is_hindi(text: str) -> bool:
        """Heuristic to detect if text contains Devanagari characters."""
        devanagari_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        return devanagari_chars >= 3

    @classmethod
    def clean_markdown(cls, text: str) -> str:
        """Strip markdown bold, italics, links, and bullets."""
        # Remove bold/italic ** or * or __
        text = re.sub(r'[*_]{1,3}(.*?)[*_]{1,3}', r'\1', text)
        # Remove markdown links [text](url)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        # Remove backticks
        text = re.sub(r'`+([^`]+)`+', r'\1', text)
        # Remove bullet markers at line starts
        text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.MULTILINE)
        # Clean extra spaces
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

    @classmethod
    def normalize_currency_en(cls, text: str) -> str:
        """Convert English currency symbols to words: ₹50.50 -> 'fifty rupees and fifty paise'."""
        def replace_rupees(match):
            val_str = match.group(1).replace(',', '')
            if '.' in val_str:
                parts = val_str.split('.', 1)
                rupees = int(parts[0]) if parts[0] else 0
                paise_str = (parts[1] + "0")[:2]
                paise = int(paise_str)
                r_word = num_to_en(rupees)
                p_word = num_to_en(paise)
                if rupees > 0 and paise > 0:
                    return f"{r_word} rupees and {p_word} paise"
                elif paise > 0:
                    return f"{p_word} paise"
                else:
                    return f"{r_word} rupees"
            else:
                amt = int(val_str)
                return f"{num_to_en(amt)} rupees"

        text = re.sub(r'(?:₹|Rs\.?|INR)\s*(\d+(?:,\d+)*(?:\.\d+)?)', replace_rupees, text, flags=re.IGNORECASE)
        
        # Replace $50
        def replace_dollars(match):
            val_str = match.group(1).replace(',', '')
            if '.' in val_str:
                parts = val_str.split('.', 1)
                dollars = int(parts[0]) if parts[0] else 0
                cents_str = (parts[1] + "0")[:2]
                cents = int(cents_str)
                d_word = num_to_en(dollars)
                c_word = num_to_en(cents)
                if dollars > 0 and cents > 0:
                    return f"{d_word} dollars and {c_word} cents"
                elif cents > 0:
                    return f"{c_word} cents"
                else:
                    return f"{d_word} dollars"
            else:
                amt = int(val_str)
                return f"{num_to_en(amt)} dollars"

        text = re.sub(r'\$\s*(\d+(?:,\d+)*(?:\.\d+)?)', replace_dollars, text)
        return text

    @classmethod
    def normalize_currency_hi(cls, text: str) -> str:
        """Convert Hindi currency representations to Devanagari: ₹50.50 -> 'पचास रुपये पचास पैसे'."""
        def replace_rupees_hi(match):
            val_str = match.group(1).replace(',', '')
            if '.' in val_str:
                parts = val_str.split('.', 1)
                rupees = int(parts[0]) if parts[0] else 0
                paise_str = (parts[1] + "0")[:2]
                paise = int(paise_str)
                r_word = num_to_hi(rupees)
                p_word = num_to_hi(paise)
                if rupees > 0 and paise > 0:
                    return f"{r_word} रुपये {p_word} पैसे"
                elif paise > 0:
                    return f"{p_word} पैसे"
                else:
                    return f"{r_word} रुपये"
            else:
                amt = int(val_str)
                return f"{num_to_hi(amt)} रुपये"

        text = re.sub(r'(?:₹|रु\.?|Rs\.?|INR)\s*(\d+(?:,\d+)*(?:\.\d+)?)', replace_rupees_hi, text, flags=re.IGNORECASE)
        return text

    @classmethod
    def normalize_plain_numbers_en(cls, text: str) -> str:
        """Convert plain decimals (3.5 -> 'three point five') and integers (15 -> 'fifteen')."""
        def replace_decimal(match):
            int_part = num_to_en(int(match.group(1)))
            dec_part = " ".join(num_to_en(int(d)) for d in match.group(2))
            return f"{int_part} point {dec_part}"

        text = re.sub(r'\b(\d+)\.(\d+)\b', replace_decimal, text)

        def replace_int(match):
            val = int(match.group(1))
            return num_to_en(val)

        text = re.sub(r'\b(\d+)\b', replace_int, text)
        return text

    @classmethod
    def normalize_plain_numbers_hi(cls, text: str) -> str:
        """Convert plain decimals (3.5 -> 'तीन दशमलव पाँच') and integers (15 -> 'पंद्रह')."""
        def replace_decimal_hi(match):
            int_part = num_to_hi(int(match.group(1)))
            dec_part = " ".join(num_to_hi(int(d)) for d in match.group(2))
            return f"{int_part} दशमलव {dec_part}"

        text = re.sub(r'\b(\d+)\.(\d+)\b', replace_decimal_hi, text)

        def replace_int_hi(match):
            val = int(match.group(1))
            return num_to_hi(val)

        text = re.sub(r'\b(\d+)\b', replace_int_hi, text)
        return text

    @classmethod
    def normalize(cls, text: str, lang: str = "auto") -> str:
        """Complete normalization pipeline for a sentence."""
        text = cls.clean_markdown(text)
        
        detected_hi = cls.is_hindi(text) if lang == "auto" else (lang.lower() in ("hi", "hindi"))
        
        if detected_hi:
            text = cls.normalize_currency_hi(text)
            text = cls.normalize_plain_numbers_hi(text)
        else:
            text = cls.normalize_currency_en(text)
            text = cls.normalize_plain_numbers_en(text)

        return text
