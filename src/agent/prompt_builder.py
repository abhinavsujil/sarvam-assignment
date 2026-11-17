"""Prompt builder for grounded civic service desk generation."""

from typing import Tuple, Optional
from src.agent.retriever import ServiceDocument


class AgentPromptBuilder:
    """Builds grounded bilingual system prompts injecting retrieved KB facts or general assistant prompts."""

    @staticmethod
    def is_hindi(text: str) -> bool:
        """Detect if text contains Devanagari characters."""
        return sum(1 for c in text if '\u0900' <= c <= '\u097F') >= 3

    @classmethod
    def build_prompt(
        cls,
        query: str,
        doc: Optional[ServiceDocument] = None,
        lang: str = "auto"
    ) -> Tuple[str, str, str]:
        """Construct system prompt and user prompt grounded in the retrieved document (if present).
        
        Returns:
            Tuple[str, str, str]: (system_prompt, user_prompt, detected_language)
        """
        is_hi = cls.is_hindi(query) if lang == "auto" else (lang.lower() in ("hi", "hindi"))
        active_lang = "hi" if is_hi else "en"

        if doc is not None:
            context_summary = doc.get_context_summary(lang=active_lang)
            if is_hi:
                system_prompt = (
                    "आप केरल के अक्षय ई-केंद्र (Akshaya Center) के आधिकारिक आवाज सहायक हैं। "
                    "नागरिक के प्रश्न का उत्तर नीचे दिए गए आधिकारिक सेवा विवरण के आधार पर 1 से 2 स्पष्ट और स्वाभाविक वाक्यों में दें। "
                    "यदि नागरिक किसी मनगढ़ंत शुल्क (जैसे 500 रुपये, 2500 रुपये) के बारे में पूछे, तो उसे बिल्कुल न दोहराएं; "
                    "स्पष्ट बताएं कि सही सरकारी शुल्क की पुष्टि केंद्र काउंटर पर करनी होगी। कोई भी संख्या न गढ़ें।\n\n"
                    f"[आधिकारिक सेवा विवरण]\n{context_summary}"
                )
            else:
                system_prompt = (
                    "You are an official Akshaya service desk voice assistant in Kerala. "
                    "Answer the citizen's query in 1-2 natural, concise sentences based strictly on the verified service details below. "
                    "Do NOT invent or fabricate any fees, charges, or numbers, and do not confirm user-suggested fees. "
                    "Always state that statutory fees must be confirmed directly at the Akshaya counter.\n\n"
                    f"[Official Service Details]\n{context_summary}"
                )
        else:
            if is_hi:
                system_prompt = (
                    "आप केरल के अक्षय ई-केंद्र के आधिकारिक नागरिक आवाज सहायक हैं। "
                    "सामान्य ज्ञान के प्रश्नों का 1-2 संक्षिप्त वाक्यों में सही उत्तर दें। "
                    "यदि नागरिक किसी ऐसी सेवा के बारे में पूछे जो अक्षय केंद्र पर उपलब्ध नहीं है (जैसे अंतरिक्ष यान, निजी बैंक ऋण), "
                    "तो विनम्रतापूर्वक बताएं कि अक्षय केंद्र केवल नागरिक एवं सरकारी सेवाएं प्रदान करता है।"
                )
            else:
                system_prompt = (
                    "You are a helpful and knowledgeable voice assistant for the Akshaya Civic Service Desk in Kerala. "
                    "For general knowledge questions, provide a direct, helpful, and concise answer in 1-2 sentences. "
                    "If asked whether an unsupported service (such as spaceships, aircraft, or non-civic commercial operations) "
                    "can be registered or done at this counter, clearly state that Akshaya centers only handle government civic services."
                )

        return system_prompt, query, active_lang
