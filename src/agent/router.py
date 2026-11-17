"""
Dialog Router & Intent Classification for Akshaya Civic Voice Assistant.
Cleanly separates conversational greetings, grounded civic queries, and out-of-scope requests.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
from src.agent.retriever import BM25Retriever, ServiceDocument


class IntentType(str, Enum):
    GREETING = "GREETING"
    CIVIC_INQUIRY = "CIVIC_INQUIRY"
    GENERAL_GOV_INQUIRY = "GENERAL_GOV_INQUIRY"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


@dataclass
class DialogDecision:
    """Structured decision output from dialog routing."""
    intent: IntentType
    system_prompt: str
    user_prompt: str
    language: str  # "en" or "hi"
    service_doc: Optional[ServiceDocument] = None
    bm25_score: float = 0.0
    direct_response: Optional[str] = None
    canonical_response: Optional[str] = None


class DialogRouter:
    """Routes user queries into structured dialog actions with proper system prompts."""

    MIN_BM25_THRESHOLD = 1.2

    GREETING_WORDS = {
        "hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening",
        "who are you", "what can you do", "help", "how are you",
        "नमस्ते", "प्रणाम", "हेलो", "आप कैसे हो", "कैसे हो", "कौन हो", "सहायता",
        "kaise ho", "kaise hain", "aap kaise ho"
    }

    CONVERSATIONAL_PHRASES = {
        "who are you", "what is your name", "what's your name", "tell me your name",
        "what can you do", "introduce yourself", "how are you", "who made you",
        "how do you do", "how are things", "how is it going",
        "आपका नाम क्या है", "तुम्हारा नाम क्या है", "आप कौन हैं", "आप कौन हो",
        "तुम कौन हो", "आप क्या कर सकते हैं", "आप कैसे हैं", "आप कैसे हो",
        "आप लोग कैसे हैं", "आप लोग कैसे हो", "आप लोग कैसे हे", "आप कैसे हे",
        "कैसे हैं आप", "कैसे हो आप", "क्या हाल है", "सब कैसा है", "सब ठीक है",
        "आपका क्या नाम है", "अपना नाम बताओ", "अपना परिचय दो", "नाम क्या है", "क्या नाम है",
        "aap log kaise", "aap kaise", "kaise he", "kaise ho", "kya haal"
    }

    SERVICE_ANCHORS = {
        'income_tax_filing': {'tax', 'itr', 'टैक्स', 'आयकर', 'आईटीआर'},
        'income_certificate': {'income', 'salary', 'आय', 'इनकम'},
        'caste_certificate': {'caste', 'community', 'जाति', 'कास्ट'},
        'new_aadhaar_enrollment': {'aadhaar', 'aadhar', 'आधार', 'नामांकन', 'enrollment', 'lena', 'लेना', 'banwana', 'बनवाना'},
        'aadhaar_mobile_update': {'mobile', 'phone', 'update', 'मोबाइल', 'फोन', 'अपडेट'},
        'pan_card_new': {'pan', 'पैन'},
        'pan_aadhaar_link': {'link', 'लिंक'},
        'passport_seva': {'passport', 'पासपोर्ट'},
        'non_creamy_layer': {'ncl', 'creamy', 'क्रीमी', 'ओबीसी'},
        'ews_certificate': {'ews', 'ईडब्ल्यूएस'},
        'ration_card_member_addition': {'ration', 'राशन', 'member', 'सदस्य'},
        'ration_card_new': {'ration', 'राशन', 'new', 'नया'},
        'driving_license_learner': {'learner', 'learning', 'llr', 'लर्नर', 'लर्निंग', 'कच्चा'},
        'driving_license_permanent': {'permanent', 'पक्का'},
        'driving_license_renewal': {'renewal', 'renew', 'नवीनीकरण', 'रिन्यू'},
        'ayushman_bharat_card': {'ayushman', 'pmjay', 'golden', 'आयुष्मान', 'गोल्डन'},
        'pm_kisan_samman': {'kisan', 'किसान', 'samman', 'सम्मान'},
        'old_age_pension': {'pension', 'पेंशन', 'vridha', 'वृद्धा'},
        'epfo_pf_withdrawal': {'epfo', 'pf', 'uan', 'पीएफ', 'ईपीएफओ'},
        'electricity_bill_payment': {'electricity', 'power', 'bijli', 'बिजली', 'लाइट'},
        'death_certificate': {'death', 'demise', 'cremation', 'मृत्यु', 'मौत', 'डेथ'},
        'birth_certificate': {'birth', 'जन्म', 'बर्थ', 'पैदाइश'},
        'residence_certificate': {'residence', 'domicile', 'nativity', 'निवास', 'डोमिसाइल', 'रेजिडेंस'},
        'marriage_certificate': {'marriage', 'wedding', 'विवाह', 'मैरिज', 'शादी'},
        'voter_id_service': {'voter', 'epic', 'election', 'मतदाता', 'वोटर'}
    }

    def __init__(self, retriever: BM25Retriever):
        self.retriever = retriever

    def resolve_service_entity(self, query: str) -> Optional[Tuple[ServiceDocument, float]]:
        """Deterministically resolves the exact civic service if the query contains its primary semantic anchors."""
        import re
        q_tokens = set(t.lower() for t in re.findall(r'[\w\u0900-\u097F]+', query))
        if not q_tokens:
            return None

        best_doc = None
        best_score = 0.0

        for doc in self.retriever.documents:
            anchors = self.SERVICE_ANCHORS.get(doc.service_id, set())
            matched_anchors = q_tokens.intersection(anchors)
            if not matched_anchors:
                continue

            score = float(len(matched_anchors) * 10)
            for alias in (doc.aliases or []):
                a_toks = set(t.lower() for t in re.findall(r'[\w\u0900-\u097F]+', alias))
                if a_toks and a_toks.issubset(q_tokens):
                    score += 20.0

            if score > best_score:
                best_score = score
                best_doc = doc

        if best_doc is not None:
            return best_doc, best_score
        return None

    @staticmethod
    def detect_language(text: str, fallback_lang: str = "en") -> str:
        """Detect language dynamically based on sentence word composition.
        
        Rules:
        1. If user speaks entirely in English -> 'en'.
        2. If user speaks in English with 1-2 Hindi words (e.g. 'Can I get आय प्रमाण पत्र here?')
           -> English words >= Hindi words -> 'en' (Respond completely in English!).
        3. If user speaks in Hindi with English loanwords (e.g. 'मुझे एक income certificate लेना है')
           -> Hindi words > English words -> 'hi' (Respond in Hindi!).
        4. If purely Devanagari -> 'hi'.
        """
        import re
        clean = text.strip()
        tokens = re.findall(r'[\u0900-\u097F]+|[a-zA-Z]+', clean)
        if not tokens:
            return "hi" if fallback_lang.lower().startswith("hi") else "en"

        hi_words = sum(1 for t in tokens if re.search(r'[\u0900-\u097F]', t))
        en_words = sum(1 for t in tokens if re.search(r'[a-zA-Z]', t))

        if hi_words == 0:
            return "en"
        if en_words == 0:
            return "hi"

        # Code-mixed turn:
        # If English words dominate or tie -> Respond in English completely
        if en_words >= hi_words:
            return "en"
        return "hi"

    def is_greeting(self, text: str) -> bool:
        """Check if user query is a greeting or casual conversational opener."""
        import re
        q_clean = re.sub(r'[^\w\s\u0900-\u097F]', '', text.strip().lower()).strip()
        tokens = q_clean.split()
        if not tokens:
            return False

        # Conversational / identity phrases
        civic_stoppers = {'certificate', 'card', 'bill', 'प्रमाण', 'पत्र', 'कार्ड', 'इनकम', 'जाति', 'निवास', 'राशन'}
        if any(phrase in q_clean for phrase in self.CONVERSATIONAL_PHRASES):
            if not any(t in civic_stoppers for t in tokens):
                return True

        # Pure greeting detection (strictly <= 2 tokens, e.g. 'hello', 'hi', 'namaste', 'नमस्ते')
        if len(tokens) == 1 and tokens[0] in self.GREETING_WORDS:
            return True
        if len(tokens) == 2 and (tokens[0] in self.GREETING_WORDS or tokens[1] in self.GREETING_WORDS):
            if not any(t in civic_stoppers for t in tokens):
                return True
        return False

    @staticmethod
    def classify_sub_intent(query: str) -> str:
        """Classify inquiry into FEE, TIMELINE, or GENERAL for focused grounding."""
        q_lower = query.lower()
        fee_kw = [
            "fee", "fees", "cost", "charge", "charges", "rate", "price", "rupee", "rupees", "paisa",
            "फीस", "शुल्क", "खर्च", "दाम", "पैसे", "रुपये", "कितना लगेगा"
        ]
        if any(w in q_lower for w in fee_kw):
            return "FEE"
        timeline_kw = [
            "how many days", "how long", "timeline", "time", "days", "when",
            "दिन", "समय", "कब", "कितने दिन", "लगते", "कितना समय"
        ]
        if any(w in q_lower for w in timeline_kw):
            return "TIMELINE"
        return "GENERAL"

    CIVIC_KEYWORDS = {
        'certificate', 'certificates', 'license', 'licence', 'application form',
        'registration', 'documents', 'ration card', 'aadhaar card', 'voter card',
        'e-district', 'akshaya center', 'akshaya kendra', 'electricity bill', 'water bill',
        'property tax', 'income tax', 'pension scheme', 'welfare scheme', 'driving license',
        'passport', 'pan card', 'scheme', 'yojana', 'subsidy', 'welfare', 'portal',
        'sarkari', 'government service', 'indian government', 'csc', 'seva', 'allowance',
        'kisan', 'shram', 'sukanya', 'epfo', 'provident fund', 'ayushman', 'pmjay', 'ration',
        'tax', 'itr', 'tax filing', 'income tax return',
        'प्रमाण पत्र', 'सर्टिफिकेट', 'राशन कार्ड', 'आधार कार्ड', 'वोटर कार्ड', 'लाइसेंस',
        'दस्तावेज', 'दस्तावेज़', 'कागजात', 'ई-डिस्ट्रिक्ट', 'अक्षय केंद्र', 'बिजली बिल',
        'पानी बिल', 'पेंशन', 'सब्सिडी', 'ड्राइविंग लाइसेंस', 'पासपोर्ट', 'पैन कार्ड',
        'योजना', 'सरकारी', 'सरकार', 'मंत्रालय', 'कल्याण', 'आवेदन', 'पंजीकरण', 'ई-सेवा',
        'भत्ता', 'अनुदान', 'नागरिक सेवा', 'सेवा केंद्र', 'श्रम', 'किसान',
        'टैक्स', 'आयकर', 'आईटीआर', 'डॉक्यूमेंट्स', 'टैक्स फाइल'
    }

    def is_civic_intent(self, text: str) -> bool:
        """Detect whether a query pertains to civic, e-governance, or public administration."""
        q_lower = text.lower()
        return any(kw in q_lower for kw in self.CIVIC_KEYWORDS)

    def route(self, query: str, client_lang: str = "en", template_mode: bool = False) -> DialogDecision:
        """Route citizen query to the appropriate system prompt and knowledge context."""
        clean_q = query.strip()
        score = 0.0
        # Respect user language selection explicitly
        if client_lang and client_lang.lower().startswith("hi"):
            lang = "hi"
        elif client_lang and client_lang.lower().startswith("en") and not any('\u0900' <= c <= '\u097F' for c in clean_q):
            lang = "en"
        else:
            lang = self.detect_language(clean_q, fallback_lang=client_lang)
        is_hi = (lang == "hi")

        # 1. Check for Greeting / Conversational intent
        if self.is_greeting(clean_q):
            is_name_query = any(w in clean_q.lower() for w in ["name", "नाम", "who are you", "कौन", "परिचय"])
            is_how_are_you = any(w in clean_q.lower() for w in ["how are you", "कैसे हैं", "कैसे हो", "कैसे हे", "हाल", "kaise", "सब कैसा"])
            if is_hi:
                if is_name_query:
                    resp = "मेरा नाम अक्षय आवाज सहायक है। बताइए मैं आपकी क्या सहायता कर सकता हूँ?"
                elif is_how_are_you:
                    resp = "नमस्ते! हम बिल्कुल ठीक हैं। अक्षय केंद्र में आपका स्वागत है, बताइए आज मैं आपकी क्या सहायता कर सकता हूँ?"
                else:
                    resp = "नमस्ते! मैं अक्षय ई-केंद्र का आधिकारिक आवाज सहायक हूँ। बताइए आज मैं आपकी क्या सहायता कर सकता हूँ?"
            else:
                if is_name_query:
                    resp = "My name is the Akshaya Voice Assistant. How can I assist you today?"
                elif is_how_are_you:
                    resp = "Hello! I am doing well, thank you. Welcome to the Akshaya Center. How can I assist you today?"
                else:
                    resp = "Hello! I am the official Akshaya Voice Assistant. How can I assist you today?"

            return DialogDecision(
                intent=IntentType.GREETING,
                system_prompt="",
                user_prompt=clean_q,
                language=lang,
                service_doc=None,
                bm25_score=0.0,
                direct_response=resp,
                canonical_response=resp
            )

        # 2. Check for Specific Civic Service Intent via Semantic Anchors
        service_match = self.resolve_service_entity(clean_q)

        # If citizen asked only for generic certificate/card without specifying which one
        GENERIC_CATEGORY_TOKENS = {'certificate', 'certificates', 'cert', 'प्रमाण पत्र', 'प्रमाण', 'सर्टिफिकेट'}
        if service_match is None and any(t in clean_q.lower() for t in GENERIC_CATEGORY_TOKENS):
            if is_hi:
                resp = "आप किस प्रमाण पत्र के बारे में जानना चाहते हैं? अक्षय केंद्र पर आय प्रमाण पत्र, जाति प्रमाण पत्र, निवास प्रमाण पत्र, जन्म, मृत्यु और विवाह प्रमाण पत्र उपलब्ध हैं।"
                sys_prompt = "You are the official Akshaya Voice Assistant. Ask the citizen in 1 polite Hindi sentence (देवनागरी में) which specific certificate they need."
            else:
                resp = "Which certificate would you like information about? We provide Income, Caste, Residence, Birth, Death, and Marriage certificates."
                sys_prompt = "You are the official Akshaya Voice Assistant. Ask the citizen in 1 polite sentence which specific certificate they need."
            return DialogDecision(
                intent=IntentType.GENERAL_GOV_INQUIRY,
                system_prompt=sys_prompt,
                user_prompt=clean_q,
                language=lang,
                service_doc=None,
                bm25_score=0.0,
                direct_response=resp if template_mode else None,
                canonical_response=resp
            )

        if service_match is not None:
            doc, score = service_match
            sub_intent = self.classify_sub_intent(clean_q)

            if is_hi:
                docs_hi = ", ".join(doc.required_documents_hi)
                devanagari_mandate = "You MUST write your entire response in Hindi using the Devanagari script (देवनागरी लिपि में ही उत्तर दें). Do NOT write in English or Romanized letters. "
                if sub_intent == "FEE":
                    canonical = f"{doc.name_hi} के लिए: {doc.fee_note_hi}"
                    sys_prompt = (
                        "You are the official Akshaya Civic Voice Assistant in Kerala. "
                        + devanagari_mandate +
                        f"The citizen is asking about fees for {doc.name_hi}. "
                        f"State clearly in 1 short polite Hindi sentence: {doc.fee_note_hi}. "
                        "Do NOT mention any specific numbers or imaginary rupee fees."
                    )
                elif sub_intent == "TIMELINE":
                    canonical = f"{doc.name_hi} के लिए आधिकारिक प्रसंस्करण समय {doc.processing_time_days} कार्य दिवस है।"
                    sys_prompt = (
                        "You are the official Akshaya Civic Voice Assistant in Kerala. "
                        + devanagari_mandate +
                        f"The citizen is asking about the time needed for {doc.name_hi}. "
                        f"State clearly in 1 short polite Hindi sentence that processing time is {doc.processing_time_days} कार्य दिवस on {doc.official_portal}. "
                        "Do NOT invent any other numbers."
                    )
                else:
                    canonical = f"{doc.name_hi} के लिए आवश्यक दस्तावेज हैं: {docs_hi}। इसका आधिकारिक प्रसंस्करण समय {doc.processing_time_days} कार्य दिवस है।"
                    sys_prompt = (
                        "You are the official Akshaya Civic Voice Assistant in Kerala. "
                        + devanagari_mandate +
                        "Answer the citizen in strictly 1-2 short, polite Hindi sentences (under 35 words). Do not use bullet points or numbered lists. Base your answer on these facts:\n"
                        f"- सेवा: {doc.name_hi}\n"
                        f"- आवश्यक दस्तावेज: {docs_hi}\n"
                        f"- प्रसंस्करण समय: {doc.processing_time_days} कार्य दिवस\n"
                        f"- पोर्टल: {doc.official_portal}\n"
                        "State only verified facts. Do NOT invent or fabricate any extra numbers."
                    )
            else:
                docs_en = ", ".join(doc.required_documents_en)
                if sub_intent == "FEE":
                    canonical = f"For {doc.name_en}: {doc.fee_note_en}"
                    sys_prompt = (
                        "You are the official Akshaya Civic Voice Assistant in Kerala. "
                        f"The citizen is asking about the fee for {doc.name_en}. "
                        f"State in 1 short concise sentence (under 25 words) that statutory fees must be verified directly at the Akshaya counter. "
                        "Do NOT invent any dollar or rupee numbers."
                    )
                elif sub_intent == "TIMELINE":
                    canonical = f"The official processing time for {doc.name_en} is {doc.processing_time_days} working days."
                    sys_prompt = (
                        "You are the official Akshaya Civic Voice Assistant in Kerala. "
                        f"State clearly in 1 concise sentence (under 25 words) that the processing time for {doc.name_en} is {doc.processing_time_days} working days on {doc.official_portal}."
                    )
                else:
                    canonical = f"To apply for {doc.name_en}, the required documents are: {docs_en}. The processing time is {doc.processing_time_days} working days on {doc.official_portal}."
                    sys_prompt = (
                        "You are the official Akshaya Civic Voice Assistant in Kerala. "
                        "Answer the citizen's query in strictly 1-2 short, natural spoken sentences (under 35 words). "
                        "Do NOT use bullet points, numbered lists, or markdown bold text. Base your answer strictly on these facts:\n"
                        f"- Service: {doc.name_en}\n"
                        f"- Required Documents: {docs_en}\n"
                        f"- Processing Time: {doc.processing_time_days} working days\n"
                        f"- Official Portal: {doc.official_portal}\n"
                        "Do NOT invent any fees or unverified numbers."
                    )

            # Optional template comparison switch
            direct_resp = canonical if template_mode else None

            return DialogDecision(
                intent=IntentType.CIVIC_INQUIRY,
                system_prompt=sys_prompt,
                user_prompt=clean_q,
                language=lang,
                service_doc=doc,
                bm25_score=score,
                direct_response=direct_resp,
                canonical_response=canonical
            )

        # 3. Check for General Indian Civic / Government Inquiry (outside the 24 pre-indexed services)
        if self.is_civic_intent(clean_q):
            if is_hi:
                canonical = (
                    "इस सरकारी योजना या नागरिक सेवा के लिए आवेदन संबंधित विभाग या राज्य ई-डिस्ट्रिक्ट पोर्टल के माध्यम से किया जाता है। "
                    "कृपया अपना मूल आधार कार्ड, बैंक पासबुक और पते का प्रमाण लेकर निकटतम नागरिक सेवा केंद्र (CSC या अक्षय केंद्र) पर संपर्क करें।"
                )
                sys_prompt = (
                    "You are an official Indian Government and Civic Voice Assistant. "
                    "Answer the citizen's government or scheme question helpfully, accurately, and politely in 1-2 concise Hindi sentences in Devanagari script (देवनागरी लिपि में ही उत्तर दें). "
                    "Direct them to the official portal (india.gov.in / state portal) or local civic center. Do NOT invent fake fees or unverified numbers."
                )
            else:
                canonical = (
                    "For this Indian government service or welfare scheme, applications are processed through the respective central or state department portal (india.gov.in). "
                    "Please carry your Aadhaar card, active bank passbook, and address proof to the local citizen service center for assisted registration."
                )
                sys_prompt = (
                    "You are an official Indian Government and Civic Voice Assistant. "
                    "Answer the citizen's civic query helpfully and politely in 1-2 concise spoken sentences. "
                    "Explain the general application procedure or required KYC documents (Aadhaar, address proof) and direct them to the official portal (india.gov.in). "
                    "Do NOT invent any fees or unverified numbers."
                )

            direct_resp = canonical if template_mode else None

            return DialogDecision(
                intent=IntentType.GENERAL_GOV_INQUIRY,
                system_prompt=sys_prompt,
                user_prompt=clean_q,
                language=lang,
                service_doc=None,
                bm25_score=score,
                direct_response=direct_resp,
                canonical_response=canonical
            )

        # 4. Open-ended Conversational & General Inquiries
        if is_hi:
            resp = "मैं अक्षय ई-केंद्र का आधिकारिक सहायक हूँ। बताइए मैं आपकी क्या सहायता कर सकता हूँ?"
            sys_prompt = (
                "You are the intelligent, polite, and helpful official voice assistant at an Akshaya Civic Center. "
                "Answer the citizen's question helpfully, politely, and concisely in 1-2 natural sentences in Hindi using Devanagari script (देवनागरी में उत्तर दें). "
                "Do not invent fake government fees or unverified numerical statistics."
            )
        else:
            resp = "I am the official Akshaya voice assistant. How can I help you today?"
            sys_prompt = (
                "You are the intelligent, polite, and helpful official voice assistant at an Akshaya Civic Service Center. "
                "Answer the citizen's question helpfully, politely, and concisely in 1-2 natural sentences based on your knowledge. "
                "Do not invent fake government fees or unverified numerical statistics."
            )

        return DialogDecision(
            intent=IntentType.OUT_OF_SCOPE,
            system_prompt=sys_prompt,
            user_prompt=clean_q,
            language=lang,
            service_doc=None,
            bm25_score=score,
            direct_response=resp if template_mode else None,
            canonical_response=resp
        )
