"""Okapi BM25 Information Retrieval engine over Akshaya service knowledge base."""

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any


@dataclass
class ServiceDocument:
    """Civic service document data model."""
    service_id: str
    name_en: str
    name_hi: str
    category_en: str
    category_hi: str
    description_en: str
    description_hi: str
    required_documents_en: List[str]
    required_documents_hi: List[str]
    processing_time_days: int
    official_portal: str
    fee_inr: Optional[int]
    fee_note_en: str
    fee_note_hi: str
    aliases: List[str] = None

    def get_search_text(self) -> str:
        """Combined bilingual text representation for indexing."""
        alias_str = " ".join(self.aliases) if self.aliases else ""
        parts = [
            self.service_id,
            self.name_en,
            self.name_hi,
            alias_str,
            self.category_en,
            self.category_hi,
            self.description_en,
            self.description_hi,
            " ".join(self.required_documents_en),
            " ".join(self.required_documents_hi),
            self.official_portal,
            str(self.processing_time_days),
        ]
        return " ".join(parts).lower()

    def get_context_summary(self, lang: str = "en") -> str:
        """Structured plain text summary for LLM prompt context injection."""
        if lang.lower() in ("hi", "hindi"):
            docs = ", ".join(self.required_documents_hi)
            fee_str = "अघोषित / काउंटर पर सत्यापित करें" if self.fee_inr is None else f"₹{self.fee_inr}"
            return (
                f"सेवा: {self.name_hi}\n"
                f"आवश्यक दस्तावेज: {docs}\n"
                f"प्रसंस्करण समय: {self.processing_time_days} कार्य दिवस\n"
                f"पोर्टल: {self.official_portal}\n"
                f"शुल्क: {fee_str}\n"
                f"शुल्क सूचना: {self.fee_note_hi}"
            )
        else:
            docs = ", ".join(self.required_documents_en)
            fee_str = "Unspecified / Confirm at counter" if self.fee_inr is None else f"INR {self.fee_inr}"
            return (
                f"Service: {self.name_en}\n"
                f"Required Documents: {docs}\n"
                f"Processing Time: {self.processing_time_days} working days\n"
                f"Portal: {self.official_portal}\n"
                f"Fee: {fee_str}\n"
                f"Fee Note: {self.fee_note_en}"
            )


class BM25Retriever:
    """Lightweight, zero-dependency Okapi BM25 retriever for English and Hindi text."""

    def __init__(self, kb_path: Path, k1: float = 1.5, b: float = 0.75):
        self.kb_path = Path(kb_path)
        self.k1 = k1
        self.b = b
        self.documents: List[ServiceDocument] = []
        self.corpus_size: int = 0
        self.avg_doc_len: float = 0.0
        self.doc_lengths: List[int] = []
        self.doc_term_freqs: List[Dict[str, int]] = []
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

        self._load_and_index()

    STOP_WORDS = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'what', 'which', 'who', 'how', 'when', 'where', 'why', 'can', 'i', 'you', 'my', 'your', 'it',
        'this', 'that', 'do', 'does', 'did', 'and', 'or', 'so', 'tell', 'me', 'about', 'need', 'want',
        'का', 'की', 'के', 'में', 'पर', 'से', 'को', 'है', 'हैं', 'था', 'थी', 'थे', 'और', 'या',
        'तो', 'भी', 'क्या', 'क्यों', 'कैसे', 'कहाँ', 'कब', 'यह', 'वह', 'इस', 'उस', 'मुझे', 'आप'
    }

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize English words and Devanagari sequences, filtering functional stopwords."""
        tokens = re.findall(r'[\w\u0900-\u097F]+', text.lower())
        return [
            t for t in tokens
            if (len(t) > 1 or '\u0900' <= t <= '\u097F') and t not in self.STOP_WORDS
        ]

    def _load_and_index(self) -> None:
        if not self.kb_path.exists():
            raise FileNotFoundError(f"Knowledge base not found: {self.kb_path}")

        with open(self.kb_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.documents = [ServiceDocument(**s) for s in data.get("services", [])]
        self.corpus_size = len(self.documents)
        if self.corpus_size == 0:
            return

        total_length = 0
        self.doc_lengths = []
        self.doc_term_freqs = []
        self.doc_freqs = {}

        for doc in self.documents:
            tokens = self._tokenize(doc.get_search_text())
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_length += doc_len

            tf: Dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self.doc_term_freqs.append(tf)

            for token in set(tokens):
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        self.avg_doc_len = total_length / float(self.corpus_size)

        # Compute Robertson-Spärck Jones IDF
        for term, df in self.doc_freqs.items():
            # idf = ln(1 + (N - df + 0.5) / (df + 0.5))
            self.idf[term] = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))

    def retrieve(self, query: str, top_k: int = 1) -> List[Tuple[ServiceDocument, float]]:
        """Compute BM25F scores with title entity boosting and return top_k matching ServiceDocuments."""
        query_tokens = self._tokenize(query)
        if not query_tokens or self.corpus_size == 0:
            # Fallback to first document with score 0.0 if empty
            return [(self.documents[0], 0.0)] if self.documents else []

        scores: List[float] = [0.0] * self.corpus_size

        for term in query_tokens:
            if term not in self.idf:
                continue
            idf_val = self.idf[term]

            for i in range(self.corpus_size):
                tf = self.doc_term_freqs[i].get(term, 0)
                if tf == 0:
                    continue
                doc_len = self.doc_lengths[i]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                scores[i] += idf_val * (numerator / denominator)

        # Field-weighting (BM25F): Primary Title / Alias matching boost
        query_token_set = set(query_tokens)
        for i in range(self.corpus_size):
            doc = self.documents[i]
            title_text = f"{doc.name_en} {doc.name_hi} {' '.join(doc.aliases if doc.aliases else [])}"
            title_tokens = set(self._tokenize(title_text))
            title_overlap = len(query_token_set.intersection(title_tokens))
            if title_overlap > 0:
                scores[i] += title_overlap * 3.5

        # Pair scores with documents and sort descending
        doc_scores = [(self.documents[i], scores[i]) for i in range(self.corpus_size)]
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        return doc_scores[:top_k]
