"""GiNZA processing pipeline and serialization logic."""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span, Token

from .models import InputItem


LOGGER = logging.getLogger(__name__)


class GiNZAProcessor:
    """GiNZA processor with lazy-loaded spaCy pipeline.

    Manages the lifecycle of the spaCy model instance and exposes a method to
    process a list of `InputItem` into a structured JSON-serializable dict.
    """

    def __init__(self, model_name: str = "ja_ginza_electra") -> None:
        self._model_name = model_name
        self._nlp: Optional[Language] = None

    @property
    def nlp(self) -> Language:
        """Return a lazy-loaded spaCy pipeline instance."""

        if self._nlp is None:
            start = time.perf_counter()
            LOGGER.info("event=load_model status=starting model=%s", self._model_name)
            try:
                self._nlp = spacy.load(self._model_name)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception(
                    "event=load_model status=error model=%s error=%s",
                    self._model_name,
                    exc.__class__.__name__,
                )
                raise
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                LOGGER.info(
                    "event=load_model status=finished model=%s latency_ms=%.2f",
                    self._model_name,
                    elapsed_ms,
                )
        return self._nlp

    def process_items(self, items: Iterable[InputItem]) -> Dict[str, Any]:
        """Process input items and return structured JSON-serializable output."""

        processed_inputs: List[Dict[str, Any]] = []
        global_document_frequency: Dict[str, int] = {}
        for item in items:
            start = time.perf_counter()
            LOGGER.info(
                "event=process_text status=starting id=%s text_len=%d",
                item.id,
                len(item.text),
            )
            try:
                doc: Doc = self.nlp(item.text)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception(
                    "event=process_text status=error id=%s error=%s",
                    item.id,
                    exc.__class__.__name__,
                )
                raise

            sentences_payload: List[Dict[str, Any]] = []
            token_i_to_sent_index: Dict[int, int] = {}
            for sent_index, sent in enumerate(doc.sents):
                for tok in sent:
                    token_i_to_sent_index[int(tok.i)] = int(sent_index)
                sentences_payload.append(self._serialize_sentence(sent, sent_index))

            (
                term_counts,
                term_occurrences,
                total_terms,
                document_terms_set,
            ) = self._extract_terms(doc, token_i_to_sent_index)

            for term in document_terms_set:
                global_document_frequency[term] = (
                    global_document_frequency.get(term, 0) + 1
                )

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            LOGGER.info(
                "event=process_text status=finished id=%s latency_ms=%.2f",
                item.id,
                elapsed_ms,
            )

            per_doc_terms: List[Dict[str, Any]] = []
            if total_terms > 0:
                for term, count in term_counts.items():
                    per_doc_terms.append(
                        {
                            "term": term,
                            "tf_raw": int(count),
                            "tf_norm": round(count / total_terms, 8),
                            "occurrences": term_occurrences.get(term, []),
                        }
                    )

            processed_inputs.append(
                {
                    "id": item.id,
                    "text": item.text,
                    "doc_char_length": len(item.text),
                    "latency_ms": round(elapsed_ms, 2),
                    "sentences": sentences_payload,
                    "term_stats": {
                        "num_terms": int(total_terms),
                        "vocab_size": int(len(term_counts)),
                        "terms": per_doc_terms,
                    },
                }
            )

        LOGGER.info("event=tfidf_compute status=starting")
        num_docs = max(len(processed_inputs), 1)
        idf_map = self._compute_idf(num_docs, global_document_frequency)

        for input_payload in processed_inputs:
            terms_list: List[Dict[str, Any]] = input_payload.get("term_stats", {}).get(
                "terms",
                [],
            )
            for entry in terms_list:
                term = entry["term"]
                idf_value = idf_map.get(term, 0.0)
                tf_raw = float(entry["tf_raw"])  # raw count
                entry["idf"] = round(idf_value, 8)
                entry["tfidf"] = round(tf_raw * idf_value, 8)

            terms_list.sort(key=lambda e: (-e.get("tfidf", 0.0), e["term"]))

        vocabulary: List[Dict[str, Any]] = []
        for term, df_value in sorted(global_document_frequency.items()):
            vocabulary.append(
                {
                    "term": term,
                    "df": int(df_value),
                    "idf": round(idf_map.get(term, 0.0), 8),
                }
            )

        LOGGER.info("event=tfidf_compute status=finished")

        output: Dict[str, Any] = {
            "model": self._model_name,
            "language": "ja",
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "inputs": processed_inputs,
            "vocabulary": vocabulary,
        }
        return output

    @staticmethod
    def _serialize_sentence(sent: Span, sent_index: int) -> Dict[str, Any]:
        tokens_payload: List[Dict[str, Any]] = []
        for idx_in_sentence, token in enumerate(sent):
            tokens_payload.append(
                GiNZAProcessor._serialize_token(token, idx_in_sentence)
            )

        payload: Dict[str, Any] = {
            "index": sent_index,
            "start_char": sent.start_char,
            "end_char": sent.end_char,
            "text": sent.text,
            "tokens": tokens_payload,
        }
        return payload

    @staticmethod
    def _serialize_token(token: Token, sent_token_index: int) -> Dict[str, Any]:
        reading_values = token.morph.get("Reading") or []
        inflection_values = token.morph.get("Inflection") or []

        payload: Dict[str, Any] = {
            "doc_index": int(token.i),
            "sent_index": int(sent_token_index),
            "start_char": int(token.idx),
            "end_char": int(token.idx + len(token)),
            "text": token.orth_,
            "lemma": token.lemma_,
            "norm": token.norm_,
            "reading": list(reading_values),
            "pos": token.pos_,
            "inflection": list(inflection_values),
            "tag": token.tag_,
            "dep": token.dep_,
            "head_doc_index": int(token.head.i),
        }
        return payload

    @staticmethod
    def _is_informative_token(token: Token) -> bool:
        if token.is_space or token.is_punct:
            return False
        text = token.orth_
        return bool(text and text.strip())

    def _extract_terms(
        self, doc: Doc, token_i_to_sent_index: Mapping[int, int]
    ) -> tuple[Dict[str, int], Dict[str, List[Dict[str, int]]], int, set[str]]:
        term_counts: Dict[str, int] = {}
        term_occurrences: Dict[str, List[Dict[str, int]]] = {}
        total_terms = 0
        for tok in doc:
            if not self._is_informative_token(tok):
                continue
            term = tok.lemma_
            term_counts[term] = term_counts.get(term, 0) + 1
            occ = {
                "doc_index": int(tok.i),
                "sent_index": int(token_i_to_sent_index.get(int(tok.i), 0)),
            }
            term_occurrences.setdefault(term, []).append(occ)
            total_terms += 1

        document_terms_set = set(term_counts.keys())
        return term_counts, term_occurrences, total_terms, document_terms_set

    @staticmethod
    def _compute_idf(num_docs: int, df_map: Mapping[str, int]) -> Dict[str, float]:
        idf_values: Dict[str, float] = {}
        for term, df in df_map.items():
            idf = math.log((1.0 + num_docs) / (1.0 + float(df))) + 1.0
            idf_values[term] = idf
        return idf_values


