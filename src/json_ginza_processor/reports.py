"""CSV report generation utilities for GiNZA processor outputs.

This module provides helpers to export statistics from the JSON/dict output of
the GiNZA processing pipeline into CSV files. All ranking reports are restricted
to nouns (spaCy POS ``NOUN``) and proper nouns (``PROPN``):

- Highest DF (document frequency) ranking [nouns only]
- Highest IDF (inverse document frequency) ranking [nouns only]
- Highest DF-IDF (df * idf) ranking [nouns only]
- List of all nouns (unique lemmas across inputs)

Public entrypoint:
- export_csv_reports
"""

from __future__ import annotations

import csv
import logging
import os
from collections import Counter
from typing import Any, Dict, Iterable, Mapping, Tuple


LOGGER = logging.getLogger(__name__)


def _term_stats_maps(
    result: Mapping[str, Any],
) -> Tuple[Dict[str, int], Dict[str, float]]:
    """Build lookup maps for df and idf from the result vocabulary.

    Parameters
    ----------
    result:
        The structured dict returned by the processing pipeline.

    Returns
    -------
    Tuple[Dict[str, int], Dict[str, float]]
        Two dictionaries mapping terms to df and idf respectively.
    """

    vocab = result.get("vocabulary", [])
    term_to_df: Dict[str, int] = {}
    term_to_idf: Dict[str, float] = {}
    for entry in vocab:
        term = entry.get("term")
        if not isinstance(term, str):
            continue
        df_val = int(entry.get("df", 0))
        idf_val = float(entry.get("idf", 0.0))
        term_to_df[term] = df_val
        term_to_idf[term] = idf_val
    return term_to_df, term_to_idf


def _collect_nouns_and_tags(
    result: Mapping[str, Any],
) -> Tuple[set[str], Dict[str, int], Dict[str, str]]:
    """Collect unique noun lemmas and their tags from token payloads.

    Nouns are identified using spaCy coarse POS tags: {NOUN, PROPN}.

    For each lemma, this function records:
    - total token frequency across all inputs (noun tokens only)
    - the most frequent fine-grained tag (e.g., "名詞-固有名詞-地名-一般")

    Returns
    -------
    Tuple[set[str], Dict[str, int], Dict[str, str]]
        A set of unique noun lemmas, a frequency counter across all noun tokens,
        and a mapping of lemma -> most frequent tag.
    """

    noun_pos = {"NOUN", "PROPN"}
    nouns: set[str] = set()
    freq: Counter[str] = Counter()
    tag_counters: Dict[str, Counter[str]] = {}

    inputs = result.get("inputs", [])
    for inp in inputs:
        for sent in inp.get("sentences", []):
            for tok in sent.get("tokens", []):
                pos = tok.get("pos")
                if pos in noun_pos:
                    lemma = tok.get("lemma")
                    if isinstance(lemma, str) and lemma:
                        nouns.add(lemma)
                        freq[lemma] += 1
                        tag_val = tok.get("tag")
                        if isinstance(tag_val, str) and tag_val:
                            tag_counters.setdefault(lemma, Counter())[tag_val] += 1

    lemma_to_tag: Dict[str, str] = {}
    for lemma, counter in tag_counters.items():
        most_common = counter.most_common(1)
        lemma_to_tag[lemma] = most_common[0][0] if most_common else ""

    return nouns, dict(freq), lemma_to_tag


def _collect_total_tf(result: Mapping[str, Any]) -> Dict[str, int]:
    """Collect total term frequency across all documents from term_stats."""

    totals: Counter[str] = Counter()
    for inp in result.get("inputs", []):
        for term_entry in inp.get("term_stats", {}).get("terms", []):
            term = term_entry.get("term")
            tf_raw = int(term_entry.get("tf_raw", 0))
            if isinstance(term, str):
                totals[term] += tf_raw
    return dict(totals)


def _write_csv(path: str, header: Iterable[str], rows: Iterable[Iterable[Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))


def export_csv_reports(result: Mapping[str, Any], output_dir: str) -> Dict[str, str]:
    """Export CSV reports for DF, IDF, DF-IDF, and nouns.

    Parameters
    ----------
    result:
        The structured dict returned by `process_payload_to_dict`.
    output_dir:
        Directory where CSV files will be written. Created if missing.

    Returns
    -------
    Dict[str, str]
        Mapping from a logical report name to the written file path.
    """

    LOGGER.info("event=export_csv_reports status=starting output_dir=%s", output_dir)

    term_to_df, term_to_idf = _term_stats_maps(result)
    totals_tf = _collect_total_tf(result)
    nouns_set, nouns_freq, lemma_to_tag = _collect_nouns_and_tags(result)

    # Restrict ranking reports to noun/proper-noun lemmas only.
    noun_terms_df = [t for t in term_to_df.keys() if t in nouns_set]
    noun_terms_idf = [t for t in term_to_idf.keys() if t in nouns_set]
    noun_terms_dfidf = (term_to_df.keys() | term_to_idf.keys()) & nouns_set

    # Highest DF ranking (nouns only)
    df_terms_sorted = sorted(noun_terms_df, key=lambda t: (-term_to_df.get(t, 0), t))
    df_rows = (
        (
            term,
            lemma_to_tag.get(term, ""),
            term_to_df.get(term, 0),
            term_to_idf.get(term, 0.0),
        )
        for term in df_terms_sorted
    )

    # Highest IDF ranking (nouns only)
    idf_terms_sorted = sorted(
        noun_terms_idf, key=lambda t: (-term_to_idf.get(t, 0.0), t)
    )
    idf_rows = (
        (
            term,
            lemma_to_tag.get(term, ""),
            term_to_idf.get(term, 0.0),
            term_to_df.get(term, 0),
        )
        for term in idf_terms_sorted
    )

    # Highest DF-IDF ranking (df * idf) (nouns only)
    df_idf_sorted = sorted(
        (
            (
                term,
                lemma_to_tag.get(term, ""),
                term_to_df.get(term, 0),
                term_to_idf.get(term, 0.0),
                float(term_to_df.get(term, 0)) * float(term_to_idf.get(term, 0.0)),
            )
            for term in noun_terms_dfidf
        ),
        key=lambda x: (-x[4], x[0]),
    )

    # Nouns list (unique lemmas) with optional counts and df/idf when available
    noun_rows = (
        (
            lemma,
            nouns_freq.get(lemma, 0),
            totals_tf.get(lemma, 0),
            term_to_df.get(lemma, 0),
            term_to_idf.get(lemma, 0.0),
        )
        for lemma in sorted(nouns_set)
    )

    paths: Dict[str, str] = {}

    df_path = os.path.join(output_dir, "highest_df.csv")
    _write_csv(df_path, ("term", "tag", "df", "idf"), df_rows)
    paths["highest_df"] = df_path
    LOGGER.info("event=export_csv file=%s rows=%d", df_path, len(df_terms_sorted))

    idf_path = os.path.join(output_dir, "highest_idf.csv")
    _write_csv(idf_path, ("term", "tag", "idf", "df"), idf_rows)
    paths["highest_idf"] = idf_path
    LOGGER.info("event=export_csv file=%s rows=%d", idf_path, len(idf_terms_sorted))

    df_idf_path = os.path.join(output_dir, "highest_df_idf.csv")
    _write_csv(df_idf_path, ("term", "tag", "df", "idf", "df_idf"), df_idf_sorted)
    paths["highest_df_idf"] = df_idf_path
    LOGGER.info("event=export_csv file=%s rows=%d", df_idf_path, len(df_idf_sorted))

    nouns_path = os.path.join(output_dir, "nouns.csv")
    noun_rows = (
        (
            lemma,
            lemma_to_tag.get(lemma, ""),
            nouns_freq.get(lemma, 0),
            totals_tf.get(lemma, 0),
            term_to_df.get(lemma, 0),
            term_to_idf.get(lemma, 0.0),
        )
        for lemma in sorted(nouns_set)
    )
    _write_csv(
        nouns_path, ("lemma", "tag", "token_freq", "total_tf", "df", "idf"), noun_rows
    )
    paths["nouns"] = nouns_path
    LOGGER.info("event=export_csv file=%s rows=%d", nouns_path, len(nouns_set))

    LOGGER.info("event=export_csv_reports status=finished")
    return paths
