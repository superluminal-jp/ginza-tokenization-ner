"""CSV report generation utilities for GiNZA processor outputs.

This module provides helpers to export statistics from the JSON/dict output of
the GiNZA processing pipeline into CSV files:

- Highest DF (document frequency) ranking
- Highest IDF (inverse document frequency) ranking
- Highest DF-IDF (df * idf) ranking
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
    result: Mapping[str, Any]
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


def _collect_nouns(result: Mapping[str, Any]) -> Tuple[set[str], Dict[str, int]]:
    """Collect unique noun lemmas from token payloads.

    Nouns are identified using spaCy coarse POS tags: {NOUN, PROPN}.

    Returns
    -------
    Tuple[set[str], Dict[str, int]]
        A set of unique noun lemmas, and a frequency counter across all tokens.
    """

    noun_pos = {"NOUN", "PROPN"}
    nouns: set[str] = set()
    freq: Counter[str] = Counter()

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

    return nouns, dict(freq)


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
    nouns_set, nouns_freq = _collect_nouns(result)

    # Highest DF ranking
    df_rows = (
        (term, term_to_df.get(term, 0), term_to_idf.get(term, 0.0))
        for term in sorted(
            term_to_df.keys(), key=lambda t: (-term_to_df[t], t)
        )
    )

    # Highest IDF ranking
    idf_rows = (
        (term, term_to_idf.get(term, 0.0), term_to_df.get(term, 0))
        for term in sorted(
            term_to_idf.keys(), key=lambda t: (-term_to_idf[t], t)
        )
    )

    # Highest DF-IDF ranking (df * idf)
    df_idf_pairs = (
        (term, term_to_df.get(term, 0), term_to_idf.get(term, 0.0))
        for term in term_to_df.keys() | term_to_idf.keys()
    )
    df_idf_sorted = sorted(
        (
            (term, df, idf, float(df) * float(idf))
            for term, df, idf in df_idf_pairs
        ),
        key=lambda x: (-x[3], x[0]),
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
    _write_csv(df_path, ("term", "df", "idf"), df_rows)
    paths["highest_df"] = df_path
    LOGGER.info("event=export_csv file=%s rows=%d", df_path, len(term_to_df))

    idf_path = os.path.join(output_dir, "highest_idf.csv")
    _write_csv(idf_path, ("term", "idf", "df"), idf_rows)
    paths["highest_idf"] = idf_path
    LOGGER.info("event=export_csv file=%s rows=%d", idf_path, len(term_to_idf))

    df_idf_path = os.path.join(output_dir, "highest_df_idf.csv")
    _write_csv(df_idf_path, ("term", "df", "idf", "df_idf"), df_idf_sorted)
    paths["highest_df_idf"] = df_idf_path
    LOGGER.info("event=export_csv file=%s rows=%d", df_idf_path, len(df_idf_sorted))

    nouns_path = os.path.join(output_dir, "nouns.csv")
    _write_csv(
        nouns_path, ("lemma", "token_freq", "total_tf", "df", "idf"), noun_rows
    )
    paths["nouns"] = nouns_path
    LOGGER.info("event=export_csv file=%s rows=%d", nouns_path, len(nouns_set))

    LOGGER.info("event=export_csv_reports status=finished")
    return paths


