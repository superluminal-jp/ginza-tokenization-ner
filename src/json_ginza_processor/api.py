"""Public API helpers for GiNZA JSON processor."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Mapping

from .models import InputItem
from .processor import GiNZAProcessor
from .reports import export_csv_reports


LOGGER = logging.getLogger(__name__)


def _normalize_input(payload: Mapping[str, Any]) -> list[InputItem]:
    """Normalize various input payload shapes into a list of `InputItem`."""

    if not isinstance(payload, Mapping):
        raise ValueError("Input payload must be a mapping/dict.")

    if "texts" not in payload:
        if "text" in payload and isinstance(payload["text"], str):
            return [InputItem(id="0", text=payload["text"])]
        raise ValueError("Missing 'texts' key in payload.")

    texts_obj = payload["texts"]
    if not hasattr(texts_obj, "__iter__") or isinstance(texts_obj, (str, bytes)):
        raise ValueError("'texts' must be an iterable of strings or objects.")

    normalized: list[InputItem] = []
    for idx, item in enumerate(texts_obj):
        if isinstance(item, str):
            normalized.append(InputItem(id=str(idx), text=item))
            continue

        if isinstance(item, Mapping):
            if "text" not in item or not isinstance(item["text"], str):
                raise ValueError(
                    "Each object in 'texts' must contain a string 'text' field."
                )
            item_id = str(item.get("id", idx))
            normalized.append(InputItem(id=item_id, text=item["text"]))
            continue

        raise ValueError(
            "Elements of 'texts' must be strings or mappings with a 'text' field."
        )

    return normalized


def process_payload_to_dict(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Process a JSON-like mapping and return a structured dict output."""

    LOGGER.info("event=process_payload_to_dict status=starting")
    items = _normalize_input(payload)
    processor = GiNZAProcessor()
    result = processor.process_items(items)
    LOGGER.info("event=process_payload_to_dict status=finished")
    return result


def process_payload_to_json(payload: Mapping[str, Any]) -> str:
    """Process a JSON-like mapping and return a JSON string."""

    result_dict = process_payload_to_dict(payload)
    return json.dumps(result_dict, ensure_ascii=False, separators=(",", ":"), indent=2)


def export_reports_from_payload(payload: Mapping[str, Any], output_dir: str) -> Dict[str, str]:
    """Process payload and write CSV reports to ``output_dir``.

    Generates four CSV files:
    - highest_df.csv: terms ranked by document frequency (df desc).
    - highest_idf.csv: terms ranked by inverse document frequency (idf desc).
    - highest_df_idf.csv: terms ranked by df * idf desc.
    - nouns.csv: unique noun lemmas with counts and df/idf when available.

    Parameters
    ----------
    payload:
        Input mapping, same as accepted by ``process_payload_to_dict``.
    output_dir:
        Directory where CSV files will be created.

    Returns
    -------
    Dict[str, str]
        Mapping of logical report names to file paths.
    """

    LOGGER.info(
        "event=export_reports_from_payload status=starting output_dir=%s", output_dir
    )
    result = process_payload_to_dict(payload)
    paths = export_csv_reports(result, output_dir)
    LOGGER.info("event=export_reports_from_payload status=finished")
    return paths

