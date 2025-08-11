"""Example usage of the json_ginza_processor package.

Run with: python src/samples/usage_example.py
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict


def build_example_payload() -> Dict[str, Any]:
    """Build a minimal example payload to process.

    Returns
    -------
    Dict[str, Any]
        A payload following the documented input schema.
    """

    return {
        "texts": [
            {"id": "utt-1", "text": "銀座でランチをご一緒しましょう。"},
            "明日、会議があります。",
        ]
    }


def main() -> None:
    """Run the example using the public API functions."""

    # Make package importable when running from project root
    sys.path.append("src")

    from json_ginza_processor import (  # pylint: disable=C0415
        process_payload_to_dict,
        process_payload_to_json,
        export_reports_from_payload,
    )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger("usage_example")

    payload = build_example_payload()
    logger.info("processing payload with %d texts", len(payload.get("texts", [])))

    result_dict = process_payload_to_dict(payload)
    first_input = result_dict["inputs"][0]
    first_sentence = first_input["sentences"][0]
    logger.info(
        "first token: %s",
        json.dumps(first_sentence["tokens"][0], ensure_ascii=False),
    )
    logger.info(
        "top terms: %s",
        json.dumps(first_input["term_stats"]["terms"][:3], ensure_ascii=False),
    )

    result_json = process_payload_to_json(payload)
    logger.info("json output preview: %s", result_json[:200].replace("\n", " ") + " …")

    # Save the result to a file
    with open("result.json", "w", encoding="utf-8") as f:
        f.write(result_json)

    # Export CSV reports
    report_paths = export_reports_from_payload(payload, output_dir="reports")
    logger.info("CSV reports written: %s", report_paths)


if __name__ == "__main__":
    main()


