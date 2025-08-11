import argparse
import json
import logging
import time
from typing import Any, Dict, Iterable, List, Tuple

import spacy
from spacy.language import Language
from spacy.tokens import DocBin
from spacy.training import Example


LOGGER = logging.getLogger(__name__)


def _read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read dataset from JSON array or JSONL.

    The Stockmark dataset file `ner.json` is a JSON array. Some workflows may
    use JSONL (one JSON object per line). This function accepts both.

    Args:
        path: Input file path to read.

    Returns:
        List of raw example dicts as loaded from the file.
    """
    start_time = time.perf_counter()
    with open(path, "r", encoding="utf-8") as file:
        content = file.read().strip()

    if not content:
        raise ValueError("Input file is empty.")

    try:
        if content[0] == "[":
            data = json.loads(content)
            if not isinstance(data, list):
                raise ValueError("Top-level JSON must be a list of objects.")
            LOGGER.info("Loaded %d examples from JSON array.", len(data))
            return data
        # Fallback: JSONL
        examples: List[Dict[str, Any]] = []
        for idx, line in enumerate(content.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {idx}: {exc.msg}") from exc
        LOGGER.info("Loaded %d examples from JSONL.", len(examples))
        return examples
    finally:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        LOGGER.info("read_input elapsed_ms=%.2f path=%s", elapsed_ms, path)


def _extract_entities(
    raw_entities: Iterable[Dict[str, Any]],
) -> List[Tuple[int, int, str]]:
    """Convert Stockmark entity dicts to spaCy format tuples.

    Stockmark uses keys: `name`, `span` (list [start, end]), `type`.

    Args:
        raw_entities: Iterable of entity dicts from the dataset.

    Returns:
        List of (start, end, label) tuples.
    """
    entities: List[Tuple[int, int, str]] = []
    for entity in raw_entities or []:
        span = entity.get("span")
        label = entity.get("type")
        if (
            not isinstance(span, (list, tuple))
            or len(span) != 2
            or not isinstance(span[0], int)
            or not isinstance(span[1], int)
            or not isinstance(label, str)
        ):
            LOGGER.warning("Skipping malformed entity: %s", entity)
            continue
        start, end = int(span[0]), int(span[1])
        if start >= end:
            LOGGER.warning("Skipping invalid span start>=end: %s", entity)
            continue
        entities.append((start, end, label))
    return entities


def convert_stockmark_to_spacy(input_file: str, output_file: str, nlp: Language) -> int:
    """Convert Stockmark NER dataset to spaCy DocBin format.

    This reads the Stockmark dataset (JSON array or JSONL), converts each
    example's entities to spaCy training tuples, and writes a `.spacy` file.

    Args:
        input_file: Path to `ner.json` or compatible JSONL file.
        output_file: Destination path for the `.spacy` DocBin.
        nlp: Loaded spaCy pipeline used to create docs (tokenizer only).

    Returns:
        Number of documents written to the DocBin.
    """
    start_time = time.perf_counter()
    raw_examples = _read_json_or_jsonl(input_file)

    doc_bin = DocBin()
    num_added = 0

    for idx, item in enumerate(raw_examples):
        try:
            text = item["text"]
            if not isinstance(text, str):
                raise ValueError("Field 'text' must be a string.")

            entities = _extract_entities(item.get("entities", []))

            # Optional sanity checks
            text_len = len(text)
            valid_entities: List[Tuple[int, int, str]] = []
            for start, end, label in entities:
                if end > text_len:
                    LOGGER.warning(
                        "Skipping out-of-range entity span [%d, %d) for idx=%d",
                        start,
                        end,
                        idx,
                    )
                    continue
                valid_entities.append((start, end, label))

            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, {"entities": valid_entities})
            doc_bin.add(example.reference)
            num_added += 1
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception(
                "Failed to process example idx=%d curid=%s: %s",
                idx,
                item.get("curid"),
                exc,
            )

    doc_bin.to_disk(output_file)
    elapsed_s = time.perf_counter() - start_time
    LOGGER.info(
        "wrote_docbin count=%d path=%s elapsed_s=%.2f",
        num_added,
        output_file,
        elapsed_s,
    )
    return num_added


def _build_nlp(model_name: str) -> Language:
    """Load spaCy pipeline, falling back to a blank Japanese tokenizer.

    Args:
        model_name: Name or path of a spaCy model to load.

    Returns:
        A spaCy `Language` object.
    """
    try:
        LOGGER.info("Loading spaCy model: %s", model_name)
        return spacy.load(model_name)
    except Exception:  # noqa: BLE001
        LOGGER.exception(
            "Failed to load model '%s'. Falling back to blank('ja').", model_name
        )
        return spacy.blank("ja")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Stockmark NER dataset (JSON/JSONL) to spaCy DocBin (.spacy)."
        )
    )
    parser.add_argument("--input", default="ner.json", help="Input JSON or JSONL file")
    parser.add_argument("--output", default="ner.spacy", help="Output .spacy file")
    parser.add_argument(
        "--model",
        default="ja_ginza_electra",
        help="spaCy model to use for tokenization",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format=("%(asctime)s %(levelname)s %(name)s op=%(message)s"),
        level=getattr(logging, args.log_level.upper(), logging.INFO),
    )

    nlp = _build_nlp(args.model)
    count = convert_stockmark_to_spacy(args.input, args.output, nlp)
    LOGGER.info("conversion_complete count=%d output=%s", count, args.output)


if __name__ == "__main__":
    main()
