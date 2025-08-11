import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import spacy
from sklearn.model_selection import train_test_split
from spacy.cli.train import train
from spacy.language import Language


def _read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read dataset from JSON array or JSONL file.

    Args:
        path: Path to a `.json` or `.jsonl` file.

    Returns:
        A list of example dictionaries.
    """
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    # JSON array
    if content[0] == "[":
        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError(f"Top-level JSON must be list: {path}")
        return data
    # JSONL
    examples: List[Dict[str, Any]] = []
    for idx, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            examples.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON on line {idx} in {path}: {exc.msg}"
            ) from exc
    return examples


def prepare_stockmark_data(
    data_dir: str | Path, train_ratio: float = 0.8
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Prepare Stockmark dataset for training.

    Supports both JSONL files and JSON array files (e.g., `ner.json`).

    Args:
        data_dir: Directory containing dataset files.
        train_ratio: Proportion of data to use for training.

    Returns:
        Tuple of (train_data, dev_data) lists.
    """
    start_time = time.perf_counter()
    data_path = Path(data_dir)
    all_data: List[Dict[str, Any]] = []

    # Read JSONL files
    for jsonl_file in data_path.glob("*.jsonl"):
        all_data.extend(_read_json_or_jsonl(jsonl_file))

    # Read JSON array files (e.g., ner.json)
    for json_file in data_path.glob("*.json"):
        all_data.extend(_read_json_or_jsonl(json_file))

    logging.info("loaded_examples count=%d dir=%s", len(all_data), str(data_path))

    if not all_data:
        raise ValueError(
            f"No dataset files found or empty data under: {data_path}. "
            "Provide JSONL files or a JSON array file like ner.json."
        )

    train_data, dev_data = train_test_split(
        all_data, train_size=train_ratio, random_state=42
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logging.info(
        "split_dataset train=%d dev=%d elapsed_ms=%.2f",
        len(train_data),
        len(dev_data),
        elapsed_ms,
    )
    return train_data, dev_data


def _extract_entities(
    raw_entities: Iterable[Dict[str, Any]],
) -> List[Tuple[int, int, str]]:
    """Map dataset entity dicts to spaCy (start, end, label) tuples.

    Supports both Stockmark format (`span`, `type`) and generic
    (`start`, `end`, `label`).
    """
    ents: List[Tuple[int, int, str]] = []
    for ent in raw_entities or []:
        if "start" in ent and "end" in ent and "label" in ent:
            ents.append((int(ent["start"]), int(ent["end"]), str(ent["label"])))
            continue
        span = ent.get("span")
        label = ent.get("type")
        if (
            isinstance(span, (list, tuple))
            and len(span) == 2
            and isinstance(span[0], int)
            and isinstance(span[1], int)
            and isinstance(label, str)
        ):
            ents.append((int(span[0]), int(span[1]), label))
    return ents


def convert_to_spacy_format(
    data: List[Dict[str, Any]], nlp: Language, output_path: str
) -> int:
    """Convert dataset list to spaCy DocBin format and write to disk.

    Args:
        data: List of input examples with `text` and `entities`.
        nlp: spaCy pipeline used for tokenization only.
        output_path: Destination `.spacy` file path.

    Returns:
        Number of documents written.
    """
    from spacy.tokens import DocBin
    from spacy.training import Example

    doc_bin = DocBin()
    num_added = 0

    for idx, item in enumerate(data):
        text = item.get("text")
        if not isinstance(text, str):
            logging.warning("skipping_non_string_text idx=%d", idx)
            continue
        entities = _extract_entities(item.get("entities", []))
        text_len = len(text)
        filtered: List[Tuple[int, int, str]] = []
        for start, end, label in entities:
            if 0 <= start < end <= text_len:
                filtered.append((start, end, label))
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, {"entities": filtered})
        doc_bin.add(example.reference)
        num_added += 1

    doc_bin.to_disk(output_path)
    logging.info("wrote_docbin count=%d path=%s", num_added, output_path)
    return num_added


def _build_nlp(model_name: str) -> Language:
    """Load spaCy pipeline, fallback to `blank('ja')` if needed."""
    try:
        logging.info("loading_model name=%s", model_name)
        return spacy.load(model_name)
    except Exception:  # noqa: BLE001
        logging.exception("model_load_failed name=%s fallback=blank('ja')", model_name)
        return spacy.blank("ja")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune GiNZA NER with Stockmark dataset"
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing Stockmark JSON/JSONL files (e.g., ner.json)",
    )
    parser.add_argument(
        "--output_dir",
        default="./model_output",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--config", default="./config.cfg", help="spaCy training config file"
    )
    parser.add_argument(
        "--base_model",
        default="ja_ginza_electra",
        help="Base model to fine-tune (e.g., ja_ginza_electra)",
    )
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Only prepare and write train/dev .spacy files, skip training",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        level=getattr(logging, args.log_level.upper(), logging.INFO),
    )

    Path(args.output_dir).mkdir(exist_ok=True)
    Path("./data").mkdir(exist_ok=True)

    logging.info("Loading base model...")
    nlp = _build_nlp(args.base_model)

    logging.info("Preparing data...")
    train_data, dev_data = prepare_stockmark_data(args.data_dir)

    convert_to_spacy_format(train_data, nlp, "./data/train.spacy")
    convert_to_spacy_format(dev_data, nlp, "./data/dev.spacy")

    if args.prepare_only:
        logging.info("prepare_only=true; skipping training phase")
        return

    logging.info("Starting training...")
    train(
        config_path=args.config,
        output_path=args.output_dir,
        use_gpu=-1,  # Change to 0 for GPU
        overrides={},
    )

    logging.info("Training completed! model_dir=%s", args.output_dir)


if __name__ == "__main__":
    main()
