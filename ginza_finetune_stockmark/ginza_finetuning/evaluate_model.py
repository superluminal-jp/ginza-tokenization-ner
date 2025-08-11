import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import spacy
from spacy.scorer import Scorer
from spacy.training import Example


def _read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read dataset from JSON array or JSONL file.

    Args:
        path: File path to JSON/JSONL.

    Returns:
        List of example dicts.
    """
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    if content[0] == "[":
        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError(f"Top-level JSON must be list: {path}")
        return data
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


def evaluate_with_json(model_name_or_path: str, test_data_path: str) -> Dict[str, Any]:
    """Evaluate model with JSON/JSONL test data.

    Args:
        model_name_or_path: spaCy model name or path.
        test_data_path: Path to JSON/JSONL file.
    """
    nlp = spacy.load(model_name_or_path)
    items = _read_json_or_jsonl(Path(test_data_path))
    examples: List[Example] = []
    for item in items:
        text = item.get("text")
        if not isinstance(text, str):
            continue
        ents = _extract_entities(item.get("entities", []))
        doc = nlp.make_doc(text)
        examples.append(Example.from_dict(doc, {"entities": ents}))
    scorer = Scorer()
    scores = scorer.score(examples)
    _print_scores(scores)
    return scores


def evaluate_with_docbin(model_name_or_path: str, docbin_path: str) -> Dict[str, Any]:
    """Evaluate model with a DocBin file containing gold annotations.

    Args:
        model_name_or_path: spaCy model name or path.
        docbin_path: Path to `.spacy` DocBin file (gold docs).
    """
    from spacy.tokens import DocBin

    nlp = spacy.load(model_name_or_path)
    doc_bin = DocBin().from_disk(docbin_path)
    gold_docs = list(doc_bin.get_docs(nlp.vocab))
    examples: List[Example] = []
    for gold_doc in gold_docs:
        pred_doc = nlp(gold_doc.text)
        examples.append(Example(pred_doc, gold_doc))
    scorer = Scorer()
    scores = scorer.score(examples)
    _print_scores(scores)
    return scores


def _print_scores(scores: Dict[str, Any]) -> None:
    print("=== Evaluation Results ===")
    print(f"Entities F-score: {scores.get('ents_f', 0.0):.4f}")
    print(f"Entities Precision: {scores.get('ents_p', 0.0):.4f}")
    print(f"Entities Recall: {scores.get('ents_r', 0.0):.4f}")
    if "ents_per_type" in scores:
        print("\n=== Per-type Scores ===")
        for entity_type, type_scores in scores["ents_per_type"].items():
            print(f"{entity_type}:")
            print(f"  F-score: {type_scores.get('f', 0.0):.4f}")
            print(f"  Precision: {type_scores.get('p', 0.0):.4f}")
            print(f"  Recall: {type_scores.get('r', 0.0):.4f}")


def test_predictions(model_name_or_path: str, test_texts: List[str]) -> None:
    """Run the model on sample texts and print entities."""
    nlp = spacy.load(model_name_or_path)
    print("\n=== Sample Predictions ===")
    for text in test_texts:
        doc = nlp(text)
        print(f"\nText: {text}")
        print("Entities:")
        for ent in doc.ents:
            print(f"  {ent.text} ({ent.label_}) [{ent.start_char}:{ent.end_char}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned GiNZA NER model")
    parser.add_argument(
        "--model",
        required=True,
        help="Path or name of the model (e.g., ./model_output/model-best or ja_ginza_electra)",
    )
    parser.add_argument(
        "--test_data",
        help="Path to test data JSON/JSONL file (Stockmark or spaCy format)",
    )
    parser.add_argument(
        "--test_docbin",
        help="Path to DocBin .spacy file with gold annotations (e.g., ./data/dev.spacy)",
    )
    parser.add_argument("--sample_texts", nargs="+", help="Sample texts to test")
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

    if args.test_docbin:
        evaluate_with_docbin(args.model, args.test_docbin)
    elif args.test_data:
        evaluate_with_json(args.model, args.test_data)

    if args.sample_texts:
        test_predictions(args.model, args.sample_texts)

    if not args.sample_texts and not args.test_data and not args.test_docbin:
        default_texts = [
            "東京駅から新宿駅まで電車で行きました。",
            "田中さんはソニー株式会社で働いています。",
            "2023年12月25日にクリスマスパーティーを開催します。",
        ]
        test_predictions(args.model, default_texts)


if __name__ == "__main__":
    main()
