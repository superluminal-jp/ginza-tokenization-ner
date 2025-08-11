## Overview

GiNZA-based tokenization and NER with JSON input/output. The package in `src/json_ginza_processor/` accepts structured JSON, processes text with GiNZA (`ja_ginza_electra`), and returns structured JSON preserving connections to the original text via ids, sentence indices, and character offsets. It also computes TF, IDF, and TF-IDF over lemmas.

No CLI is provided; use it programmatically in Python.

## What is GiNZA?

GiNZA is a Japanese NLP pipeline for spaCy. It delivers high‑quality Japanese
tokenization, sentence segmentation, part‑of‑speech tagging, dependency
parsing, and named entity recognition (NER). This project uses the
`ja_ginza_electra` model, a transformer‑based variant focused on accuracy for
modern Japanese text.

### Key features

- **JSON in/JSON out**: simple, predictable schema.
- **Sentence and token alignment**: indices and character offsets.
- **Lemma-based statistics**: TF, IDF, TF-IDF with term occurrences.
- **Structured logging** and error handling.
- **CSV reports**: export noun-only DF, IDF, DF\*IDF rankings and all nouns.

## How it works (behind the scenes)

The API normalizes your input, runs GiNZA via spaCy, and serializes results
into a predictable JSON structure.

- Input normalization:

  - Accepts `{ "text": "…" }`, `{ "texts": ["…", {"id": "…", "text": "…"}] }`.
  - Auto‑assigns a numeric string id when `id` is omitted.

- Model loading:

  - `ja_ginza_electra` is lazily loaded on first use and cached.
  - Load and per‑text processing latencies are logged.

- Sentences and tokens:

  - Sentences come from `doc.sents` with `index`, `start_char`, `end_char`,
    and `text`.
  - Tokens carry stable indices and character offsets to align to the original
    text.

- Token fields (from spaCy `Token`):

  - `doc_index` (`token.i`), `sent_index` (index within sentence),
    `start_char`/`end_char`, `text` (`orth_`), `lemma`, `norm`.
  - `reading` = `token.morph.get("Reading")`, `inflection` =
    `token.morph.get("Inflection")`.
  - `pos`, `tag`, `dep`, and `head_doc_index` (`token.head.i`).

- Term statistics (lemma‑based):

  - Spaces and punctuation are excluded from counts.
  - Per document: `tf_raw`, `tf_norm = tf_raw / num_terms`.
  - Global: smoothed `idf = log((1 + N) / (1 + df)) + 1`, and `tfidf = tf_raw * idf`.
  - `occurrences` keeps `{doc_index, sent_index}` links back to tokens.

- NER note:
  - While GiNZA produces entities (`doc.ents`), this package’s JSON currently
    omits entity spans. They can be added in a future extension if needed.

## Install

See `INSTALL.md` for environment setup and GiNZA model installation. Ensure the spaCy model `ja_ginza_electra` is available.

## Usage

### Programmatic (recommended)

```python
# Run from project root
import sys
sys.path.append("src")

# Preferred import from the package
from json_ginza_processor import (
    process_payload_to_dict,
    process_payload_to_json,
    export_reports_from_payload,
)

payload = {
    "texts": [
        {"id": "utt-1", "text": "銀座でランチをご一緒しましょう。"},
        "明日、会議があります。",
    ]
}

result_dict = process_payload_to_dict(payload)
print(result_dict["inputs"][0]["sentences"][0]["tokens"][0])
print(result_dict["inputs"][0]["term_stats"]["terms"][:3])

result_json = process_payload_to_json(payload)
print(result_json)

# Export CSV reports to ./reports
paths = export_reports_from_payload(payload, output_dir="reports")
print(paths)
```

### Example script

- Run the sample script from the project root:

```bash
python src/samples/usage_example.py
```

- What it does:
  - Uses the package API to process two example texts.
  - Prints a preview of the first token and top terms to stdout.
  - Writes the full JSON output to `result.json` in the project root.
- Writes CSV reports under `reports/` directory (rankings are nouns/proper nouns only):
  - `highest_df.csv` (noun term, df, idf)
  - `highest_idf.csv` (noun term, idf, df)
  - `highest_df_idf.csv` (noun term, df, idf, df_idf)
  - `nouns.csv` (lemma, token_freq, total_tf, df, idf)

### Minimal input variants

- **Single string**: `{ "text": "銀座でランチをご一緒しましょう。" }`
- **List of strings**: `{ "texts": ["…", "…"] }`
- **List of objects**: `{ "texts": [{"id": "utt-1", "text": "…"}, …] }`

## Input schema

```json
{
  "texts": [
    "string" |
    { "id": "string (optional)", "text": "string" }
  ]
}
```

If `id` is omitted, a numeric string index is auto-assigned.

## Output schema (highlights)

- **Top-level**

  - **model**: spaCy model name (e.g., `ja_ginza_electra`).
  - **language**: `ja`.
  - **processed_at**: ISO8601 UTC timestamp.
  - **inputs**: list of per-input results.
  - **vocabulary**: global list of `{term, df, idf}` across all inputs.

- **Per input** (`inputs[i]`)
  - **id**: input id.
  - **text**: original text.
  - **sentences**: sentence list with `index`, `start_char`, `end_char`, `text`.
  - Each sentence contains **tokens** with indices and offsets.
  - **term_stats**:
    - **num_terms**: total counted terms (after filtering punctuation/spaces).
    - **vocab_size**: number of unique terms in this document.
    - **terms**: list of `{term, tf_raw, tf_norm, idf, tfidf, occurrences}`.

### Token object

```json
{
  "doc_index": 0,
  "sent_index": 0,
  "start_char": 0,
  "end_char": 2,
  "text": "銀座",
  "lemma": "銀座",
  "norm": "銀座",
  "reading": ["ギンザ"],
  "pos": "PROPN",
  "inflection": [],
  "tag": "名詞-固有名詞-地名-一般",
  "dep": "nmod",
  "head_doc_index": 4
}
```

### Term statistics

- **tf_raw**: raw count of the lemma in the document.
- **tf_norm**: normalized term frequency, `tf_raw / num_terms`.
- **idf**: smoothed inverse document frequency, `log((1 + N) / (1 + df)) + 1`.
- **tfidf**: `tf_raw * idf`.
- **occurrences**: list of `{doc_index, sent_index}` to link back to tokens.

## Notes

- The processor filters out punctuation and spaces for term statistics.
- TF/IDF/TF-IDF are computed on lemmas to reduce sparsity.
- Logging is enabled via Python `logging`; configure levels/handlers as needed.
- The legacy module `src/json_ginza_processor.py` is deprecated and now re-exports
  the public API from the package. Prefer `from json_ginza_processor import ...`.

## Development

- Code style: PEP8, Black (line length 88), isort, type hints.
- Linting: flake8 (ignore E203, W503), no unused imports or commented-out code.
- Tests: add tests mirroring public functions.
