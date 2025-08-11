## Installation Guide

This project uses GiNZA and the `ja_ginza_electra` model.
For the most reliable experience, use Python 3.11 so dependencies like
`tokenizers` install from prebuilt wheels.

### Prerequisites

- OS: macOS, Linux, or Windows.
- Python 3.11 (recommended; 3.8–3.11 generally work).
- pip and venv (bundled with Python).

If you need a specific Python version, consider using a version manager
such as pyenv (macOS/Linux) or the official Python installer (Windows).

### Create and activate a virtual environment

macOS/Linux (bash/zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -V
pip install --upgrade pip setuptools wheel
```

Windows (PowerShell):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -V
pip install --upgrade pip setuptools wheel
```

### Keep Python 3.11 for this project

Keeping Python at 3.11 avoids building `tokenizers` from source and ensures a
reliable installation.

Option A: with pyenv (recommended on macOS/Linux)

```bash
# Install pyenv (macOS example with Homebrew)
brew install pyenv

# Install a specific 3.11.x and pin it to this project directory
pyenv install 3.11.9
pyenv local 3.11.9   # creates a .python-version file

# Verify and create a venv using the pinned interpreter
python -V             # should show Python 3.11.9
python -m venv .venv
source .venv/bin/activate
```

Alternative (pyenv-virtualenv)

```bash
brew install pyenv pyenv-virtualenv
pyenv install 3.11.9
pyenv virtualenv 3.11.9 ginza-3.11
pyenv local ginza-3.11
python -V             # should show Python 3.11.x
```

Option B: without pyenv (if `python3.11` is available on your system)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -V             # should show Python 3.11.x
```

Tip: Commit the generated `.python-version` to ensure collaborators also use
Python 3.11 when using pyenv.

### Install dependencies

Option A: install from the repository requirements file

```bash
pip install -r src/requirements.txt
```

Option B: install core packages directly

```bash
pip install -U ginza ja_ginza_electra
```

Note: The first run of the model will download weights from Hugging Face
automatically.

### Verify the installation

Run this one-liner (works on macOS/Linux/Windows):

```bash
python -c "import spacy; nlp=spacy.load('ja_ginza_electra'); doc=nlp('東京都で自然言語処理を学習しています。'); print([t.text for t in doc]); print([(e.text, e.label_) for e in doc.ents])"
```

You should see Japanese tokens and named entities printed without errors.

### Troubleshooting

- Tokenizers build error (common on Python 3.12)

  - Symptom: pip attempts to build `tokenizers` from source and fails.
  - Fix: use Python 3.11 in your virtualenv and reinstall. If you must
    stay on 3.12, install a Rust toolchain and `setuptools-rust`, then
    retry (may still be slower/less reliable):
    ```bash
    pip install setuptools-rust
    ```

- Offline or restricted network environments

  - The model weights are downloaded on first use. If online access is
    restricted, use the “with-model” release artifact from GiNZA and
    install it from a pre-downloaded file. See: [GiNZA releases](https://github.com/megagonlabs/ginza/releases).

- PyTorch wheel selection
  - The default installation pulls a CPU-only PyTorch wheel. For custom
    builds, see: [PyTorch Get Started](https://pytorch.org/get-started/locally/).

### Useful commands

- Activate the environment (macOS/Linux):

```bash
source .venv/bin/activate
```

- Activate the environment (Windows PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

- Deactivate the environment (all platforms):

```bash
deactivate
```

- Validate spaCy installation:

```bash
python -m spacy validate
```

### Notes

- Prefer Python 3.11 for this project to avoid building `tokenizers`.
- If you change dependencies, update `src/requirements.txt` and test the
  install steps again.
