# Fine-tune GiNZA NER with Stockmark dataset

## Step 1: Data Preparation

```bash
# Clone the Stockmark dataset
git clone https://github.com/stockmarkteam/ner-wikipedia-dataset.git
```

```bash
# Create project structure
mkdir ginza_finetuning
cd ginza_finetuning
mkdir data model_output
```

## Step 2: Convert and Train

```bash
# Convert data and train model
python train_ginza_ner.py \
 --data_dir ../ner-wikipedia-dataset/ \
 --output_dir ./model_output \
 --config ./config.cfg \
 --base_model ja_ginza_electra
```

## Step 3: Evaluate Model

```bash
# Evaluate with test data
python evaluate_model.py \
 --model ./model_output/model-best \
 --test_data ./test_data.jsonl
```

```bash
# Test with sample texts
python evaluate_model.py \
 --model ./model_output/model-best \
 --sample_texts "田中太郎は東京大学の学生です。" "ソニー株式会社は日本の会社です。" \
```

## Step 4: Use the Fine-tuned Model

```bash
# Load your fine-tuned model
pythonimport spacy
nlp = spacy.load("./model_output/model-best")
```

```bash
# Test on new text
text = "山田花子さんはトヨタ自動車で働いています。"
doc = nlp(text)
```

```bash
for ent in doc.ents:
print(f"{ent.text} -> {ent.label\_}")
```

## Key Configuration Options

### Adjust Training Parameters

In config.cfg, you can modify:

- max_steps: Training duration (default: 20000)
- eval_frequency: How often to evaluate (default: 200)
- dropout: Regularization (default: 0.1)
- learn_rate: Learning rate (default: 0.001)

### GPU Training

To use GPU, change in config.cfg:

```ini
[system]
gpu_allocator = "pytorch"
```

And use GPU ID in training:

```bash
python -m spacy train config.cfg --output ./model_output --gpu-id 0
```
