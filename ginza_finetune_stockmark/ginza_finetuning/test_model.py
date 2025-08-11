import spacy

# Load your fine-tuned model
nlp = spacy.load("./model_output/model-best")

# Test on new text
text = "山田花子さんはトヨタ自動車で働いています。"
doc = nlp(text)

print("doc", doc)
print("doc.ents", doc.ents)
for ent in doc.ents:
    print(f"{ent.text} -> {ent.label_}")
