import os, pathlib
import re

# Folder with your cleaned plain text files
folder = pathlib.Path("data")
texts = []

for file in folder.glob("*.plain.txt"):
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()
        # Clean the text to remove standalone numbers followed by commas
        text = re.sub(r"\b\d+\b[,\s]*", "", text)  # Remove standalone numbers
        texts.append(text)

# Combine all texts into one corpus
full_text = "\n".join(texts)
print("Total chars:", len(full_text))

# Split 80/10/10
n = len(full_text)
train_text = full_text[:int(n * 0.8)]
val_text = full_text[int(n * 0.8):int(n * 0.9)]
test_text = full_text[int(n * 0.9):]

# Save the splits
outdir = pathlib.Path("monkeypox_ds")
outdir.mkdir(exist_ok=True)

with open(outdir / "train.txt", "w", encoding="utf-8") as f:
    f.write(train_text)
with open(outdir / "val.txt", "w", encoding="utf-8") as f:
    f.write(val_text)
with open(outdir / "test.txt", "w", encoding="utf-8") as f:
    f.write(test_text)

print("Saved:", outdir)