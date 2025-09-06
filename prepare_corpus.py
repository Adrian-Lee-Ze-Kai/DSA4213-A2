import os, numpy as np, pathlib

# folder with your cleaned plain text files
folder = pathlib.Path("data")
texts = []

for file in folder.glob("*.plain.txt"):
    with open(file, "r", encoding="utf-8") as f:
        texts.append(f.read())

full_text = "\n".join(texts)
print("Total chars:", len(full_text))

# encode to bytes (character-level)
data = np.frombuffer(full_text.encode("utf-8"), dtype=np.uint8)

# split 80/10/10
n = len(data)
train, val, test = np.split(data, [int(n*0.8), int(n*0.9)])

outdir = pathlib.Path("monkeypox_ds")
outdir.mkdir(exist_ok=True)

train.tofile(outdir/"train.bin")
val.tofile(outdir/"val.bin")
test.tofile(outdir/"test.bin")

print("Saved:", outdir)
