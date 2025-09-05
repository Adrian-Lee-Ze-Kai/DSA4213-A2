
import argparse, json, numpy as np
from pathlib import Path

DOC_SEP = b"\n\n<|doc|>\n\n"  # separator between docs

def read_all_texts(indir: Path) -> bytes:
    parts = []
    for p in sorted(indir.glob("*.txt")):
        b = p.read_bytes()
        b = b.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        parts.append(b)
    return DOC_SEP.join(parts)

def split_bytes(b: bytes, train=0.8, val=0.1):
    n = len(b)
    n_train = int(n * train)
    n_val   = int(n * val)
    return b[:n_train], b[n_train:n_train+n_val], b[n_train+n_val:]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="folder with cleaned .txt files")
    ap.add_argument("--out_dir", default="dataset", help="where to save .bin files")
    args = ap.parse_args()

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # combine
    blob = read_all_texts(in_dir)
    print(f"Total size: {len(blob):,} bytes")

    # split
    train_b, val_b, test_b = split_bytes(blob)

    # save as uint8 (byte-level tokens 0..255)
    def write_bin(name, data):
        arr = np.frombuffer(data, dtype=np.uint8)
        arr.tofile(out_dir / name)

    write_bin("train.bin", train_b)
    write_bin("val.bin",   val_b)
    write_bin("test.bin",  test_b)

    meta = {
        "vocab_size": 256,
        "encoding": "byte-level (0–255)",
        "separator": DOC_SEP.decode("utf-8", errors="ignore")
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Saved to {out_dir.resolve()}")

if __name__ == "__main__":
    main()
