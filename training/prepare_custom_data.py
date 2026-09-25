"""Build a fine-tuning dataset from your own handwriting.

Step 1 - extract line crops from your note photos and make a labels file:
    python training/prepare_custom_data.py extract --photos my_photos --out my_data

    -> my_data/lines/*.png   (one image per handwritten line)
    -> my_data/labels.tsv    (filename <TAB> text)  <- open it and type the correct text
                                                      for each line (delete rows you can't use)

Step 2 - turn the labelled lines into a train/val dataset:
    python training/prepare_custom_data.py build --data my_data

    -> my_data/dataset/train/{images, metadata.jsonl}
    -> my_data/dataset/val/{images, metadata.jsonl}

Load it in your training script and mix it with IAM:
    from datasets import load_dataset, concatenate_datasets
    mine = load_dataset("imagefolder", data_dir="my_data/dataset")
    train = concatenate_datasets([iam_train] + [mine["train"]] * 5)   # oversample your own lines x5
Aim for 100-200 labelled lines; more is better.
"""
import argparse
import csv
import json
import os
import random
import shutil
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from core.line_extractor import prepare_lines  # noqa: E402

IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def extract(photos, out):
    lines_dir = os.path.join(out, "lines")
    os.makedirs(lines_dir, exist_ok=True)
    rows = []
    for name in sorted(os.listdir(photos)):
        if not name.lower().endswith(IMG_EXT):
            continue
        img = cv2.imread(os.path.join(photos, name))
        if img is None:
            print("skip (unreadable):", name)
            continue
        stem = os.path.splitext(name)[0]
        for i, line in enumerate(prepare_lines(img)):
            fn = f"{stem}_{i:02d}.png"
            line.save(os.path.join(lines_dir, fn))
            rows.append((fn, ""))
    with open(os.path.join(out, "labels.tsv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f, delimiter="\t").writerows(rows)
    print(f"Saved {len(rows)} line crops. Now fill in the 2nd column of {out}/labels.tsv")


def build(data, val_frac=0.1, seed=42):
    lines_dir = os.path.join(data, "lines")
    with open(os.path.join(data, "labels.tsv"), encoding="utf-8") as f:
        rows = [(r[0], r[1].strip()) for r in csv.reader(f, delimiter="\t") if len(r) >= 2 and r[1].strip()]
    if not rows:
        sys.exit("No labelled rows found - fill in the text column first.")

    random.Random(seed).shuffle(rows)
    n_val = max(1, int(len(rows) * val_frac))
    splits = {"val": rows[:n_val], "train": rows[n_val:]}

    for split, items in splits.items():
        d = os.path.join(data, "dataset", split)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "metadata.jsonl"), "w", encoding="utf-8") as meta:
            for fn, text in items:
                shutil.copy(os.path.join(lines_dir, fn), os.path.join(d, fn))
                meta.write(json.dumps({"file_name": fn, "text": text}, ensure_ascii=False) + "\n")
        print(f"{split}: {len(items)} lines")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--photos", required=True)
    e.add_argument("--out", required=True)
    b = sub.add_parser("build")
    b.add_argument("--data", required=True)
    args = ap.parse_args()
    extract(args.photos, args.out) if args.cmd == "extract" else build(args.data)