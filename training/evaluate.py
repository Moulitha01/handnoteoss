"""Test the model on the held-out TEST split and report accuracy.
Character accuracy = 100 x (1 - CER).  Target for this project: >= 94 %.
Run:  python training/evaluate.py [--model models/trocr-finetuned] [--limit 500]
"""
import argparse, os, torch
from datasets import load_dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import jiwer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/trocr-finetuned")
ap.add_argument("--dataset", default="Teklia/IAM-line")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

name = a.model if os.path.isdir(a.model) else "microsoft/trocr-base-handwritten"
print("Evaluating:", name)
proc = TrOCRProcessor.from_pretrained(name)
model = VisionEncoderDecoderModel.from_pretrained(name).eval()
dev = "cuda" if torch.cuda.is_available() else "cpu"; model.to(dev)
test = load_dataset(a.dataset, split="test")
if a.limit: test = test.select(range(a.limit))

preds, refs = [], []
for i in range(0, len(test), 16):
    b = test[i:i + 16]
    pv = proc(images=[im.convert("RGB") for im in b["image"]], return_tensors="pt").pixel_values.to(dev)
    with torch.no_grad():
        ids = model.generate(pv, max_new_tokens=64, num_beams=4)
    preds += proc.batch_decode(ids, skip_special_tokens=True); refs += b["text"]

cer = jiwer.cer(refs, preds); wer = jiwer.wer(refs, preds)
print("Test samples        :", len(refs))
print("Character error rate: %.2f %%" % (100 * cer))
print("Word error rate     : %.2f %%" % (100 * wer))
print("CHARACTER ACCURACY  : %.2f %%  -> %s" % (100 * (1 - cer), "PASS (>=94)" if 1 - cer >= 0.94 else "BELOW TARGET - train more epochs"))
