"""Fine-tune TrOCR on an open handwriting dataset from HuggingFace.
Default dataset: Teklia/IAM-line (IAM handwriting database, one text line per image).
Run:  python training/train_trocr.py --epochs 3
Needs a GPU for reasonable speed (free Google Colab / Kaggle T4 is enough, ~1-2 h).
"""
import argparse, os
import torch
from datasets import load_dataset
from transformers import (TrOCRProcessor, VisionEncoderDecoderModel,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator)
import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="Teklia/IAM-line")
ap.add_argument("--base", default="microsoft/trocr-base-handwritten")
ap.add_argument("--out", default="models/trocr-finetuned")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--limit", type=int, default=0, help="use only N train samples (quick test)")
a = ap.parse_args()

proc = TrOCRProcessor.from_pretrained(a.base)
model = VisionEncoderDecoderModel.from_pretrained(a.base)
model.config.decoder_start_token_id = proc.tokenizer.cls_token_id
model.config.pad_token_id = proc.tokenizer.pad_token_id
model.config.eos_token_id = proc.tokenizer.sep_token_id

ds = load_dataset(a.dataset)
train = ds["train"]; val = ds["validation"]
if a.limit:
    train = train.select(range(a.limit)); val = val.select(range(min(a.limit // 4, len(val))))


def prep(batch):
    pv = proc(images=[i.convert("RGB") for i in batch["image"]], return_tensors="pt").pixel_values
    lab = proc.tokenizer(batch["text"], padding="max_length", max_length=64, truncation=True).input_ids
    lab = [[t if t != proc.tokenizer.pad_token_id else -100 for t in l] for l in lab]
    return {"pixel_values": list(pv), "labels": lab}


train = train.with_transform(prep); val = val.with_transform(prep)
cer = evaluate.load("cer")


def metrics(p):
    pred = proc.batch_decode(p.predictions, skip_special_tokens=True)
    p.label_ids[p.label_ids == -100] = proc.tokenizer.pad_token_id
    ref = proc.batch_decode(p.label_ids, skip_special_tokens=True)
    c = cer.compute(predictions=pred, references=ref)
    return {"cer": c, "char_accuracy": 100 * (1 - c)}


args = Seq2SeqTrainingArguments(
    output_dir=a.out + "_ckpt", per_device_train_batch_size=a.bs, per_device_eval_batch_size=a.bs,
    predict_with_generate=True, eval_strategy="epoch", save_strategy="epoch",
    num_train_epochs=a.epochs, learning_rate=4e-5, fp16=torch.cuda.is_available(),
    logging_steps=50, load_best_model_at_end=True, metric_for_best_model="cer",
    greater_is_better=False, remove_unused_columns=False, report_to="none")
tr = Seq2SeqTrainer(model=model, args=args, train_dataset=train, eval_dataset=val,
                    data_collator=default_data_collator, compute_metrics=metrics,
                    processing_class=proc)
tr.train()
os.makedirs(a.out, exist_ok=True)
model.save_pretrained(a.out); proc.save_pretrained(a.out)
print("Saved to", a.out, "- now run: python training/evaluate.py")
