# Handwritten Notes Digitizer  (Software Freedom Day demo, problem #627)
Converts photos of handwritten notes into editable, searchable digital text and exports to DOCX / XML / TXT / MD / JSON.
**Every component is open source.**

## Open-source stack
| Layer | Tool | Licence |
|---|---|---|
| Language | Python 3 | PSF |
| Image cleaning | OpenCV | Apache-2.0 |
| Classic OCR | Tesseract (pytesseract) | Apache-2.0 |
| Handwriting AI | TrOCR via HuggingFace Transformers + PyTorch | MIT / Apache-2.0 / BSD |
| Dataset | IAM handwriting (HF: Teklia/IAM-line) | research licence, free to use |
| Database | SQLite (+FTS5 search) | Public domain |
| Web UI | Flask | BSD |
| Exports | python-docx, xml.etree | MIT / PSF |

## Setup
```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr python3-venv
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python test_pipeline.py      # smoke test
python app.py                # open http://localhost:5000  (phones on same Wi-Fi: http://<laptop-ip>:5000)
```
Windows: install Tesseract from the UB-Mannheim build, add it to PATH.

## Getting >= 94 % accuracy (do this BEFORE the event)
Accuracy here = character accuracy = 100 x (1 - CER) on the IAM **test** split.
```bash
python training/evaluate.py                     # baseline: pretrained microsoft/trocr-base-handwritten
python training/train_trocr.py --epochs 3       # fine-tune on GPU (Colab/Kaggle free T4)
python training/evaluate.py                     # final test score, prints PASS/BELOW TARGET
```
The fine-tuned model is saved to `models/trocr-finetuned` and loaded automatically by the app.
Pretrained TrOCR-base is published with ~3 % CER on IAM, so ~97 % character accuracy is realistic; word-level accuracy is lower (~85-90 %).
Report the numbers YOUR run prints - do not quote figures you have not measured.
Tips if below 94: more epochs (5-8), `--bs 16`, use the `trocr-large-handwritten` base.

## Project layout
```
app.py                Flask server + REST API      core/preprocess.py  OpenCV cleaning + line segmentation
core/ocr_engine.py    Tesseract + TrOCR            core/database.py    SQLite + FTS5
core/exporter.py      docx/xml/txt/md/json         training/           train + evaluate scripts
templates/index.html  UI                           docs/architecture.svg  chart to print
```
Note: real handwriting works best with dark pen on plain/ruled paper, good light, phone held flat.
