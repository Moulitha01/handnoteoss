# Stall guide (first-timer friendly)

## Your 60-second pitch
"Old paper notes are impossible to search. We photograph a page, OpenCV cleans it, an open-source AI (TrOCR from HuggingFace) reads the handwriting, and you get editable text stored in SQLite - searchable and exportable to Word or XML. Everything is open source, so you can run, study and improve it - that is software freedom."

## Live demo script (3 min)
1. Show a handwritten page. 2. Upload/scan it. 3. Text appears - fix one word by hand. 4. Save, search a word from an older note. 5. Export DOCX + XML and open them.

## Thermocol cut-outs (hang above the stall, label each with its logo + licence)
Python - OpenCV - Tesseract - HuggingFace - PyTorch - Flask - SQLite - python-docx - Linux (if you use it)
Plus 5 bigger boards for the flow: INPUT -> OpenCV -> OCR -> SQLite -> EXPORT, joined with string/arrows.
How: print logos on A4, glue on 10 mm thermocol sheet, cut with a hobby knife, paint, punch a hole and hang with fishing line. Print `docs/architecture.svg` at A1/A2 as the main poster.

## Checklist
- Laptop charged + charger, extension cord, run `python app.py` beforehand (first TrOCR run downloads ~1.3 GB - do it at home).
- Pre-loaded 5 note photos as backup, offline (no venue Wi-Fi needed).
- Printed accuracy result screenshot from `training/evaluate.py`.
- Likely questions: why open source? (freedom to inspect/modify) - why TrOCR over Tesseract? (Tesseract is weak on cursive) - what data? (IAM) - what's the limit? (messy writing lowers accuracy).
