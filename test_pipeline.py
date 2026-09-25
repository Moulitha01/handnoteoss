"""Quick self-test (no TrOCR needed):  python test_pipeline.py"""
from core import database as db
from core.ocr_engine import extract_text
from core.exporter import FORMATS
db.init()
text, eng = extract_text("samples/sample_note.png", "tesseract")
print("Engine:", eng, "\nText:\n" + text)
nid = db.add("Self-test", text or "hello world", eng, "sample_note.png", "test")
print("Search hit:", [n["id"] for n in db.search("self")])
for f, (fn, _) in FORMATS.items():
    print(f, len(fn(db.get(nid))), "bytes")
db.delete(nid)
print("ALL OK")
