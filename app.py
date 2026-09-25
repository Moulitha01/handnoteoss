"""Handwritten Notes Digitizer - Flask web app (run: python app.py)."""
import os
import io
import uuid
import time
from flask import Flask, request, jsonify, render_template, send_file
from core import database as db
from core.ocr_engine import extract_text, trocr_available, preload
from core.exporter import FORMATS
from core import llm_helper

app = Flask(__name__)
UPLOADS = os.path.join(os.path.dirname(__file__), "data", "uploads")
os.makedirs(UPLOADS, exist_ok=True)
db.init()
preload()  # load TrOCR in the background so the first upload is fast


@app.route("/")
def index():
    return render_template("index.html", trocr=trocr_available(), llm=llm_helper.is_available())


@app.post("/api/upload")
def upload():
    f = request.files.get("image")
    if not f:
        return jsonify(error="no image"), 400
    ext = os.path.splitext(f.filename)[1].lower() or ".png"
    path = os.path.join(UPLOADS, uuid.uuid4().hex + ext)
    f.save(path)
    try:
        raw_text, engine = extract_text(path, request.form.get("engine", "auto"))
    except Exception as e:
        return jsonify(error=str(e)), 500

    # Local LLM: correct OCR mistakes, then generate title / tags / summary.
    # Skipped (notes still save fine) if Ollama isn't running or ?llm=0 is sent.
    text, tags, summary, llm_title = raw_text, "", "", ""
    if request.form.get("llm", "1") != "0" and raw_text.strip() and llm_helper.is_available():
        t_llm = time.perf_counter()
        fixed = llm_helper.correct_text(raw_text)
        text = fixed["corrected"]
        meta = llm_helper.organize_note(text)
        tags = ", ".join(meta["tags"])
        summary = meta["summary"]
        llm_title = meta["title"]
        print("[llm] correction + organizing took %.1fs" % (time.perf_counter() - t_llm), flush=True)

    title = (request.form.get("title") or llm_title
             or os.path.splitext(f.filename)[0] or "Untitled note")
    nid = db.add(title, text, engine, os.path.basename(path),
                 tags=tags, summary=summary, original_text=raw_text)
    return jsonify(db.get(nid))


@app.get("/api/notes")
def notes():
    return jsonify(db.search(request.args.get("q", "")))


@app.route("/api/notes/<int:nid>", methods=["GET", "PUT", "DELETE"])
def note(nid):
    if request.method == "DELETE":
        db.delete(nid)
        return jsonify(ok=True)
    if request.method == "PUT":
        d = request.json
        db.update(nid, d["title"], d["text"], d.get("tags", ""), d.get("summary"))
    n = db.get(nid)
    return (jsonify(n), 200) if n else (jsonify(error="not found"), 404)


@app.get("/api/notes/<int:nid>/export/<fmt>")
def export(nid, fmt):
    n = db.get(nid)
    if not n or fmt not in FORMATS:
        return jsonify(error="bad request"), 404
    fn, mime = FORMATS[fmt]
    return send_file(io.BytesIO(fn(n)), mimetype=mime, as_attachment=True,
                     download_name="note_%d.%s" % (nid, fmt))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)