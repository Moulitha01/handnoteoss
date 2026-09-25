"""
HandNote OSS
Handwritten Notes Digitizer - Flask Web Application

Run:
    python app.py
"""

import os
import io
import uuid
import time

from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    send_file,
)

from core import database as db

# ------------------------------------------------------------
# OCR
# ------------------------------------------------------------

# We still import these because:
#   trocr_available() -> tells the frontend if TrOCR exists
#   preload()         -> loads English TrOCR in background
from core.ocr_engine import (
    trocr_available,
    preload,
)

# NEW:
# All OCR requests now go through the orchestrator.
from core.orchestrator import (
    process_image,
)

from core.exporter import FORMATS
from core import llm_helper


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# UPLOAD DIRECTORY
# ============================================================

UPLOADS = os.path.join(
    os.path.dirname(__file__),
    "data",
    "uploads",
)

os.makedirs(
    UPLOADS,
    exist_ok=True,
)


# ============================================================
# DATABASE
# ============================================================

db.init()


# ============================================================
# PRELOAD ENGLISH TrOCR
# ============================================================

# Loads TrOCR in a background thread so that the first
# English upload is faster.

preload()


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html",
        trocr=trocr_available(),
        llm=llm_helper.is_available(),
    )


# ============================================================
# OCR IMAGE UPLOAD
# ============================================================

@app.post("/api/upload")
def upload():

    # --------------------------------------------------------
    # GET IMAGE
    # --------------------------------------------------------

    # static/js/app.js sends the uploaded file using
    # the form-data key "image".

    f = request.files.get(
        "image"
    )

    if not f:

        return jsonify(
            error="no image"
        ), 400


    # --------------------------------------------------------
    # CREATE UNIQUE FILE NAME
    # --------------------------------------------------------

    ext = (
        os.path.splitext(
            f.filename
        )[1].lower()
        or ".png"
    )

    filename = (
        uuid.uuid4().hex
        + ext
    )

    path = os.path.join(
        UPLOADS,
        filename,
    )


    # --------------------------------------------------------
    # SAVE IMAGE LOCALLY
    # --------------------------------------------------------

    f.save(
        path
    )


    # ========================================================
    # OCR ORCHESTRATION
    # ========================================================

    try:

        # The frontend sends:
        #
        #   trocr -> English handwriting
        #   tamil -> Tamil handwriting

        requested_engine = (
            request.form.get(
                "engine",
                "trocr",
            )
        )

        print(
            "[app] OCR request:",
            requested_engine,
            flush=True,
        )


        # ----------------------------------------------------
        # NEW ORCHESTRATOR
        # ----------------------------------------------------
        #
        # OLD CODE:
        #
        # raw_text, engine = extract_text(
        #     path,
        #     requested_engine
        # )
        #
        # NEW CODE:
        #
        # app.py no longer decides how OCR works.
        # orchestrator.py handles that decision.
        # ----------------------------------------------------

        result = process_image(
            image_path=path,
            engine=requested_engine,
        )


        # ----------------------------------------------------
        # GET RESULT FROM ORCHESTRATOR
        # ----------------------------------------------------

        raw_text = result.get(
            "text",
            "",
        )

        engine = result.get(
            "engine",
            requested_engine,
        )

        language = result.get(
            "language",
            "",
        )

        processing_time = result.get(
            "time",
            0,
        )


        print(
            "[app] OCR completed:",
            f"engine={engine},",
            f"language={language},",
            f"time={processing_time}s",
            flush=True,
        )


    except Exception as e:

        print(
            "[ocr] error:",
            str(e),
            flush=True,
        )

        return jsonify(
            error=str(e)
        ), 500


    # ========================================================
    # OPTIONAL LOCAL LLM
    # ========================================================

    # The OCR result works without the LLM.
    #
    # If a local LLM is available and enabled,
    # it can:
    #
    #   - correct OCR text
    #   - generate title
    #   - generate tags
    #   - generate summary

    text = raw_text

    tags = ""

    summary = ""

    llm_title = ""


    if (
        request.form.get(
            "llm",
            "1",
        ) != "0"

        and raw_text.strip()

        and llm_helper.is_available()
    ):

        t_llm = (
            time.perf_counter()
        )


        # ----------------------------------------------------
        # CORRECT OCR TEXT
        # ----------------------------------------------------

        fixed = (
            llm_helper.correct_text(
                raw_text
            )
        )

        text = fixed[
            "corrected"
        ]


        # ----------------------------------------------------
        # ORGANIZE NOTE
        # ----------------------------------------------------

        meta = (
            llm_helper.organize_note(
                text
            )
        )


        tags = ", ".join(
            meta.get(
                "tags",
                [],
            )
        )


        summary = meta.get(
            "summary",
            "",
        )


        llm_title = meta.get(
            "title",
            "",
        )


        print(
            "[llm] correction + organizing took %.1fs"
            % (
                time.perf_counter()
                - t_llm
            ),
            flush=True,
        )


    # ========================================================
    # NOTE TITLE
    # ========================================================

    title = (
        request.form.get(
            "title"
        )

        or llm_title

        or os.path.splitext(
            f.filename
        )[0]

        or "Untitled note"
    )


    # ========================================================
    # SAVE NOTE TO SQLITE
    # ========================================================

    nid = db.add(
        title,
        text,
        engine,
        os.path.basename(
            path
        ),
        tags=tags,
        summary=summary,
        original_text=raw_text,
    )


    # ========================================================
    # RETURN NOTE TO FRONTEND
    # ========================================================

    return jsonify(
        db.get(
            nid
        )
    )


# ============================================================
# GET / SEARCH NOTES
# ============================================================

@app.get("/api/notes")
def notes():

    query = request.args.get(
        "q",
        "",
    )

    return jsonify(
        db.search(
            query
        )
    )


# ============================================================
# GET / UPDATE / DELETE NOTE
# ============================================================

@app.route(
    "/api/notes/<int:nid>",
    methods=[
        "GET",
        "PUT",
        "DELETE",
    ],
)
def note(nid):


    # ========================================================
    # DELETE NOTE
    # ========================================================

    if request.method == "DELETE":

        db.delete(
            nid
        )

        return jsonify(
            ok=True
        )


    # ========================================================
    # UPDATE NOTE
    # ========================================================

    if request.method == "PUT":

        data = (
            request.get_json(
                silent=True
            )
            or {}
        )


        existing = db.get(
            nid
        )


        if not existing:

            return jsonify(
                error="not found"
            ), 404


        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------

        title = data.get(
            "title",
            existing.get(
                "title",
                "Untitled note",
            ),
        )


        # ----------------------------------------------------
        # TEXT
        # ----------------------------------------------------

        text = data.get(
            "text",
            existing.get(
                "text",
                "",
            ),
        )


        # ----------------------------------------------------
        # TAGS
        # ----------------------------------------------------

        tags = data.get(
            "tags",
            existing.get(
                "tags",
                "",
            ),
        )


        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        summary = data.get(
            "summary",
            existing.get(
                "summary",
                "",
            ),
        )


        # ----------------------------------------------------
        # UPDATE DATABASE
        # ----------------------------------------------------

        db.update(
            nid,
            title,
            text,
            tags,
            summary,
        )


    # ========================================================
    # RETURN NOTE
    # ========================================================

    saved_note = db.get(
        nid
    )


    if saved_note:

        return jsonify(
            saved_note
        ), 200


    return jsonify(
        error="not found"
    ), 404


# ============================================================
# EXPORT NOTE
# ============================================================

@app.get(
    "/api/notes/<int:nid>/export/<fmt>"
)
def export(nid, fmt):


    # --------------------------------------------------------
    # GET NOTE
    # --------------------------------------------------------

    note_data = db.get(
        nid
    )


    # --------------------------------------------------------
    # VALIDATE FORMAT
    # --------------------------------------------------------

    if (
        not note_data
        or fmt not in FORMATS
    ):

        return jsonify(
            error="bad request"
        ), 404


    # --------------------------------------------------------
    # GET EXPORT FUNCTION
    # --------------------------------------------------------

    export_function, mime_type = (
        FORMATS[
            fmt
        ]
    )


    # --------------------------------------------------------
    # CREATE EXPORT FILE
    # --------------------------------------------------------

    file_data = (
        export_function(
            note_data
        )
    )


    # --------------------------------------------------------
    # SEND FILE
    # --------------------------------------------------------

    return send_file(

        io.BytesIO(
            file_data
        ),

        mimetype=mime_type,

        as_attachment=True,

        download_name=(
            "note_%d.%s"
            % (
                nid,
                fmt,
            )
        ),
    )


# ============================================================
# RUN FLASK APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
    )