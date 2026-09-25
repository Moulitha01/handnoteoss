"""
HandNote OSS - OCR Orchestrator

Routes an uploaded handwritten-note image to the correct
local OCR engine.

Current engines:
    English -> TrOCR
    Tamil   -> Tamil Handwriting OCR

All OCR runs locally.
No cloud OCR API is required.
"""

import time

from .ocr_engine import (
    extract_text as extract_english_text,
    trocr_available,
)

from .tamil_ocr_engine import (
    extract_tamil_text,
    tamil_status,
)


# ============================================================
# LOGGING
# ============================================================

def _log(message):
    print(
        "[orchestrator] " + message,
        flush=True,
    )


# ============================================================
# ENGINE STATUS
# ============================================================

def get_engine_status():
    """
    Return the availability of the OCR engines.
    """

    tamil = tamil_status()

    return {
        "english": {
            "engine": "trocr",
            "available": trocr_available(),
        },

        "tamil": {
            "engine": "tamil",
            "available": tamil.get(
                "available",
                False,
            ),
            "loaded": tamil.get(
                "loaded",
                False,
            ),
            "loading": tamil.get(
                "loading",
                False,
            ),
            "error": tamil.get(
                "error",
                None,
            ),
        },
    }


# ============================================================
# ENGLISH OCR
# ============================================================

def _run_english(image_path):
    """
    Run the existing English TrOCR pipeline.
    """

    _log(
        "routing image -> English TrOCR"
    )

    text, engine = extract_english_text(
        image_path,
        "trocr",
    )

    return {
        "text": text,
        "engine": engine,
        "language": "english",
    }


# ============================================================
# TAMIL OCR
# ============================================================

def _run_tamil(image_path):
    """
    Run the Tamil handwriting OCR pipeline.
    """

    _log(
        "routing image -> Tamil handwriting OCR"
    )

    text = extract_tamil_text(
        image_path
    )

    return {
        "text": text,
        "engine": "tamil",
        "language": "tamil",
    }


# ============================================================
# MAIN ORCHESTRATOR
# ============================================================

def process_image(
    image_path,
    engine="trocr",
):
    """
    Main HandNote OCR orchestration function.

    Parameters
    ----------
    image_path:
        Path to the uploaded image.

    engine:
        "trocr"
            English handwriting.

        "english"
            Alias for TrOCR.

        "tamil"
            Tamil handwriting.

        "auto"
            Currently defaults to English TrOCR.

    Returns
    -------
    dict

    Example:

        {
            "text": "...",
            "engine": "trocr",
            "language": "english",
            "time": 8.2
        }
    """

    start = time.perf_counter()

    requested_engine = (
        engine or "trocr"
    ).strip().lower()

    _log(
        "request received: %r"
        % requested_engine
    )

    # --------------------------------------------------------
    # NORMALIZE ENGINE NAME
    # --------------------------------------------------------

    aliases = {
        "english": "trocr",
        "en": "trocr",
        "eng": "trocr",

        "tamil": "tamil",
        "ta": "tamil",

        "auto": "trocr",
    }

    selected_engine = aliases.get(
        requested_engine,
        requested_engine,
    )

    # --------------------------------------------------------
    # ROUTE TO OCR ENGINE
    # --------------------------------------------------------

    if selected_engine == "trocr":

        result = _run_english(
            image_path
        )

    elif selected_engine == "tamil":

        result = _run_tamil(
            image_path
        )

    else:

        raise ValueError(
            "Unsupported OCR engine: %s"
            % requested_engine
        )

    # --------------------------------------------------------
    # ADD PROCESSING INFORMATION
    # --------------------------------------------------------

    elapsed = (
        time.perf_counter()
        - start
    )

    result["time"] = round(
        elapsed,
        2,
    )

    result["requested_engine"] = (
        requested_engine
    )

    _log(
        "%s pipeline completed in %.2fs"
        % (
            result["language"],
            elapsed,
        )
    )

    return result