"""
Local OCR engines for HandNote OSS.

Engines:

    trocr
        Primary handwriting recognition engine.
        Runs locally using Hugging Face Transformers.

    tesseract
        Optional fallback for printed / very neat text.

For the Software Freedom Day build:
    - no cloud OCR is required
    - no API key is required
    - handwriting recognition can run locally
"""

import os
import time
import threading

import cv2
import numpy as np

from PIL import Image, ImageOps

from .preprocess import (
    load_image,
    preprocess,
    segment_lines,
)

from .line_extractor import (
    prepare_lines,
    get_page,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_DIR = os.environ.get(
    "TROCR_MODEL",
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "models",
        "trocr-finetuned",
    ),
)

BASE_MODEL = "microsoft/trocr-base-handwritten"

BATCH = int(
    os.environ.get(
        "TROCR_BATCH",
        "4",
    )
)

BEAMS = int(
    os.environ.get(
        "TROCR_BEAMS",
        "2",
    )
)

MAX_LINES = int(
    os.environ.get(
        "MAX_LINES",
        "40",
    )
)

_trocr = {}

_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(msg):

    print(
        "[ocr] " + msg,
        flush=True,
    )


# ---------------------------------------------------------------------------
# TrOCR loading
# ---------------------------------------------------------------------------

def trocr_available():

    try:

        import torch
        import transformers

        return True

    except Exception:

        return False


def _load_trocr():

    with _lock:

        if _trocr:
            return _trocr

        import torch

        from transformers import (
            TrOCRProcessor,
            VisionEncoderDecoderModel,
        )

        start = time.perf_counter()

        config_path = os.path.join(
            MODEL_DIR,
            "config.json",
        )

        has_finetuned = os.path.isfile(
            config_path
        )

        if has_finetuned:

            model_name = MODEL_DIR

            _log(
                "using fine-tuned model: %s"
                % model_name
            )

        else:

            model_name = BASE_MODEL

            _log(
                "no valid fine-tuned model at %s - using base model"
                % MODEL_DIR
            )

        try:

            processor = (
                TrOCRProcessor
                .from_pretrained(
                    model_name
                )
            )

            model = (
                VisionEncoderDecoderModel
                .from_pretrained(
                    model_name
                )
                .eval()
            )

        except Exception as exc:

            if model_name == BASE_MODEL:
                raise

            _log(
                "failed to load fine-tuned model (%s); "
                "falling back to base model: %s"
                % (
                    model_name,
                    exc,
                )
            )

            model_name = BASE_MODEL

            processor = (
                TrOCRProcessor
                .from_pretrained(
                    model_name
                )
            )

            model = (
                VisionEncoderDecoderModel
                .from_pretrained(
                    model_name
                )
                .eval()
            )

        # Optional CPU optimisation.
        if (
            os.environ.get(
                "TROCR_QUANTIZE"
            )
            == "1"
        ):

            model = (
                torch.quantization
                .quantize_dynamic(
                    model,
                    {torch.nn.Linear},
                    dtype=torch.qint8,
                )
            )

        _trocr.update(
            proc=processor,
            model=model,
            name=model_name,
        )

        _log(
            "TrOCR ready (%s) in %.1fs"
            % (
                model_name,
                time.perf_counter()
                - start,
            )
        )

    return _trocr


def preload():
    """
    Load TrOCR in background when server starts.
    """

    if trocr_available():

        threading.Thread(
            target=_load_trocr,
            daemon=True,
        ).start()


# ---------------------------------------------------------------------------
# TrOCR image preparation
# ---------------------------------------------------------------------------

def _to_pil(line):
    """
    Convert a line crop into an OCR-friendly RGB PIL image.

    IMPORTANT:

    The old implementation padded every line into a giant square.

        1400 x 90
             ↓
        1400 x 1400

    The processor would then shrink that square, making the handwriting
    occupy a very small vertical region.

    Here we keep a useful line-like aspect ratio instead.
    """

    if isinstance(
        line,
        Image.Image,
    ):

        pil = line.convert(
            "RGB"
        )

    else:

        arr = np.asarray(line)

        if arr.ndim == 2:

            pil = Image.fromarray(
                arr.astype(
                    np.uint8
                )
            ).convert(
                "RGB"
            )

        else:

            pil = Image.fromarray(
                cv2.cvtColor(
                    arr,
                    cv2.COLOR_BGR2RGB,
                )
            )

    # Remove unnecessary outer white space.
    gray = np.array(
        pil.convert("L")
    )

    # Detect non-white content conservatively.
    content = gray < 245

    ys, xs = np.where(
        content
    )

    if (
        len(xs) > 0
        and len(ys) > 0
    ):

        x0 = max(
            0,
            int(xs.min()) - 10,
        )

        x1 = min(
            pil.width,
            int(xs.max()) + 11,
        )

        y0 = max(
            0,
            int(ys.min()) - 8,
        )

        y1 = min(
            pil.height,
            int(ys.max()) + 9,
        )

        pil = pil.crop(
            (
                x0,
                y0,
                x1,
                y1,
            )
        )

    # Add modest white border.
    pil = ImageOps.expand(
        pil,
        border=12,
        fill="white",
    )

    # ---------------------------------------------------------------
    # Do NOT convert to a giant square.
    #
    # Keep the line aspect ratio while ensuring handwriting isn't
    # extremely tiny.
    # ---------------------------------------------------------------

    target_height = 96

    width, height = pil.size

    if height > 0:

        scale = (
            target_height
            / height
        )

        new_width = max(
            32,
            int(width * scale),
        )

        # Prevent pathological huge images.
        new_width = min(
            new_width,
            1600,
        )

        pil = pil.resize(
            (
                new_width,
                target_height,
            ),
            Image.Resampling.LANCZOS,
        )

    return pil


# ---------------------------------------------------------------------------
# TrOCR inference
# ---------------------------------------------------------------------------

def run_trocr(line_imgs):

    import torch

    if not line_imgs:
        return []

    trocr = _load_trocr()

    pils = [
        _to_pil(line)
        for line in line_imgs
    ]

    texts = []

    with torch.inference_mode():

        for index in range(
            0,
            len(pils),
            BATCH,
        ):

            start = (
                time.perf_counter()
            )

            batch = pils[
                index:index + BATCH
            ]

            pixel_values = (
                trocr["proc"](
                    images=batch,
                    return_tensors="pt",
                )
                .pixel_values
            )

            generated_ids = (
                trocr["model"]
                .generate(
                    pixel_values,
                    max_new_tokens=64,
                    num_beams=BEAMS,
                    early_stopping=True,
                )
            )

            decoded = (
                trocr["proc"]
                .batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )
            )

            for text in decoded:

                texts.append(
                    text.strip()
                )

            _log(
                "lines %d-%d of %d done (%.1fs)"
                % (
                    index + 1,
                    min(
                        index + BATCH,
                        len(pils),
                    ),
                    len(pils),
                    time.perf_counter()
                    - start,
                )
            )

    return texts


# ---------------------------------------------------------------------------
# Tesseract
# ---------------------------------------------------------------------------

def _configure_tesseract():

    import pytesseract

    current = (
        pytesseract
        .pytesseract
        .tesseract_cmd
    )

    if current not in (
        None,
        "tesseract",
    ):
        return

    env = os.environ.get(
        "TESSERACT_CMD"
    )

    if (
        env
        and os.path.isfile(env)
    ):

        pytesseract.pytesseract.tesseract_cmd = env

        return

    if os.name == "nt":

        candidates = [

            r"C:\Program Files\Tesseract-OCR\tesseract.exe",

            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",

            os.path.expandvars(
                r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"
            ),
        ]

        for candidate in candidates:

            if os.path.isfile(
                candidate
            ):

                pytesseract.pytesseract.tesseract_cmd = candidate

                return


def run_tesseract(
    lines,
    page_gray,
):
    """
    Optional engine for printed or very neat text.
    """

    import pytesseract

    _configure_tesseract()

    start = time.perf_counter()

    try:

        if lines:

            texts = []

            for line in lines:

                text = (
                    pytesseract
                    .image_to_string(
                        _to_pil(line),
                        config="--oem 1 --psm 7",
                    )
                    .strip()
                )

                texts.append(
                    text
                )

        else:

            _log(
                "no lines detected - falling back to whole-page Tesseract"
            )

            texts = (
                pytesseract
                .image_to_string(
                    page_gray,
                    config="--oem 1 --psm 6",
                )
                .splitlines()
            )

    except pytesseract.TesseractNotFoundError:

        raise RuntimeError(
            "Tesseract is not installed or could not be found."
        )

    _log(
        "tesseract done, %d line(s) (%.1fs)"
        % (
            len(lines)
            if lines
            else 1,
            time.perf_counter()
            - start,
        )
    )

    return texts


# ---------------------------------------------------------------------------
# Main OCR pipeline
# ---------------------------------------------------------------------------

def extract_text(
    image_path,
    engine="auto",
):
    """
    HandNote OSS local OCR pipeline.

    auto:
        Use TrOCR for handwriting.

    trocr:
        Force TrOCR.

    tesseract:
        Force Tesseract.

    Returns:
        (text, engine_used)
    """

    start = time.perf_counter()

    img = load_image(
        image_path
    )

    gray, binary = preprocess(
        img
    )

    requested = engine

    # ---------------------------------------------------------------
    # IMPORTANT CHANGE
    #
    # HandNote OSS is a HANDWRITING digitizer.
    # Therefore auto should use TrOCR.
    # ---------------------------------------------------------------

    if engine == "auto":

        if trocr_available():
            engine = "trocr"

        else:
            _log(
                "TrOCR unavailable - falling back to Tesseract"
            )

            engine = "tesseract"

    # Cloud OCR is intentionally not part of this build.
    if engine not in (
        "trocr",
        "tesseract",
    ):

        _log(
            "unsupported engine %r - using TrOCR"
            % engine
        )

        engine = (
            "trocr"
            if trocr_available()
            else "tesseract"
        )

    _log(
        "engine requested=%r -> using %r"
        % (
            requested,
            engine,
        )
    )

    # ---------------------------------------------------------------
    # Detect handwriting lines.
    # ---------------------------------------------------------------

    lines = prepare_lines(
        img
        if img.ndim == 3
        else cv2.cvtColor(
            img,
            cv2.COLOR_GRAY2BGR,
        )
    )

    # Old/fallback segmentation path.
    if not lines:

        _log(
            "smart line extraction found no lines; "
            "trying fallback segmentation"
        )

        lines = segment_lines(
            gray,
            binary,
        )

    _log(
        "found %d text lines (%.1fs preprocessing)"
        % (
            len(lines),
            time.perf_counter()
            - start,
        )
    )

    if len(lines) > MAX_LINES:

        _log(
            "too many lines - keeping first %d"
            % MAX_LINES
        )

        lines = lines[
            :MAX_LINES
        ]

    # ---------------------------------------------------------------
    # Recognition
    # ---------------------------------------------------------------

    if engine == "trocr":

        if not lines:

            _log(
                "no handwriting lines detected"
            )

            return "", engine

        texts = run_trocr(
            lines
        )

    else:

        texts = run_tesseract(
            lines,
            gray,
        )

    final_lines = []

    for text in texts:

        text = text.strip()

        if text:
            final_lines.append(
                text
            )

    result = "\n".join(
        final_lines
    )

    _log(
        "%s finished in %.1fs total"
        % (
            engine,
            time.perf_counter()
            - start,
        )
    )

    return result, engine