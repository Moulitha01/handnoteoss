"""
HandNote OSS - Tamil Handwriting OCR Engine

Tamil pipeline:

    uploaded image
        ↓
    OpenCV segmentation
        ↓
    lines
        ↓
    words
        ↓
    Tamil handwriting model
        ↓
    reconstruct sentence

Model:
    sabaridsnfuji/Tamil_Offline_Handwritten_OCR

Everything runs locally after the model has been downloaded once.
"""

import os
import time
import threading

import cv2
import torch

from PIL import Image

from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

from core.tamil_segmenter import segment_tamil_words


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "sabaridsnfuji/Tamil_Offline_Handwritten_OCR"

ENCODER_NAME = "google/vit-base-patch16-224-in21k"

DECODER_NAME = "d42kw01f/Tamil-RoBERTa"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# MODEL STATE
# ============================================================

_processor = None
_model = None

_load_error = None
_loading = False

_model_lock = threading.Lock()


# ============================================================
# AVAILABILITY
# ============================================================

def tamil_available():
    return True


# ============================================================
# LOAD MODEL
# ============================================================

def _load_model():

    global _processor
    global _model
    global _load_error
    global _loading

    if (
        _processor is not None
        and _model is not None
    ):
        return _processor, _model

    with _model_lock:

        if (
            _processor is not None
            and _model is not None
        ):
            return _processor, _model

        _loading = True

        try:

            print(
                "[tamil-ocr] loading Tamil handwriting model...",
                flush=True
            )

            print(
                f"[tamil-ocr] device: {DEVICE}",
                flush=True
            )

            start = time.perf_counter()

            # --------------------------------------------
            # IMAGE PROCESSOR
            # --------------------------------------------

            image_processor = (
                AutoImageProcessor.from_pretrained(
                    ENCODER_NAME
                )
            )

            # --------------------------------------------
            # TOKENIZER
            # --------------------------------------------

            tokenizer = (
                AutoTokenizer.from_pretrained(
                    DECODER_NAME
                )
            )

            # --------------------------------------------
            # PROCESSOR
            # --------------------------------------------

            processor = TrOCRProcessor(
                image_processor=image_processor,
                tokenizer=tokenizer,
            )

            # --------------------------------------------
            # MODEL
            # --------------------------------------------

            model = (
                VisionEncoderDecoderModel
                .from_pretrained(
                    MODEL_NAME
                )
            )

            model.to(
                DEVICE
            )

            model.eval()

            # --------------------------------------------
            # TOKEN CONFIGURATION
            # --------------------------------------------

            model.config.decoder_start_token_id = (
                processor.tokenizer.cls_token_id
            )

            model.config.pad_token_id = (
                processor.tokenizer.pad_token_id
            )

            model.config.eos_token_id = (
                processor.tokenizer.sep_token_id
            )

            model.config.vocab_size = (
                model.config.decoder.vocab_size
            )

            _processor = processor
            _model = model

            _load_error = None

            elapsed = (
                time.perf_counter()
                - start
            )

            print(
                "[tamil-ocr] handwriting model ready "
                f"in {elapsed:.1f}s",
                flush=True
            )

        except Exception as error:

            _load_error = str(
                error
            )

            print(
                "[tamil-ocr] model loading failed:",
                error,
                flush=True
            )

            raise

        finally:

            _loading = False

    return _processor, _model


# ============================================================
# RECOGNIZE ONE WORD
# ============================================================

def _recognize_word(
    image,
    processor,
    model
):
    """
    Recognize one segmented Tamil word.

    image is an OpenCV BGR image.
    """

    if image is None:
        return ""

    if image.size == 0:
        return ""

    # OpenCV BGR -> RGB
    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(
        rgb
    )

    pixel_values = (
        processor(
            images=pil_image,
            return_tensors="pt"
        )
        .pixel_values
        .to(DEVICE)
    )

    with torch.inference_mode():

        generated_ids = model.generate(
            pixel_values,
            max_length=64,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    text = (
        processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        .strip()
    )

    return text


# ============================================================
# FALLBACK — WHOLE IMAGE
# ============================================================

def _recognize_whole_image(
    image_path,
    processor,
    model
):
    """
    If segmentation completely fails, run the model on the
    original image instead of returning nothing.
    """

    image = cv2.imread(
        image_path
    )

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    print(
        "[tamil-ocr] segmentation returned no words; "
        "using whole-image fallback",
        flush=True
    )

    return _recognize_word(
        image,
        processor,
        model
    )


# ============================================================
# MAIN OCR FUNCTION
# ============================================================

def extract_tamil_text(
    image_path
):
    """
    Recognize Tamil handwriting from an uploaded image.

    Full image
        -> segment lines
        -> segment words
        -> recognize every word
        -> reconstruct text
    """

    if not os.path.exists(
        image_path
    ):
        raise FileNotFoundError(
            f"Image does not exist: {image_path}"
        )

    processor, model = _load_model()

    print(
        "[tamil-ocr] processing:",
        os.path.basename(
            image_path
        ),
        flush=True
    )

    start = time.perf_counter()

    # ========================================================
    # SEGMENT
    # ========================================================

    lines = segment_tamil_words(
        image_path
    )

    # ========================================================
    # FALLBACK
    # ========================================================

    if not lines:

        text = _recognize_whole_image(
            image_path,
            processor,
            model
        )

        print(
            "[tamil-ocr] result:",
            text,
            flush=True
        )

        return text

    # ========================================================
    # RECOGNIZE EACH WORD
    # ========================================================

    recognized_lines = []

    total_words = sum(
        len(line)
        for line in lines
    )

    current_word = 0

    print(
        f"[tamil-ocr] recognizing "
        f"{total_words} segmented word(s)...",
        flush=True
    )

    for line_number, words in enumerate(
        lines,
        start=1
    ):

        recognized_words = []

        for word_number, word_image in enumerate(
            words,
            start=1
        ):

            current_word += 1
            debug_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "data",
                "tamil_debug"
            )

            os.makedirs(
                debug_dir,
                exist_ok=True
            )

            debug_path = os.path.join(
                debug_dir,
                f"line_{line_number}_word_{word_number}.png"
            )

            cv2.imwrite(
                debug_path,
                word_image
            )

            print(
                f"[tamil-debug] saved: {debug_path}",
                flush=True
            )

            word_start = (
                time.perf_counter()
            )

            word_start = (
                time.perf_counter()
            )

            text = _recognize_word(
                word_image,
                processor,
                model
            )

            word_elapsed = (
                time.perf_counter()
                - word_start
            )

            print(
                f"[tamil-ocr] "
                f"line {line_number}, "
                f"word {word_number} "
                f"({current_word}/{total_words}) "
                f"-> {text!r} "
                f"[{word_elapsed:.2f}s]",
                flush=True
            )

            if text:
                recognized_words.append(
                    text
                )

        if recognized_words:

            recognized_lines.append(
                " ".join(
                    recognized_words
                )
            )

    # ========================================================
    # RECONSTRUCT NOTE
    # ========================================================

    final_text = "\n".join(
        recognized_lines
    ).strip()

    # Safety fallback
    if not final_text:

        final_text = _recognize_whole_image(
            image_path,
            processor,
            model
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        "[tamil-ocr] final result:",
        final_text,
        flush=True
    )

    print(
        "[tamil-ocr] finished in "
        f"{elapsed:.2f}s",
        flush=True
    )

    return final_text


# ============================================================
# PRELOAD
# ============================================================

def preload_tamil():

    try:

        _load_model()

        return True

    except Exception:

        return False


# ============================================================
# STATUS
# ============================================================

def tamil_status():

    return {

        "available":
            tamil_available(),

        "loaded":
            (
                _processor is not None
                and
                _model is not None
            ),

        "loading":
            _loading,

        "device":
            str(DEVICE),

        "model":
            MODEL_NAME,

        "error":
            _load_error,
    }