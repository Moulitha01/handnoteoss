"""Basic image helpers used by ocr_engine.py.

  load_image(path)          -> BGR numpy image (EXIF rotation applied, huge photos downscaled)
  preprocess(img)           -> (gray, binary)  gray = lighting-corrected grey, binary = ink is 255
  segment_lines(gray, bin)  -> list of grayscale line images (numpy) - fallback line splitter

The smarter pipeline (paper crop, ruled-line removal, ...) lives in core/line_extractor.py.
"""
import cv2
import numpy as np
from PIL import Image, ImageOps

from .line_extractor import ink_mask, remove_ruled_lines, remove_small_blobs
from .line_extractor import segment_lines as _segment_cleaned

MAX_SIDE = 3000


def load_image(path):
    pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    binary, norm = ink_mask(gray)
    return norm, binary


def segment_lines(gray, binary):
    cleaned = remove_small_blobs(remove_ruled_lines(binary))
    lines = _segment_cleaned(cleaned, gray)
    return [np.array(im.convert("L")) for im in lines]