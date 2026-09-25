"""
Page cleanup + line segmentation for handwritten notes.

The important design rule in this file is:

    MASK IMAGE -> used to FIND handwriting
    GRAYSCALE IMAGE -> used by OCR

We do not feed the aggressively cleaned binary mask directly to TrOCR.

Pipeline:
    1. correct_orientation
    2. crop_paper
    3. resize page
    4. create lighting-normalized grayscale
    5. create ink mask
    6. remove ruled lines/noise from MASK ONLY
    7. detect line bounding boxes
    8. crop natural grayscale handwriting for OCR

Main entry point:
    prepare_lines(image_bgr)
"""

import os
import re

import cv2
import numpy as np
from PIL import Image


WORK_WIDTH = 1600


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def _rotate(img, angle_cw):
    if angle_cw == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    if angle_cw == 180:
        return cv2.rotate(img, cv2.ROTATE_180)

    if angle_cw == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return img


def _osd_rotation(img):
    """
    Ask Tesseract for an orientation hint.

    This is only used as a fallback.
    """

    try:
        import pytesseract

        gray = (
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img.ndim == 3
            else img
        )

        osd = pytesseract.image_to_osd(gray)

        m = re.search(r"Rotate:\s*(\d+)", osd)
        conf = re.search(
            r"Orientation confidence:\s*([\d.]+)",
            osd
        )

        if m:
            return (
                int(m.group(1)),
                float(conf.group(1)) if conf else 0.0,
            )

    except Exception:
        pass

    return None, 0.0


# ---------------------------------------------------------------------------
# Ink detection
# ---------------------------------------------------------------------------

def ink_mask(gray):
    """
    Produce:

        ink -> binary mask where handwriting is white (255)
        norm -> lighting-normalized grayscale image

    'ink' is used for detection.

    'norm' is suitable for OCR because it retains grayscale stroke
    information instead of converting everything into hard black/white.
    """

    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    # Estimate page background.
    background = cv2.medianBlur(
        cv2.dilate(
            gray,
            np.ones((7, 7), np.uint8),
        ),
        21,
    )

    # Correct uneven illumination.
    diff = 255 - cv2.absdiff(gray, background)

    norm = cv2.normalize(
        diff,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    # Detect dark writing.
    _, ink = cv2.threshold(
        norm,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    return ink, norm


# ---------------------------------------------------------------------------
# Orientation detection
# ---------------------------------------------------------------------------

def _line_peakiness(gray):
    """
    Measure how strongly horizontal text-line bands exist.
    """

    ink, _ = ink_mask(gray)

    profile = (ink > 0).sum(axis=1).astype(float)

    if profile.max() <= 0:
        return 0.0

    profile /= profile.max()

    return float(profile.std())


def _text_confidence(gray):
    """
    Use Tesseract only as a rough relative orientation check.
    """

    try:
        import pytesseract

        data = pytesseract.image_to_data(
            gray,
            config="--psm 6",
            output_type=pytesseract.Output.DICT,
        )

        confs = []

        for c in data["conf"]:
            c = str(c).strip()

            if c not in ("-1", ""):
                confs.append(float(c))

        if not confs:
            return -1.0, 0

        return sum(confs) / len(confs), len(confs)

    except Exception:
        return -1.0, 0


def correct_orientation(img):
    """
    Rotate page so text is approximately left-to-right.

    We intentionally keep this conservative because handwriting can
    confuse Tesseract's orientation detector.
    """

    MIN_WORDS = 4
    MIN_MARGIN = 15.0

    if max(img.shape[:2]) > 1200:
        small = cv2.resize(
            img,
            None,
            fx=0.5,
            fy=0.5,
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = img

    gray0 = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2GRAY,
    )

    rotated90 = _rotate(small, 90)

    gray90 = cv2.cvtColor(
        rotated90,
        cv2.COLOR_BGR2GRAY,
    )

    score0 = _line_peakiness(gray0)
    score90 = _line_peakiness(gray90)

    # Page appears to already be portrait/upright.
    if score90 <= score0 * 1.15:

        c0, n0 = _text_confidence(gray0)

        gray180 = cv2.rotate(
            gray0,
            cv2.ROTATE_180,
        )

        c180, n180 = _text_confidence(gray180)

        if (
            max(n0, n180) >= MIN_WORDS
            and c180 - c0 >= MIN_MARGIN
        ):
            return _rotate(img, 180)

        return img

    # Page appears sideways.
    rotated270 = _rotate(small, 270)

    gray270 = cv2.cvtColor(
        rotated270,
        cv2.COLOR_BGR2GRAY,
    )

    c90, n90 = _text_confidence(gray90)
    c270, n270 = _text_confidence(gray270)

    if (
        max(n90, n270) >= MIN_WORDS
        and abs(c90 - c270) >= MIN_MARGIN
    ):
        if c90 > c270:
            return _rotate(img, 90)

        return _rotate(img, 270)

    # Fall back to Tesseract OSD.
    angle, confidence = _osd_rotation(img)

    if angle in (90, 270) and confidence >= 1.0:
        return _rotate(img, angle)

    return _rotate(img, 90)


# ---------------------------------------------------------------------------
# Paper detection
# ---------------------------------------------------------------------------

def _order_points(pts):

    rect = np.zeros((4, 2), dtype="float32")

    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).ravel()

    rect[0] = pts[np.argmin(sums)]
    rect[2] = pts[np.argmax(sums)]

    rect[1] = pts[np.argmin(diffs)]
    rect[3] = pts[np.argmax(diffs)]

    return rect


def _four_point_warp(img, pts):

    tl, tr, br, bl = _order_points(pts)

    width = int(
        max(
            np.linalg.norm(br - bl),
            np.linalg.norm(tr - tl),
        )
    )

    height = int(
        max(
            np.linalg.norm(tr - br),
            np.linalg.norm(tl - bl),
        )
    )

    if width < 100 or height < 100:
        return img

    destination = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(
        np.array(
            [tl, tr, br, bl],
            dtype="float32",
        ),
        destination,
    )

    return cv2.warpPerspective(
        img,
        matrix,
        (width, height),
        borderValue=(255, 255, 255),
    )


def crop_paper(img):
    """
    Detect the largest paper-like area.

    If reliable paper detection fails, return the original image.
    """

    h, w = img.shape[:2]

    scale = min(
        1.0,
        800 / max(h, w),
    )

    if scale < 1:
        small = cv2.resize(
            img,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = img.copy()

    gray = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2GRAY,
    )

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    _, threshold = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_CLOSE,
        np.ones((15, 15), np.uint8),
    )

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return img

    contour = max(
        contours,
        key=cv2.contourArea,
    )

    area = cv2.contourArea(contour)

    if area < 0.25 * small.shape[0] * small.shape[1]:
        return img

    approx = cv2.approxPolyDP(
        contour,
        0.02 * cv2.arcLength(contour, True),
        True,
    )

    if len(approx) == 4:

        points = (
            approx
            .reshape(4, 2)
            .astype("float32")
            / scale
        )

        return _four_point_warp(
            img,
            points,
        )

    x, y, bw, bh = cv2.boundingRect(contour)

    return img[
        int(y / scale):int((y + bh) / scale),
        int(x / scale):int((x + bw) / scale),
    ]


# ---------------------------------------------------------------------------
# Mask cleanup
# ---------------------------------------------------------------------------

def remove_ruled_lines(
    ink,
    min_len_frac=1 / 12,
):
    """
    Remove notebook ruling from the DETECTION MASK.

    Important:
    this cleaned mask is used to LOCATE text.
    It is not used directly as the final TrOCR image.
    """

    _, width = ink.shape

    horizontal_size = max(
        int(width * min_len_frac),
        40,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (horizontal_size, 1),
    )

    rules = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        kernel,
    )

    rules = cv2.dilate(
        rules,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, 3),
        ),
    )

    cleaned = cv2.bitwise_and(
        ink,
        cv2.bitwise_not(rules),
    )

    # Only reconnect the segmentation mask.
    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, 3),
        ),
    )

    return cleaned


def remove_small_blobs(
    mask,
    min_area=12,
):
    """
    Remove tiny noise from the detection mask.

    12 is intentionally conservative so dots over i/j and punctuation
    have a better chance of surviving.
    """

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    )

    if count <= 1:
        return mask

    keep = (
        stats[:, cv2.CC_STAT_AREA]
        >= min_area
    )

    keep[0] = False

    return (
        keep[labels].astype(np.uint8)
        * 255
    )


# ---------------------------------------------------------------------------
# Line segmentation
# ---------------------------------------------------------------------------

def segment_lines(
    mask,
    norm,
    min_height=10,
    min_width=30,
    min_ink=0.004,
    pad_x=18,
    pad_y=10,
):
    """
    Detect separate handwritten text rows.

    mask:
        cleaned binary image, ink = 255.
        Used ONLY for locating handwriting.

    norm:
        normalized grayscale page.
        Actual OCR crops are taken from this image.

    Returns:
        list of PIL RGB images, one image per handwritten row.
    """

    if mask is None or mask.size == 0:
        return []

    if norm is None or norm.size == 0:
        return []

    height, width = mask.shape

    # ---------------------------------------------------------
    # 1. Horizontal projection
    # ---------------------------------------------------------
    #
    # Count how much handwriting exists in every image row.
    #
    profile = np.count_nonzero(mask, axis=1).astype(np.float32)

    if profile.max() <= 0:
        return []

    # ---------------------------------------------------------
    # 2. Smooth VERY lightly
    # ---------------------------------------------------------
    #
    # Previously stronger smoothing could fill the whitespace
    # between two handwritten rows and make them look like one.
    #
    kernel_size = 3

    kernel = (
        np.ones(kernel_size, dtype=np.float32)
        / kernel_size
    )

    smooth = np.convolve(
        profile,
        kernel,
        mode="same",
    )

    # ---------------------------------------------------------
    # 3. Detect rows containing meaningful ink
    # ---------------------------------------------------------
    #
    # Keep this threshold low enough for thin handwriting.
    #
    threshold = max(
        1.0,
        smooth.max() * 0.018,
    )

    active = smooth > threshold

    # ---------------------------------------------------------
    # 4. Find continuous active regions
    # ---------------------------------------------------------

    raw_runs = []

    start = None

    for y, is_active in enumerate(active):

        if is_active and start is None:

            start = y

        elif not is_active and start is not None:

            raw_runs.append(
                [start, y]
            )

            start = None

    if start is not None:

        raw_runs.append(
            [start, height]
        )

    if not raw_runs:
        return []

    # ---------------------------------------------------------
    # 5. Estimate handwriting height
    # ---------------------------------------------------------
    #
    # This makes merging adaptive rather than using one fixed
    # merge_gap for every handwriting size.
    #

    candidate_heights = [
        y1 - y0
        for y0, y1 in raw_runs
        if y1 - y0 >= 3
    ]

    if candidate_heights:

        median_height = float(
            np.median(candidate_heights)
        )

    else:

        median_height = 15.0

    # Only merge VERY small internal gaps.
    #
    # Example:
    #
    # letter stroke
    #    2px gap       <- probably same text row
    # letter stroke
    #
    # But:
    #
    # first sentence
    #
    #       large gap  <- must remain separate
    #
    # second sentence

    merge_gap = max(
        2,
        min(
            6,
            int(median_height * 0.18),
        ),
    )

    merged = []

    for run in raw_runs:

        if not merged:

            merged.append(
                run.copy()
            )

            continue

        gap = (
            run[0]
            - merged[-1][1]
        )

        if gap <= merge_gap:

            merged[-1][1] = run[1]

        else:

            merged.append(
                run.copy()
            )

    # ---------------------------------------------------------
    # 6. Build OCR crops
    # ---------------------------------------------------------

    lines = []

    for y0, y1 in merged:

        if y1 - y0 < min_height:
            continue

        strip = mask[
            y0:y1,
            :
        ]

        columns = np.where(
            strip.any(axis=0)
        )[0]

        if len(columns) == 0:
            continue

        x0 = int(
            columns[0]
        )

        x1 = int(
            columns[-1] + 1
        )

        if x1 - x0 < min_width:
            continue

        ink_density = (
            strip[:, x0:x1] > 0
        ).mean()

        if ink_density < min_ink:
            continue

        # ---------------------------------------------
        # Add margins around the handwriting.
        # ---------------------------------------------

        crop_x0 = max(
            0,
            x0 - pad_x,
        )

        crop_x1 = min(
            width,
            x1 + pad_x,
        )

        crop_y0 = max(
            0,
            y0 - pad_y,
        )

        crop_y1 = min(
            height,
            y1 + pad_y,
        )

        # IMPORTANT:
        # Use natural normalized grayscale,
        # NOT the binary segmentation mask.
        crop = norm[
            crop_y0:crop_y1,
            crop_x0:crop_x1,
        ].copy()

        if crop.size == 0:
            continue

        # Gentle contrast enhancement.
        low = np.percentile(
            crop,
            1
        )

        high = np.percentile(
            crop,
            99
        )

        if high > low + 5:

            crop = np.clip(
                (crop.astype(np.float32) - low)
                * 255.0
                / (high - low),
                0,
                255,
            ).astype(np.uint8)

        # Add white breathing room.
        crop = cv2.copyMakeBorder(
            crop,
            8,
            8,
            14,
            14,
            cv2.BORDER_CONSTANT,
            value=255,
        )

        lines.append(
            Image.fromarray(
                crop
            ).convert("RGB")
        )

    return lines
# ---------------------------------------------------------------------------
# Page preparation
# ---------------------------------------------------------------------------

def get_page(image_bgr):
    """
    Orientation-correct, detect paper, and resize the page.
    """

    image_bgr = correct_orientation(
        image_bgr
    )

    page = crop_paper(
        image_bgr
    )

    if page is None or page.size == 0:
        page = image_bgr

    scale = (
        WORK_WIDTH
        / page.shape[1]
    )

    if scale < 1:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC

    page = cv2.resize(
        page,
        None,
        fx=scale,
        fy=scale,
        interpolation=interpolation,
    )

    h, w = page.shape[:2]

    margin_y = int(
        h * 0.015
    )

    margin_x = int(
        w * 0.015
    )

    # Avoid invalid slicing on unusual images.
    if (
        margin_y * 2 < h
        and margin_x * 2 < w
    ):
        page = page[
            margin_y:h - margin_y,
            margin_x:w - margin_x,
        ]

    return page


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def prepare_lines(
    image_bgr,
    debug_dir=None,
):
    """
    Complete local handwriting preprocessing pipeline.
    """

    page = get_page(
        image_bgr
    )

    gray = cv2.cvtColor(
        page,
        cv2.COLOR_BGR2GRAY,
    )

    # Generate detection mask + OCR-friendly grayscale.
    ink, norm = ink_mask(
        gray
    )

    # IMPORTANT:
    # cleaning happens to the detection mask.
    cleaned = remove_ruled_lines(
        ink
    )

    cleaned = remove_small_blobs(
        cleaned
    )

    # OCR crops come from norm, not cleaned.
    lines = segment_lines(
        cleaned,
        norm,
    )

    # ---------------------------------------------------------------
    # Debug output
    # ---------------------------------------------------------------

    if debug_dir:

        os.makedirs(
            debug_dir,
            exist_ok=True,
        )

        cv2.imwrite(
            os.path.join(
                debug_dir,
                "1_page.png",
            ),
            page,
        )

        cv2.imwrite(
            os.path.join(
                debug_dir,
                "2_normalized.png",
            ),
            norm,
        )

        cv2.imwrite(
            os.path.join(
                debug_dir,
                "3_ink_raw.png",
            ),
            255 - ink,
        )

        cv2.imwrite(
            os.path.join(
                debug_dir,
                "4_ink_cleaned.png",
            ),
            255 - cleaned,
        )

        for index, line in enumerate(lines):

            line.save(
                os.path.join(
                    debug_dir,
                    f"line_{index:02d}.png",
                )
            )

    return lines