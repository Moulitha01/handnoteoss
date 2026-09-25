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
    min_height=8,
    min_width=25,
    min_ink=0.002,
    pad_x=20,
    pad_y=8,
):
    """
    Detect handwritten rows and return one grayscale OCR crop per row.

    Uses connected-component vertical clustering first, which prevents two
    nearby handwritten rows from being merged by projection smoothing.
    """

    if mask is None or mask.size == 0 or norm is None or norm.size == 0:
        return []

    mask = (mask > 0).astype(np.uint8) * 255
    height, width = mask.shape[:2]

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    components = []
    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        if area < 8 or (w <= 1 and h <= 2):
            continue
        if h > height * 0.30:
            continue

        components.append({
            "x0": x, "y0": y, "x1": x + w, "y1": y + h,
            "cy": float(centroids[i][1]), "h": h, "area": area,
        })

    rows = []

    if components:
        hs = [
            c["h"] for c in components
            if 3 <= c["h"] <= max(8, int(height * 0.08))
        ]
        char_h = float(np.median(hs)) if hs else 18.0
        centre_tol = max(7.0, min(28.0, char_h * 0.80))

        for c in sorted(components, key=lambda z: (z["cy"], z["x0"])):
            best = None
            best_score = None

            for idx, row in enumerate(rows):
                overlap = max(
                    0, min(c["y1"], row["y1"]) - max(c["y0"], row["y0"])
                )
                smaller_h = max(1, min(c["h"], row["y1"] - row["y0"]))
                overlap_ratio = overlap / smaller_h
                centre_distance = abs(c["cy"] - row["cy"])

                if overlap_ratio >= 0.28 or centre_distance <= centre_tol:
                    score = centre_distance - overlap_ratio * char_h
                    if best_score is None or score < best_score:
                        best_score = score
                        best = idx

            if best is None:
                rows.append({
                    "x0": c["x0"], "y0": c["y0"],
                    "x1": c["x1"], "y1": c["y1"],
                    "cy": c["cy"], "weight": max(1, c["area"]),
                    "count": 1,
                })
            else:
                row = rows[best]
                old = row["weight"]
                add = max(1, c["area"])
                total = old + add
                row["cy"] = (row["cy"] * old + c["cy"] * add) / total
                row["weight"] = total
                row["x0"] = min(row["x0"], c["x0"])
                row["y0"] = min(row["y0"], c["y0"])
                row["x1"] = max(row["x1"], c["x1"])
                row["y1"] = max(row["y1"], c["y1"])
                row["count"] += 1

        rows.sort(key=lambda r: r["cy"])

        # Merge only fragments that are clearly on the same baseline.
        merged = []
        for row in rows:
            if not merged:
                merged.append(row.copy())
                continue

            prev = merged[-1]
            centre_gap = abs(row["cy"] - prev["cy"])
            overlap = max(
                0, min(row["y1"], prev["y1"]) - max(row["y0"], prev["y0"])
            )
            min_h = max(
                1, min(row["y1"] - row["y0"], prev["y1"] - prev["y0"])
            )
            overlap_ratio = overlap / min_h

            if centre_gap <= max(5.0, char_h * 0.38) and overlap_ratio >= 0.45:
                total = prev["weight"] + row["weight"]
                prev["cy"] = (
                    prev["cy"] * prev["weight"] + row["cy"] * row["weight"]
                ) / total
                prev["weight"] = total
                prev["x0"] = min(prev["x0"], row["x0"])
                prev["y0"] = min(prev["y0"], row["y0"])
                prev["x1"] = max(prev["x1"], row["x1"])
                prev["y1"] = max(prev["y1"], row["y1"])
                prev["count"] += row["count"]
            else:
                merged.append(row.copy())

        rows = [
            r for r in merged
            if (r["y1"] - r["y0"] >= 3)
            and ((r["x1"] - r["x0"] >= min_width) or r["count"] > 1)
        ]

    # Projection fallback.
    if not rows:
        profile = np.count_nonzero(mask, axis=1).astype(np.float32)
        if profile.max() <= 0:
            return []

        smooth = np.convolve(
            profile, np.ones(3, dtype=np.float32) / 3.0, mode="same"
        )
        active = smooth > max(1.0, smooth.max() * 0.012)

        runs = []
        run_start = None
        for y, on in enumerate(active):
            if on and run_start is None:
                run_start = y
            elif not on and run_start is not None:
                runs.append([run_start, y])
                run_start = None
        if run_start is not None:
            runs.append([run_start, height])

        for y0, y1 in runs:
            if y1 - y0 < 3:
                continue
            strip = mask[y0:y1, :]
            xs = np.where(strip.any(axis=0))[0]
            if len(xs):
                rows.append({
                    "x0": int(xs[0]), "y0": y0,
                    "x1": int(xs[-1] + 1), "y1": y1,
                    "cy": (y0 + y1) / 2.0,
                    "weight": int(np.count_nonzero(strip)), "count": 1,
                })

    lines = []

    for row in sorted(rows, key=lambda r: r["cy"]):
        x0, y0 = int(row["x0"]), int(row["y0"])
        x1, y1 = int(row["x1"]), int(row["y1"])

        if y1 - y0 < min_height and row.get("count", 1) < 2:
            continue
        if x1 - x0 < min_width:
            continue

        density = (mask[y0:y1, x0:x1] > 0).mean()
        if density < min_ink:
            continue

        cx0 = max(0, x0 - pad_x)
        cx1 = min(width, x1 + pad_x)
        cy0 = max(0, y0 - pad_y)
        cy1 = min(height, y1 + pad_y)

        crop = norm[cy0:cy1, cx0:cx1].copy()
        if crop.size == 0:
            continue

        low = float(np.percentile(crop, 1))
        high = float(np.percentile(crop, 99))
        if high > low + 5:
            crop = np.clip(
                (crop.astype(np.float32) - low) * 255.0 / (high - low),
                0, 255
            ).astype(np.uint8)

        crop = cv2.copyMakeBorder(
            crop, 10, 10, 18, 18,
            cv2.BORDER_CONSTANT, value=255
        )

        lines.append(Image.fromarray(crop).convert("RGB"))

    print(
        f"[segment] detected {len(lines)} handwriting line"
        f"{'' if len(lines) == 1 else 's'}",
        flush=True,
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