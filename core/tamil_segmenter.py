"""
HandNote OSS
Tamil Handwriting Segmenter

Goal:
    Tamil handwritten image
        -> clean image
        -> find handwriting components
        -> group components into text lines
        -> reject false/background lines
        -> split real lines into words
        -> return word crops

This module is used only by the Tamil OCR pipeline.
"""

import cv2
import numpy as np


# ============================================================
# 1. LOAD AND PREPROCESS IMAGE
# ============================================================

def _load_and_prepare(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    height, width = image.shape[:2]

    # --------------------------------------------------------
    # Resize only extremely large images
    # --------------------------------------------------------

    max_width = 1800

    if width > max_width:

        scale = max_width / float(width)

        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Correct uneven lighting / page shadows
    # --------------------------------------------------------

    background = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=25,
        sigmaY=25
    )

    normalized = cv2.divide(
        gray,
        background,
        scale=255
    )

    # --------------------------------------------------------
    # Small blur
    # --------------------------------------------------------

    normalized = cv2.GaussianBlur(
        normalized,
        (3, 3),
        0
    )

    # --------------------------------------------------------
    # Convert handwriting to white pixels
    # --------------------------------------------------------

    _, binary = cv2.threshold(
        normalized,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # --------------------------------------------------------
    # Remove isolated tiny noise
    # --------------------------------------------------------

    number_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    cleaned = np.zeros_like(binary)

    for index in range(1, number_labels):

        area = stats[
            index,
            cv2.CC_STAT_AREA
        ]

        width_component = stats[
            index,
            cv2.CC_STAT_WIDTH
        ]

        height_component = stats[
            index,
            cv2.CC_STAT_HEIGHT
        ]

        # Tiny dots caused by camera/page noise
        if area < 6:
            continue

        if (
            width_component <= 1
            and
            height_component <= 2
        ):
            continue

        cleaned[
            labels == index
        ] = 255

    return image, cleaned


# ============================================================
# 2. FIND HANDWRITING COMPONENTS
# ============================================================

def _find_components(binary):

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    components = []

    image_height, image_width = binary.shape

    for contour in contours:

        x, y, w, h = cv2.boundingRect(
            contour
        )

        area = cv2.contourArea(
            contour
        )

        # ----------------------------------------------------
        # Reject microscopic noise
        # ----------------------------------------------------

        if area < 4:
            continue

        if w < 2 and h < 2:
            continue

        # ----------------------------------------------------
        # Reject giant page-border regions
        # ----------------------------------------------------

        if (
            w > image_width * 0.90
            and
            h > image_height * 0.90
        ):
            continue

        components.append(
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,

                "x2": x + w,
                "y2": y + h,

                "cx": x + (w / 2.0),
                "cy": y + (h / 2.0),

                "area": max(
                    1,
                    int(area)
                )
            }
        )

    return components


# ============================================================
# 3. ESTIMATE HANDWRITING SIZE
# ============================================================

def _estimate_character_height(components):

    heights = [
        component["h"]
        for component in components
        if component["h"] >= 4
    ]

    if not heights:
        return 20.0

    return float(
        np.median(heights)
    )


# ============================================================
# 4. GROUP COMPONENTS INTO TEXT LINES
# ============================================================

def _group_into_lines(
    components,
    char_height
):

    if not components:
        return []

    components = sorted(
        components,
        key=lambda item: (
            item["cy"],
            item["x"]
        )
    )

    lines = []

    # --------------------------------------------------------
    # Tamil characters contain upper/lower marks.
    #
    # Therefore this tolerance deliberately allows components
    # above/below the main character body to remain in one line.
    # --------------------------------------------------------

    vertical_tolerance = max(
        18.0,
        char_height * 2.4
    )

    for component in components:

        best_line = None
        best_distance = None

        for line in lines:

            distance = abs(
                component["cy"]
                -
                line["center_y"]
            )

            overlap = max(
                0,
                min(
                    component["y2"],
                    line["bottom"]
                )
                -
                max(
                    component["y"],
                    line["top"]
                )
            )

            overlap_ratio = (
                overlap
                /
                max(
                    1,
                    component["h"]
                )
            )

            belongs = (
                distance <= vertical_tolerance
                or
                overlap_ratio >= 0.15
            )

            if belongs:

                if (
                    best_distance is None
                    or
                    distance < best_distance
                ):

                    best_line = line
                    best_distance = distance

        # ----------------------------------------------------
        # Create new line
        # ----------------------------------------------------

        if best_line is None:

            lines.append(
                {
                    "components": [
                        component
                    ],

                    "top":
                        component["y"],

                    "bottom":
                        component["y2"],

                    "center_y":
                        component["cy"]
                }
            )

        # ----------------------------------------------------
        # Add component to existing line
        # ----------------------------------------------------

        else:

            best_line[
                "components"
            ].append(
                component
            )

            best_line["top"] = min(
                best_line["top"],
                component["y"]
            )

            best_line["bottom"] = max(
                best_line["bottom"],
                component["y2"]
            )

            centers = [
                item["cy"]
                for item
                in best_line["components"]
            ]

            best_line["center_y"] = float(
                np.median(centers)
            )

    # --------------------------------------------------------
    # Second pass:
    # merge groups that are vertically extremely close.
    # --------------------------------------------------------

    lines.sort(
        key=lambda line:
            line["center_y"]
    )

    merged = []

    merge_distance = max(
        22.0,
        char_height * 2.8
    )

    for line in lines:

        if not merged:

            merged.append(line)
            continue

        previous = merged[-1]

        distance = abs(
            line["center_y"]
            -
            previous["center_y"]
        )

        if distance <= merge_distance:

            previous[
                "components"
            ].extend(
                line["components"]
            )

            previous["top"] = min(
                previous["top"],
                line["top"]
            )

            previous["bottom"] = max(
                previous["bottom"],
                line["bottom"]
            )

            centers = [
                item["cy"]
                for item
                in previous["components"]
            ]

            previous["center_y"] = float(
                np.median(centers)
            )

        else:

            merged.append(line)

    for line in merged:

        line["components"].sort(
            key=lambda item:
                item["x"]
        )

    return merged


# ============================================================
# 5. FILTER FALSE / BACKGROUND LINES
# ============================================================

def _filter_false_lines(
    lines,
    char_height,
    image_width
):
    """
    Remove weak groups caused by:

        page shadows
        folds
        faint background writing
        isolated noise

    IMPORTANT:
    We do NOT simply keep the first line.

    Real multi-line Tamil notes should still be possible.
    """

    if not lines:
        return []

    scored_lines = []

    for line in lines:

        components = line[
            "components"
        ]

        if not components:
            continue

        x1 = min(
            component["x"]
            for component in components
        )

        x2 = max(
            component["x2"]
            for component in components
        )

        line_width = max(
            1,
            x2 - x1
        )

        component_count = len(
            components
        )

        total_area = sum(
            component["area"]
            for component in components
        )

        substantial_components = [
            component
            for component in components
            if (
                component["h"]
                >= char_height * 0.45
                or
                component["w"]
                >= char_height * 0.35
            )
        ]

        substantial_count = len(
            substantial_components
        )

        # ----------------------------------------------------
        # Score line quality
        # ----------------------------------------------------

        score = (
            component_count
            +
            (substantial_count * 2.0)
            +
            min(
                line_width / max(
                    char_height,
                    1.0
                ),
                20.0
            )
            +
            min(
                total_area / 150.0,
                20.0
            )
        )

        scored_lines.append(
            {
                "line": line,
                "score": score,
                "components":
                    component_count,
                "substantial":
                    substantial_count,
                "width":
                    line_width,
                "area":
                    total_area
            }
        )

    if not scored_lines:
        return []

    # --------------------------------------------------------
    # Find strongest handwriting line.
    # --------------------------------------------------------

    strongest_score = max(
        item["score"]
        for item in scored_lines
    )

    filtered = []

    for item in scored_lines:

        score_ratio = (
            item["score"]
            /
            max(
                strongest_score,
                1.0
            )
        )

        component_count = item[
            "components"
        ]

        substantial_count = item[
            "substantial"
        ]

        line_width = item[
            "width"
        ]

        # ----------------------------------------------------
        # A candidate line should contain meaningful structure.
        #
        # Weak one-component / narrow page artifacts are
        # rejected unless their quality is comparable to the
        # strongest line.
        # ----------------------------------------------------

        enough_components = (
            component_count >= 3
        )

        enough_structure = (
            substantial_count >= 2
        )

        enough_width = (
            line_width
            >= max(
                char_height * 1.3,
                image_width * 0.025
            )
        )

        comparable_to_best = (
            score_ratio >= 0.40
        )

        keep = (
            enough_components
            and
            enough_structure
            and
            enough_width
            and
            comparable_to_best
        )

        print(
            "[tamil-segment] candidate "
            f"components={component_count}, "
            f"strong={substantial_count}, "
            f"width={line_width}px, "
            f"score={item['score']:.1f}, "
            f"ratio={score_ratio:.2f}, "
            f"keep={keep}",
            flush=True
        )

        if keep:

            filtered.append(
                item["line"]
            )

    # --------------------------------------------------------
    # Safety fallback.
    #
    # Never throw away everything.
    # --------------------------------------------------------

    if not filtered:

        strongest = max(
            scored_lines,
            key=lambda item:
                item["score"]
        )

        filtered = [
            strongest["line"]
        ]

        print(
            "[tamil-segment] "
            "filter fallback: keeping strongest line",
            flush=True
        )

    filtered.sort(
        key=lambda line:
            line["center_y"]
    )

    return filtered


# ============================================================
# 6. SPLIT ONE LINE INTO WORDS
# ============================================================

def _group_line_into_words(
    line,
    char_height
):

    components = sorted(
        line["components"],
        key=lambda item:
            item["x"]
    )

    if not components:
        return []

    if len(components) == 1:
        return [
            components
        ]

    # --------------------------------------------------------
    # Calculate gaps between components
    # --------------------------------------------------------

    positive_gaps = []

    for index in range(
        len(components) - 1
    ):

        current = components[
            index
        ]

        following = components[
            index + 1
        ]

        gap = (
            following["x"]
            -
            current["x2"]
        )

        if gap > 0:

            positive_gaps.append(
                gap
            )

    # --------------------------------------------------------
    # Dynamic word-gap threshold
    # --------------------------------------------------------

    if positive_gaps:

        median_gap = float(
            np.median(
                positive_gaps
            )
        )

        gap_threshold = max(
            12.0,
            char_height * 0.85,
            median_gap * 2.2
        )

    else:

        gap_threshold = max(
            12.0,
            char_height * 0.85
        )

    words = []

    current_word = [
        components[0]
    ]

    for index in range(
        1,
        len(components)
    ):

        previous = components[
            index - 1
        ]

        current = components[
            index
        ]

        gap = (
            current["x"]
            -
            previous["x2"]
        )

        if gap > gap_threshold:

            words.append(
                current_word
            )

            current_word = [
                current
            ]

        else:

            current_word.append(
                current
            )

    if current_word:

        words.append(
            current_word
        )

    return words


# ============================================================
# 7. CREATE WORD IMAGE
# ============================================================

def _crop_word(
    image,
    components
):

    x1 = min(
        item["x"]
        for item in components
    )

    y1 = min(
        item["y"]
        for item in components
    )

    x2 = max(
        item["x2"]
        for item in components
    )

    y2 = max(
        item["y2"]
        for item in components
    )

    word_height = max(
        1,
        y2 - y1
    )

    # --------------------------------------------------------
    # Tamil characters need generous padding because vowel
    # signs can extend beyond the main body.
    # --------------------------------------------------------

    pad_x = max(
        12,
        int(
            word_height * 0.35
        )
    )

    pad_y = max(
        12,
        int(
            word_height * 0.40
        )
    )

    image_height, image_width = (
        image.shape[:2]
    )

    x1 = max(
        0,
        x1 - pad_x
    )

    y1 = max(
        0,
        y1 - pad_y
    )

    x2 = min(
        image_width,
        x2 + pad_x
    )

    y2 = min(
        image_height,
        y2 + pad_y
    )

    return image[
        y1:y2,
        x1:x2
    ]


# ============================================================
# 8. MAIN FUNCTION
# ============================================================

def segment_tamil_words(
    image_path
):

    # --------------------------------------------------------
    # Prepare image
    # --------------------------------------------------------

    image, binary = (
        _load_and_prepare(
            image_path
        )
    )

    image_height, image_width = (
        image.shape[:2]
    )

    # --------------------------------------------------------
    # Find components
    # --------------------------------------------------------

    components = (
        _find_components(
            binary
        )
    )

    if not components:

        print(
            "[tamil-segment] "
            "no handwriting components found",
            flush=True
        )

        return []

    # --------------------------------------------------------
    # Estimate handwriting size
    # --------------------------------------------------------

    char_height = (
        _estimate_character_height(
            components
        )
    )

    print(
        "[tamil-segment] "
        f"{len(components)} component(s), "
        f"estimated component height="
        f"{char_height:.1f}px",
        flush=True
    )

    # --------------------------------------------------------
    # Find candidate lines
    # --------------------------------------------------------

    lines = _group_into_lines(
        components,
        char_height
    )

    print(
        "[tamil-segment] "
        f"{len(lines)} candidate line(s) "
        "before filtering",
        flush=True
    )

    # --------------------------------------------------------
    # Remove page/background false lines
    # --------------------------------------------------------

    lines = _filter_false_lines(
        lines,
        char_height,
        image_width
    )

    print(
        "[tamil-segment] "
        f"{len(lines)} real line(s) "
        "after filtering",
        flush=True
    )

    # --------------------------------------------------------
    # Split real lines into words
    # --------------------------------------------------------

    result = []

    for line_number, line in enumerate(
        lines,
        start=1
    ):

        word_groups = (
            _group_line_into_words(
                line,
                char_height
            )
        )

        word_images = []

        for word_components in word_groups:

            crop = _crop_word(
                image,
                word_components
            )

            if (
                crop is not None
                and
                crop.size > 0
            ):

                word_images.append(
                    crop
                )

        if word_images:

            result.append(
                word_images
            )

            print(
                "[tamil-segment] "
                f"line {line_number}: "
                f"{len(word_images)} word(s)",
                flush=True
            )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    total_words = sum(
        len(line)
        for line in result
    )

    print(
        "[tamil-segment] detected "
        f"{len(result)} line(s), "
        f"{total_words} word(s)",
        flush=True
    )

    return result