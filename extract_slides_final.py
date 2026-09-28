import cv2
import numpy as np
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import sys
import time
import shutil


VIDEO = sys.argv[1]
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/data/output")

SLIDES_DIR = OUT / "slides"


# ============================================================
# CONTENT-FIRST / FAST EXTRACTOR
#
# PRIORITY:
#   1. Never lose genuine slide content
#   2. Never lose completed annotations
#   3. Keep chronological order
#   4. Avoid expensive feature matching
#
# Repeated zoom/pan/annotation states are acceptable.
# Missing content is NOT acceptable.
# ============================================================


SAMPLE_FPS = 2.0

# Comparison resolution.
# The original full-resolution frame is always saved.
COMPARE_WIDTH = 320
COMPARE_HEIGHT = 180

# A change larger than this is considered meaningful.
#
# Deliberately conservative so that genuine slides are
# not accidentally merged.
CHANGE_RATIO = 0.055
CHANGE_MEAN = 7.0

# Require the change to persist for several sampled frames.
# This removes most transition-animation noise.
CONFIRM_FRAMES = 3

# New state must remain reasonably stable before we accept it.
STABLE_RATIO = 0.045
STABLE_MEAN = 5.5

# Minimum time between accepted states.
# Kept low because we prefer extra states over lost content.
MIN_STATE_SECONDS = 0.75


# ============================================================
# OUTPUT CLEANUP
# ============================================================

if OUT.exists():
    for item in OUT.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

SLIDES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def format_time(seconds):
    seconds = int(max(0, seconds))

    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    return f"{h:02d}:{m:02d}:{s:02d}"


def compare_image(frame):
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    return cv2.resize(
        gray,
        (COMPARE_WIDTH, COMPARE_HEIGHT),
        interpolation=cv2.INTER_AREA
    )


def difference(a, b):
    diff = cv2.absdiff(a, b)

    ratio = float(
        np.mean(diff > 20)
    )

    mean = float(
        np.mean(diff)
    )

    return ratio, mean


def save_slide(frame, number):
    path = (
        SLIDES_DIR
        / f"slide_{number:04d}.png"
    )

    ok = cv2.imwrite(
        str(path),
        frame
    )

    if not ok:
        raise RuntimeError(
            f"Could not save slide: {path}"
        )

    return path


def make_pdf(slide_files):
    if not slide_files:
        return None

    pdf_path = (
        OUT
        / f"{Path(VIDEO).stem}.pdf"
    )

    first = cv2.imread(
        str(slide_files[0])
    )

    if first is None:
        return None

    height, width = first.shape[:2]

    pdf = canvas.Canvas(
        str(pdf_path),
        pagesize=(width, height)
    )

    for slide_file in slide_files:

        pdf.drawImage(
            ImageReader(
                str(slide_file)
            ),
            0,
            0,
            width=width,
            height=height
        )

        pdf.showPage()

    pdf.save()

    return pdf_path


def make_timestamps(slide_times):
    path = OUT / "timestamps.txt"

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for number, timestamp in slide_times:

            f.write(
                f"{number:04d}  "
                f"{format_time(timestamp)}\n"
            )

    return path


# ============================================================
# EXTRACTION
# ============================================================

def extract():

    wall_start = time.time()

    cap = cv2.VideoCapture(VIDEO)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    duration = (
        total_frames / fps
    )

    sample_every = max(
        1,
        int(round(
            fps / SAMPLE_FPS
        ))
    )

    print()
    print("=" * 78)
    print("FAST CONTENT-FIRST SLIDE EXTRACTION")
    print("=" * 78)
    print()

    print(
        f"Video duration : {format_time(duration)}"
    )

    print(
        f"Video FPS      : {fps:.2f}"
    )

    print(
        f"Sampling       : {SAMPLE_FPS:.1f} FPS"
    )

    print()

    print("IMPORTANT RULES:")
    print(
        "  ✓ Never use ORB/feature matching"
    )
    print(
        "  ✓ Never globally deduplicate slides"
    )
    print(
        "  ✓ Zoom/pan repetitions are allowed"
    )
    print(
        "  ✓ Annotation changes are preserved"
    )
    print(
        "  ✓ Latest state becomes the saved state"
    )
    print(
        "  ✓ Genuine changes are deliberately kept"
    )

    print()
    print("=" * 78)
    print()

    frame_number = 0

    current_frame = None
    current_compare = None
    current_start = 0.0

    candidate_frame = None
    candidate_compare = None
    candidate_start = None
    candidate_count = 0

    slide_number = 1

    slide_files = []
    slide_times = []

    last_saved_time = -999.0
    last_progress = -30.0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        timestamp = frame_number / fps

        if frame_number % sample_every != 0:
            frame_number += 1
            continue

        small = compare_image(frame)

        # ----------------------------------------------------
        # FIRST FRAME
        # ----------------------------------------------------

        if current_frame is None:

            current_frame = frame.copy()
            current_compare = small.copy()
            current_start = timestamp

            frame_number += 1
            continue

        # ----------------------------------------------------
        # COMPARE CURRENT STATE TO CURRENT FRAME
        # ----------------------------------------------------

        ratio, mean = difference(
            current_compare,
            small
        )

        meaningful_change = (
            ratio >= CHANGE_RATIO
            or
            mean >= CHANGE_MEAN
        )

        # ----------------------------------------------------
        # NO SIGNIFICANT CHANGE
        #
        # Still update the frame.
        #
        # This is critical:
        #
        # If someone is writing/highlighting on the slide,
        # the newest frame becomes the current final state.
        # ----------------------------------------------------

        if not meaningful_change:

            current_frame = frame.copy()
            current_compare = small.copy()

            candidate_frame = None
            candidate_compare = None
            candidate_start = None
            candidate_count = 0

        # ----------------------------------------------------
        # POSSIBLE NEW STATE
        # ----------------------------------------------------

        else:

            if candidate_frame is None:

                candidate_frame = frame.copy()
                candidate_compare = small.copy()
                candidate_start = timestamp
                candidate_count = 1

            else:

                # Check whether the candidate itself is stable.
                candidate_ratio, candidate_mean = (
                    difference(
                        candidate_compare,
                        small
                    )
                )

                candidate_is_stable = (
                    candidate_ratio <= STABLE_RATIO
                    and
                    candidate_mean <= STABLE_MEAN
                )

                if candidate_is_stable:

                    candidate_count += 1

                else:

                    # New change is still happening.
                    # Keep the newest frame as the candidate.
                    candidate_frame = frame.copy()
                    candidate_compare = small.copy()
                    candidate_start = timestamp
                    candidate_count = 1

            # ------------------------------------------------
            # CONFIRM NEW STATE
            # ------------------------------------------------

            candidate_duration = (
                timestamp - candidate_start
                if candidate_start is not None
                else 0
            )

            if (
                candidate_count >= CONFIRM_FRAMES
                and
                candidate_duration >= MIN_STATE_SECONDS
                and
                timestamp - last_saved_time
                >= MIN_STATE_SECONDS
            ):

                # ------------------------------------------------
                # SAVE THE LAST COMPLETE STATE
                #
                # This is the key requirement.
                #
                # Everything that happened to the previous state
                # up until this transition is represented by
                # current_frame.
                # ------------------------------------------------

                path = save_slide(
                    current_frame,
                    slide_number
                )

                slide_files.append(path)

                slide_times.append(
                    (
                        slide_number,
                        current_start
                    )
                )

                print(
                    f"[{format_time(timestamp)}] "
                    f"Saved slide/state "
                    f"{slide_number:04d} "
                    f"(change={ratio:.3f}, "
                    f"mean={mean:.1f})"
                )

                slide_number += 1
                last_saved_time = timestamp

                # ------------------------------------------------
                # THE CANDIDATE BECOMES THE NEW CURRENT STATE
                # ------------------------------------------------

                current_frame = candidate_frame.copy()
                current_compare = candidate_compare.copy()
                current_start = candidate_start

                candidate_frame = None
                candidate_compare = None
                candidate_start = None
                candidate_count = 0

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if timestamp - last_progress >= 30:

            percent = (
                timestamp / duration * 100
                if duration > 0
                else 0
            )

            print(
                f"Progress: {percent:5.1f}% | "
                f"time {format_time(timestamp)} | "
                f"states captured {slide_number - 1}"
            )

            last_progress = timestamp

        frame_number += 1

    cap.release()

    # ========================================================
    # ALWAYS SAVE FINAL STATE
    # ========================================================

    if current_frame is not None:

        path = save_slide(
            current_frame,
            slide_number
        )

        slide_files.append(path)

        slide_times.append(
            (
                slide_number,
                current_start
            )
        )

        print(
            f"[END] Saved final state "
            f"{slide_number:04d}"
        )

    # ========================================================
    # PDF
    # ========================================================

    pdf_path = make_pdf(
        slide_files
    )

    timestamps_path = (
        make_timestamps(
            slide_times
        )
    )

    elapsed = (
        time.time() - wall_start
    )

    print()
    print("=" * 78)
    print("EXTRACTION COMPLETE")
    print("=" * 78)
    print()

    print(
        f"States/slides saved : {len(slide_files)}"
    )

    print(
        f"Processing time     : {format_time(elapsed)}"
    )

    print(
        f"Slides folder       : {SLIDES_DIR}"
    )

    if pdf_path:
        print(
            f"PDF                 : {pdf_path}"
        )

    print(
        f"Timestamps          : {timestamps_path}"
    )

    print()
    print("CONTENT-FIRST POLICY:")
    print(
        "  ✓ No feature matching"
    )
    print(
        "  ✓ No ORB"
    )
    print(
        "  ✓ No global duplicate removal"
    )
    print(
        "  ✓ Zoom repetitions allowed"
    )
    print(
        "  ✓ Pan repetitions allowed"
    )
    print(
        "  ✓ Annotation changes preserved"
    )
    print(
        "  ✓ Final annotated state preserved"
    )
    print(
        "  ✓ Chronological order preserved"
    )
    print(
        "  ✓ Full-resolution images saved"
    )
    print()
    print("=" * 78)


if __name__ == "__main__":
    extract()
