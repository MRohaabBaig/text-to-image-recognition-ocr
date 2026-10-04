"""
Project 4: Basic Text Recognition (OCR) Pipeline
DecodeLabs AI Internship - Optional Mastery Task

I built this to satisfy the four validation gates DecodeLabs set for
Project 4: library integration, pre-processing integrity, an 80%+
confidence benchmark, and a clean visual output.
"""

import os
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------
# CONFIG - I keep all the knobs up here so I do not have to hunt 
# through the rest of the script every time I want to tweak knobs.
# --------------------------------------------------------------------------
ASSETS_DIR = "assets"
OUTPUT_DIR = "output"
SAMPLE_IMAGE_PATH = os.path.join(ASSETS_DIR, "sample_input.jpg")
CONFIDENCE_THRESHOLD = 80
TESSERACT_PSM = 6

# On Windows, pytesseract can't find the Tesseract engine unless I point it
# to the .exe explicitly. Leaving this commented out because on Mac/Linux
# the engine is usually already on PATH after installing via brew/apt.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def ensure_dirs():
    """I call this first so the script never crashes on a missing folder."""
    os.makedirs(ASSETS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_sample_image(path):
    """
    I generate my own test image instead of requiring a supplied one.
    That way the script is fully self-contained and runs the moment
    it's pasted in, with zero extra setup besides installing deps.
    """
    width, height = 800, 300
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)

    # Pillow 10+ lets me size the built-in font directly, so I don't need
    # to ship a separate .ttf file with the project.
    font = ImageFont.load_default(size=48)

    lines = ["DecodeLabs AI Internship", "Project 4 - OCR Test", "Confidence Target: 80%"]
    y = 30
    for line in lines:
        draw.text((40, y), line, fill="black", font=font)
        y += 70

    image.save(path)
    return path


def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"I couldn't load an image from {path}")
    return image


def preprocess_image(image):
    """
    This is the pre-processing pipeline from my project brief: grayscale ->
    blur -> adaptive threshold. I save each intermediate step to disk so I
    have visual proof of the pre-processing for the milestone review.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "01_grayscale.jpg"), gray)

    # I use a small Gaussian blur to smooth out compression artifacts before
    # binarizing - it stops thresholding from picking up single-pixel noise
    # as if it were text.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding instead of a single global cutoff (like Otsu)
    # because it recalculates the black/white cutoff per local region, so
    # it holds up better if lighting isn't perfectly even across the image.
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,  # block size - must be odd; controls the size of the local region
        10,  # constant subtracted from the mean - fine-tunes sensitivity
    )
    cv2.imwrite(os.path.join(OUTPUT_DIR, "02_threshold.jpg"), thresh)

    return thresh


def run_ocr(processed_image):
    """
    I use image_to_data instead of image_to_string because I need
    per-word confidence scores to enforce my 80% gate - a plain string
    output doesn't give me that.
    """
    config = f"--oem 3 --psm {TESSERACT_PSM}"
    data = pytesseract.image_to_data(processed_image, config=config, output_type=Output.DICT)
    return data


def filter_by_confidence(data, threshold=CONFIDENCE_THRESHOLD):
    """
    This is my confidence gate. I drop anything below the threshold so a
    low-confidence misread doesn't make it into my final output.
    """
    kept_words = []
    kept_boxes = []
    kept_confidences = []

    for i, word in enumerate(data["text"]):
        confidence = float(data["conf"][i])
        # Tesseract returns -1 for regions it didn't attempt to read at all,
        # so I filter those out along with genuinely empty strings.
        if word.strip() and confidence >= threshold:
            kept_words.append(word)
            kept_confidences.append(confidence)
            box = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
            kept_boxes.append(box)

    return kept_words, kept_boxes, kept_confidences


def draw_results(original_image, boxes, output_path):
    """I draw a green box around every word that passed my confidence gate,
    so the visual output doubles as proof the filter is actually working."""
    annotated = original_image.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 2)
    cv2.imwrite(output_path, annotated)


def main():
    ensure_dirs()

    print("Step 1: Generating my sample input image...")
    generate_sample_image(SAMPLE_IMAGE_PATH)

    print("Step 2: Loading the image...")
    image = load_image(SAMPLE_IMAGE_PATH)

    print("Step 3: Pre-processing (grayscale -> blur -> adaptive threshold)...")
    processed = preprocess_image(image)

    print("Step 4: Running OCR with pytesseract...")
    data = run_ocr(processed)

    print(f"Step 5: Filtering results below {CONFIDENCE_THRESHOLD}% confidence...")
    words, boxes, confidences = filter_by_confidence(data)

    output_path = os.path.join(OUTPUT_DIR, "03_ocr_result.jpg")
    draw_results(image, boxes, output_path)

    print("\n--- RESULTS ---")
    print("Extracted text:", " ".join(words) if words else "(nothing passed the confidence gate)")
    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        total_words = len([w for w in data["text"] if w.strip()])
        print(f"Average confidence: {avg_conf:.1f}%")
        print(f"Words kept: {len(words)} / {total_words}")
    else:
        print("No words passed the confidence threshold - try lowering "
              "CONFIDENCE_THRESHOLD or using a clearer image.")

    print(f"\nOutputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()