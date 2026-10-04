# Project 4: Basic Text Recognition (OCR)

## Overview
This is my submission for Project 4 (Image or Text Recognition - Basic) at
DecodeLabs. It's a Python pipeline that takes a raw image, cleans it up,
runs Optical Character Recognition on it using Tesseract, and only keeps
results the model is at least 80% confident about.

The script is self-contained: it generates its own sample test image on
first run, so there's nothing you need to supply to see it work end to end.

## Features
- Generates a sample input image automatically (no external files needed)
- Full pre-processing pipeline: grayscale conversion -> Gaussian blur ->
  adaptive thresholding
- OCR via `pytesseract` (Google's Tesseract engine)
- Per-word confidence filtering (80% minimum, configurable)
- Draws bounding boxes on every word that passes the confidence gate
- Saves every intermediate step as an image, so you can see exactly what
  each stage of the pipeline did

## Installation

### 1. Install the Tesseract engine (required - not a pip package)
`pytesseract` is just a Python wrapper. The actual OCR engine has to be
installed separately:

- **Windows**: install the "UB Mannheim" Tesseract build, then uncomment
  and update the `tesseract_cmd` line near the top of `ocr_pipeline.py`
  with your install path.
- **macOS**: `brew install tesseract`
- **Linux (Debian/Ubuntu)**: `sudo apt install tesseract-ocr`

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

## Usage
```bash
python ocr_pipeline.py
```

That's it. On first run it will:
1. Create an `assets/` folder and generate a sample test image inside it.
2. Create an `output/` folder with three images: grayscale, thresholded,
   and the final result with green bounding boxes.
3. Print the extracted text and average confidence to the console.

## Expected output