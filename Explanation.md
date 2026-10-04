# Code Walkthrough - Explanation.md

A plain-English walkthrough of `ocr_pipeline.py`, grouped by block since a
strict line-by-line table would be mostly repetition. Each row maps a piece
of code to what it does and the concept behind it.

| Code | What it does | Underlying concept |
|---|---|---|
| `import cv2` | Loads OpenCV, used for image loading, filtering, and drawing | OpenCV is the standard Python library for computer vision |
| `import pytesseract` / `from pytesseract import Output` | Loads the OCR wrapper and a helper that formats results as a Python dict | `pytesseract` calls the Tesseract OCR engine under the hood |
| `from PIL import Image, ImageDraw, ImageFont` | Loads Pillow, used only to *create* the sample test image | Pillow is a general image-creation/editing library, separate from OpenCV |
| `CONFIDENCE_THRESHOLD = 80` | Sets the minimum confidence percentage a word must have to be kept | This is the "confidence gate" - AI models output probabilities, not certainties, so you decide how much doubt is acceptable |
| `TESSERACT_PSM = 6` | Tells Tesseract to expect "a single uniform block of text" | PSM = Page Segmentation Mode; it tells the OCR engine what kind of layout to expect (a receipt vs. a book page vs. a license plate need different modes) |
| `ensure_dirs()` | Creates `assets/` and `output/` folders if they don't exist | Basic defensive programming - the script shouldn't crash just because a folder is missing |
| `generate_sample_image()` | Draws three lines of text onto a blank white image and saves it | This makes the script "self-contained" - it needs no external test file to demonstrate the pipeline |
| `Image.new("RGB", (width, height), color="white")` | Creates a blank white canvas | An image is fundamentally a grid of pixel values; this starts with every pixel set to white (255,255,255) |
| `ImageFont.load_default(size=48)` | Loads a built-in font at 48pt | Avoids needing to ship a font file - relies on Pillow's bundled default |
| `draw.text((40, y), line, ...)` | Writes each line of text onto the canvas | Simulates a real "document photo" for the pipeline to process |
| `load_image(path)` | Reads the image file into a NumPy array using `cv2.imread` | To a computer, an image *is* a NumPy array of numbers - this is the "Input" step of Input -> Process -> Output |
| `if image is None: raise FileNotFoundError(...)` | Fails loudly if the image didn't load | `cv2.imread` silently returns `None` on failure instead of raising an error, so I check manually |
| `cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)` | Converts the 3-channel color image to a single-channel grayscale image | Removes color information the OCR engine doesn't need, cutting the data to 1/3 the size and reducing noise from color variation |
| `cv2.GaussianBlur(gray, (5, 5), 0)` | Smooths the image slightly by averaging each pixel with its neighbors | Removes tiny specks of noise that could otherwise get mistaken for text edges |
| `cv2.adaptiveThreshold(...)` | Converts grayscale to pure black-and-white, using a *locally* calculated cutoff instead of one global cutoff | Adaptive thresholding handles uneven lighting better than a single global threshold (like Otsu's method) because it recalculates the black/white boundary for each local region of the image |
| `31` (block size argument) | Size of the local neighborhood used to calculate each region's threshold | Must be odd; a bigger block size adapts more slowly, a smaller one is more sensitive to local detail |
| `10` (constant argument) | A value subtracted from the local mean before comparing | Fine-tunes how aggressively pixels get pushed to black vs. white |
| `pytesseract.image_to_data(...)` | Runs Tesseract OCR and returns results with per-word bounding boxes and confidence scores | Unlike `image_to_string` (which just returns raw text), `image_to_data` exposes the confidence score needed for the gate |
| `config = f"--oem 3 --psm {TESSERACT_PSM}"` | `--oem 3` selects Tesseract's default OCR engine mode (LSTM-based); `--psm 6` sets the layout mode | OEM = OCR Engine Mode, controls which internal recognition engine runs; PSM controls layout assumptions |
| `for i, word in enumerate(data["text"]):` | Loops through every detected text region | `image_to_data` returns parallel lists - `data["text"][i]` and `data["conf"][i]` refer to the same word |
| `confidence = float(data["conf"][i])` | Converts Tesseract's confidence string to a number | Tesseract returns confidence as a string; `-1` means "not attempted," 0-100 means an actual confidence score |
| `if word.strip() and confidence >= threshold:` | Keeps a word only if it's non-empty AND meets the confidence bar | This is the actual "gatekeeper" logic - two conditions must both be true |
| `box = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])` | Reads the bounding box position and size for that word | Tesseract gives box coordinates in raw pixels already, unlike some deep-learning models that return normalized 0-1 coordinates |
| `cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 2)` | Draws a green rectangle from the top-left corner to the bottom-right corner | `(x, y)` is the top-left origin point; adding width/height gets the bottom-right corner; `(0, 200, 0)` is a BGR color (OpenCV uses BGR, not RGB) and `2` is the line thickness |
| `avg_conf = sum(confidences) / len(confidences)` | Computes the average confidence across all kept words | Gives a single number to report as "how confident was this run overall" |
| `if __name__ == "__main__":` | Ensures `main()` only runs when the script is executed directly | Standard Python convention - lets the file be imported elsewhere later without auto-running |