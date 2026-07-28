# Baby Picture Scanner

A standalone Tkinter desktop utility, inspired by dupeGuru's scan-and-review
workflow. It uses the local HuggingFace `nateraw/vit-age-classifier` model to
score each image for the `0-2` age class.

## Run

Python 3.10+ and Tkinter are required. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

On startup the app checks the local HuggingFace cache. If the model is missing,
it downloads it in the background with a byte-level progress indicator; Scan
stays disabled until this finishes. Internet access is required only for this
first download, and a retry button is shown if it fails. Choose a folder, set
the confidence threshold, optionally include subfolders, and click **Scan**.
Results can be
reviewed with thumbnails, marked/unmarked, revealed in the file manager, or
moved to the desktop trash.

Supported formats are JPEG, PNG, WebP, and BMP. Classification is image-level,
with OpenCV's built-in Haar cascade used to detect faces first. Each detected
face is cropped with context around it and classified independently; the
strongest `0-2` score becomes the image confidence. Images with no detected
faces, or whose face crops produce no baby probability, fall back to
whole-image classification. The model is an age estimator rather than a
dedicated baby detector, so results should be reviewed before deletion.

## Headless smoke test

After installing dependencies:

```bash
python scripts/smoke_test.py
```

This creates a tiny synthetic-image fixture and verifies the scanner's file
discovery and classifier interface. The included `test_assets/` folder also
contains a few Wikimedia Commons test images used during verification:
`baby_sleeping.jpg` scored 94.6% as `0-2`, while the apple image scored 0% as a
baby. Face-level classification is also applied to the baby portrait test
image, improving results for faces that are small within a larger image.
Synthetic drawings are not expected to be classified as real babies; use real,
legally obtained photographs for an accuracy evaluation.
