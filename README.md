# Baby Picture Scanner

A standalone Tkinter desktop utility, inspired by dupeGuru's scan-and-review
workflow. It uses the local HuggingFace `nateraw/vit-age-classifier` model to
score each image for babies or kids.

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

Supported formats are JPEG, PNG, WebP, and BMP. Classification is face-first,
with OpenCV's built-in Haar cascade used to detect faces. Each detected
face is cropped with context around it and classified independently; the
strongest face score becomes the image confidence; the whole image is also
classified as a candidate so sleeping or occluded faces cannot hide a strong
image-level result. The model is an age estimator rather than a dedicated baby
detector, so results should be reviewed before deletion.
The default scan mode is **Kids 8 and under**, which adds the model's `0-2`
and `3-9` probabilities. The `3-9` model class extends beyond age 8, so this
is an approximate range. **Babies (0-2)** mode uses only the `0-2` class.

Scanning prepares image decoding and face detection concurrently, downsizes
large images before detection, and batches face crops through the classifier
to reduce per-image overhead. PyTorch CPU threads are capped to a small
parallel count to avoid oversubscription.

## Headless smoke test

After installing dependencies:

```bash
python scripts/smoke_test.py
```

This creates a tiny synthetic-image fixture and verifies the scanner's file
discovery and classifier interface. The included `test_assets/` folder also
contains a few Wikimedia Commons test images used during verification:
`baby_sleeping.jpg` scored 94.6% in Babies mode and 97.6% in Kids mode, while
the apple image scored 0% and 13.5%, respectively. The Kids-mode apple score
comes from the model's broad `3-9` class on a non-face whole-image candidate.
Face-level classification is also applied to the baby portrait test image,
which scored 61.1% in Babies mode and 79.7% in Kids mode.
Synthetic drawings are not expected to be classified as real babies; use real,
legally obtained photographs for an accuracy evaluation.
