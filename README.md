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
After downloading, the app loads the model into memory once at startup and
keeps that pipeline for all subsequent scans. Startup shows an animated
“Loading model into memory…” indicator; scan preparation does not reload the
model.

During results review, select an image and drag a rectangle over its preview.
Use **Crop & Overwrite Original** to confirm and save the crop back to the
original file, or **Reset** to clear the selection. The crop button remains
disabled until a valid rectangle is selected.

The app remembers the last folder, recursive-scan setting, mode, and threshold
in `~/.baby_picture_scanner.json`. Press **Enter** to start a scan, **Delete**
to delete marked results, and double-click a result to open the image. Click
column headings to sort results.

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
large images before detection, and processes bounded eight-image inference
chunks so progress remains visible throughout large folders. Face crops are
batched through the classifier to reduce per-image overhead. PyTorch CPU
threads are capped to a small parallel count to avoid oversubscription.

The Windows bundle includes a small application icon. Deletion uses the system
trash when `send2trash` is available and otherwise offers a permanent-delete
fallback. Corrupt or unreadable image files are skipped and do not abort the
rest of a scan.

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
